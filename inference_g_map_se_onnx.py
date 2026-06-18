import argparse
import glob
import json
import os

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from rich.progress import track

from dataset import mag_pha_istft, mag_pha_stft


embedding_session = None


def load_metadata(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_providers(provider):
    if provider:
        return [provider]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_embedding_session(ecapa_model_path):
    global embedding_session
    if embedding_session is None:
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        embedding_session = ort.InferenceSession(
            ecapa_model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return embedding_session


def extract_noisy_embedding(input_file, metadata, allow_zero_fallback):
    ecapa_model_path = metadata["ecapa_model_path"]
    embed_dim = metadata["embed_dim"]
    sampling_rate = metadata["sampling_rate"]

    if not os.path.isfile(ecapa_model_path):
        if allow_zero_fallback:
            return np.zeros((1, embed_dim), dtype=np.float32)
        raise FileNotFoundError(
            "ECAPA ONNX model not found at '{}'. Download it or pass "
            "--allow_zero_embedding_fallback for a low-quality fallback.".format(
                ecapa_model_path
            )
        )

    waveform, sample_rate = torchaudio.load(input_file)
    if sample_rate != sampling_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sampling_rate
        )(waveform)

    waveform = waveform * (1 << 15)
    features = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=sampling_rate,
        window_type="hamming",
        use_energy=False,
    )
    features = features - torch.mean(features, dim=0)
    session = get_embedding_session(ecapa_model_path)
    return session.run(
        output_names=["embs"],
        input_feed={"feats": features.unsqueeze(0).numpy()},
    )[0].astype(np.float32)


def run_model(session, noisy_amp, noisy_pha, prior_embedding):
    amp_g, pha_g = session.run(
        output_names=["amp_g", "pha_g"],
        input_feed={
            "noisy_amp": noisy_amp.cpu().numpy().astype(np.float32),
            "noisy_pha": noisy_pha.cpu().numpy().astype(np.float32),
            "prior_embedding": prior_embedding.astype(np.float32),
        },
    )
    return torch.from_numpy(amp_g), torch.from_numpy(pha_g)


def enhance_chunk(session, noisy_wav, prior_embedding, metadata):
    noisy_wav = noisy_wav.unsqueeze(0)
    noisy_amp, noisy_pha, _ = mag_pha_stft(
        noisy_wav,
        metadata["n_fft"],
        metadata["hop_size"],
        metadata["win_size"],
        metadata["compress_factor"],
    )
    amp_g, pha_g = run_model(session, noisy_amp, noisy_pha, prior_embedding)
    audio_g = mag_pha_istft(
        amp_g,
        pha_g,
        metadata["n_fft"],
        metadata["hop_size"],
        metadata["win_size"],
        metadata["compress_factor"],
    )
    return audio_g.squeeze(0)


def enhance_chunk_to_length(
    session, noisy_wav, model_chunk_size, output_length, prior_embedding, metadata
):
    if noisy_wav.numel() < model_chunk_size:
        pad = noisy_wav.new_zeros(model_chunk_size - noisy_wav.numel())
        noisy_wav = torch.cat((noisy_wav, pad))

    enhanced = enhance_chunk(session, noisy_wav, prior_embedding, metadata)
    if enhanced.numel() < output_length:
        pad = enhanced.new_zeros(output_length - enhanced.numel())
        enhanced = torch.cat((enhanced, pad))

    return enhanced[:output_length]


def enhance_audio(session, noisy_wav, prior_embedding, metadata):
    chunk_size = metadata["chunk_size"]
    overlap_size = metadata["overlap_size"]
    audio_len = noisy_wav.size(0)
    if audio_len <= chunk_size:
        return enhance_chunk_to_length(
            session, noisy_wav, chunk_size, audio_len, prior_embedding, metadata
        )

    hop_size = chunk_size - overlap_size
    output = torch.zeros_like(noisy_wav)
    weight = torch.zeros_like(noisy_wav)

    for start in range(0, audio_len, hop_size):
        end = min(start + chunk_size, audio_len)
        chunk_len = end - start
        enhanced_chunk = enhance_chunk_to_length(
            session,
            noisy_wav[start:end],
            chunk_size,
            chunk_len,
            prior_embedding,
            metadata,
        )

        window = torch.ones(chunk_len)
        if overlap_size > 0:
            if start > 0:
                fade_len = min(overlap_size, window.numel())
                window[:fade_len] = torch.linspace(0, 1, fade_len + 2)[1:-1]
            if end < audio_len:
                fade_len = min(overlap_size, window.numel())
                window[-fade_len:] = torch.linspace(1, 0, fade_len + 2)[1:-1]

        output[start:end] += enhanced_chunk * window
        weight[start:end] += window

        if end == audio_len:
            break

    return output / weight.clamp_min(1e-8)


def inference(args):
    metadata = load_metadata(args.metadata)
    onnx_file = args.onnx_file or metadata.get("slim_onnx_file") or metadata["onnx_file"]
    session = ort.InferenceSession(onnx_file, providers=get_providers(args.provider))

    input_files = sorted(glob.glob(os.path.join(args.input_noisy_wavs_dir, "*.wav")))
    if not input_files:
        raise FileNotFoundError(
            "No .wav files found in '{}'".format(args.input_noisy_wavs_dir)
        )

    os.makedirs(args.output_dir, exist_ok=True)

    for input_file in track(input_files):
        prior_embedding = extract_noisy_embedding(
            input_file, metadata, args.allow_zero_embedding_fallback
        )
        noisy_wav, _ = librosa.load(input_file, sr=metadata["sampling_rate"])
        noisy_wav = torch.FloatTensor(noisy_wav)
        norm_factor = torch.sqrt(
            len(noisy_wav) / torch.sum(noisy_wav**2.0).clamp_min(1e-12)
        )
        audio_g = enhance_audio(
            session, noisy_wav * norm_factor, prior_embedding, metadata
        )
        audio_g = audio_g / norm_factor

        output_file = os.path.join(args.output_dir, os.path.basename(input_file))
        sf.write(
            output_file,
            audio_g.squeeze().cpu().numpy(),
            metadata["sampling_rate"],
            "PCM_16",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_noisy_wavs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata", default="onnx/g_map_se.onnx.json")
    parser.add_argument("--onnx_file", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--allow_zero_embedding_fallback", action="store_true")
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
