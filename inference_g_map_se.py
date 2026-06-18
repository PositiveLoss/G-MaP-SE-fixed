from __future__ import absolute_import, division, print_function, unicode_literals
import sys

sys.path.append("..")
import glob
import os
import argparse
import json
import torch
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.model_g_map_se import MPNet
import soundfile as sf
from rich.progress import track

h = None
device = None
embedding_session = None
warned_missing_embedding_model = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_embedding_session():
    global embedding_session
    if embedding_session is None:
        ecapa_model_path = getattr(h, "ecapa_model_path", "ecapa_models/voxceleb_ECAPA512.onnx")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        embedding_session = ort.InferenceSession(
            ecapa_model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return embedding_session


def extract_noisy_embedding(input_file):
    global warned_missing_embedding_model
    ecapa_model_path = getattr(h, "ecapa_model_path", "ecapa_models/voxceleb_ECAPA512.onnx")
    if not os.path.isfile(ecapa_model_path):
        if not warned_missing_embedding_model:
            print(
                "Warning: ECAPA model not found at '{}'; using checkpoint prior mean."
                .format(ecapa_model_path)
            )
            warned_missing_embedding_model = True
        return None

    waveform, sample_rate = torchaudio.load(input_file)
    if sample_rate != h.sampling_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=h.sampling_rate
        )(waveform)

    waveform = waveform * (1 << 15)
    features = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=h.sampling_rate,
        window_type="hamming",
        use_energy=False,
    )
    features = features - torch.mean(features, dim=0)
    session = get_embedding_session()
    embedding = session.run(
        output_names=["embs"],
        input_feed={"feats": features.unsqueeze(0).numpy()},
    )[0].squeeze(0)
    embedding = torch.from_numpy(embedding).float().unsqueeze(0)
    return embedding.to(device)


def enhance_chunk(model, noisy_wav, prior_embedding):
    noisy_wav = noisy_wav.unsqueeze(0)
    noisy_amp, noisy_pha, _ = mag_pha_stft(
        noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor
    )
    amp_g, pha_g, _ = model(noisy_wav, noisy_amp, noisy_pha, prior_embedding)
    audio_g = mag_pha_istft(
        amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
    )
    return audio_g.squeeze(0)


def enhance_chunk_to_length(
    model, noisy_wav, model_chunk_size, output_length, prior_embedding
):
    if noisy_wav.numel() < model_chunk_size:
        pad = noisy_wav.new_zeros(model_chunk_size - noisy_wav.numel())
        noisy_wav = torch.cat((noisy_wav, pad))

    enhanced = enhance_chunk(model, noisy_wav, prior_embedding)
    if enhanced.numel() < output_length:
        pad = enhanced.new_zeros(output_length - enhanced.numel())
        enhanced = torch.cat((enhanced, pad))

    return enhanced[:output_length]


def enhance_audio(model, noisy_wav, chunk_size, overlap_size, prior_embedding):
    audio_len = noisy_wav.size(0)
    if audio_len <= chunk_size:
        return enhance_chunk_to_length(
            model, noisy_wav, chunk_size, audio_len, prior_embedding
        )

    hop_size = chunk_size - overlap_size
    output = torch.zeros_like(noisy_wav)
    weight = torch.zeros_like(noisy_wav)

    for start in range(0, audio_len, hop_size):
        end = min(start + chunk_size, audio_len)
        chunk_len = end - start
        enhanced_chunk = enhance_chunk_to_length(
            model, noisy_wav[start:end], chunk_size, chunk_len, prior_embedding
        )

        window = torch.ones(chunk_len, device=noisy_wav.device)
        if overlap_size > 0:
            if start > 0:
                fade_len = min(overlap_size, window.numel())
                window[:fade_len] = torch.linspace(
                    0, 1, fade_len + 2, device=noisy_wav.device
                )[1:-1]
            if end < audio_len:
                fade_len = min(overlap_size, window.numel())
                window[-fade_len:] = torch.linspace(
                    1, 0, fade_len + 2, device=noisy_wav.device
                )[1:-1]

        output[start:end] += enhanced_chunk * window
        weight[start:end] += window

        if end == audio_len:
            break

    return output / weight.clamp_min(1e-8)


def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict["generator"])

    input_files = sorted(glob.glob(os.path.join(a.input_noisy_wavs_dir, "*.wav")))
    if not input_files:
        raise FileNotFoundError(
            "No .wav files found in '{}'".format(a.input_noisy_wavs_dir)
        )

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    chunk_size = a.chunk_size or h.segment_size
    overlap_size = a.overlap_size or min(h.win_size, chunk_size // 4)
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if overlap_size < 0:
        raise ValueError("--overlap_size cannot be negative")
    if overlap_size >= chunk_size:
        raise ValueError("--overlap_size must be smaller than --chunk_size")

    with torch.no_grad():
        for input_file in track(input_files):
            prior_embedding = extract_noisy_embedding(input_file)
            noisy_wav, _ = librosa.load(input_file, sr=h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(
                len(noisy_wav) / torch.sum(noisy_wav**2.0).clamp_min(1e-12)
            )
            audio_g = enhance_audio(
                model,
                noisy_wav * norm_factor,
                chunk_size,
                overlap_size,
                prior_embedding,
            )
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, os.path.basename(input_file))

            sf.write(
                output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, "PCM_16"
            )


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_noisy_wavs_dir", default="VoiceBank+DEMAND/wav_noisy")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Inference chunk size in samples. Defaults to config segment_size.",
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=0,
        help="Chunk overlap in samples. Defaults to min(config win_size, chunk_size / 4).",
    )
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(a)


if __name__ == "__main__":
    main()
