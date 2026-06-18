import argparse
import json
import os

import onnx
import onnxslim
import torch

from dataset import mag_pha_stft
from env import AttrDict
from models.model_g_map_se import MPNet

DEFAULT_ONNX_OPSET = min(26, onnx.defs.onnx_opset_version())


class GMapSEOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy_amp, noisy_pha, prior_embedding):
        noisy_wav = noisy_amp.new_zeros((noisy_amp.size(0), 1))
        amp_g, pha_g, _ = self.model(
            noisy_wav, noisy_amp, noisy_pha, prior_embedding
        )
        return amp_g, pha_g


def load_config(checkpoint_file):
    config_file = os.path.join(os.path.dirname(checkpoint_file), "config.json")
    with open(config_file, encoding="utf-8") as f:
        return AttrDict(json.load(f)), config_file


def load_model(h, checkpoint_file):
    model = MPNet(h)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model


def build_dummy_inputs(h, chunk_size):
    dummy_wav = torch.zeros(1, chunk_size)
    noisy_amp, noisy_pha, _ = mag_pha_stft(
        dummy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor
    )
    prior_embedding = torch.zeros(1, getattr(h, "embed_dim", 192))
    return noisy_amp, noisy_pha, prior_embedding


def save_metadata(args, h, config_file, onnx_path, slim_path, metadata_path):
    metadata = {
        "checkpoint_file": args.checkpoint_file,
        "source_config_file": config_file,
        "onnx_file": onnx_path,
        "slim_onnx_file": slim_path,
        "opset": args.opset,
        "chunk_size": args.chunk_size,
        "overlap_size": args.overlap_size,
        "sampling_rate": h.sampling_rate,
        "n_fft": h.n_fft,
        "hop_size": h.hop_size,
        "win_size": h.win_size,
        "compress_factor": h.compress_factor,
        "embed_dim": getattr(h, "embed_dim", 192),
        "ecapa_model_path": getattr(
            h, "ecapa_model_path", "ecapa_models/voxceleb_ECAPA512.onnx"
        ),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def export_onnx(args):
    h, config_file = load_config(args.checkpoint_file)
    args.chunk_size = args.chunk_size or h.segment_size
    args.overlap_size = args.overlap_size or min(h.win_size, args.chunk_size // 4)

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.overlap_size < 0:
        raise ValueError("--overlap_size cannot be negative")
    if args.overlap_size >= args.chunk_size:
        raise ValueError("--overlap_size must be smaller than --chunk_size")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model = GMapSEOnnxWrapper(load_model(h, args.checkpoint_file)).eval()
    noisy_amp, noisy_pha, prior_embedding = build_dummy_inputs(h, args.chunk_size)

    torch.onnx.export(
        model,
        (noisy_amp, noisy_pha, prior_embedding),
        args.output,
        input_names=["noisy_amp", "noisy_pha", "prior_embedding"],
        output_names=["amp_g", "pha_g"],
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=args.verify,
    )

    exported_model = onnx.load(args.output)
    onnx.checker.check_model(exported_model)

    slim_output = args.slim_output
    if slim_output:
        os.makedirs(os.path.dirname(slim_output) or ".", exist_ok=True)
        slimmed_model = onnxslim.slim(exported_model)
        if slimmed_model is None:
            raise RuntimeError("onnxslim returned no model")
        onnx.checker.check_model(slimmed_model)
        onnx.save(slimmed_model, slim_output)

    metadata_path = args.metadata or f"{args.output}.json"
    save_metadata(args, h, config_file, args.output, slim_output, metadata_path)
    print(f"Exported ONNX: {args.output}")
    if slim_output:
        print(f"Slimmed ONNX: {slim_output}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--output", default="onnx/g_map_se.onnx")
    parser.add_argument("--slim_output", default="onnx/g_map_se.slim.onnx")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--overlap_size", type=int, default=0)
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()
