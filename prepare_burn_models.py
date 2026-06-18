"""Prepare Burn model files for the Rust inference app."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ECAPA_SOURCE = ROOT / "ecapa_models" / "voxceleb_ECAPA512.onnx"
ECAPA_SLIM = ROOT / "ecapa_models" / "voxceleb_ECAPA512_321.slim.onnx"


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def slim_ecapa() -> None:
    sys.path = [
        path for path in sys.path if path not in ("", ".", str(ROOT))
    ]
    from onnxslim import slim

    ECAPA_SLIM.parent.mkdir(parents=True, exist_ok=True)
    slim(
        str(ECAPA_SOURCE),
        str(ECAPA_SLIM),
        input_shapes=["feats:1,321,80"],
        model_check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enhancement_batch_size",
        type=int,
        default=1,
        help=(
            "Fixed batch size for the G-MaP-SE Burn model. Values greater "
            "than 1 export a separate batch-specific ONNX before conversion."
        ),
    )
    return parser.parse_args()


def prepare_enhancement(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("--enhancement_batch_size must be positive")

    if batch_size == 1:
        onnx_path = "onnx/g_map_se.slim.onnx"
        model_name = "g_map_se"
    else:
        onnx_path = f"onnx/g_map_se_b{batch_size}.slim.onnx"
        metadata_path = f"onnx/g_map_se_b{batch_size}.onnx.json"
        run(
            [
                sys.executable,
                "export_g_map_se_onnx.py",
                "--checkpoint_file",
                "ckpt/g_best",
                "--output",
                f"onnx/g_map_se_b{batch_size}.onnx",
                "--slim_output",
                onnx_path,
                "--metadata",
                metadata_path,
                "--batch_size",
                str(batch_size),
            ]
        )
        model_name = f"g_map_se_b{batch_size}"

    run(
        [
            sys.executable,
            "convert_onnx_to_burn.py",
            onnx_path,
            "--output_dir",
            "burn_models",
            "--model_name",
            model_name,
        ]
    )


def main() -> None:
    args = parse_args()
    slim_ecapa()
    prepare_enhancement(args.enhancement_batch_size)
    run(
        [
            sys.executable,
            "convert_onnx_to_burn.py",
            str(ECAPA_SLIM),
            "--output_dir",
            "burn_models",
            "--model_name",
            "voxceleb_ecapa512",
        ]
    )


if __name__ == "__main__":
    main()
