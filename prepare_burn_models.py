"""Prepare Burn model files for the Rust inference app."""

from __future__ import annotations

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


def main() -> None:
    slim_ecapa()
    run(
        [
            sys.executable,
            "convert_onnx_to_burn.py",
            "onnx/g_map_se.slim.onnx",
            "--output_dir",
            "burn_models",
            "--model_name",
            "g_map_se",
        ]
    )
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
