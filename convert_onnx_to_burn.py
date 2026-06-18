"""Convert ONNX models to Burn-generated Rust code and .bpk weights."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_BURN_ONNX_VERSION = "0.21"
DEFAULT_TARGET_OPSET = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ONNX file to Burn model files by running "
            "tracel-ai/burn-onnx in a temporary Cargo project."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the ONNX model to convert.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("burn_models"),
        help="Directory where generated Burn files will be copied.",
    )
    parser.add_argument(
        "--model_name",
        help=(
            "Rust-safe output model name. Defaults to a sanitized ONNX "
            "filename stem."
        ),
    )
    parser.add_argument(
        "--burn_onnx_version",
        default=DEFAULT_BURN_ONNX_VERSION,
        help="burn-onnx crate version for the temporary Cargo project.",
    )
    parser.add_argument(
        "--target_opset",
        type=int,
        default=DEFAULT_TARGET_OPSET,
        help=(
            "Maximum ai.onnx opset to pass to burn-onnx. Models with a "
            "higher opset are converted with onnx.version_converter first."
        ),
    )
    parser.add_argument(
        "--no_opset_convert",
        action="store_true",
        help="Skip ONNX opset conversion and pass the input model as-is.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Generate development-mode Burn output.",
    )
    parser.add_argument(
        "--embed_states",
        action="store_true",
        help="Embed model weights into generated Rust code instead of .bpk.",
    )
    parser.add_argument(
        "--no_simplify",
        action="store_true",
        help="Disable burn-onnx graph simplification passes.",
    )
    parser.add_argument(
        "--no_partition",
        action="store_true",
        help="Disable burn-onnx submodule partitioning for large models.",
    )
    parser.add_argument(
        "--cargo",
        default="cargo",
        help="Cargo executable to use.",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep the temporary Cargo project for debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print Cargo output while building.",
    )
    return parser.parse_args()


def rust_bool(value: bool) -> str:
    return "true" if value else "false"


def sanitize_rust_identifier(name: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in name.lower())
    sanitized = "_".join(part for part in sanitized.split("_") if part)

    if not sanitized:
        sanitized = "model"

    if sanitized[0].isdigit():
        sanitized = f"model_{sanitized}"

    return sanitized


def get_ai_onnx_opset(model_path: Path) -> int | None:
    import onnx

    model = onnx.load(model_path)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version

    return None


def prepare_onnx_for_burn(
    input_path: Path,
    output_path: Path,
    target_opset: int,
    convert_opset: bool,
) -> int | None:
    if not convert_opset:
        shutil.copy2(input_path, output_path)
        return get_ai_onnx_opset(output_path)

    import onnx
    from onnx import shape_inference, version_converter

    source_opset = get_ai_onnx_opset(input_path)
    if source_opset is None or source_opset <= target_opset:
        shutil.copy2(input_path, output_path)
        return source_opset

    model = onnx.load(input_path)

    try:
        converted = version_converter.convert_version(model, target_opset)
        converted = shape_inference.infer_shapes(converted)
    except Exception as exc:
        msg = (
            f"Failed to convert ONNX opset {source_opset} to {target_opset}. "
            "Try re-exporting the model with a supported opset, or pass "
            "--no_opset_convert to let burn-onnx handle the file directly."
        )
        raise RuntimeError(msg) from exc

    onnx.save(converted, output_path)
    return target_opset


def write_cargo_project(
    project_dir: Path,
    model_name: str,
    burn_onnx_version: str,
    development: bool,
    embed_states: bool,
    simplify: bool,
    partition: bool,
) -> None:
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True)

    cargo_toml = f"""[package]
name = "burn-onnx-converter-{model_name.replace("_", "-")}"
version = "0.1.0"
edition = "2021"
publish = false

[build-dependencies]
burn-onnx = {json.dumps(burn_onnx_version)}
"""
    load_strategy = "Embedded" if embed_states else "File"

    build_rs = f"""use burn_onnx::{{LoadStrategy, ModelGen}};

fn main() {{
    ModelGen::new()
        .input({json.dumps(f"model/{model_name}.onnx")})
        .out_dir("model/")
        .development({rust_bool(development)})
        .simplify({rust_bool(simplify)})
        .partition({rust_bool(partition)})
        .load_strategy(LoadStrategy::{load_strategy})
        .run_from_script();
}}
"""

    (project_dir / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")
    (project_dir / "build.rs").write_text(build_rs, encoding="utf-8")
    (src_dir / "lib.rs").write_text("", encoding="utf-8")


def run_cargo_build(project_dir: Path, cargo: str, verbose: bool) -> None:
    command = [cargo, "build"]
    if not verbose:
        command.append("--quiet")

    result = subprocess.run(
        command,
        cwd=project_dir,
        check=False,
        text=True,
        capture_output=not verbose,
    )

    if result.returncode == 0:
        return

    detail = ""
    if result.stdout:
        detail += result.stdout
    if result.stderr:
        detail += result.stderr

    raise RuntimeError(f"cargo build failed:\n{detail.strip()}")


def find_generated_model_dir(project_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in (project_dir / "target" / "debug" / "build").glob("*/out/model")
        if path.is_dir()
    )

    if not candidates:
        raise FileNotFoundError(
            "burn-onnx did not produce target/debug/build/*/out/model."
        )

    return candidates[-1]


def copy_generated_files(generated_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for source in sorted(generated_dir.rglob("*")):
        if source.is_file():
            destination = output_dir / source.relative_to(generated_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)

    if not copied:
        raise FileNotFoundError(f"No generated files found in {generated_dir}.")

    return copied


def convert(args: argparse.Namespace) -> list[Path]:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"ONNX input file does not exist: {input_path}")

    if shutil.which(args.cargo) is None:
        raise FileNotFoundError(f"Cargo executable not found: {args.cargo}")

    model_name = sanitize_rust_identifier(args.model_name or input_path.stem)
    output_dir = args.output_dir.expanduser().resolve() / model_name

    temp_dir = Path(tempfile.mkdtemp(prefix="burn_onnx_"))
    try:
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        prepared_onnx = model_dir / f"{model_name}.onnx"

        opset = prepare_onnx_for_burn(
            input_path=input_path,
            output_path=prepared_onnx,
            target_opset=args.target_opset,
            convert_opset=not args.no_opset_convert,
        )

        write_cargo_project(
            project_dir=temp_dir,
            model_name=model_name,
            burn_onnx_version=args.burn_onnx_version,
            development=args.development,
            embed_states=args.embed_states,
            simplify=not args.no_simplify,
            partition=not args.no_partition,
        )
        run_cargo_build(temp_dir, args.cargo, args.verbose)

        generated_dir = find_generated_model_dir(temp_dir)
        copied = copy_generated_files(generated_dir, output_dir)

        if opset is not None:
            print(f"ai.onnx opset: {opset}")
        print(f"Burn output: {output_dir}")
        for path in copied:
            print(f"  {path}")

        return copied
    finally:
        if args.keep_temp:
            print(f"Temporary Cargo project kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    try:
        convert(parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
