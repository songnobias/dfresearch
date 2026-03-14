#!/usr/bin/env python3
"""
export.py — Export trained models to gasbench-compatible safetensors ZIP format.

Packages a trained model into the exact submission format required by
BitMind Subnet 34:

    my_detector.zip
    ├── model_config.yaml
    ├── model.py
    └── model.safetensors

Usage:
    uv run export.py --modality image --model efficientnet-b4
    uv run export.py --modality video --model r3d-18
    uv run export.py --modality audio --model wav2vec2
    uv run export.py --modality image --model efficientnet-b4 --weights path/to/model.safetensors
"""

import argparse
import shutil
import textwrap
import zipfile
from pathlib import Path

import yaml

# Maps model names to their source module paths for the model.py export
MODEL_MODULES = {
    "image": {
        "efficientnet-b4": "dfresearch.models.image.efficientnet",
        "clip-vit-l14": "dfresearch.models.image.clip_vit",
    },
    "video": {
        "r3d-18": "dfresearch.models.video.r3d",
        "videomae": "dfresearch.models.video.videomae",
    },
    "audio": {
        "wav2vec2": "dfresearch.models.audio.wav2vec2",
        "ast": "dfresearch.models.audio.ast_model",
    },
}

# Default preprocessing configs per modality
PREPROCESSING_CONFIGS = {
    "image": {
        "resize": [224, 224],
    },
    "video": {
        "resize": [224, 224],
        "num_frames": 16,
    },
    "audio": {
        "sample_rate": 16000,
        "duration_seconds": 6.0,
    },
}


def generate_model_config(modality: str, model_name: str) -> dict:
    """Generate model_config.yaml content."""
    return {
        "name": f"dfresearch-{model_name}",
        "version": "1.0.0",
        "modality": modality,
        "preprocessing": PREPROCESSING_CONFIGS[modality],
        "model": {
            "num_classes": 2,
            "weights_file": "model.safetensors",
        },
    }


def generate_model_py(modality: str, model_name: str) -> str:
    """
    Generate a standalone model.py for submission.

    This reads the original model source and appends only the load_model function
    that gasbench requires, making it self-contained.
    """
    module_name = MODEL_MODULES[modality][model_name]
    module_path = Path("src") / module_name.replace(".", "/")
    source_file = module_path.with_suffix(".py")

    if source_file.exists():
        return source_file.read_text()

    raise FileNotFoundError(
        f"Model source not found: {source_file}\n"
        f"Expected module: {module_name}"
    )


def export_model(
    modality: str,
    model_name: str,
    weights_path: Path | None = None,
    output_dir: Path = Path("results/exports"),
) -> Path:
    """
    Export a trained model to a gasbench-compatible ZIP.

    Args:
        modality: "image", "video", or "audio".
        model_name: Model identifier (e.g., "efficientnet-b4").
        weights_path: Path to model.safetensors. If None, uses default checkpoint.
        output_dir: Directory to write the ZIP file.

    Returns:
        Path to the output ZIP file.
    """
    if weights_path is None:
        weights_path = Path("results/checkpoints") / modality / "model.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Run train_{modality}.py first to generate a checkpoint."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage files in a temp directory
    stage_dir = output_dir / f".stage-{modality}-{model_name}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    # 1. model_config.yaml
    config = generate_model_config(modality, model_name)
    with open(stage_dir / "model_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # 2. model.py
    model_py = generate_model_py(modality, model_name)
    (stage_dir / "model.py").write_text(model_py)

    # 3. model.safetensors
    shutil.copy2(weights_path, stage_dir / "model.safetensors")

    # 4. Create ZIP
    zip_name = f"{modality}_detector_{model_name}.zip"
    zip_path = output_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in stage_dir.iterdir():
            zf.write(f, f.name)

    shutil.rmtree(stage_dir)

    print(f"Exported: {zip_path}")
    print(f"  Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"    {info.filename:<30} {info.file_size:>10} bytes")

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to gasbench-compatible ZIP"
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "video", "audio"],
        help="Model modality",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., efficientnet-b4, clip-vit-l14, r3d-18, videomae, wav2vec2, ast)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to model.safetensors (default: results/checkpoints/{modality}/model.safetensors)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exports"),
        help="Output directory (default: results/exports)",
    )
    args = parser.parse_args()

    if args.modality not in MODEL_MODULES:
        parser.error(f"Unknown modality: {args.modality}")
    if args.model not in MODEL_MODULES[args.modality]:
        available = list(MODEL_MODULES[args.modality].keys())
        parser.error(f"Unknown {args.modality} model: {args.model}. Available: {available}")

    zip_path = export_model(
        modality=args.modality,
        model_name=args.model,
        weights_path=args.weights,
        output_dir=args.output_dir,
    )

    print(f"\nTo test locally with gasbench:")
    print(f"  gasbench run --{args.modality}-model {zip_path} --small")
    print(f"\nTo push to BitMind Subnet 34:")
    print(f"  gascli d push --{args.modality}-model {zip_path} --wallet-name <NAME> --wallet-hotkey <KEY>")


if __name__ == "__main__":
    main()
