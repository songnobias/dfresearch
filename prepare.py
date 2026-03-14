#!/usr/bin/env python3
"""
prepare.py — Data preparation and evaluation utilities for dfresearch.

THIS FILE IS READ-ONLY. Do not modify it during autoresearch experiments.
It contains fixed evaluation metrics, data downloading, and constants that
ensure experiments are fairly comparable.

Usage:
    uv run prepare.py                         # Download all modalities
    uv run prepare.py --modality image        # Download image datasets only
    uv run prepare.py --modality video        # Download video datasets only
    uv run prepare.py --modality audio        # Download audio datasets only
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, brier_score_loss, accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# Constants (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────

TIME_BUDGET = 600             # 10-minute training budget (wall clock)
TARGET_IMAGE_SIZE = (224, 224)
TARGET_VIDEO_SIZE = (224, 224)
NUM_VIDEO_FRAMES = 16
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 6.0          # seconds
AUDIO_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)  # 96000

DEFAULT_IMAGE_BATCH_SIZE = 32
DEFAULT_VIDEO_BATCH_SIZE = 4
DEFAULT_AUDIO_BATCH_SIZE = 16

LABEL_REAL = 0
LABEL_SYNTHETIC = 1


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sn34_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 1.2,
    beta: float = 1.8,
) -> dict:
    """
    Compute the sn34_score used by BitMind Subnet 34.

    sn34_score = sqrt(MCC_norm^alpha * Brier_norm^beta)

    where:
        MCC_norm = (MCC + 1) / 2        (normalized to [0, 1])
        Brier_norm = 1 - brier_score    (higher is better)

    Args:
        y_true: Ground truth labels (0 = real, 1 = synthetic).
        y_prob: Predicted probability of synthetic class.
        alpha: Exponent for MCC component (default: 1.2).
        beta: Exponent for Brier component (default: 1.8).

    Returns:
        Dictionary with sn34_score, mcc, brier, accuracy, and component scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    mcc_norm = (mcc + 1.0) / 2.0
    brier_norm = 1.0 - brier

    # Avoid domain errors with very small values
    mcc_norm = max(mcc_norm, 1e-10)
    brier_norm = max(brier_norm, 1e-10)

    sn34 = (mcc_norm ** alpha * brier_norm ** beta) ** 0.5

    return {
        "sn34_score": float(sn34),
        "mcc": float(mcc),
        "mcc_norm": float(mcc_norm),
        "brier": float(brier),
        "brier_norm": float(brier_norm),
        "accuracy": float(acc),
    }


def evaluate_model(model, dataloader, device="cuda") -> dict:
    """
    Evaluate a model on a dataloader and return sn34 metrics.

    The model should output [B, 2] logits. We apply softmax to get
    probabilities and use p(synthetic) for scoring.

    Args:
        model: PyTorch model producing [B, 2] logits.
        dataloader: DataLoader yielding (inputs, labels) batches.
        device: Device string.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)  # [B, 2]
            probs = torch.softmax(logits.float(), dim=-1)
            p_synthetic = probs[:, 1].cpu().numpy()

            all_labels.extend(batch_labels.numpy().tolist())
            all_probs.extend(p_synthetic.tolist())

    return compute_sn34_score(np.array(all_labels), np.array(all_probs))


# ──────────────────────────────────────────────────────────────────────────────
# Data download
# ──────────────────────────────────────────────────────────────────────────────

def download_datasets(modality: str):
    """Download and cache all datasets for a modality."""
    from dfresearch.data import load_dataset_config, download_and_cache_dataset

    config = load_dataset_config(modality)
    datasets = config["datasets"]

    print(f"\nDownloading {len(datasets)} {modality} datasets...")
    print("=" * 60)

    for ds_cfg in datasets:
        name = ds_cfg["name"]
        path = ds_cfg["path"]
        max_samples = ds_cfg.get("max_samples", 1000)

        print(f"\n[{name}] from {path} (max {max_samples} samples)")
        t0 = time.time()
        download_and_cache_dataset(
            dataset_name=name,
            hf_path=path,
            modality=modality,
            max_samples=max_samples,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"All {modality} datasets downloaded.")


def verify_cache(modality: str):
    """Print cache status for all datasets of a modality."""
    from dfresearch.data import load_dataset_config, CACHE_DIR

    config = load_dataset_config(modality)
    total_samples = 0

    print(f"\nCache status for {modality} datasets:")
    print(f"{'Dataset':<35} {'Samples':>8} {'Status'}")
    print("-" * 60)

    for ds_cfg in config["datasets"]:
        name = ds_cfg["name"]
        cache_path = CACHE_DIR / "datasets" / modality / name
        marker = cache_path / ".download_complete"

        if marker.exists():
            files = [
                f for f in cache_path.iterdir()
                if not f.name.startswith(".")
            ]
            n = len(files)
            total_samples += n
            status = "OK"
        elif cache_path.exists():
            files = [
                f for f in cache_path.iterdir()
                if not f.name.startswith(".")
            ]
            n = len(files)
            total_samples += n
            status = "PARTIAL"
        else:
            n = 0
            status = "MISSING"

        print(f"{name:<35} {n:>8} {status}")

    print("-" * 60)
    print(f"{'Total':<35} {total_samples:>8}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for dfresearch"
    )
    parser.add_argument(
        "--modality",
        choices=["image", "video", "audio", "all"],
        default="all",
        help="Which modality to download (default: all)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify cache status instead of downloading",
    )
    args = parser.parse_args()

    modalities = ["image", "video", "audio"] if args.modality == "all" else [args.modality]

    for mod in modalities:
        if args.verify:
            verify_cache(mod)
        else:
            download_datasets(mod)


if __name__ == "__main__":
    main()
