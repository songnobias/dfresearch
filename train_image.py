#!/usr/bin/env python3
"""
train_image.py — Image deepfake detection training script.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Everything is fair game: model choice, hyperparameters, optimizer, augmentations,
architecture tweaks, batch size, learning rate schedule, etc.

The goal: maximize sn34_score on the validation set within the time budget.

Usage:
    uv run train_image.py
    uv run train_image.py --model clip-vit-l14
    uv run train_image.py > run.log 2>&1
"""

import argparse
import time
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

from prepare import (
    TIME_BUDGET,
    TARGET_IMAGE_SIZE,
    DEFAULT_IMAGE_BATCH_SIZE,
    evaluate_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — The agent tunes these
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "efficientnet-b4"       # "efficientnet-b4" or "clip-vit-l14"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = DEFAULT_IMAGE_BATCH_SIZE  # 32
AUGMENT_LEVEL = 2                      # 0=none, 1=basic, 2=medium, 3=hard
MAX_PER_CLASS = 5000                   # samples per class for training
WARMUP_STEPS = 100
GRAD_ACCUM_STEPS = 1
USE_AMP = True                         # mixed precision
FREEZE_BACKBONE = False                # freeze pretrained backbone layers
DROPOUT = 0.3

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    # ── Model ──
    from dfresearch.models import get_model
    model = get_model("image", args.model, num_classes=2, pretrained=True, dropout=DROPOUT)

    if FREEZE_BACKBONE and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({num_params / 1e6:.1f}M params, {num_trainable / 1e6:.1f}M trainable)")

    # ── Data ──
    from dfresearch.data import make_dataloader

    train_loader = make_dataloader(
        "image", split="train", batch_size=args.batch_size,
        target_size=TARGET_IMAGE_SIZE, augment_level=AUGMENT_LEVEL,
        max_per_class=MAX_PER_CLASS,
    )
    val_loader = make_dataloader(
        "image", split="val", batch_size=args.batch_size * 2,
        target_size=TARGET_IMAGE_SIZE, augment_level=0,
        max_per_class=MAX_PER_CLASS,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        print("ERROR: No training data. Run `uv run prepare.py --modality image` first.")
        sys.exit(1)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    amp_enabled = USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── Training loop ──
    model.train()
    step = 0
    epoch = 0
    total_loss = 0.0
    t_start = time.time()

    print(f"\nTraining for {args.time_budget}s time budget...", flush=True)
    print("-" * 60, flush=True)

    while True:
        epoch += 1
        for batch_inputs, batch_labels in train_loader:
            elapsed = time.time() - t_start
            if elapsed >= args.time_budget:
                break

            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            # LR warmup
            if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
                lr_scale = (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch_inputs)
                loss = F.cross_entropy(logits, batch_labels)
                loss = loss / GRAD_ACCUM_STEPS

            # Fast fail on NaN
            if torch.isnan(loss):
                print("ERROR: NaN loss detected, aborting.")
                sys.exit(1)

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            step += 1

            if step % 10 == 0 or step <= 5:
                avg_loss = total_loss / max(step, 1)
                print(f"  step {step:>5d} | loss {avg_loss:.4f} | elapsed {elapsed:.0f}s", flush=True)

        elapsed = time.time() - t_start
        print(f"  epoch {epoch:>3d} done | steps {step} | elapsed {elapsed:.0f}s", flush=True)
        if elapsed >= args.time_budget:
            break

    training_seconds = time.time() - t_start

    # ── Evaluation ──
    print("\nEvaluating...")
    t_eval = time.time()
    metrics = evaluate_model(model, val_loader, device=device)
    eval_seconds = time.time() - t_eval

    total_seconds = training_seconds + eval_seconds
    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ── Output summary ──
    print(f"\n{'=' * 60}")
    print("---")
    print(f"model:            {args.model}")
    print(f"sn34_score:       {metrics['sn34_score']:.6f}")
    print(f"accuracy:         {metrics['accuracy']:.6f}")
    print(f"mcc:              {metrics['mcc']:.6f}")
    print(f"brier:            {metrics['brier']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"batch_size:       {args.batch_size}")
    print(f"learning_rate:    {args.lr}")
    print(f"augment_level:    {AUGMENT_LEVEL}")

    # Save submission-ready checkpoint directory
    from safetensors.torch import save_file
    from pathlib import Path
    import json
    import shutil
    from datetime import datetime
    from export import generate_model_config, generate_model_py

    ckpt_dir = Path("results") / "checkpoints" / "image"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), ckpt_dir / "model.safetensors")

    config = generate_model_config("image", args.model)
    with open(ckpt_dir / "model_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    (ckpt_dir / "model.py").write_text(generate_model_py("image", args.model))

    print(f"\nCheckpoint saved to {ckpt_dir}/")
    print(f"  model.safetensors, model.py, model_config.yaml — ready for submission")

    # Save run artifact with timestamp for experiment history
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "timestamp": ts,
        "modality": "image",
        "model": args.model,
        "sn34_score": metrics["sn34_score"],
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "brier": metrics["brier"],
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "num_steps": step,
        "num_params_M": round(num_params / 1e6, 1),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "augment_level": AUGMENT_LEVEL,
    }
    (runs_dir / f"{ts}_meta.json").write_text(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
