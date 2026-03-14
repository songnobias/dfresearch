#!/usr/bin/env python3
"""
train_audio.py — Audio deepfake detection training script.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Everything is fair game: model choice, hyperparameters, optimizer,
spectrogram parameters, pooling strategy, architecture tweaks, etc.

The goal: maximize sn34_score on the validation set within the time budget.

Usage:
    uv run train_audio.py
    uv run train_audio.py --model ast
    uv run train_audio.py > run.log 2>&1
"""

import argparse
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import (
    TIME_BUDGET,
    AUDIO_SAMPLE_RATE,
    AUDIO_DURATION,
    DEFAULT_AUDIO_BATCH_SIZE,
    evaluate_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — The agent tunes these
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "wav2vec2"                  # "wav2vec2" or "ast"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = DEFAULT_AUDIO_BATCH_SIZE    # 16
MAX_PER_CLASS = 3000
WARMUP_STEPS = 100
GRAD_ACCUM_STEPS = 2
USE_AMP = True
FREEZE_FEATURE_ENCODER = True            # freeze wav2vec2 CNN feature extractor
DROPOUT = 0.2

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
        torch.cuda.reset_peak_memory_stats()

    # ── Model ──
    from dfresearch.models import get_model

    model_kwargs = {"num_classes": 2, "pretrained": True, "dropout": DROPOUT}
    if args.model == "wav2vec2":
        model_kwargs["freeze_feature_encoder"] = FREEZE_FEATURE_ENCODER

    model = get_model("audio", args.model, **model_kwargs)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({num_params / 1e6:.1f}M params, {num_trainable / 1e6:.1f}M trainable)")

    # ── Data ──
    from dfresearch.data import make_dataloader

    train_loader = make_dataloader(
        "audio", split="train", batch_size=args.batch_size,
        max_per_class=MAX_PER_CLASS,
    )
    val_loader = make_dataloader(
        "audio", split="val", batch_size=args.batch_size * 2,
        max_per_class=MAX_PER_CLASS,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device == "cuda"))

    # ── Training loop ──
    model.train()
    step = 0
    epoch = 0
    total_loss = 0.0
    t_start = time.time()

    print(f"\nTraining for {args.time_budget}s time budget...")
    print("-" * 60)

    while True:
        epoch += 1
        for batch_inputs, batch_labels in train_loader:
            elapsed = time.time() - t_start
            if elapsed >= args.time_budget:
                break

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # LR warmup
            if step < WARMUP_STEPS:
                lr_scale = (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast("cuda", enabled=(USE_AMP and device == "cuda")):
                logits = model(batch_inputs)
                loss = F.cross_entropy(logits, batch_labels)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            step += 1

            if step % 50 == 0:
                avg_loss = total_loss / step
                print(f"  step {step:>5d} | loss {avg_loss:.4f} | elapsed {elapsed:.0f}s")

            if torch.isnan(loss):
                print("ERROR: NaN loss detected, aborting.")
                sys.exit(1)

        elapsed = time.time() - t_start
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

    # Save checkpoint
    from safetensors.torch import save_file
    from pathlib import Path

    ckpt_dir = Path("results") / "checkpoints" / "audio"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), ckpt_dir / "model.safetensors")
    print(f"\nCheckpoint saved to {ckpt_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
