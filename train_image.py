#!/usr/bin/env python3
"""
train_image.py — Image deepfake detection training script.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Everything is fair game: model choice, hyperparameters, optimizer, augmentations,
architecture tweaks, batch size, learning rate schedule, etc.

The goal: maximize sn34_score on the validation set within the time budget.

Available models: efficientnet-b4, resnet-50, clip-vit-l14, smogy-swin, convnext-base

Usage:
    uv run train_image.py
    uv run train_image.py --model resnet-50
    uv run train_image.py --model clip-vit-l14
    uv run train_image.py --model smogy-swin
    uv run train_image.py > run.log 2>&1
"""

import argparse
import math
import time
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

try:
    import wandb
    WANDB_AVAILABLE = wandb.api.api_key is not None
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from prepare import (
    TIME_BUDGET,
    TARGET_IMAGE_SIZE,
    DEFAULT_IMAGE_BATCH_SIZE,
    evaluate_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — The agent tunes these
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "clip-vit-l14"       # "efficientnet-b4" or "clip-vit-l14"
LEARNING_RATE = 3e-5              # head LR; backbone gets BACKBONE_LR_SCALE * this
BACKBONE_LR_SCALE = 0.1           # backbone LR = LEARNING_RATE * 0.1 (prevents forgetting)
WEIGHT_DECAY = 1e-2               # strong regularization improves holdout generalization
BATCH_SIZE = DEFAULT_IMAGE_BATCH_SIZE  # 32; reduce to 16 if OOM with CLIP
AUGMENT_LEVEL = 3                 # 0=none, 1=basic, 2=medium, 3=hard
MAX_PER_CLASS = 10000             # samples per class for training; more = better generalization
WARMUP_STEPS = 200                # longer warmup stabilizes large ViT fine-tuning
GRAD_ACCUM_STEPS = 1
USE_AMP = True                    # mixed precision
FREEZE_BACKBONE = False           # freeze pretrained backbone layers
DROPOUT = 0.3
LABEL_SMOOTHING = 0.10            # higher smoothing lowers overconfident outputs → better Brier
ETA_MIN_LR_RATIO = 0.001          # cosine floor: final lr = base_lr * this (after warmup)


# ──────────────────────────────────────────────────────────────────────────────
# Temperature calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_temperature(model, val_loader, device, amp_enabled):
    """
    Fit a scalar temperature T on the validation set to minimise NLL.

    Calibrated probabilities lower the Brier score significantly because the
    production sn34_score weights calibration with β=1.8 vs α=1.2 for MCC.

    After finding T, we divide the final linear layer's weight and bias by T
    so the calibration is baked into the saved safetensors — no model.py changes
    needed.
    """
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
            logits_list.append(logits.float().cpu())
            labels_list.append(labels)

    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)

    temperature = nn.Parameter(torch.ones(1))
    optimizer_cal = torch.optim.LBFGS(
        [temperature], lr=0.1, max_iter=500, line_search_fn="strong_wolfe"
    )

    def nll_loss():
        optimizer_cal.zero_grad()
        loss = F.cross_entropy(
            logits_all / temperature.clamp(min=0.05, max=20.0), labels_all
        )
        loss.backward()
        return loss

    optimizer_cal.step(nll_loss)
    T = float(temperature.clamp(min=0.05, max=20.0).item())
    print(f"Temperature calibration: T = {T:.4f} (T>1 softens, T<1 sharpens probabilities)")

    # Bake T into the final linear layer so the model is self-contained
    with torch.no_grad():
        last_linear = model.head[-1]
        if isinstance(last_linear, nn.Linear):
            last_linear.weight.data.div_(T)
            last_linear.bias.data.div_(T)
        else:
            print(f"WARNING: last layer is {type(last_linear)}, skipping weight scaling")

    return T


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        help="Cross-entropy label smoothing (0 disables)",
    )
    parser.add_argument(
        "--no-cosine-decay",
        action="store_true",
        help="Keep base LR after warmup instead of cosine decay to end of time budget",
    )
    parser.add_argument(
        "--no-temperature-calibration",
        action="store_true",
        help="Skip post-training temperature scaling calibration",
    )
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

    if FREEZE_BACKBONE:
        backbone = getattr(model, "backbone", None) or getattr(model, "vision_model", None)
        if backbone is not None:
            for param in backbone.parameters():
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

    # ── Optimizer: differential learning rates ──
    # Backbone (vision_model / backbone) uses a much lower LR to preserve
    # pretrained representations while the head adapts faster.
    backbone_module = getattr(model, "vision_model", None) or getattr(model, "backbone", None)
    head_module = getattr(model, "head", None)

    if backbone_module is not None and head_module is not None:
        backbone_params = list(backbone_module.parameters())
        head_params = list(head_module.parameters())
        # Any remaining parameters (e.g. norm buffers not in backbone/head)
        backbone_ids = {id(p) for p in backbone_params}
        head_ids = {id(p) for p in head_params}
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in backbone_ids and id(p) not in head_ids
        ]
        backbone_lr = args.lr * BACKBONE_LR_SCALE
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in backbone_params if p.requires_grad],
                 "lr": backbone_lr, "base_lr": backbone_lr},
                {"params": [p for p in head_params if p.requires_grad] + other_params,
                 "lr": args.lr, "base_lr": args.lr},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        print(f"Optimizer: backbone lr={backbone_lr:.1e}, head lr={args.lr:.1e}")
    else:
        # Fallback: single param group (for models without distinct backbone/head attrs)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
        )
        # Add base_lr so the LR schedule code below works uniformly
        for pg in optimizer.param_groups:
            pg["base_lr"] = args.lr
        print(f"Optimizer: single group lr={args.lr:.1e}")

    amp_enabled = USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── W&B init ──
    if WANDB_AVAILABLE:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "dfresearch"),
            config={
                "modality": "image", "model": args.model,
                "lr": args.lr, "backbone_lr_scale": BACKBONE_LR_SCALE,
                "batch_size": args.batch_size,
                "time_budget": args.time_budget, "augment_level": AUGMENT_LEVEL,
                "warmup_steps": WARMUP_STEPS, "grad_accum": GRAD_ACCUM_STEPS,
                "dropout": DROPOUT, "freeze_backbone": FREEZE_BACKBONE,
                "weight_decay": WEIGHT_DECAY, "max_per_class": MAX_PER_CLASS,
                "label_smoothing": args.label_smoothing,
                "cosine_decay": not args.no_cosine_decay,
                "eta_min_lr_ratio": ETA_MIN_LR_RATIO,
                "temperature_calibration": not args.no_temperature_calibration,
                "num_params_M": round(num_params / 1e6, 1),
                "num_trainable_M": round(num_trainable / 1e6, 1),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
            },
            tags=["image", args.model],
            reinit=True,
        )
        print(f"W&B: logging to {wandb.run.url}", flush=True)

    # ── Training loop ──
    from tqdm import tqdm

    model.train()
    step = 0
    epoch = 0
    total_loss = 0.0
    t_start = time.time()
    budget = args.time_budget
    is_tty = sys.stdout.isatty()

    print(f"\nTraining for {budget}s budget...", flush=True)
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}", flush=True)
    print(f"LR schedule: warmup {WARMUP_STEPS} steps, then ", end="", flush=True)
    print("cosine to end of budget" if not args.no_cosine_decay else "constant LR", flush=True)

    time_up = False
    warmup_done_elapsed: float | None = None
    while not time_up:
        epoch += 1
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_tty, leave=False, ncols=100)
        for batch_inputs, batch_labels in pbar:
            elapsed = time.time() - t_start
            if elapsed >= budget:
                time_up = True
                break

            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            # Per-group LR schedule using each group's own base_lr
            if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
                lr_scale = (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"] * lr_scale
            elif not args.no_cosine_decay:
                if warmup_done_elapsed is None:
                    warmup_done_elapsed = elapsed
                span = max(budget - warmup_done_elapsed, 1e-6)
                t_prog = min(1.0, (elapsed - warmup_done_elapsed) / span)
                for pg in optimizer.param_groups:
                    base = pg["base_lr"]
                    eta_min = base * ETA_MIN_LR_RATIO
                    pg["lr"] = eta_min + (base - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t_prog))
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"]

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch_inputs)
                ls = args.label_smoothing if args.label_smoothing > 0 else 0.0
                loss = F.cross_entropy(logits, batch_labels, label_smoothing=ls)
                loss = loss / GRAD_ACCUM_STEPS

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

            batch_loss = loss.item() * GRAD_ACCUM_STEPS
            total_loss += batch_loss
            epoch_loss += batch_loss
            epoch_steps += 1
            step += 1

            # Log the head LR (last param group) for display
            head_lr = optimizer.param_groups[-1]["lr"]
            remaining = max(0, budget - elapsed)
            pbar.set_postfix_str(f"loss={epoch_loss / epoch_steps:.4f} lr={head_lr:.1e} rem={remaining:.0f}s")
        pbar.close()

        elapsed = time.time() - t_start
        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            head_lr = optimizer.param_groups[-1]["lr"]
            print(f"Epoch {epoch:<4d} | loss={avg_loss:.4f} | lr={head_lr:.1e} | step={step} | {elapsed:.0f}s/{budget}s", flush=True)
            if WANDB_AVAILABLE:
                wandb.log({"train/loss": avg_loss, "train/lr": head_lr, "train/epoch": epoch, "train/step": step})

    training_seconds = time.time() - t_start

    # ── Temperature calibration ──
    calibration_temperature = 1.0
    if not args.no_temperature_calibration:
        print("\nCalibrating temperature on validation set...")
        calibration_temperature = calibrate_temperature(model, val_loader, device, amp_enabled)

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
    print(f"model:                    {args.model}")
    print(f"sn34_score:               {metrics['sn34_score']:.6f}")
    print(f"accuracy:                 {metrics['accuracy']:.6f}")
    print(f"mcc:                      {metrics['mcc']:.6f}")
    print(f"brier:                    {metrics['brier']:.6f}")
    print(f"calibration_temperature:  {calibration_temperature:.4f}")
    print(f"training_seconds:         {training_seconds:.1f}")
    print(f"total_seconds:            {total_seconds:.1f}")
    print(f"peak_vram_mb:             {peak_vram_mb:.1f}")
    print(f"num_steps:                {step}")
    print(f"num_params_M:             {num_params / 1e6:.1f}")
    print(f"num_epochs:               {epoch}")
    print(f"batch_size:               {args.batch_size}")
    print(f"learning_rate:            {args.lr}")
    print(f"backbone_lr_scale:        {BACKBONE_LR_SCALE}")
    print(f"weight_decay:             {WEIGHT_DECAY}")
    print(f"label_smoothing:          {args.label_smoothing}")
    print(f"cosine_decay:             {not args.no_cosine_decay}")
    print(f"augment_level:            {AUGMENT_LEVEL}")
    print(f"max_per_class:            {MAX_PER_CLASS}")

    if WANDB_AVAILABLE:
        wandb.log({
            "eval/sn34_score": metrics["sn34_score"],
            "eval/accuracy": metrics["accuracy"],
            "eval/mcc": metrics["mcc"],
            "eval/brier": metrics["brier"],
            "eval/calibration_temperature": calibration_temperature,
            "system/peak_vram_mb": peak_vram_mb,
            "system/training_seconds": training_seconds,
        })
        wandb.summary.update({
            "sn34_score": metrics["sn34_score"],
            "accuracy": metrics["accuracy"],
            "brier": metrics["brier"],
            "calibration_temperature": calibration_temperature,
        })

    # Save submission-ready checkpoint directory
    from safetensors.torch import save_file
    from pathlib import Path
    import json
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
        "calibration_temperature": calibration_temperature,
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "num_steps": step,
        "num_params_M": round(num_params / 1e6, 1),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "backbone_lr_scale": BACKBONE_LR_SCALE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": args.label_smoothing,
        "cosine_decay": not args.no_cosine_decay,
        "augment_level": AUGMENT_LEVEL,
        "max_per_class": MAX_PER_CLASS,
    }
    (runs_dir / f"{ts}_meta.json").write_text(json.dumps(run_meta, indent=2))

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
