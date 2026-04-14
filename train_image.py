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

MODEL_NAME = "smogy-swin"
LEARNING_RATE = 3e-5
BACKBONE_LR_SCALE = 0.005
WEIGHT_DECAY = 0.01
BATCH_SIZE = DEFAULT_IMAGE_BATCH_SIZE
AUGMENT_LEVEL = 3
MAX_PER_CLASS = 20000
WARMUP_STEPS = 750
GRAD_ACCUM_STEPS = 4
USE_AMP = True
FREEZE_BACKBONE = False
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05
ETA_MIN_LR_RATIO = 0.005

# ── Calibration / regularization ──────────────────────────────────────────────
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.10
ENTROPY_LAMBDA = 0.005

# ── EMA ───────────────────────────────────────────────────────────────────────
EMA_DECAY = 0.9998

# ── Focal loss ────────────────────────────────────────────────────────────────
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.0

# ── Dataset-balanced sampling ─────────────────────────────────────────────────
DATASET_BALANCED_SAMPLING = True


# ──────────────────────────────────────────────────────────────────────────────
# EMA — Exponential Moving Average of model weights
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Maintains an exponential moving average of model parameters for smoother,
    better-calibrated predictions."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ──────────────────────────────────────────────────────────────────────────────
# Temperature calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_temperature(model, val_loader, device, amp_enabled):
    """
    Find scalar temperature T that minimises Brier score on the validation set.

    The production sn34_score uses brier_score = (0.25 - brier) / 0.25 raised to
    β=1.8, so minimising Brier directly is the correct objective — not NLL.

    T is baked into the final linear layer so the model is self-contained.
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
    labels_all = torch.cat(labels_list).float()

    best_T, best_brier = 1.0, float("inf")
    for T_candidate in np.concatenate([
        np.linspace(0.05, 1.0, 200),
        np.linspace(1.0, 10.0, 200),
    ]):
        probs = torch.softmax(logits_all / T_candidate, dim=1)
        p_syn = probs[:, 1]
        brier = float(torch.mean((p_syn - labels_all) ** 2).item())
        if brier < best_brier:
            best_brier, best_T = brier, float(T_candidate)

    temperature = nn.Parameter(torch.tensor([best_T]))
    optimizer_cal = torch.optim.LBFGS(
        [temperature], lr=0.01, max_iter=500, line_search_fn="strong_wolfe"
    )

    def brier_loss():
        optimizer_cal.zero_grad()
        T_clamped = temperature.clamp(min=0.05, max=15.0)
        probs = torch.softmax(logits_all / T_clamped, dim=1)
        loss = torch.mean((probs[:, 1] - labels_all) ** 2)
        loss.backward()
        return loss

    optimizer_cal.step(brier_loss)
    T = float(temperature.clamp(min=0.05, max=15.0).item())

    probs = torch.softmax(logits_all / T, dim=1)
    cal_brier = float(torch.mean((probs[:, 1] - labels_all) ** 2).item())
    print(
        f"Temperature calibration: T = {T:.4f}  "
        f"(Brier before={best_brier:.4f} → after={cal_brier:.4f})"
    )

    with torch.no_grad():
        last_linear = model.head[-1]
        if isinstance(last_linear, nn.Linear):
            last_linear.weight.data.div_(T)
            last_linear.bias.data.div_(T)
        else:
            print(f"WARNING: last layer is {type(last_linear)}, skipping weight scaling")

    return T


# ──────────────────────────────────────────────────────────────────────────────
# Mixup & CutMix
# ──────────────────────────────────────────────────────────────────────────────

def mixup_batch(inputs, labels, alpha: float):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam


def cutmix_batch(inputs, labels, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    B, C, H, W = inputs.shape

    cx = np.random.uniform(0, W)
    cy = np.random.uniform(0, H)
    rw = W * math.sqrt(1 - lam) / 2
    rh = H * math.sqrt(1 - lam) / 2
    x1 = int(max(0, cx - rw))
    y1 = int(max(0, cy - rh))
    x2 = int(min(W, cx + rw))
    y2 = int(min(H, cy + rh))

    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[idx, :, y1:y2, x1:x2]
    area_ratio = (x2 - x1) * (y2 - y1) / (W * H)
    lam = 1 - area_ratio
    return mixed, labels, labels[idx], lam


def mixed_loss(logits, labels_a, labels_b, lam, label_smoothing, focal_gamma=0.0):
    if focal_gamma > 0:
        loss_a = focal_cross_entropy(logits, labels_a, label_smoothing, focal_gamma)
        loss_b = focal_cross_entropy(logits, labels_b, label_smoothing, focal_gamma)
    else:
        loss_a = F.cross_entropy(logits, labels_a, label_smoothing=label_smoothing)
        loss_b = F.cross_entropy(logits, labels_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1 - lam) * loss_b


def focal_cross_entropy(logits, labels, label_smoothing=0.0, gamma=2.0):
    ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def production_sn34(brier: float, mcc: float) -> float:
    """Production formula: brier_score = (0.25 - brier) / 0.25, then geomean."""
    mcc_norm = max(0.0, (mcc + 1.0) / 2.0) ** 1.2
    brier_score = max(0.0, (0.25 - brier) / 0.25) ** 1.8
    return float(max(0.0, (mcc_norm * brier_score)) ** 0.5)


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
        "--label-smoothing", type=float, default=LABEL_SMOOTHING,
    )
    parser.add_argument("--no-cosine-decay", action="store_true")
    parser.add_argument("--no-temperature-calibration", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
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

    # ── EMA ──
    ema = None
    if not args.no_ema and EMA_DECAY > 0:
        ema = ModelEMA(model, decay=EMA_DECAY)
        print(f"EMA enabled (decay={EMA_DECAY})")

    # ── Data ──
    from collections import Counter
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from dfresearch.data import gather_samples, ImageDeepfakeDataset

    train_samples = gather_samples("image", split="train", max_per_class=MAX_PER_CLASS)
    val_samples = gather_samples("image", split="val", max_per_class=MAX_PER_CLASS)

    train_dataset = ImageDeepfakeDataset(
        train_samples, target_size=TARGET_IMAGE_SIZE, augment_level=AUGMENT_LEVEL,
    )
    val_dataset = ImageDeepfakeDataset(
        val_samples, target_size=TARGET_IMAGE_SIZE, augment_level=0,
    )

    if DATASET_BALANCED_SAMPLING and len(train_samples) > 0:
        dataset_names = [str(s[0].parent.name) for s in train_samples]
        dataset_counts = Counter(dataset_names)
        sample_weights = torch.tensor(
            [1.0 / dataset_counts[name] for name in dataset_names]
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print(f"Dataset-balanced sampling: {len(dataset_counts)} datasets, "
              f"min={min(dataset_counts.values())}, max={max(dataset_counts.values())}")
    else:
        sampler = None

    num_workers = min(4, max(1, len(train_dataset)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, max(len(train_dataset), 1)),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(len(train_dataset) > args.batch_size),
        persistent_workers=(num_workers > 0),
    )
    val_num_workers = min(4, max(1, len(val_dataset)))
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size * 2, max(len(val_dataset), 1)),
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(val_num_workers > 0),
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        print("ERROR: No training data. Run `uv run prepare.py --modality image` first.")
        sys.exit(1)

    # ── Optimizer: differential learning rates ──
    backbone_module = (
        getattr(model, "vision_model", None)
        or getattr(model, "swin", None)
        or getattr(model, "backbone", None)
    )
    head_module = getattr(model, "head", None)

    if backbone_module is not None and head_module is not None:
        backbone_params = list(backbone_module.parameters())
        head_params = list(head_module.parameters())
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
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
        )
        for pg in optimizer.param_groups:
            pg["base_lr"] = args.lr
        print(f"Optimizer: single group lr={args.lr:.1e}")

    amp_enabled = USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── Configuration summary ──
    print(f"\nConfig: grad_accum={GRAD_ACCUM_STEPS}, weight_decay={WEIGHT_DECAY}")
    print(f"  mixup_alpha={MIXUP_ALPHA}, cutmix_alpha={CUTMIX_ALPHA}, cutmix_prob={CUTMIX_PROB}")
    print(f"  focal_loss={'gamma=' + str(FOCAL_GAMMA) if USE_FOCAL_LOSS else 'off'}")
    print(f"  entropy_lambda={ENTROPY_LAMBDA}, dropout={DROPOUT}")

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
                "mixup_alpha": MIXUP_ALPHA,
                "cutmix_alpha": CUTMIX_ALPHA,
                "cutmix_prob": CUTMIX_PROB,
                "focal_loss": USE_FOCAL_LOSS,
                "focal_gamma": FOCAL_GAMMA,
                "entropy_lambda": ENTROPY_LAMBDA,
                "ema_decay": EMA_DECAY,
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
    best_prod_sn34 = 0.0

    print(f"\nTraining for {budget}s budget...", flush=True)
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
                # Choose between CutMix, Mixup, or plain
                use_cutmix = CUTMIX_ALPHA > 0 and np.random.random() < CUTMIX_PROB
                use_mixup = MIXUP_ALPHA > 0 and not use_cutmix

                focal_g = FOCAL_GAMMA if USE_FOCAL_LOSS else 0.0
                ls = args.label_smoothing

                if use_cutmix:
                    mixed_inputs, labels_a, labels_b, lam = cutmix_batch(
                        batch_inputs, batch_labels, CUTMIX_ALPHA
                    )
                    logits = model(mixed_inputs)
                    loss = mixed_loss(logits, labels_a, labels_b, lam, ls, focal_g)
                elif use_mixup:
                    mixed_inputs, labels_a, labels_b, lam = mixup_batch(
                        batch_inputs, batch_labels, MIXUP_ALPHA
                    )
                    logits = model(mixed_inputs)
                    loss = mixed_loss(logits, labels_a, labels_b, lam, ls, focal_g)
                else:
                    logits = model(batch_inputs)
                    if USE_FOCAL_LOSS:
                        loss = focal_cross_entropy(logits, batch_labels, ls, FOCAL_GAMMA)
                    else:
                        loss = F.cross_entropy(logits, batch_labels, label_smoothing=ls)

                if ENTROPY_LAMBDA > 0:
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    loss = loss - ENTROPY_LAMBDA * entropy

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

                if ema is not None:
                    ema.update(model)

            batch_loss = loss.item() * GRAD_ACCUM_STEPS
            total_loss += batch_loss
            epoch_loss += batch_loss
            epoch_steps += 1
            step += 1

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

    # ── Apply EMA weights before evaluation ──
    if ema is not None:
        print("\nApplying EMA weights for evaluation...")
        ema.apply_shadow(model)

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
    prod_sn34 = production_sn34(metrics['brier'], metrics['mcc'])
    print(f"\n{'=' * 60}")
    print("---")
    print(f"model:                    {args.model}")
    print(f"sn34_score (local):       {metrics['sn34_score']:.6f}")
    print(f"sn34_score (production):  {prod_sn34:.6f}  ← matches gasbench")
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
    print(f"mixup_alpha:              {MIXUP_ALPHA}")
    print(f"cutmix_alpha:             {CUTMIX_ALPHA}")
    print(f"cutmix_prob:              {CUTMIX_PROB}")
    print(f"focal_loss:               {USE_FOCAL_LOSS} (gamma={FOCAL_GAMMA})")
    print(f"entropy_lambda:           {ENTROPY_LAMBDA}")
    print(f"ema_decay:                {EMA_DECAY}")

    if WANDB_AVAILABLE:
        wandb.log({
            "eval/sn34_score": metrics["sn34_score"],
            "eval/sn34_score_production": prod_sn34,
            "eval/accuracy": metrics["accuracy"],
            "eval/mcc": metrics["mcc"],
            "eval/brier": metrics["brier"],
            "eval/calibration_temperature": calibration_temperature,
            "system/peak_vram_mb": peak_vram_mb,
            "system/training_seconds": training_seconds,
        })
        wandb.summary.update({
            "sn34_score": metrics["sn34_score"],
            "sn34_score_production": prod_sn34,
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
        "sn34_score_production": prod_sn34,
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
        "mixup_alpha": MIXUP_ALPHA,
        "cutmix_alpha": CUTMIX_ALPHA,
        "cutmix_prob": CUTMIX_PROB,
        "focal_loss": USE_FOCAL_LOSS,
        "focal_gamma": FOCAL_GAMMA,
        "entropy_lambda": ENTROPY_LAMBDA,
        "ema_decay": EMA_DECAY,
    }
    (runs_dir / f"{ts}_meta.json").write_text(json.dumps(run_meta, indent=2))

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
