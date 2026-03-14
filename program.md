# dfresearch — Autonomous Deepfake Detection Research

This is an experiment to have an AI agent autonomously research and improve deepfake detection models for [BitMind Subnet 34](https://bitmind.ai/subnet).

## Context

BitMind runs a competitive deepfake detection subnet on Bittensor where miners submit models that detect AI-generated content across **image**, **video**, and **audio** modalities. Models are scored using the `sn34_score` metric — a geometric mean of normalized MCC and Brier score. Higher is better.

This repo provides baseline models and a training setup. Your job is to make them better.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a modality and run tag**: propose a tag based on today's date and modality (e.g. `image-mar14`). The branch `autoresearch/{tag}` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/{tag}` from current master/main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context and model descriptions.
   - `prepare.py` — fixed constants, data downloading, evaluation metrics. **Do not modify.**
   - `train_image.py` / `train_video.py` / `train_audio.py` — the file you modify (pick one based on modality).
   - `src/dfresearch/models/` — baseline model implementations.
   - `src/dfresearch/data.py` — data loading pipeline.
   - `src/dfresearch/transforms.py` — augmentation pipeline.
4. **Verify data exists**: Check that `~/.cache/dfresearch/datasets/{modality}/` contains cached data. If not, tell the human to run `uv run prepare.py --modality {modality}`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock). You launch it as:

```bash
uv run train_image.py > run.log 2>&1    # for image experiments
uv run train_video.py > run.log 2>&1    # for video experiments
uv run train_audio.py > run.log 2>&1    # for audio experiments
```

**What you CAN do:**
- Modify the training script for the chosen modality (`train_image.py`, `train_video.py`, or `train_audio.py`). Everything is fair game: model choice, architecture changes, optimizer, hyperparameters, batch size, augmentation level, etc.
- Modify model files in `src/dfresearch/models/` — add new architectures, change existing ones, modify forward passes, add layers, etc.
- Modify `src/dfresearch/transforms.py` to add or tweak augmentations.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation metrics and constants.
- Modify `src/dfresearch/data.py`. The data pipeline is fixed.
- Modify `evaluate.py` or `export.py`. These are stable tooling scripts.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` and `compute_sn34_score` functions are ground truth.

**The goal: get the highest `sn34_score` on the validation set.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the augmentations.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful sn34_score gains, but it should not blow up dramatically (stay under 40GB ideally).

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating changes, weigh the complexity cost against improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline — run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
model:            efficientnet-b4
sn34_score:       0.723456
accuracy:         0.856000
mcc:              0.712000
brier:            0.134500
training_seconds: 600.1
total_seconds:    615.3
peak_vram_mb:     12340.0
num_steps:        1500
num_params_M:     19.3
num_epochs:       5
batch_size:       32
learning_rate:    0.0001
augment_level:    2
```

Extract the key metrics:

```bash
grep "^sn34_score:\|^accuracy:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 6 columns:

```
commit	sn34_score	accuracy	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. sn34_score achieved (e.g. 0.723456) — use 0.000000 for crashes
3. accuracy (e.g. 0.856000) — use 0.000000 for crashes
4. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	sn34_score	accuracy	memory_gb	status	description
a1b2c3d	0.723456	0.856000	12.1	keep	baseline efficientnet-b4
b2c3d4e	0.741230	0.872000	12.3	keep	increase LR to 3e-4 and augment_level=3
c3d4e5f	0.718000	0.845000	12.1	discard	switch to clip-vit-l14 (worse on this data size)
d4e5f6g	0.000000	0.000000	0.0	crash	double batch size (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify the training script with an experimental idea.
3. git commit
4. Run the experiment: `uv run train_{modality}.py > run.log 2>&1`
5. Read out the results: `grep "^sn34_score:\|^accuracy:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. Give up after a few attempts.
7. Record the results in results.tsv (do NOT commit results.tsv — leave it untracked)
8. If sn34_score improved (higher is better), keep the commit and advance the branch
9. If sn34_score is equal or worse, `git reset --hard` back to where you started

## Experiment ideas (starting points)

### Image
- Switch between efficientnet-b4 and clip-vit-l14, compare baselines
- Try different learning rates: 5e-5, 1e-4, 3e-4, 5e-4
- Increase augment_level (more aggressive data augmentation)
- Freeze backbone and only train the head (linear probing)
- Try cosine annealing LR schedule
- Add label smoothing to cross-entropy loss
- Increase model capacity: try efficientnet_b5 or efficientnet_b7
- Add mixup or cutmix augmentation
- Try focal loss instead of cross-entropy

### Video
- Switch between r3d-18 and videomae
- Increase num_frames (more temporal context)
- Try different frame sampling strategies
- Add temporal attention on top of frame features
- Try 2D backbone + temporal pooling instead of 3D convolutions
- Experiment with different augmentation levels per frame

### Audio
- Switch between wav2vec2 and ast
- Try unfreezing the feature encoder (FREEZE_FEATURE_ENCODER = False)
- Experiment with larger wav2vec2 models (wav2vec2-large)
- Add SpecAugment-style augmentation
- Try weighted pooling instead of mean pooling over time
- Experiment with different spectrogram parameters for AST

## Competition context

When you achieve a good sn34_score, the model can be exported for BitMind competition:

```bash
uv run export.py --modality {modality} --model {model_name}
gasbench run --{modality}-model results/exports/{modality}_detector_{model_name}.zip --small
```

The entrance exam requires >=80% accuracy. The full benchmark uses sn34_score for TAO emissions ranking.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the model code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
