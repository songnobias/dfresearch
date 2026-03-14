# dfresearch

Autonomous deepfake detection research for [BitMind Subnet 34](https://bitmind.ai/subnet).

Give an AI agent a deepfake detection training setup and let it experiment autonomously. It modifies the training code, trains for 10 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and a better model.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Datasets sourced from [BitMind-AI/gasbench](https://github.com/BitMind-AI/gasbench).

---

## How it works

```
┌──────────────────────────────────────────────────────────┐
│  SETUP                                                    │
│  1. Download datasets (uv run prepare.py)                 │
│  2. Create experiment branch                              │
│  3. Run baseline                                          │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  LOOP FOREVER                                             │
│                                                           │
│  1. Edit train_{modality}.py with an idea                 │
│  2. git commit                                            │
│  3. uv run train_{modality}.py > run.log 2>&1             │
│  4. Parse results: grep sn34_score run.log                │
│  5. sn34_score improved? → KEEP (advance branch)          │
│     sn34_score same/worse? → DISCARD (git reset)          │
│  6. Log to results.tsv                                    │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  EXPORT                                                   │
│  1. uv run export.py --modality image --model effnet-b4   │
│  2. gasbench run --image-model detector.zip --small       │
│  3. gascli d push --image-model detector.zip              │
└──────────────────────────────────────────────────────────┘
```

The repo has three types of files:

- **`prepare.py`** — Fixed constants, data downloading, evaluation metrics. Not modified.
- **`train_*.py`** — Training scripts the agent edits. One per modality. Contains model choice, hyperparameters, optimizer, training loop. **This is what the agent modifies.**
- **`program.md`** — Agent instructions. **This is what the human iterates on.**

## Quick start

**Requirements:** NVIDIA GPU (tested on A100/H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download datasets (uses 4 concurrent workers by default)
uv run prepare.py --modality image              # image datasets
uv run prepare.py --modality video              # video datasets
uv run prepare.py --modality audio              # audio datasets
uv run prepare.py                               # all modalities
uv run prepare.py --modality image --workers 8  # faster with more workers

# 4. Verify data was cached correctly
uv run prepare.py --verify --modality image

# 5. Run a training experiment (~10 min on GPU)
uv run train_image.py                 # trains EfficientNet-B4 baseline
uv run train_video.py                 # trains R3D-18 baseline
uv run train_audio.py                 # trains Wav2Vec2 baseline

# 6. Evaluate
uv run evaluate.py --modality image

# 7. Export for competition
uv run export.py --modality image --model efficientnet-b4
```

### Setting up results tracking

Before running the autoresearch loop, create a `results.tsv` file:

```bash
echo -e "commit\tsn34_score\taccuracy\tmemory_gb\tstatus\tdescription" > results.tsv
```

This file tracks all experiments. See `program.md` for the full autoresearch workflow.

### CLI entry point

The `dfresearch` CLI wraps all commands:

```bash
dfresearch prepare --modality image          # download datasets
dfresearch train --modality image            # train default model
dfresearch evaluate --modality image         # evaluate checkpoint
dfresearch export --modality image --model efficientnet-b4  # export for competition
```

## Running the agent

Point your AI coding agent (Claude, Codex, etc.) at this repo and prompt:

```
Hi, have a look at program.md and let's kick off a new image experiment!
Let's do the setup first.
```

The agent will create a branch, establish a baseline, and start experimenting autonomously. Each experiment takes ~10 minutes, so you get ~6/hour or ~50 overnight.

## Baseline models

### Image (2 baselines)

| Model | Params | Source | Description |
|-------|--------|--------|-------------|
| **EfficientNet-B4** | 19M | [timm](https://github.com/huggingface/pytorch-image-models) | ImageNet-pretrained CNN. The standard baseline from FaceForensics++ and DFDC challenge literature. Proven, efficient, well-understood. |
| **CLIP ViT-L/14** | 304M | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | OpenAI's vision-language model. CLIP features generalize exceptionally well across unseen generators ([UniversalFakeDetect, CVPR 2023](https://arxiv.org/abs/2302.10174)). Fine-tune the vision encoder with a classification head. |

### Video (2 baselines)

| Model | Params | Source | Description |
|-------|--------|--------|-------------|
| **R3D-18** | 33M | [torchvision](https://pytorch.org/vision/stable/models/video_resnet.html) | 3D ResNet-18 pretrained on Kinetics-400. 3D convolutions capture temporal artifacts between frames. Lightweight, fast to train. |
| **VideoMAE** | 87M | [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) | Self-supervised masked autoencoder pretrained on video. Strong spatiotemporal representations with excellent transfer learning. |

### Audio (2 baselines)

| Model | Params | Source | Description |
|-------|--------|--------|-------------|
| **Wav2Vec2** | 95M | [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | Self-supervised pretrained on 960h of speech. Dominant approach in ASVspoof challenges. Takes raw waveform input (16kHz, 6s). |
| **AST** | 87M | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Audio Spectrogram Transformer — applies ViT to mel-spectrograms. AudioSet pretrained. Different inductive bias from Wav2Vec2 (frequency-domain vs time-domain). |

## Switching models

Each training script defaults to one model. To switch:

```bash
# Image: switch to CLIP
uv run train_image.py --model clip-vit-l14

# Video: switch to VideoMAE
uv run train_video.py --model videomae

# Audio: switch to AST
uv run train_audio.py --model ast
```

Or edit the `MODEL_NAME` constant at the top of the training script.

## Project structure

```
dfresearch/
├── README.md                          # This file
├── pyproject.toml                     # Dependencies
├── program.md                         # Autoresearch agent instructions
│
├── prepare.py                         # Data download & evaluation (DO NOT MODIFY)
├── train_image.py                     # Image training (agent modifies)
├── train_video.py                     # Video training (agent modifies)
├── train_audio.py                     # Audio training (agent modifies)
├── evaluate.py                        # Evaluate a checkpoint
├── export.py                          # Export to gasbench ZIP format
├── analysis.ipynb                     # Experiment visualization
│
├── configs/
│   ├── image_datasets.yaml            # Training datasets for images
│   ├── video_datasets.yaml            # Training datasets for videos
│   └── audio_datasets.yaml            # Training datasets for audio
│
└── src/dfresearch/
    ├── __init__.py
    ├── cli.py                         # CLI entry point
    ├── data.py                        # Dataset download & DataLoaders
    ├── transforms.py                  # Augmentations
    └── models/
        ├── __init__.py                # Model registry
        ├── image/
        │   ├── efficientnet.py        # EfficientNet-B4
        │   └── clip_vit.py            # CLIP ViT-L/14
        ├── video/
        │   ├── r3d.py                 # R3D-18
        │   └── videomae.py            # VideoMAE
        └── audio/
            ├── wav2vec2.py            # Wav2Vec2
            └── ast_model.py           # Audio Spectrogram Transformer
```

## Datasets

Training datasets are curated subsets of the [gasbench](https://github.com/BitMind-AI/gasbench) benchmark suite, selected for diversity, accessibility, and training utility.

### Image datasets

| Type | Datasets | Samples |
|------|----------|---------|
| Real | flickr30k, celeb-a-hq, ffhq-256, ms-coco-unique, open-image-v7, lfw | ~7,000 |
| Synthetic | SFHQ, DALL-E 3, MidJourney, JourneyDB, FakeClue, Nano-banana | ~5,500 |
| Semisynthetic | face-swap | ~500 |

### Video datasets

| Type | Datasets | Samples |
|------|----------|---------|
| Real | imagenet-vidvrd, ucf101, moments-in-time, dfd-real | ~2,000 |
| Synthetic | aislop-videos, vidprom, deepaction, moviegen-bench | ~2,000 |
| Semisynthetic | semisynthetic-video, dfd-fake, fakeparts-faceswap | ~800 |

### Audio datasets

| Type | Datasets | Samples |
|------|----------|---------|
| Real | common-voice-17, mls-eng-10k, crema-d, slurp, english-dialects, MediaSpeech | ~4,000 |
| Synthetic | arabic-deepfake, emovoice-db, elevenlabs, fake-voices, ShiftySpeech, thorsten | ~3,000 |

All datasets are publicly available on HuggingFace. See `configs/*.yaml` for full details.

## Competition workflow

### Scoring

Models are scored on BitMind Subnet 34 using the `sn34_score`:

```
sn34_score = sqrt(MCC_norm^1.2 * Brier_norm^1.8)
```

Where `MCC_norm = (MCC + 1) / 2` and `Brier_norm = 1 - brier_score`. This rewards both discrimination accuracy (MCC) and calibration quality (Brier).

### Submission format

Models must be submitted as a ZIP containing:

```
detector.zip
├── model_config.yaml    # Metadata and preprocessing config
├── model.py             # Model architecture with load_model() function
└── model.safetensors    # Trained weights
```

The `export.py` script generates this automatically.

### End-to-end competition flow

```bash
# 1. Train your model
uv run train_image.py

# 2. Evaluate locally
uv run evaluate.py --modality image

# 3. Export to competition format
uv run export.py --modality image --model efficientnet-b4

# 4. Test with gasbench (requires gasbench installed)
pip install gasbench
gasbench run --image-model results/exports/image_detector_efficientnet-b4.zip --small

# 5. Push to BitMind Subnet 34
gascli d push \
  --image-model results/exports/image_detector_efficientnet-b4.zip \
  --wallet-name your_wallet \
  --wallet-hotkey your_hotkey
```

### Entrance exam

Before your model is scored on the network, it must pass an entrance exam:
- Runs `gasbench run --small` internally
- Must achieve **>= 80% accuracy** to pass
- Maximum 1 hour 25 minute timeout

### Model requirements

- **Format**: Safetensors only (ONNX no longer accepted)
- **Input**: uint8 images/video, float32 audio (see [Safetensors spec](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md))
- **Output**: `[batch_size, 2]` logits for `[real, synthetic]`
- **Allowed imports**: torch, torchvision, torchaudio, transformers, timm, einops, safetensors, PIL, cv2, numpy, scipy (see full list in [gasbench docs](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md#allowed-imports))

## Design choices

- **Single training file per modality.** The agent only touches `train_{modality}.py`. This keeps scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for 10 minutes. This makes experiments directly comparable regardless of architecture changes.
- **Real competition metric.** We use the exact `sn34_score` formula from Subnet 34, so improvements here translate directly to better competition rankings.
- **Gasbench-compatible export.** One command to package your model in the exact format validators expect.
- **Self-contained.** No external dependencies beyond what's in `pyproject.toml`. One GPU, one training file, one metric.

## Tips for miners

1. **Start with the baseline.** Run the default training first to establish your starting point.
2. **Image is the easiest win.** More training data, faster iteration, simpler models. Start here.
3. **Augmentations matter.** Level 2-3 augmentations significantly improve generalization to unseen generators.
4. **CLIP features generalize well.** If you're struggling with cross-generator performance, try the CLIP baseline.
5. **Don't over-fit to the training set.** The competition includes private holdout datasets you can't see. Diverse training data and augmentations help.
6. **Submit early, iterate fast.** Push your first model as soon as it passes 80% accuracy. You can always update it.
7. **Watch memory.** The competition runs evaluation on cloud GPUs. Models that OOM will fail.

## Troubleshooting

**"No cached data found"** / **"No training data"**: Run `uv run prepare.py --modality {modality}` to download datasets. Use `--verify` to check cache status.

**Downloads are slow**: Increase concurrent workers: `uv run prepare.py --modality image --workers 8`. Check your internet connection and HuggingFace Hub access.

**Some datasets fail to download**: This is expected — some HuggingFace datasets require authentication or have rate limits. The system skips failures gracefully and continues with remaining datasets. You can re-run prepare to retry failed ones.

**OOM during training**: Reduce `BATCH_SIZE` or `MAX_PER_CLASS` in the training script. Increase `GRAD_ACCUM_STEPS` to compensate. For video models, reduce `NUM_FRAMES`.

**"No CUDA GPU found"**: Training will fall back to CPU but will be very slow. Ensure NVIDIA drivers and CUDA are installed. Check with `nvidia-smi`.

**Low sn34_score**: Check that your model outputs proper logits (not probabilities). Make sure augmentation is enabled during training but disabled during evaluation. Ensure both real and synthetic samples exist in the cache.

**Export fails**: Ensure you've run the training script first to generate `results/checkpoints/{modality}/model.safetensors`.

**gasbench entrance exam fails**: You need >= 80% accuracy. Check `uv run evaluate.py --modality {modality}` locally first.

## License

MIT
