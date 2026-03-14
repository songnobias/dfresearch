"""
Dataset download, caching, and DataLoader utilities for dfresearch.

Supports image, video, and audio modalities.
Downloads from HuggingFace Hub using the datasets library, with local caching.
"""

import io
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

CACHE_DIR = Path(os.environ.get("DFRESEARCH_CACHE", "~/.cache/dfresearch")).expanduser()
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"

LABEL_MAP = {"real": 0, "synthetic": 1, "semisynthetic": 1}


def load_dataset_config(modality: str) -> dict:
    """Load dataset YAML config for a modality."""
    config_path = CONFIGS_DIR / f"{modality}_datasets.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_and_cache_dataset(
    dataset_name: str,
    hf_path: str,
    modality: str,
    max_samples: int = 1000,
    split: str = "train",
) -> Path:
    """
    Download a dataset from HuggingFace and cache locally.

    Returns the cache directory path containing the samples.
    """
    cache_path = CACHE_DIR / "datasets" / modality / dataset_name
    marker = cache_path / ".download_complete"

    if marker.exists():
        return cache_path

    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        ds = load_dataset(hf_path, split=split, streaming=True, trust_remote_code=True)
        count = 0
        for item in ds:
            if count >= max_samples:
                break

            if modality == "image":
                _cache_image_sample(item, cache_path, count)
            elif modality == "video":
                _cache_video_sample(item, cache_path, count)
            elif modality == "audio":
                _cache_audio_sample(item, cache_path, count)
            count += 1

        marker.touch()
        print(f"  Cached {count} samples for {dataset_name}")
    except Exception as e:
        print(f"  Warning: Could not download {dataset_name}: {e}")

    return cache_path


def _cache_image_sample(item: dict, cache_path: Path, idx: int):
    """Cache a single image sample."""
    img = None
    for key in ("image", "img", "src_img", "photo", "base_image"):
        if key in item and item[key] is not None:
            img = item[key]
            break

    if img is None:
        return

    if isinstance(img, Image.Image):
        img.save(cache_path / f"{idx:06d}.png")
    elif isinstance(img, bytes):
        Image.open(io.BytesIO(img)).save(cache_path / f"{idx:06d}.png")


def _cache_video_sample(item: dict, cache_path: Path, idx: int):
    """Cache a single video sample as raw bytes."""
    video = None
    for key in ("video", "video_bytes", "mp4"):
        if key in item and item[key] is not None:
            video = item[key]
            break

    if video is None:
        return

    if isinstance(video, bytes):
        (cache_path / f"{idx:06d}.mp4").write_bytes(video)


def _cache_audio_sample(item: dict, cache_path: Path, idx: int):
    """Cache a single audio sample."""
    audio = None
    for key in ("audio", "speech", "input_values"):
        if key in item and item[key] is not None:
            audio = item[key]
            break

    if audio is None:
        return

    if isinstance(audio, dict) and "array" in audio:
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio.get("sampling_rate", 16000)
        np.savez_compressed(cache_path / f"{idx:06d}.npz", audio=arr, sr=sr)
    elif isinstance(audio, bytes):
        (cache_path / f"{idx:06d}.wav").write_bytes(audio)


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class ImageDeepfakeDataset(Dataset):
    """Binary classification dataset for image deepfake detection."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        target_size: tuple[int, int] = (224, 224),
        augment_level: int = 0,
    ):
        self.samples = samples
        self.target_size = target_size
        self.augment_level = augment_level

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))

        from dfresearch.transforms import resize_image, apply_random_augmentations, random_horizontal_flip

        img = resize_image(img, self.target_size)
        if self.augment_level > 0:
            img = apply_random_augmentations(img, level=self.augment_level)
            img = random_horizontal_flip(img)

        # HWC uint8 -> CHW uint8 tensor (gasbench convention: model receives uint8)
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W] uint8
        return tensor, label


class VideoDeepfakeDataset(Dataset):
    """Binary classification dataset for video deepfake detection."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        target_size: tuple[int, int] = (224, 224),
        num_frames: int = 16,
        augment_level: int = 0,
    ):
        self.samples = samples
        self.target_size = target_size
        self.num_frames = num_frames
        self.augment_level = augment_level

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(path), ctx=cpu(0))
            total = len(vr)

            if total >= self.num_frames:
                indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            else:
                indices = list(range(total)) + [total - 1] * (self.num_frames - total)

            frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C] uint8
        except Exception:
            frames = np.zeros(
                (self.num_frames, self.target_size[0], self.target_size[1], 3),
                dtype=np.uint8,
            )

        from dfresearch.transforms import resize_image, apply_random_augmentations

        processed = []
        for i in range(frames.shape[0]):
            frame = resize_image(frames[i], self.target_size)
            if self.augment_level > 0:
                frame = apply_random_augmentations(frame, level=self.augment_level)
            processed.append(frame)

        frames_arr = np.stack(processed)  # [T, H, W, C]
        tensor = torch.from_numpy(frames_arr).permute(0, 3, 1, 2)  # [T, C, H, W] -> [T, 3, H, W]
        return tensor, label


class AudioDeepfakeDataset(Dataset):
    """Binary classification dataset for audio deepfake detection."""

    TARGET_SR = 16000
    TARGET_DURATION = 6.0

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        sample_rate: int = 16000,
        duration_seconds: float = 6.0,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration_seconds)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            if path.suffix == ".npz":
                data = np.load(path)
                waveform = data["audio"].astype(np.float32)
                sr = int(data.get("sr", self.sample_rate))
            else:
                import soundfile as sf
                waveform, sr = sf.read(str(path), dtype="float32")

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)

            if sr != self.sample_rate:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        except Exception:
            waveform = np.zeros(self.target_length, dtype=np.float32)

        # Pad or crop to target length
        if len(waveform) > self.target_length:
            start = random.randint(0, len(waveform) - self.target_length)
            waveform = waveform[start : start + self.target_length]
        elif len(waveform) < self.target_length:
            pad = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, pad), mode="constant")

        waveform = np.clip(waveform, -1.0, 1.0)
        tensor = torch.from_numpy(waveform)  # [96000] float32
        return tensor, label


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def gather_samples(
    modality: str,
    split: str = "train",
    max_per_class: Optional[int] = None,
) -> list[tuple[Path, int]]:
    """
    Gather all cached samples for a modality with labels.

    Returns list of (filepath, label) tuples.
    """
    config = load_dataset_config(modality)
    all_samples = []

    for ds_cfg in config["datasets"]:
        media_type = ds_cfg["media_type"]
        label = LABEL_MAP.get(media_type, 1)
        cache_path = CACHE_DIR / "datasets" / modality / ds_cfg["name"]

        if not cache_path.exists():
            continue

        extensions = {
            "image": {".png", ".jpg", ".jpeg", ".webp"},
            "video": {".mp4", ".avi", ".mkv", ".mov"},
            "audio": {".wav", ".mp3", ".flac", ".npz"},
        }[modality]

        files = sorted(
            f for f in cache_path.iterdir()
            if f.suffix.lower() in extensions and f.name != ".download_complete"
        )

        for f in files:
            all_samples.append((f, label))

    random.shuffle(all_samples)

    if max_per_class is not None:
        real = [(p, l) for p, l in all_samples if l == 0]
        fake = [(p, l) for p, l in all_samples if l == 1]
        real = real[:max_per_class]
        fake = fake[:max_per_class]
        all_samples = real + fake
        random.shuffle(all_samples)

    # Split into train/val (80/20)
    n = len(all_samples)
    split_idx = int(0.8 * n)

    if split == "train":
        return all_samples[:split_idx]
    elif split == "val":
        return all_samples[split_idx:]
    return all_samples


def make_dataloader(
    modality: str,
    split: str = "train",
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
    num_frames: int = 16,
    augment_level: int = 1,
    max_per_class: Optional[int] = None,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for the specified modality and split."""
    samples = gather_samples(modality, split=split, max_per_class=max_per_class)

    if modality == "image":
        dataset = ImageDeepfakeDataset(
            samples,
            target_size=target_size,
            augment_level=augment_level if split == "train" else 0,
        )
    elif modality == "video":
        dataset = VideoDeepfakeDataset(
            samples,
            target_size=target_size,
            num_frames=num_frames,
            augment_level=augment_level if split == "train" else 0,
        )
    elif modality == "audio":
        dataset = AudioDeepfakeDataset(samples)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
