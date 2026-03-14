"""
Dataset download, caching, and DataLoader utilities for dfresearch.

Supports image, video, and audio modalities.
Downloads from HuggingFace Hub using the datasets library, with local caching.
Supports concurrent downloads across multiple datasets.
"""

import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Broadened column name detection for HuggingFace datasets
IMAGE_COLUMNS = ("image", "img", "src_img", "photo", "base_image", "tgt_img",
                 "file", "pixel_values", "input_image", "picture", "frame",
                 "image1", "image2")
VIDEO_COLUMNS = ("video", "video_bytes", "mp4", "clip", "video_file",
                 "video_path", "content")
AUDIO_COLUMNS = ("audio", "speech", "input_values", "waveform", "signal",
                 "sound", "wav", "recording")


def load_dataset_config(modality: str) -> dict:
    """Load dataset YAML config for a modality."""
    config_path = CONFIGS_DIR / f"{modality}_datasets.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------

def download_and_cache_dataset(
    dataset_name: str,
    hf_path: str,
    modality: str,
    max_samples: int = 1000,
    split: str = "train",
    include_paths: Optional[list[str]] = None,
    exclude_paths: Optional[list[str]] = None,
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

        kwargs = dict(streaming=True, trust_remote_code=True)

        # Try requested split first, fall back to common alternatives
        ds = None
        for try_split in (split, "train", "test", "validation", None):
            try:
                if try_split is None:
                    ds = load_dataset(hf_path, **kwargs)
                    # Take first available split from the IterableDatasetDict
                    if hasattr(ds, "keys"):
                        first_key = next(iter(ds.keys()))
                        ds = ds[first_key]
                else:
                    ds = load_dataset(hf_path, split=try_split, **kwargs)
                break
            except (ValueError, KeyError):
                continue

        if ds is None:
            print(f"  Warning: No usable split found for {dataset_name}")
            return cache_path

        saved = 0
        skipped = 0
        for item in ds:
            if saved >= max_samples:
                break

            # Apply include/exclude path filtering on row metadata
            if include_paths or exclude_paths:
                row_path = _extract_path_hint(item)
                if row_path:
                    if include_paths and not any(inc in row_path for inc in include_paths):
                        skipped += 1
                        continue
                    if exclude_paths and any(exc in row_path for exc in exclude_paths):
                        skipped += 1
                        continue

            ok = False
            if modality == "image":
                ok = _cache_image_sample(item, cache_path, saved)
            elif modality == "video":
                ok = _cache_video_sample(item, cache_path, saved)
            elif modality == "audio":
                ok = _cache_audio_sample(item, cache_path, saved)

            if ok:
                saved += 1
            else:
                skipped += 1
                if skipped > 50 and saved == 0:
                    print(f"  Warning: 50+ rows skipped with no successful caches for {dataset_name}")
                    print(f"  Available columns: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
                    break

        marker.touch()
        print(f"  Cached {saved} samples for {dataset_name} (skipped {skipped})")
    except Exception as e:
        print(f"  Warning: Could not download {dataset_name}: {e}")

    return cache_path


def _extract_path_hint(item: dict) -> Optional[str]:
    """Try to extract a file path from a dataset row for include/exclude filtering."""
    for key in ("file", "filename", "path", "filepath", "source", "url", "id"):
        val = item.get(key)
        if isinstance(val, str):
            return val
    return None


def download_all_datasets(
    modality: str,
    max_workers: int = 4,
    progress: bool = True,
) -> dict[str, int]:
    """
    Download all datasets for a modality using concurrent workers.

    Returns dict mapping dataset name to number of cached samples.
    """
    config = load_dataset_config(modality)
    datasets_cfg = config["datasets"]

    if progress:
        print(f"\nDownloading {len(datasets_cfg)} {modality} datasets "
              f"(up to {max_workers} concurrent)...")
        print("=" * 60)

    results = {}

    def _download_one(ds_cfg):
        name = ds_cfg["name"]
        t0 = time.time()
        cache_path = download_and_cache_dataset(
            dataset_name=name,
            hf_path=ds_cfg["path"],
            modality=modality,
            max_samples=ds_cfg.get("max_samples", 1000),
            include_paths=ds_cfg.get("include_paths"),
            exclude_paths=ds_cfg.get("exclude_paths"),
        )
        elapsed = time.time() - t0
        # Count actual cached files
        n = _count_media_files(cache_path, modality)
        return name, n, elapsed

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, cfg): cfg["name"] for cfg in datasets_cfg}

        for future in as_completed(futures):
            name = futures[future]
            try:
                name, count, elapsed = future.result()
                results[name] = count
                if progress:
                    print(f"  [{name}] {count} samples in {elapsed:.1f}s")
            except Exception as e:
                results[name] = 0
                if progress:
                    print(f"  [{name}] FAILED: {e}")

    if progress:
        total = sum(results.values())
        print(f"\n{'=' * 60}")
        print(f"Total: {total} {modality} samples across {len(results)} datasets")

    return results


def _count_media_files(cache_path: Path, modality: str) -> int:
    """Count media files in a cache directory."""
    if not cache_path.exists():
        return 0
    extensions = _modality_extensions(modality)
    return sum(1 for f in cache_path.iterdir()
               if f.suffix.lower() in extensions and not f.name.startswith("."))


def _modality_extensions(modality: str) -> set[str]:
    return {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        "video": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
        "audio": {".wav", ".mp3", ".flac", ".npz", ".ogg"},
    }[modality]


# ---------------------------------------------------------------------------
# Cache helpers — return True if sample was successfully saved
# ---------------------------------------------------------------------------

def _cache_image_sample(item: dict, cache_path: Path, idx: int) -> bool:
    """Cache a single image sample. Returns True on success."""
    img = None
    for key in IMAGE_COLUMNS:
        val = item.get(key)
        if val is not None:
            img = val
            break

    if img is None:
        return False

    try:
        out = cache_path / f"{idx:06d}.png"
        if isinstance(img, Image.Image):
            img.convert("RGB").save(out)
            return True
        elif isinstance(img, bytes):
            Image.open(io.BytesIO(img)).convert("RGB").save(out)
            return True
        elif isinstance(img, dict) and "bytes" in img:
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(out)
            return True
        elif isinstance(img, dict) and "path" in img:
            # Some datasets give a local path instead of bytes
            Image.open(img["path"]).convert("RGB").save(out)
            return True
    except Exception:
        pass
    return False


def _cache_video_sample(item: dict, cache_path: Path, idx: int) -> bool:
    """Cache a single video sample as raw bytes. Returns True on success."""
    video = None
    for key in VIDEO_COLUMNS:
        val = item.get(key)
        if val is not None:
            video = val
            break

    if video is None:
        return False

    try:
        out = cache_path / f"{idx:06d}.mp4"
        if isinstance(video, bytes):
            out.write_bytes(video)
            return True
        elif isinstance(video, dict) and "bytes" in video:
            out.write_bytes(video["bytes"])
            return True
        elif isinstance(video, dict) and "path" in video:
            import shutil
            shutil.copy2(video["path"], out)
            return True
    except Exception:
        pass
    return False


def _cache_audio_sample(item: dict, cache_path: Path, idx: int) -> bool:
    """Cache a single audio sample. Returns True on success."""
    audio = None
    for key in AUDIO_COLUMNS:
        val = item.get(key)
        if val is not None:
            audio = val
            break

    if audio is None:
        return False

    try:
        if isinstance(audio, dict) and "array" in audio:
            arr = np.array(audio["array"], dtype=np.float32)
            sr = audio.get("sampling_rate", 16000)
            np.savez_compressed(cache_path / f"{idx:06d}.npz", audio=arr, sr=sr)
            return True
        elif isinstance(audio, dict) and "bytes" in audio:
            (cache_path / f"{idx:06d}.wav").write_bytes(audio["bytes"])
            return True
        elif isinstance(audio, bytes):
            (cache_path / f"{idx:06d}.wav").write_bytes(audio)
            return True
        elif isinstance(audio, dict) and "path" in audio:
            import shutil
            ext = Path(audio["path"]).suffix or ".wav"
            shutil.copy2(audio["path"], cache_path / f"{idx:06d}{ext}")
            return True
    except Exception:
        pass
    return False


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
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        from dfresearch.transforms import resize_image, apply_random_augmentations, random_horizontal_flip

        img = resize_image(img, self.target_size)
        if self.augment_level > 0:
            img = apply_random_augmentations(img, level=self.augment_level)
            img = random_horizontal_flip(img)

        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # [3, H, W] uint8
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
        for i in range(min(frames.shape[0], self.num_frames)):
            frame = resize_image(frames[i], self.target_size)
            if self.augment_level > 0:
                frame = apply_random_augmentations(frame, level=self.augment_level)
            processed.append(frame)

        # Pad if we got fewer frames than expected
        while len(processed) < self.num_frames:
            processed.append(processed[-1] if processed else
                             np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))

        frames_arr = np.stack(processed)  # [T, H, W, C]
        tensor = torch.from_numpy(np.ascontiguousarray(frames_arr)).permute(0, 3, 1, 2)
        return tensor, label


class AudioDeepfakeDataset(Dataset):
    """Binary classification dataset for audio deepfake detection."""

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
    seed: int = 42,
) -> list[tuple[Path, int]]:
    """
    Gather all cached samples for a modality with labels.

    Uses a fixed seed for reproducible train/val splits.
    Applies class balancing before splitting so both train and val are balanced.
    """
    config = load_dataset_config(modality)
    real_samples = []
    fake_samples = []

    extensions = _modality_extensions(modality)

    for ds_cfg in config["datasets"]:
        media_type = ds_cfg["media_type"]
        label = LABEL_MAP.get(media_type, 1)
        cache_path = CACHE_DIR / "datasets" / modality / ds_cfg["name"]

        if not cache_path.exists():
            continue

        files = sorted(
            f for f in cache_path.iterdir()
            if f.suffix.lower() in extensions and not f.name.startswith(".")
        )

        for f in files:
            if label == 0:
                real_samples.append((f, label))
            else:
                fake_samples.append((f, label))

    # Deterministic shuffle for reproducible splits
    rng = random.Random(seed)
    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    # Apply per-class cap
    if max_per_class is not None:
        real_samples = real_samples[:max_per_class]
        fake_samples = fake_samples[:max_per_class]

    # Split each class independently to maintain balance in both splits
    def _split(samples):
        n = len(samples)
        split_idx = int(0.8 * n)
        return samples[:split_idx], samples[split_idx:]

    real_train, real_val = _split(real_samples)
    fake_train, fake_val = _split(fake_samples)

    if split == "train":
        out = real_train + fake_train
    elif split == "val":
        out = real_val + fake_val
    else:
        out = real_samples + fake_samples

    rng.shuffle(out)

    if len(out) == 0:
        print(f"WARNING: No {modality} samples found for split='{split}'. "
              f"Run `uv run prepare.py --modality {modality}` first.")

    return out


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

    # Don't drop_last if it would yield zero batches
    use_drop_last = (split == "train") and (len(dataset) > batch_size)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=use_drop_last,
        persistent_workers=(num_workers > 0),
    )
