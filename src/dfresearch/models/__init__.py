"""Baseline deepfake detection models for image, video, and audio modalities."""

from dfresearch.models.image.efficientnet import EfficientNetB4Detector
from dfresearch.models.image.clip_vit import CLIPViTDetector
from dfresearch.models.video.r3d import R3D18Detector
from dfresearch.models.video.videomae import VideoMAEDetector
from dfresearch.models.audio.wav2vec2 import Wav2Vec2Detector
from dfresearch.models.audio.ast_model import ASTDetector

MODEL_REGISTRY = {
    "image": {
        "efficientnet-b4": EfficientNetB4Detector,
        "clip-vit-l14": CLIPViTDetector,
    },
    "video": {
        "r3d-18": R3D18Detector,
        "videomae": VideoMAEDetector,
    },
    "audio": {
        "wav2vec2": Wav2Vec2Detector,
        "ast": ASTDetector,
    },
}


def get_model(modality: str, name: str, **kwargs):
    """Instantiate a model by modality and name."""
    if modality not in MODEL_REGISTRY:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(MODEL_REGISTRY)}")
    models = MODEL_REGISTRY[modality]
    if name not in models:
        raise ValueError(f"Unknown {modality} model: {name}. Choose from {list(models)}")
    return models[name](**kwargs)
