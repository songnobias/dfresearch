"""
ResNet-50 image deepfake detector (timm, ImageNet-pretrained).

Smaller and faster than EfficientNet-B4; same I/O contract as other image models.

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet50Detector(nn.Module):
    """ResNet-50 binary deepfake detector."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
        )
        self.feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        features = self.backbone(x)
        return self.head(features)


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    model = ResNet50Detector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
