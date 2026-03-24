"""
ConvNeXt-Base image deepfake detector for BitMind SN34.

Modern CNN backbone (Liu et al., "A ConvNet for the 2020s") via timm.
Good balance of accuracy, speed, and VRAM — fits comfortably on 16 GB GPUs
and finishes the gasbench entrance exam well within the time limit.

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]

Allowed imports only (torch, timm, safetensors) — safe for gasbench sandbox.
"""

import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvNeXtDetector(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone_name: str = "convnext_base",
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] uint8 [0, 255]
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        features = self.backbone(x)
        return self.head(features)


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """gasbench entry point — no network, no pretrained downloads."""
    model = ConvNeXtDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model
