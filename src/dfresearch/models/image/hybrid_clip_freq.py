"""
Hybrid CLIP + lightweight forensic detector for image deepfake detection.

Combines:
1. CLIP ViT-L/14 pooled semantic embedding
2. A compact residual-frequency branch built from fixed high-pass filters

Input:  [B, 3, H, W] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ResidualFrequencyBranch(nn.Module):
    """Lightweight forensic branch using fixed residual filters plus a CNN head."""

    def __init__(self, out_dim: int = 256):
        super().__init__()

        kernels = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],   # laplacian
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],   # sobel y
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],    # sobel x
                [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],   # second derivative
            ],
            dtype=torch.float32,
        )
        self.register_buffer("fixed_kernels", kernels.unsqueeze(1))

        in_channels = 3 * kernels.shape[0] + 1  # residual filter maps + fft magnitude
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def _residual_maps(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [0, 1]
        b, c, _, _ = x.shape
        kernel_bank = self.fixed_kernels.repeat(c, 1, 1, 1)
        residual = F.conv2d(x, kernel_bank, padding=1, groups=c)
        return residual

    def _fft_map(self, x: torch.Tensor) -> torch.Tensor:
        # grayscale FFT magnitude summary map
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._residual_maps(x)
        fft_mag = self._fft_map(x)
        feats = torch.cat([residual, fft_mag], dim=1)
        feats = self.encoder(feats)
        return self.proj(feats)


class HybridCLIPFreqDetector(nn.Module):
    """CLIP semantic features fused with a compact forensic branch."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        model_name: str = "openai/clip-vit-large-patch14",
        forensic_dim: int = 256,
    ):
        super().__init__()

        if pretrained:
            from transformers import CLIPVisionModel

            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        else:
            from transformers import CLIPVisionConfig, CLIPVisionModel

            config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=224,
                patch_size=14,
            )
            self.vision_model = CLIPVisionModel(config)

        self.feat_dim = self.vision_model.config.hidden_size
        self.forensic_branch = ResidualFrequencyBranch(out_dim=forensic_dim)

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        fused_dim = self.feat_dim + self.forensic_branch.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        forensic_features = self.forensic_branch(x)

        clip_x = (x - self.mean) / self.std
        outputs = self.vision_model(pixel_values=clip_x)
        pooled = outputs.pooler_output

        fused = torch.cat([pooled, forensic_features], dim=-1)
        return self.head(fused)


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with saved weights (gasbench entry point)."""
    model = HybridCLIPFreqDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
