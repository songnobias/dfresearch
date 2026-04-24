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

    def __init__(self, out_dim: int = 384):
        super().__init__()

        kernels = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],   # laplacian
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],   # sobel y
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],    # sobel x
                [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],   # second derivative
                [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],  # high pass
                [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]],       # diagonal emboss
            ],
            dtype=torch.float32,
        )
        self.register_buffer("fixed_kernels", kernels.unsqueeze(1))

        in_channels = 3 * kernels.shape[0] + 3  # residual maps + fft + gray + local variance
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.GELU(),
            nn.Conv2d(160, 224, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(224),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def _residual_maps(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [0, 1]
        _, c, _, _ = x.shape
        kernel_bank = self.fixed_kernels.repeat(c, 1, 1, 1)
        residual = torch.abs(F.conv2d(x, kernel_bank, padding=1, groups=c))
        return residual

    def _fft_map(self, x: torch.Tensor) -> torch.Tensor:
        # grayscale FFT magnitude summary map
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return mag

    def _local_variance_map(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        mean_sq = F.avg_pool2d(gray * gray, kernel_size=5, stride=1, padding=2)
        var = torch.relu(mean_sq - mean * mean)
        var = var / (var.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._residual_maps(x)
        fft_mag = self._fft_map(x)
        gray = x.mean(dim=1, keepdim=True)
        local_var = self._local_variance_map(x)
        feats = torch.cat([residual, fft_mag, gray, local_var], dim=1)
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
        forensic_dim: int = 384,
        eval_tta: bool = True,
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
        self.eval_tta = eval_tta

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        fused_dim = self.feat_dim + self.forensic_branch.out_dim
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, num_classes),
        )
        self.forensic_head = nn.Sequential(
            nn.LayerNorm(self.forensic_branch.out_dim),
            nn.Linear(self.forensic_branch.out_dim, num_classes),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def _document_view(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        gray = x.mean(dim=1, keepdim=True)
        gray_rgb = gray.repeat(1, 3, 1, 1)
        blur = F.avg_pool2d(gray_rgb, kernel_size=3, stride=1, padding=1)
        sharpened = torch.clamp(gray_rgb + 0.8 * (gray_rgb - blur), 0.0, 1.0)
        down = F.interpolate(sharpened, scale_factor=0.6, mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return up

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        forensic_features = self.forensic_branch(x)

        clip_x = (x - self.mean) / self.std
        outputs = self.vision_model(pixel_values=clip_x)
        pooled = outputs.pooler_output

        fused = torch.cat([pooled, forensic_features], dim=-1)
        fused_logits = self.head(fused)
        semantic_logits = self.semantic_head(pooled)
        forensic_logits = self.forensic_head(forensic_features)
        return fused_logits + 0.35 * semantic_logits + 0.55 * forensic_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._forward_single(x)
        if self.training or not self.eval_tta:
            return logits

        doc_view = torch.clamp(self._document_view(x) * 255.0, 0.0, 255.0)
        doc_logits = self._forward_single(doc_view)
        return 0.65 * logits + 0.35 * doc_logits


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with saved weights (gasbench entry point)."""
    model = HybridCLIPFreqDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
