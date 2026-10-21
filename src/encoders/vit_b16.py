"""Frozen ImageNet ViT-B/16 features for preprocessed BlueBoat sonar RGB.

Matches the ResNet50/ConvNeXt baseline: CHW float32 224x224 images in [0, 1],
per-observation SONAR or MASK-ONLY input, and ImageNet channel normalization.
Does not resize or fine-tune. The pretrained classification head is removed,
retaining the class-token representation produced by the final transformer.

Input: preprocessed (3,224,224) float RGB sonar or mask patch in [0,1].
Output: L2-normalized 768-dimensional CLS-token visual embedding.
Usage: imported by scripts.vit_b16_benchmark; no standalone CLI.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms.functional import normalize


class FrozenViTB16:
    """Extract L2-normalized 768D class-token features without training."""

    def __init__(self, device=None):
        """Load pretrained ViT-B/16 and configure evaluation-only inference."""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1

        self.model = vit_b_16(weights=self.weights)
        self.model.heads = nn.Identity()
        self.model.requires_grad_(False)
        self.model = self.model.eval().to(self.device)

        transform = self.weights.transforms()
        self.mean = transform.mean
        self.std = transform.std

    @torch.inference_mode()
    def encode(self, rgb):
        """Return a finite L2-normalized NumPy feature vector of shape (768,)."""
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != (3, 224, 224):
            raise ValueError("Expected RGB array of shape (3, 224, 224)")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be finite and in [0, 1]")

        x = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(self.device)
        x = normalize(x, mean=self.mean, std=self.std)
        features = self.model(x)
        if features.shape != (1, 768):
            raise ValueError(f"Unexpected ViT-B/16 feature shape: {tuple(features.shape)}")
        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()
