"""Frozen ImageNet ConvNeXt-Tiny features for preprocessed sonar RGB.

Input matches the ResNet50 baseline: CHW float32, 224x224, values in [0, 1].
Preserve patch geometry and apply only the weights' ImageNet channel statistics.
No fine-tuning or encoder-specific resampling is performed.

Input: preprocessed (3,224,224) float RGB sonar or mask patch in [0,1].
Output: L2-normalized 768-dimensional visual embedding.
Usage: imported by scripts.convnext_tiny_benchmark; no standalone CLI.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms.functional import normalize


class FrozenConvNeXtTiny:
    """Extract L2-normalized 768D embeddings without training."""

    def __init__(self, device=None):
        """Load pretrained ConvNeXt-Tiny weights and configure inference."""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        self.model = convnext_tiny(weights=self.weights)
        # Keep pretrained LayerNorm2d and Flatten; remove only classification.
        self.model.classifier[-1] = nn.Identity()
        self.model.requires_grad_(False)
        self.model = self.model.eval().to(self.device)

        transform = self.weights.transforms()
        self.mean = transform.mean
        self.std = transform.std

    @torch.inference_mode()
    def encode(self, rgb):
        """Return one L2-normalized (768,) NumPy feature vector."""
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != (3, 224, 224):
            raise ValueError("Expected RGB array of shape (3, 224, 224)")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be finite and in [0, 1]")

        x = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(self.device)
        x = normalize(x, mean=self.mean, std=self.std)
        features = self.model(x)
        if features.shape != (1, 768):
            raise ValueError(f"Unexpected ConvNeXt-Tiny feature shape: {tuple(features.shape)}")
        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()
