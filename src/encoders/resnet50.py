"""Frozen ImageNet ResNet50 embeddings for already-preprocessed sonar RGB.

Input comes from prepare_sonar_rgb: CHW float32, 224x224, in [0, 1].
Keep the geometry unchanged; apply only ImageNet channel normalization.

Input: preprocessed (3,224,224) float RGB sonar or mask patch in [0,1].
Output: L2-normalized 2048-dimensional visual embedding.
Usage: imported by scripts.resnet50_benchmark; no standalone CLI.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms.functional import normalize


class FrozenResNet50:
    """Extract L2-normalized 2048D embeddings without training."""

    def __init__(self, device=None):
        """Load the ImageNet V2 weights and configure the non-training inference device."""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = ResNet50_Weights.IMAGENET1K_V2

        self.model = resnet50(weights=self.weights)
        self.model.fc = nn.Identity()
        self.model.requires_grad_(False)
        self.model = self.model.eval().to(self.device)

        transform = self.weights.transforms()
        self.mean = transform.mean
        self.std = transform.std

    @torch.inference_mode()
    def encode(self, rgb):
        """Return an L2-normalized (2048,) NumPy embedding for one RGB patch."""
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != (3, 224, 224):
            raise ValueError("Expected RGB array of shape (3, 224, 224)")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be finite and in [0, 1]")

        x = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(self.device)
        x = normalize(x, mean=self.mean, std=self.std)
        features = self.model(x)
        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()
