"""Frozen OpenAI CLIP ViT-B/32 image embeddings for BlueBoat sonar patches.

Use the shared (3, 224, 224) float32 [0, 1] input and existing SONAR/MASK-ONLY
protocol. Only CLIP's checkpoint-specific channel normalization is applied;
there is no additional crop, resize, training, or use of text prompts.

Input: preprocessed (3,224,224) float RGB sonar or mask patch in [0,1].
Output: L2-normalized 512-dimensional visual embedding (no text features).
Usage: imported by scripts.clip_vit_b32_benchmark; no standalone CLI.
"""

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import normalize


class FrozenCLIPViTB32:
    """Return L2-normalized 512D image embeddings from CLIP ViT-B/32."""

    # OpenAI CLIP checkpoints were trained with QuickGELU, not native GELU.
    model_name = "ViT-B-32-quickgelu"
    weights = "openai"

    def __init__(self, device=None):
        """Load OpenAI's ViT-B-32-quickgelu checkpoint and select the compute device."""
        import open_clip

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.weights
        )
        self.model = model.eval().to(self.device)
        self.model.requires_grad_(False)

        # Read actual checkpoint normalization from OpenCLIP's validation
        # transform; skip its resize/crop to retain the shared patch geometry.
        channel_normalization = preprocess.transforms[-1]
        self.mean = tuple(float(v) for v in channel_normalization.mean)
        self.std = tuple(float(v) for v in channel_normalization.std)
        if len(self.mean) != 3 or len(self.std) != 3 or any(x <= 0 for x in self.std):
            raise ValueError("CLIP preprocessing has invalid channel statistics")

    @torch.inference_mode()
    def encode(self, rgb):
        """Return one finite, unit-length (512,) NumPy float32 vector."""
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != (3, 224, 224):
            raise ValueError("Expected RGB array of shape (3, 224, 224)")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be finite and in [0, 1]")

        x = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(self.device)
        x = normalize(x, mean=self.mean, std=self.std)
        features = self.model.encode_image(x)
        if features.shape != (1, 512):
            raise ValueError(f"Unexpected CLIP image feature shape: {tuple(features.shape)}")
        features = F.normalize(features.float(), p=2, dim=1)
        if not torch.isfinite(features).all():
            raise ValueError("CLIP returned non-finite embeddings")
        return features.squeeze(0).cpu().numpy()
