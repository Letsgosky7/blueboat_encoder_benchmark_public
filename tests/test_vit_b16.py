"""Check ViT-B/16 inference interface, frozen weights and CLS feature vectors.

Run: python -m unittest discover -s tests -p 'test_*.py' -v (repository root).
Input: synthetic in-memory arrays and, where needed, temporary CSV/NPZ fixtures.
Output: unittest pass/fail assertions; temporary files are discarded after tests.
No CLI flags; these tests do not require raw BlueBoat XTF recordings.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights

from src.encoders.vit_b16 import FrozenViTB16


class _StubViT(nn.Module):
    """Minimal ViT interface: a 768D representation followed by heads."""

    def __init__(self):
        """Initialize the test-only encoder stub."""
        super().__init__()
        self.heads = nn.Linear(768, 1000)
        self.last_input = None

    def forward(self, x):
        """Return a fixed-shape feature vector for the mocked backbone."""
        self.last_input = x.detach().clone()
        features = torch.arange(1, 769, dtype=x.dtype, device=x.device)
        features = features.unsqueeze(0).expand(x.shape[0], -1)
        return self.heads(features)


class FrozenViTB16Tests(unittest.TestCase):
    """Replace pretrained ViT with a stub to verify CLS encoding and input validation."""
    def setUp(self):
        """Construct small synthetic inputs and isolated fixtures for these tests."""
        self.stub = _StubViT()
        self.patcher = patch("src.encoders.vit_b16.vit_b_16", return_value=self.stub)
        self.factory = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.encoder = FrozenViTB16(device="cpu")

    def test_frozen_model_and_pretrained_weights(self):
        self.factory.assert_called_once_with(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.assertIsInstance(self.encoder.model.heads, nn.Identity)
        self.assertFalse(self.encoder.model.training)
        self.assertFalse(any(p.requires_grad for p in self.encoder.model.parameters()))

    def test_768d_embedding_and_channel_normalization(self):
        rgb = np.full((3, 224, 224), 0.5, dtype=np.float32)
        output = self.encoder.encode(rgb)
        self.assertEqual(output.shape, (768,))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        self.assertAlmostEqual(float(np.linalg.norm(output)), 1.0, places=5)
        self.assertEqual(tuple(self.stub.last_input.shape), (1, 3, 224, 224))
        for channel, (mean, std) in enumerate(zip(self.encoder.mean, self.encoder.std)):
            self.assertAlmostEqual(
                self.stub.last_input[0, channel, 0, 0].item(),
                (0.5 - mean) / std,
                places=5,
            )

    def test_invalid_inputs_are_rejected(self):
        for rgb in (
            np.zeros((224, 224, 3), dtype=np.float32),
            np.full((3, 224, 224), np.nan, dtype=np.float32),
            np.full((3, 224, 224), -0.1, dtype=np.float32),
            np.full((3, 224, 224), 1.1, dtype=np.float32),
        ):
            with self.subTest(shape=rgb.shape):
                with self.assertRaises(ValueError):
                    self.encoder.encode(rgb)
        self.assertIsNone(self.stub.last_input)


if __name__ == "__main__":
    unittest.main()
