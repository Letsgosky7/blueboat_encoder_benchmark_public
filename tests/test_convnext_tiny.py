"""Check frozen ConvNeXt-Tiny pretrained interface and normalized embeddings.

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
from torchvision.models import ConvNeXt_Tiny_Weights

from src.encoders.convnext_tiny import FrozenConvNeXtTiny


class _StubConvNeXt(nn.Module):
    """Minimal classifier interface matching torchvision ConvNeXt."""

    def __init__(self):
        """Initialize the test-only encoder stub."""
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Identity(), nn.Flatten(1), nn.Linear(768, 1)
        )
        self.last_input = None

    def forward(self, x):
        """Return a fixed-shape feature vector for the mocked backbone."""
        self.last_input = x.detach().clone()
        features = torch.arange(768, dtype=x.dtype, device=x.device)
        features = features.reshape(1, 768, 1, 1).expand(x.shape[0], -1, -1, -1)
        return self.classifier(features)


class FrozenConvNeXtTinyTests(unittest.TestCase):
    """Replace pretrained ConvNeXt with a stub to verify preprocessing and output shape."""
    def setUp(self):
        """Construct small synthetic inputs and isolated fixtures for these tests."""
        self.stub = _StubConvNeXt()
        self.patcher = patch(
            "src.encoders.convnext_tiny.convnext_tiny", return_value=self.stub
        )
        self.factory = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.encoder = FrozenConvNeXtTiny(device="cpu")

    def test_frozen_backbone_and_pretrained_weights(self):
        self.factory.assert_called_once_with(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        self.assertIsInstance(self.encoder.model.classifier[-1], nn.Identity)
        self.assertFalse(self.encoder.model.training)
        self.assertFalse(any(p.requires_grad for p in self.encoder.model.parameters()))

    def test_768d_l2_normalized_embedding_and_imagenet_normalization(self):
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
            with self.subTest(shape=rgb.shape, invalid=float(rgb.flat[0])):
                with self.assertRaises(ValueError):
                    self.encoder.encode(rgb)
        self.assertIsNone(self.stub.last_input)


if __name__ == "__main__":
    unittest.main()
