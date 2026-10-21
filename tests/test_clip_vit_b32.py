"""Check frozen OpenAI CLIP architecture and 512D visual input/feature behavior.

Run: python -m unittest discover -s tests -p 'test_*.py' -v (repository root).
Input: synthetic in-memory arrays and, where needed, temporary CSV/NPZ fixtures.
Output: unittest pass/fail assertions; temporary files are discarded after tests.
No CLI flags; these tests do not require raw BlueBoat XTF recordings.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torchvision.transforms import Normalize

from src.encoders.clip_vit_b32 import FrozenCLIPViTB32


class _StubCLIP(nn.Module):
    """Expose only the CLIP image encoding interface needed by unit tests."""
    def __init__(self):
        """Initialize the test-only encoder stub."""
        super().__init__()
        self.visual_param = nn.Parameter(torch.ones(1))
        self.last_input = None

    def encode_image(self, x):
        """Return a controlled image embedding without invoking pretrained weights."""
        self.last_input = x.detach().clone()
        return torch.arange(1, 513, dtype=x.dtype, device=x.device).unsqueeze(0)


class FrozenCLIPViTB32Tests(unittest.TestCase):
    """Substitute a minimal CLIP factory to verify frozen image-only behavior."""
    def setUp(self):
        """Construct small synthetic inputs and isolated fixtures for these tests."""
        self.stub = _StubCLIP()
        normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.factory = Mock(return_value=(
            self.stub, None, SimpleNamespace(transforms=[normalize])
        ))
        module = types.ModuleType("open_clip")
        module.create_model_and_transforms = self.factory
        patcher = patch.dict(sys.modules, {"open_clip": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.encoder = FrozenCLIPViTB32(device="cpu")

    def test_pretrained_frozen_model(self):
        self.factory.assert_called_once_with("ViT-B-32-quickgelu", pretrained="openai")
        self.assertFalse(self.encoder.model.training)
        self.assertFalse(any(p.requires_grad for p in self.encoder.model.parameters()))

    def test_512d_unit_features_and_clip_normalization(self):
        rgb = np.full((3, 224, 224), 0.5, dtype=np.float32)
        features = self.encoder.encode(rgb)
        self.assertEqual(features.shape, (512,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0, places=5)
        self.assertEqual(tuple(self.stub.last_input.shape), (1, 3, 224, 224))
        for ch, (mean, std) in enumerate(zip(self.encoder.mean, self.encoder.std)):
            self.assertAlmostEqual(
                self.stub.last_input[0, ch, 0, 0].item(),
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
