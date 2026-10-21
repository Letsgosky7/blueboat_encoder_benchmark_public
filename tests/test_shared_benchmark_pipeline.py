"""Regression tests for the common four-encoder benchmark.

Input: synthetic in-memory SONAR patches and temporary NPZ/CSV fixtures.
Output: unit assertions, with all files deleted after the tests.
Run: python -m unittest discover -s tests -p 'test_*.py' -v.
No real model weights or XTF dataset are required.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.benchmark import BANK_SURVEYS, POSITIVE_PAIRS
from src.benchmark.pipeline import ENCODER_CLASSES, evaluate_results, main


class FakeEncoder:
    """Produce unit-norm vectors of the selected model dimension without weights."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.weights = SimpleNamespace(name="synthetic")
        self.device = "cpu"
        self.calls = 0

    def encode(self, rgb):
        """Compute a deterministic embedding from a validated three-channel image."""
        self.calls += 1
        assert rgb.shape == (3, 224, 224)
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[:2] = (float(rgb.mean()), 1.0)
        return vector / np.linalg.norm(vector)


class SharedBenchmarkPipelineTests(unittest.TestCase):
    """Exercise common score generation and cached-only CLI routing."""

    @staticmethod
    def write_fixture(root):
        """Create 3 positive pairs, 4 bank observations, and 2 negative pairs."""
        patches = root / "outputs/patches"
        tables = root / "outputs/tables"
        patches.mkdir(parents=True)
        tables.mkdir(parents=True)
        for patch_id, (survey_a, survey_b) in enumerate(POSITIVE_PAIRS, 1):
            p = np.full((20, 20), patch_id, dtype=np.float32)
            valid = np.ones((20, 20), dtype=bool)
            valid[4:8, 5:9] = False
            np.savez_compressed(
                patches / f"positive_pairs_{survey_a}_{survey_b}.npz",
                patch_id=np.array([patch_id]),
                patch_a=p[None], patch_b=(p + 1)[None],
                mask_a=valid[None], mask_b=valid[None],
                heading_difference_deg=np.array([90.0]),
            )
        for i, survey in enumerate(BANK_SURVEYS):
            p = np.full((20, 20), i + 1, dtype=np.float32)
            valid = np.ones((20, 20), dtype=bool)
            np.savez_compressed(
                patches / f"patch_bank_{survey}.npz",
                bank_index=np.array([0]), patch=p[None], mask=valid[None],
            )
        pd.DataFrame([
            dict(null_pair_id=101, null_type="within-site-noncolocated",
                 survey_a="pass_a", survey_b="pass_b",
                 bank_index_a=0, bank_index_b=0, heading_difference_deg=30.0),
            dict(null_pair_id=102, null_type="isolated-control",
                 survey_a="control", survey_b="pass_c",
                 bank_index_a=0, bank_index_b=0, heading_difference_deg=120.0),
        ]).to_csv(tables / "null_pair_metadata.csv", index=False)

    def test_stub_inference_saves_and_verifies_four_model_archives(self):
        """All backbones share exactly the same pair and condition protocol."""
        dims = {"resnet50": 2048, "convnext_tiny": 768,
                "vit_b16": 768, "clip_vit_b32": 512}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            for model_name, dimension in dims.items():
                with self.subTest(encoder=model_name):
                    fake = FakeEncoder(dimension)
                    evaluate_results(
                        root, model_name, encoder=fake, expected_counts=(3, 4, 2)
                    )
                    # Six positive observations plus four bank observations,
                    # each encoded once per SONAR and MASK-ONLY condition.
                    self.assertEqual(fake.calls, 20)
                    outputs = root / "outputs/embeddings"
                    scores = pd.read_csv(
                        outputs / f"{model_name}_sonar_mask_pair_scores.csv"
                    )
                    self.assertEqual(len(scores), 10)  # (3+2) pairs, two conditions
                    self.assertEqual(set(scores["condition"]), {"sonar", "mask-only"})
                    with np.load(outputs / f"{model_name}_embeddings.npz") as archive:
                        self.assertEqual(str(archive["model"]), model_name)
                        self.assertEqual(archive["positive_sonar_a"].shape, (3, dimension))
                        self.assertEqual(archive["bank_mask_only"].shape, (4, dimension))
                        if model_name == "clip_vit_b32":
                            self.assertEqual(
                                str(archive["model_architecture"]), "ViT-B-32-quickgelu"
                            )

    def test_cached_modes_do_not_run_encoder(self):
        """All model CLI flags route to cached operations only."""
        with (
            patch("src.benchmark.pipeline.evaluate_results") as inference,
            patch("src.benchmark.reporting.analyze_results") as analyze,
            patch("src.benchmark.reporting.report_retrieval") as retrieval,
            patch("src.benchmark.pipeline.verify_embeddings") as verification,
        ):
            for model_name in ENCODER_CLASSES:
                main(model_name, argv=["--analyze-only"])
                main(model_name, argv=["--retrieval-only"])
                main(model_name, argv=["--verify-embeddings"])
            inference.assert_not_called()
            self.assertEqual(analyze.call_count, 4)
            self.assertEqual(retrieval.call_count, 8)
            self.assertEqual(verification.call_count, 4)

    def test_unknown_model_rejected(self):
        """Prevent accidental use of an unregistered model without loading it."""
        with self.assertRaisesRegex(ValueError, "Unknown benchmark encoder"):
            main("invalid", argv=["--verify-embeddings"])


if __name__ == "__main__":
    unittest.main()
