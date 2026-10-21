"""Synthetic tests for frozen-embedding PCA; no model inference or XTF inputs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.embedding_pca import main
from src.analysis.embedding_pca import (
    compute_embedding_pca,
    fit_pca,
    read_positive_embeddings,
    read_reference_pairs,
)
from src.encoders.embedding_store import ARCHIVE_SPECS


class EmbeddingPCATests(unittest.TestCase):
    """Keep all sources and generated artifacts inside a temporary fake project."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.rng = np.random.default_rng(38)
        pair_names = (
            ["pass_a-pass_b"] * 4 +
            ["pass_a-pass_c"] * 5 +
            ["pass_b-pass_c"] * 16
        )
        self.names = np.array(pair_names)
        self.ids = np.arange(1, 26, dtype=np.int64)
        geo = pd.DataFrame({
            "pair": self.names, "patch_id": self.ids,
            "heading_difference_deg": np.linspace(2, 179, 25),
            "look_difference_deg": np.linspace(4, 175, 25),
        })
        quality = pd.DataFrame({
            "pair": self.names, "patch_id": self.ids,
            "joint_coverage": np.linspace(0.52, 0.88, 25),
            "requires_review": np.array([i % 5 == 0 for i in range(25)]),
        })
        tables = self.root / "outputs/tables"
        tables.mkdir(parents=True)
        # Shuffle independently: joins must not depend on CSV row position.
        geo.sample(frac=1, random_state=19).to_csv(
            tables / "positive_pair_metadata.csv", index=False
        )
        quality.sample(frac=1, random_state=41).to_csv(
            tables / "positive_pair_quality.csv", index=False
        )
        self._write_model("resnet50")

    def _write_model(self, model):
        """Create four unit-normalized feature arrays matching the real schema."""
        dim, _, architecture = ARCHIVE_SPECS[model]

        def normalized(a):
            return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)

        base = normalized(self.rng.normal(size=(25, dim)))
        close = normalized(base + self.rng.normal(scale=0.12, size=(25, dim)))
        mask = normalized(self.rng.normal(size=(25, dim)))
        mask_b = normalized(mask + self.rng.normal(scale=0.17, size=(25, dim)))
        arrays = {
            "model": np.asarray(model),
            "weights": np.asarray("fixture"),
            "preprocessing": np.asarray("fixture-L2-unit"),
            "positive_survey_pair": self.names,
            "positive_patch_id": self.ids,
            "positive_sonar_a": base,
            "positive_sonar_b": close,
            "positive_mask_only_a": mask,
            "positive_mask_only_b": mask_b,
        }
        if architecture is not None:
            arrays["model_architecture"] = np.asarray(architecture)
        target = self.root / "outputs/embeddings" / f"{model}_embeddings.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        return target

    def test_pca_rank_and_variance_and_translation_invariance(self):
        """Center, fit two PCs and preserve all variance of a rank-2 dataset."""
        x = np.array([
            [1, 0, 0], [0, 1, 0], [-2, -1, 0],
            [3, -1, 0], [-1, 2, 0],
        ], dtype=float)
        scores, ratios = fit_pca(x)
        self.assertEqual(scores.shape, (5, 2))
        self.assertAlmostEqual(float(ratios.sum()), 1, places=10)
        np.testing.assert_allclose(scores.mean(axis=0), [0, 0], atol=1e-10)
        translated, translated_ratios = fit_pca(x + [100, -50, 9])
        np.testing.assert_allclose(translated, scores, atol=1e-9)
        np.testing.assert_allclose(translated_ratios, ratios, atol=1e-12)
        for i, j in ((0, 1), (2, 3), (1, 4)):
            np.testing.assert_allclose(
                np.linalg.norm(scores[i] - scores[j]),
                np.linalg.norm(x[i] - x[j]), atol=1e-9
            )

    def test_pca_rejects_degenerate_and_nonfinite_vectors(self):
        """PCA is not scientifically meaningful for constant features."""
        with self.assertRaisesRegex(ValueError, "constant"):
            fit_pca(np.ones((4, 8)))
        invalid = np.ones((4, 8))
        invalid[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fit_pca(invalid)

    def test_input_alignment_and_provenance(self):
        """Archive IDs must match quality/geometry despite shuffled metadata."""
        reference = read_reference_pairs(self.root)
        self.assertEqual(len(reference), 25)
        keys, vectors, provenance = read_positive_embeddings(
            self.root / "outputs/embeddings/resnet50_embeddings.npz",
            "resnet50", set(reference.index.to_list()),
        )
        self.assertEqual(len(keys), 25)
        self.assertEqual(vectors[("sonar", "a")].shape, (25, 2048))
        self.assertEqual(provenance["weights"], "fixture")

        coords, variance, pairs, provenance_table, _ = compute_embedding_pca(
            self.root, models=("resnet50",)
        )
        self.assertEqual(len(coords), 100)
        self.assertEqual(len(variance), 2)
        self.assertEqual(len(pairs), 50)
        self.assertEqual(len(provenance_table), 2)
        self.assertTrue(variance["two_pc_explained_variance_ratio"].between(0, 1).all())
        self.assertTrue(pairs["euclidean_2d_pca"].ge(0).all())
        for record in pairs.itertuples():
            self.assertAlmostEqual(
                record.euclidean_full_embedding**2,
                2 * (1 - record.cosine_full_embedding),
                places=4,
            )
            expected_coverage = (0.52 + (record.patch_id - 1) * (0.88 - 0.52) / 24)
            self.assertAlmostEqual(record.joint_coverage, expected_coverage, places=7)

    def test_missing_or_invalid_archive_does_not_produce_output(self):
        """Reject missing model and incorrect identity without writing reports."""
        with self.assertRaises(FileNotFoundError):
            compute_embedding_pca(self.root, models=("clip_vit_b32",))
        archive = self.root / "outputs/embeddings/resnet50_embeddings.npz"
        with np.load(archive, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["positive_patch_id"][0] = 99
        np.savez_compressed(archive, **arrays)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            compute_embedding_pca(self.root, models=("resnet50",))
        self.assertFalse((self.root / "outputs/analysis/final_evaluation/embeddings/pca").exists())

    def test_cli_creates_only_pca_reports_without_changing_source(self):
        """One selected model writes 2 plots, 4 CSVs and leaves NPZ untouched."""
        original = self.root / "outputs/embeddings/resnet50_embeddings.npz"
        before = original.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(self.root), "--models", "resnet50"]), 0)
        path = self.root / "outputs/analysis/final_evaluation/embeddings/pca"
        for name in (
            "pca_coordinates.csv", "pca_variance.csv",
            "pca_pair_distances.csv", "pca_provenance.csv",
            "resnet50_sonar_pca.png", "resnet50_mask_only_pca.png",
        ):
            self.assertTrue((path / name).is_file(), name)
        self.assertEqual(len(pd.read_csv(path / "pca_coordinates.csv")), 100)
        self.assertEqual(len(pd.read_csv(path / "pca_pair_distances.csv")), 50)
        self.assertEqual(original.read_bytes(), before)
        self.assertFalse((self.root / "outputs/analysis/final_evaluation/embeddings/clustering").exists())


if __name__ == "__main__":
    unittest.main()
