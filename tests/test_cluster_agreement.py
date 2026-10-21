"""Synthetic ARI/NMI tests for saved-embedding cluster stability diagnostics."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cluster_agreement import main
from src.analysis.cluster_agreement import (
    compute_cluster_agreement,
    partition_agreement,
    ward_partition,
)
from src.encoders.embedding_store import ARCHIVE_SPECS


class ClusterAgreementTests(unittest.TestCase):
    """All fixtures are isolated; pretrained inference and real XTF are unused."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.rng = np.random.default_rng(20261014)
        self.names = np.array(
            ["pass_a-pass_b"] * 4 + ["pass_a-pass_c"] * 5 +
            ["pass_b-pass_c"] * 16
        )
        self.ids = np.arange(1, 26, dtype=np.int64)
        geo = pd.DataFrame({
            "pair": self.names,
            "patch_id": self.ids,
            "heading_difference_deg": np.linspace(10, 177, 25),
            "look_difference_deg": np.linspace(4, 179, 25),
        })
        audit = pd.DataFrame({
            "pair": self.names,
            "patch_id": self.ids,
            "joint_coverage": np.linspace(0.40, 0.90, 25),
            "requires_review": [(i % 4 == 0) for i in range(25)],
        })
        metadata = self.root / "outputs/tables"
        metadata.mkdir(parents=True)
        geo.sample(frac=1, random_state=7).to_csv(
            metadata / "positive_pair_metadata.csv", index=False
        )
        audit.sample(frac=1, random_state=11).to_csv(
            metadata / "positive_pair_quality.csv", index=False
        )
        self._archive("resnet50")

    @staticmethod
    def _normalise(x):
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    def _archive(self, model, *, identical_mask_b=False):
        dim, _, architecture = ARCHIVE_SPECS[model]
        sonar = self._normalise(self.rng.normal(size=(25, dim)))
        mask = self._normalise(self.rng.normal(size=(25, dim)))
        mask_b = mask.copy()
        if identical_mask_b:
            mask_b = np.tile(mask_b[0:1], (25, 1))
        arrays = {
            "model": np.asarray(model),
            "weights": np.asarray("synthetic-unit-tests"),
            "preprocessing": np.asarray("fixture-unit"),
            "positive_survey_pair": self.names.copy(),
            "positive_patch_id": self.ids.copy(),
            "positive_sonar_a": sonar.copy(),
            "positive_sonar_b": sonar.copy(),
            "positive_mask_only_a": mask,
            "positive_mask_only_b": mask_b,
        }
        if architecture is not None:
            arrays["model_architecture"] = np.asarray(architecture)
        path = self.root / f"outputs/embeddings/{model}_embeddings.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def test_ari_nmi_identity_and_cluster_label_permutation(self):
        """Equivalent partitions score one despite arbitrary label names."""
        a = [0, 0, 1, 1, 2, 2]
        b = [9, 9, 7, 7, 4, 4]
        self.assertEqual(partition_agreement(a, b), (1.0, 1.0))
        self.assertEqual(partition_agreement([0] * 4, [1] * 4), (1.0, 1.0))

    def test_known_independent_partitions(self):
        """Crossing balanced 2+2 partitions have ARI=-0.5 and NMI=0."""
        ari, nmi = partition_agreement([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertAlmostEqual(ari, -0.5, places=12)
        self.assertAlmostEqual(nmi, 0.0, places=12)
        with self.assertRaises(ValueError):
            partition_agreement([0, 1], [0, 1, 2])

    def test_ward_matches_obvious_groups_and_flags_ambiguity(self):
        """Ward scores original features, not a 2D PCA projection."""
        x = np.array([[0, 0], [0.01, 0], [5, 5], [5.01, 5]])
        labels, status, unique, gap = ward_partition(x, k=2)
        self.assertEqual(status, "ok")
        self.assertEqual(unique, 4)
        self.assertGreater(gap, 0)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        square = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels, status, _, _ = ward_partition(square, k=3)
        self.assertIsNone(labels)
        self.assertEqual(status, "ambiguous_tied_cut")
        labels, status, _, _ = ward_partition(np.ones((5, 3)), k=2)
        self.assertIsNone(labels)
        self.assertEqual(status, "insufficient_unique_vectors")

    def test_real_schema_metrics_have_correct_denominators(self):
        """Evaluate n=4/5/16 galleries independently; one model gives 16 rows."""
        metrics, assignments, provenance = compute_cluster_agreement(
            self.root, models=("resnet50",)
        )
        self.assertEqual(len(metrics), 16)  # 2 conditions × (2+3+3 k settings)
        self.assertEqual(len(assignments), 142)  # 2 × (4*2 + 5*3 + 16*3)
        self.assertEqual(len(provenance), 2)
        self.assertTrue(metrics["status"].eq("ok").all())
        self.assertTrue(metrics["ari"].between(-1, 1).all())
        self.assertTrue(metrics["nmi"].between(0, 1).all())
        # Identical SONAR features produce identical partitions by definition.
        sonar = metrics[metrics["condition"] == "sonar"]
        np.testing.assert_allclose(sonar["ari"], 1, atol=1e-12)
        np.testing.assert_allclose(sonar["nmi"], 1, atol=1e-12)
        expected = {"pass_a-pass_b": 2, "pass_a-pass_c": 3, "pass_b-pass_c": 3}
        self.assertEqual(metrics[metrics["condition"] == "sonar"].groupby(
            "survey_pair"
        ).size().to_dict(), expected)
        self.assertEqual(
            assignments[assignments["condition"] == "sonar"]
            .drop_duplicates(["survey_pair", "patch_id"]).shape[0], 25
        )

    def test_degenerate_mask_features_are_reported_not_scored(self):
        """Do not treat identical validity masks as meaningful agreement."""
        self._archive("resnet50", identical_mask_b=True)
        metrics, assignments, _ = compute_cluster_agreement(
            self.root, models=("resnet50",)
        )
        mask = metrics[metrics["condition"] == "mask-only"]
        self.assertTrue(mask["ari"].isna().all())
        self.assertTrue(mask["nmi"].isna().all())
        self.assertTrue(mask["status"].str.contains("insufficient_unique_vectors").all())
        self.assertFalse((assignments["condition"] == "mask-only").any())
        self.assertEqual(len(assignments), 71)  # SONAR only

    def test_incorrect_ids_fail_before_writing_results(self):
        """Archive pair/key mismatch must not create cluster outputs."""
        path = self.root / "outputs/embeddings/resnet50_embeddings.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["positive_patch_id"][1] = 999
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            compute_cluster_agreement(self.root, models=("resnet50",))
        self.assertFalse(
            (self.root / "outputs/analysis/final_evaluation/embeddings/clustering").exists()
        )

    def test_cli_exports_four_annotated_heatmaps_and_preserves_npz(self):
        """Full four-encoder run generates 64 settings, plots, and source-safe CSVs."""
        for model in ("convnext_tiny", "vit_b16", "clip_vit_b32"):
            self._archive(model)
        originals = {
            model: (self.root / f"outputs/embeddings/{model}_embeddings.npz").read_bytes()
            for model in ARCHIVE_SPECS
        }
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(self.root)]), 0)

        out = self.root / "outputs/analysis/final_evaluation/embeddings/clustering"
        names = (
            "cluster_agreement.csv", "cluster_assignments.csv",
            "cluster_provenance.csv",
            "ari_sonar_k2.png", "ari_mask_only_k2.png",
            "nmi_sonar_k2.png", "nmi_mask_only_k2.png",
        )
        for filename in names:
            self.assertTrue((out / filename).is_file(), filename)
        self.assertEqual(len(pd.read_csv(out / "cluster_agreement.csv")), 64)
        self.assertEqual(len(pd.read_csv(out / "cluster_assignments.csv")), 568)
        self.assertEqual(len(pd.read_csv(out / "cluster_provenance.csv")), 8)
        for model, original in originals.items():
            self.assertEqual(
                (self.root / f"outputs/embeddings/{model}_embeddings.npz").read_bytes(),
                original,
            )
        self.assertFalse(
            (self.root / "outputs/analysis/final_evaluation/embeddings/pca").exists()
        )


if __name__ == "__main__":
    unittest.main()
