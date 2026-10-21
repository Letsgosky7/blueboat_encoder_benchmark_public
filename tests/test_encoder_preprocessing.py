"""Check mask-safe RGB preprocessing, cosine verification and tie-aware retrieval.

Run: python -m unittest discover -s tests -p 'test_*.py' -v (repository root).
Input: synthetic in-memory arrays and, where needed, temporary CSV/NPZ fixtures.
Output: unittest pass/fail assertions; temporary files are discarded after tests.
No CLI flags; these tests do not require raw BlueBoat XTF recordings.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoders.preprocessing import prepare_sonar_rgb
from src.analysis.retrieval import (
    rank_queries, evaluate_positive_retrieval, rank_geographic_candidates,
)
from src.encoders.embedding_store import (
    save_embeddings,
    verify_embeddings,
)


class SonarPreprocessingTests(unittest.TestCase):
    """Check mask-aware resizing and intensity handling on synthetic patch arrays."""
    def test_rgb_shape_dtype_and_range(self):
        patch = np.arange(400, dtype=np.float32).reshape(20, 20)
        mask = np.ones((20, 20), dtype=bool)
        rgb, valid = prepare_sonar_rgb(patch, mask)
        self.assertEqual(rgb.shape, (3, 224, 224))
        self.assertEqual(valid.shape, (224, 224))
        self.assertEqual(rgb.dtype, np.float32)
        self.assertTrue(np.isfinite(rgb).all())
        self.assertGreaterEqual(float(rgb.min()), 0)
        self.assertLessEqual(float(rgb.max()), 1)
        np.testing.assert_array_equal(rgb[0], rgb[1])
        np.testing.assert_array_equal(rgb[1], rgb[2])

    def test_real_patch_grid_200_to_224_with_nodata(self):
        """The production 50m/0.25m raster can enter each frozen benchmark."""
        patch = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
        mask = np.ones_like(patch, dtype=bool)
        mask[40:80, 50:100] = False
        patch[~mask] = np.nan

        rgb, resized_mask = prepare_sonar_rgb(patch, mask)
        self.assertEqual(rgb.shape, (3, 224, 224))
        self.assertEqual(resized_mask.shape, (224, 224))
        self.assertTrue(np.isfinite(rgb).all())
        self.assertGreaterEqual(float(rgb.min()), 0.0)
        self.assertLessEqual(float(rgb.max()), 1.0)
        self.assertFalse(resized_mask.all())
        np.testing.assert_array_equal(rgb[0], rgb[1])
        np.testing.assert_array_equal(rgb[1], rgb[2])

    def test_nodata_is_filled_but_mask_is_preserved(self):
        patch = np.arange(400, dtype=np.float32).reshape(20, 20)
        mask = np.ones((20, 20), dtype=bool)
        mask[:, 8:12] = False
        patch[~mask] = np.nan
        rgb, valid = prepare_sonar_rgb(patch, mask, target_size=(20, 20))
        self.assertTrue(np.isfinite(rgb).all())
        np.testing.assert_array_equal(valid, mask)
        self.assertTrue(np.all(rgb[0, ~mask] == np.median(rgb[0, mask])))

    def test_original_arrays_are_unchanged(self):
        patch = np.arange(400, dtype=np.float32).reshape(20, 20)
        mask = np.ones((20, 20), dtype=bool)
        mask[0, :] = False
        patch[~mask] = np.nan
        before = patch.copy()
        prepare_sonar_rgb(patch, mask)
        np.testing.assert_array_equal(patch, before)

    def test_uniform_patch_is_neutral_gray(self):
        patch = np.full((20, 20), 7.0, dtype=np.float32)
        rgb, _ = prepare_sonar_rgb(patch, np.ones_like(patch, dtype=bool))
        self.assertTrue(np.allclose(rgb, 0.5))

    def test_invalid_masks_and_values_are_rejected(self):
        patch = np.ones((20, 20), dtype=np.float32)
        with self.assertRaises(ValueError):
            prepare_sonar_rgb(patch, np.zeros((20, 20), dtype=bool))
        with self.assertRaises(ValueError):
            prepare_sonar_rgb(patch, np.ones((19, 20), dtype=bool))
        patch[0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_sonar_rgb(patch, np.ones((20, 20), dtype=bool))

    def test_invalid_configuration_is_rejected(self):
        patch = np.ones((20, 20), dtype=np.float32)
        mask = np.ones((20, 20), dtype=bool)
        with self.assertRaises(ValueError):
            prepare_sonar_rgb(patch, mask, percentiles=(99, 1))
        with self.assertRaises(ValueError):
            prepare_sonar_rgb(patch, mask, nodata_fill="unknown")
        with self.assertRaises(ValueError):
            prepare_sonar_rgb(patch, mask, target_size=(0, 224))



class EmbeddingArchiveTests(unittest.TestCase):
    """Check that persisted unit vectors reproduce their scored pair cosines."""

    def test_generic_store_rejects_unknown_or_mismatched_model(self):
        """An archive cannot silently be mislabeled as another encoder."""
        vector = np.zeros(2048, dtype=np.float32)
        vector[0] = 1.0
        features = {"sonar": vector, "mask-only": vector}
        positives = [("pass_a-pass_b", 1, features, features)]
        bank = {("pass_a", 0): features}

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "vectors.npz"
            with self.assertRaisesRegex(ValueError, "Unsupported encoder"):
                save_embeddings(
                    archive, positives, bank, "IMAGENET1K_V2", "cpu",
                    model="unknown_encoder",
                )
            self.assertFalse(archive.exists())

            save_embeddings(
                archive, positives, bank, "IMAGENET1K_V2", "cpu",
                model="resnet50",
            )
            with self.assertRaisesRegex(ValueError, "convnext_tiny"):
                verify_embeddings(
                    archive, "not_needed.csv", "not_needed.csv",
                    model="convnext_tiny",
                )

    def test_saved_vectors_reproduce_pair_cosines(self):
        def unit_vector(index):
            """Build a deterministic unit-norm embedding for a synthetic observation."""
            vector = np.zeros(2048, dtype=np.float32)
            vector[index] = 1.0
            return vector

        pos_a = {"sonar": unit_vector(0), "mask-only": unit_vector(1)}
        pos_b = {"sonar": unit_vector(0), "mask-only": unit_vector(2)}
        bank = {
            ("pass_a", 0): {"sonar": unit_vector(0), "mask-only": unit_vector(1)},
            ("pass_b", 0): {"sonar": unit_vector(2), "mask-only": unit_vector(1)},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "features.npz"
            scores_csv = root / "scores.csv"
            null_csv = root / "nulls.csv"
            save_embeddings(
                archive, [("pass_a-pass_b", 1, pos_a, pos_b)],
                bank, "IMAGENET1K_V2", "cpu", model="resnet50",
            )
            pd.DataFrame([
                {"group": "positive", "condition": "sonar",
                 "survey_pair": "pass_a-pass_b", "pair_id": 1, "cosine": 1.0},
                {"group": "positive", "condition": "mask-only",
                 "survey_pair": "pass_a-pass_b", "pair_id": 1, "cosine": 0.0},
                {"group": "within-site-noncolocated", "condition": "sonar",
                 "survey_pair": "pass_a-pass_b", "pair_id": 2, "cosine": 0.0},
                {"group": "within-site-noncolocated", "condition": "mask-only",
                 "survey_pair": "pass_a-pass_b", "pair_id": 2, "cosine": 1.0},
            ]).to_csv(scores_csv, index=False)
            pd.DataFrame([{
                "null_pair_id": 2, "null_type": "within-site-noncolocated",
                "survey_a": "pass_a", "survey_b": "pass_b",
                "bank_index_a": 0, "bank_index_b": 0,
            }]).to_csv(null_csv, index=False)

            with np.load(archive, allow_pickle=False) as saved:
                self.assertEqual(saved["positive_sonar_a"].shape, (1, 2048))
                self.assertEqual(saved["bank_mask_only"].shape, (2, 2048))
                self.assertEqual(str(saved["weights"]), "IMAGENET1K_V2")

            self.assertLessEqual(
                verify_embeddings(archive, scores_csv, null_csv, model="resnet50"), 1e-6
            )
            scores = pd.read_csv(scores_csv)
            scores.loc[0, "cosine"] = 0.5
            scores.to_csv(scores_csv, index=False)
            with self.assertRaises(ValueError):
                verify_embeddings(archive, scores_csv, null_csv, model="resnet50")



class RetrievalMetricTests(unittest.TestCase):
    """Check exact-ID rankings, tie handling and nominal geographic diagnostics."""
    def test_perfect_retrieval_and_uniform_ties(self):
        unique = np.eye(4, dtype=np.float32)
        same = np.tile(unique[0], (4, 1))
        ids = np.arange(1, 5)

        correct = rank_queries(
            unique, unique, ids, survey_pair="pass_a-pass_b",
            condition="sonar", direction="pass_a->pass_b",
        )
        self.assertTrue(all(row["expected_recall_at_1"] == 1 for row in correct))
        self.assertTrue(all(row["expected_recall_at_3"] == 1 for row in correct))
        self.assertTrue(all(row["tie_count"] == 1 for row in correct))

        tied = rank_queries(
            same, same, ids, survey_pair="pass_a-pass_b",
            condition="mask-only", direction="pass_a->pass_b",
        )
        self.assertTrue(all(row["tie_count"] == 4 for row in tied))
        self.assertTrue(all(row["expected_recall_at_1"] == 0.25 for row in tied))
        self.assertTrue(all(row["expected_recall_at_3"] == 0.75 for row in tied))

    def test_retrieval_reads_saved_archive_both_directions(self):
        unit = np.eye(4, dtype=np.float32)
        same = np.tile(unit[0], (4, 1))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "vectors.npz"
            np.savez_compressed(
                archive,
                positive_survey_pair=np.asarray(["pass_a-pass_b"] * 4),
                positive_patch_id=np.arange(1, 5),
                positive_sonar_a=unit,
                positive_sonar_b=unit,
                positive_mask_only_a=same,
                positive_mask_only_b=same,
            )
            queries, summary = evaluate_positive_retrieval(archive)

        self.assertEqual(len(queries), 16)
        self.assertEqual(len(summary), 4)
        sonar = summary.loc[summary["condition"] == "sonar"]
        mask = summary.loc[summary["condition"] == "mask-only"]
        self.assertTrue(np.allclose(sonar["expected_recall_at_1"], 1.0))
        self.assertTrue(np.allclose(mask["expected_recall_at_1"], 0.25))
        self.assertTrue(np.allclose(mask["expected_recall_at_3"], 0.75))
        self.assertTrue(np.allclose(summary["chance_recall_at_1"], 0.25))
        self.assertTrue(np.allclose(summary["chance_recall_at_3"], 0.75))
        self.assertEqual(set(summary["direction"]), {"pass_a->pass_b", "pass_b->pass_a"})


    def test_geographic_candidates_from_saved_vectors(self):
        # IDs 1 and 2 overlap by 50% of the nominal 50m square;
        # ResNet50-like features deliberately swap their best matches.
        identity = np.eye(3, dtype=np.float32)
        swapped = identity[[1, 0, 2]]
        flat = np.tile(identity[0], (3, 1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "features.npz"
            positions = root / "positive_metadata.csv"
            np.savez_compressed(
                archive,
                positive_survey_pair=np.asarray(["pass_a-pass_b"] * 3),
                positive_patch_id=np.asarray([1, 2, 3]),
                positive_sonar_a=identity,
                positive_sonar_b=swapped,
                positive_mask_only_a=flat,
                positive_mask_only_b=flat,
            )
            pd.DataFrame({
                "pair": ["pass_a-pass_b"] * 3,
                "patch_id": [1, 2, 3],
                "midpoint_east": [0.0, 25.0, 200.0],
                "midpoint_north": [0.0, 0.0, 0.0],
                "patch_size_m": [50.0] * 3,
            }).to_csv(positions, index=False)
            candidates = rank_geographic_candidates(archive, positions)
            query = candidates.loc[
                (candidates["condition"] == "sonar")
                & (candidates["direction"] == "pass_a->pass_b")
                & (candidates["query_patch_id"] == 1)
            ].sort_values("candidate_rank")
            best = query.iloc[0]
            self.assertEqual(len(candidates), 36)  # 3 queries x 2 dirs x 2 conditions x 3 ranks
            self.assertEqual(int(best["candidate_patch_id"]), 2)
            self.assertFalse(bool(best["exact_id_match"]))
            self.assertAlmostEqual(float(best["center_distance_m"]), 25.0)
            self.assertAlmostEqual(float(best["nominal_footprint_overlap_fraction"]), 0.5)
            self.assertTrue(bool(best["center_within_50m"]))
            self.assertTrue(bool(best["nominal_overlap_ge_25pct"]))

            distant = query.loc[query["candidate_patch_id"] == 3].iloc[0]
            self.assertAlmostEqual(float(distant["nominal_footprint_overlap_fraction"]), 0.0)
            self.assertFalse(bool(distant["center_within_50m"]))

            mask = candidates.loc[
                (candidates["condition"] == "mask-only")
                & (candidates["direction"] == "pass_a->pass_b")
                & (candidates["query_patch_id"] == 2)
                & (candidates["candidate_rank"] == 1)
            ].iloc[0]
            self.assertEqual(int(mask["candidate_patch_id"]), 1)
            self.assertEqual(int(mask["near_tied_candidates"]), 3)

            missing = pd.read_csv(positions).iloc[:2]
            missing.to_csv(positions, index=False)
            with self.assertRaisesRegex(ValueError, "Missing geographic location"):
                rank_geographic_candidates(archive, positions)



if __name__ == "__main__":
    unittest.main()
