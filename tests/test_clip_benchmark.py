"""Check CLIP archive provenance, saved-vector score integrity and retrieval.

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

from src.benchmark.pipeline import append_scores, positive_pair_path
from src.benchmark.reporting import analyze_results, report_retrieval
from src.encoders.embedding_store import save_embeddings, verify_embeddings
from src.analysis.retrieval import evaluate_positive_retrieval


def observation(index):
    """Create a synthetic archive observation with an explicit patch identity."""
    def unit_vector(i):
        """Build a deterministic unit-norm embedding for a synthetic observation."""
        result = np.zeros(512, dtype=np.float32)
        result[i] = 1.0
        return result

    return {"sonar": unit_vector(index), "mask-only": unit_vector(index + 100)}


class CLIPBenchmarkTests(unittest.TestCase):
    """Verify CLIP archive and retrieval workflows without real sonar or pretrained weights."""
    def test_positive_pair_filenames_match_stored_npz_files(self):
        directory = Path("outputs/patches")
        expected = (
            ("pass_a", "pass_b", "positive_pairs_pass_a_pass_b.npz"),
            ("pass_a", "pass_c", "positive_pairs_pass_a_pass_c.npz"),
            ("pass_b", "pass_c", "positive_pairs_pass_b_pass_c.npz"),
        )
        for survey_a, survey_b, filename in expected:
            with self.subTest(surveys=(survey_a, survey_b)):
                self.assertEqual(
                    positive_pair_path(directory, survey_a, survey_b),
                    directory / filename,
                )

    def test_archive_and_retrieval_use_the_original_schema(self):
        positives = [
            ("pass_a-pass_b", 1, observation(0), observation(0)),
            ("pass_a-pass_b", 2, observation(1), observation(1)),
        ]
        bank = {
            ("pass_a", 0): observation(20),
            ("pass_b", 0): observation(21),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs/embeddings"
            tables = root / "outputs/tables"
            tables.mkdir(parents=True)
            archive = output / "clip_vit_b32_embeddings.npz"
            score_path = output / "clip_vit_b32_sonar_mask_pair_scores.csv"
            null_path = tables / "null_pair_metadata.csv"

            negatives = pd.DataFrame([
                {
                    "null_pair_id": null_id,
                    "null_type": "within-site-noncolocated",
                    "survey_a": "pass_a", "survey_b": "pass_b",
                    "bank_index_a": 0, "bank_index_b": 0,
                    "heading_difference_deg": 154.0,
                    "matched_positive_patch_id": patch_id,
                }
                for null_id, patch_id in ((101, 1), (102, 2))
            ])
            negatives.to_csv(null_path, index=False)
            rows = []
            for pair, patch_id, left, right in positives:
                append_scores(
                    rows, left, right, group="positive",
                    pair=pair, pair_id=patch_id, heading_deg=154.0,
                )
            for null_id in (101, 102):
                append_scores(
                    rows, bank[("pass_a", 0)], bank[("pass_b", 0)],
                    group="within-site-noncolocated", pair="pass_a-pass_b",
                    pair_id=null_id, heading_deg=154.0,
                )
            output.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(score_path, index=False)

            save_embeddings(
                archive, positives, bank, "openai", "cpu", model="clip_vit_b32"
            )
            with np.load(archive, allow_pickle=False) as saved:
                self.assertEqual(str(saved["model"]), "clip_vit_b32")
                self.assertEqual(str(saved["model_architecture"]), "ViT-B-32-quickgelu")
                self.assertEqual(str(saved["weights"]), "openai")
                self.assertEqual(saved["positive_sonar_a"].shape, (2, 512))
                self.assertEqual(saved["bank_mask_only"].shape, (2, 512))
            self.assertLessEqual(
                verify_embeddings(archive, score_path, null_path, model="clip_vit_b32"), 1e-6
            )

            pd.DataFrame([
                {"pair": "pass_a-pass_b", "patch_id": patch_id,
                 "requires_review": False}
                for patch_id in (1, 2)
            ]).to_csv(tables / "positive_pair_quality.csv", index=False)
            pd.DataFrame([
                {"pair": "pass_a-pass_b", "patch_id": 1,
                 "midpoint_east": 0.0, "midpoint_north": 0.0, "patch_size_m": 50.0},
                {"pair": "pass_a-pass_b", "patch_id": 2,
                 "midpoint_east": 25.0, "midpoint_north": 0.0, "patch_size_m": 50.0},
            ]).to_csv(tables / "positive_pair_metadata.csv", index=False)

            analyze_results(root, "clip_vit_b32")
            report_retrieval(root, "clip_vit_b32")
            queries, summary = evaluate_positive_retrieval(archive)
            self.assertEqual(len(queries), 8)  # 2 pairs x 2 directions x 2 conditions
            self.assertEqual(len(summary), 4)
            self.assertTrue(np.allclose(summary["expected_recall_at_1"], 1.0))
            for filename in (
                "clip_vit_b32_matched_null_margins.csv",
                "clip_vit_b32_retrieval_queries.csv",
                "clip_vit_b32_retrieval_summary.csv",
                "clip_vit_b32_retrieval_candidates.csv",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertTrue(
                (root / "outputs/figures/clip_vit_b32_similarity_distributions.png").is_file()
            )

    def test_old_non_quickgelu_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outdated_clip_archive.npz"
            np.savez_compressed(path, model="clip_vit_b32")
            with self.assertRaisesRegex(ValueError, "QuickGELU"):
                verify_embeddings(path, "unused.csv", "unused_nulls.csv", model="clip_vit_b32")

    def test_rejects_wrong_feature_size_before_writing(self):
        positives = [(
            "pass_a-pass_b", 1, observation(0),
            {"sonar": np.zeros(768, dtype=np.float32),
             "mask-only": observation(0)["mask-only"]},
        )]
        bank = {("pass_a", 0): observation(10)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            with self.assertRaises(ValueError):
                save_embeddings(
                    path, positives, bank, "openai", "cpu", model="clip_vit_b32"
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
