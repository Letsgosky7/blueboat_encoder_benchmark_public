"""Check four-encoder cached report generation, input identity and plot annotations.

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

from matplotlib import pyplot as plt

from scripts.benchmark_comparison import (
    CONDITIONS, ENCODERS, SURVEY_PAIRS, generate_report, load_results,
    figure_retrieval_by_pair, figure_matched_margins,
    figure_shared_failures, query_matrix,
)


class ComparisonTests(unittest.TestCase):
    """Build small synthetic cached outputs with aligned query identities."""
    def setUp(self):
        """Construct small synthetic inputs and isolated fixtures for these tests."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for stem, _ in ENCODERS:
            queries, scores, margins = [], [], []
            for condition in CONDITIONS:
                for pair in SURVEY_PAIRS:
                    for patch_id in (1, 2):
                        margins.append(dict(condition=condition, survey_pair=pair,
                                            pair_id=patch_id, margin=0.2,
                                            requires_review=False))
                        for grp in ("positive", "within-site-noncolocated", "isolated-control"):
                            scores.append(dict(condition=condition, survey_pair=pair,
                                               pair_id=patch_id, group=grp,
                                               cosine=0.8 if grp == "positive" else 0.2))
                        a, b = pair.split("-")
                        for direction in (f"{a}->{b}", f"{b}->{a}"):
                            queries.append(dict(condition=condition, survey_pair=pair,
                                                direction=direction, query_patch_id=patch_id,
                                                gallery_size=2, expected_recall_at_1=float(patch_id == 1),
                                                expected_recall_at_3=1.0, tie_count=1))
            queries = pd.DataFrame(queries)
            summary = queries.groupby(["condition", "survey_pair", "direction"], as_index=False).agg(
                queries=("query_patch_id", "size"),
                expected_recall_at_1=("expected_recall_at_1", "mean"),
                expected_recall_at_3=("expected_recall_at_3", "mean"),
            )
            for key, frame in (
                ("retrieval_queries", queries),
                ("retrieval_summary", summary),
                ("matched_null_margins", pd.DataFrame(margins)),
                ("sonar_mask_pair_scores", pd.DataFrame(scores)),
            ):
                frame.to_csv(self.root / f"{stem}_{key}.csv", index=False)
        np.savez_compressed(self.root / "clip_vit_b32_embeddings.npz",
                            model_architecture="ViT-B-32-quickgelu", weights="openai")

    def test_valid_fixtures_generate_pdf_and_five_figures(self):
        output = self.root / "figures"
        metrics, queries = generate_report(self.root, output)
        self.assertEqual(len(metrics), 8)
        self.assertEqual(len(queries), 24)
        self.assertTrue((output / "four_encoder_comparison_report.pdf").is_file())
        self.assertEqual(len(list(output.glob("four_encoder_*.png"))), 5)
        self.assertEqual(len(list(output.glob("four_encoder_*.csv"))), 3)
        margins = pd.read_csv(output / "four_encoder_margin_summary.csv")
        self.assertEqual(len(margins), len(ENCODERS) * len(CONDITIONS) * len(SURVEY_PAIRS))
        self.assertTrue((margins["n_pairs"] == 2).all())
        self.assertTrue(np.allclose(margins["mean_margin"], 0.2))

    def test_annotations_include_baseline_means_and_small_counts(self):
        loaded = load_results(self.root)
        retrieval = figure_retrieval_by_pair(loaded)
        margins = figure_matched_margins(loaded)
        failures = figure_shared_failures(query_matrix(loaded))
        try:
            for ax in retrieval.axes:
                self.assertEqual(len(ax.lines), 1)
                self.assertEqual(ax.lines[0].get_linestyle(), "--")
                self.assertEqual(len(ax.lines[0].get_xdata()), len(SURVEY_PAIRS))
            for ax in margins.axes:
                self.assertTrue(any("Mean:" in label.get_text() for label in ax.texts))
            counts = [label.get_text() for label in failures.axes[1].texts]
            # Synthetic fixture: each pass has 2 rank-1 successes and 2 failures,
            # so the small segment labels must not disappear.
            self.assertIn("2", counts)
            self.assertIn("4", counts)  # total above each survey-pair bar
        finally:
            plt.close(retrieval)
            plt.close(margins)
            plt.close(failures)


    def _write_diagnostic_patches(self):
        """Supply the additional patch/bank/quality inputs required by opt-in mode.

        The ordinary comparison test fixtures intentionally contain cached
        encoder CSVs only: the default report must never require raw patches.
        """
        patch_dir = self.root / "outputs/patches"
        table_dir = self.root / "outputs/tables"
        patch_dir.mkdir(parents=True)
        table_dir.mkdir(parents=True)
        quality = []
        for pair in SURVEY_PAIRS:
            a, b = pair.split("-")
            sonar = np.arange(400, dtype=np.float32).reshape(20, 20)
            mask = np.ones((20, 20), dtype=bool)
            mask[4:7, 4:9] = False
            np.savez_compressed(
                patch_dir / f"positive_pairs_{a}_{b}.npz",
                patch_id=np.array([1, 2], dtype=np.int64),
                patch_a=np.stack([sonar, sonar + 10]),
                patch_b=np.stack([sonar + 3, sonar + 13]),
                mask_a=np.stack([mask, mask]),
                mask_b=np.stack([mask, mask]),
            )
            for patch_id in (1, 2):
                quality.append({
                    "pair": pair, "patch_id": patch_id,
                    "heading_difference_deg": 18.0,
                    "joint_coverage": 0.95, "mask_iou": 1.0,
                    "requires_review": patch_id == 2,
                })
        pd.DataFrame(quality).to_csv(
            table_dir / "positive_pair_quality.csv", index=False
        )
        for survey in ("control", "pass_a", "pass_b", "pass_c"):
            np.savez_compressed(
                patch_dir / f"patch_bank_{survey}.npz",
                bank_index=np.array([0]),
                patch=sonar[None],
                mask=mask[None],
            )
        pd.DataFrame([{
            "null_pair_id": 1, "null_type": "within-site-noncolocated",
            "survey_a": "pass_a", "survey_b": "pass_b",
            "bank_index_a": 0, "bank_index_b": 0,
        }]).to_csv(table_dir / "null_pair_metadata.csv", index=False)

    def test_opt_in_diagnostics_generates_common_review_and_failure_pages(self):
        """One workflow covers all four models without duplicating SONAR images."""
        self._write_diagnostic_patches()
        # Trigger a single near-unity MASK-ONLY null for ResNet50 only.
        path = self.root / "resnet50_sonar_mask_pair_scores.csv"
        scores = pd.read_csv(path)
        index = scores.index[
            (scores["condition"] == "mask-only")
            & (scores["group"] == "within-site-noncolocated")
            & (scores["survey_pair"] == "pass_a-pass_b")
            & (scores["pair_id"] == 1)
        ]
        self.assertEqual(len(index), 1)
        scores.loc[index, "cosine"] = 1.0
        scores.to_csv(path, index=False)

        output = self.root / "comparison_with_diagnostics"
        metrics, comparison = generate_report(
            self.root, output, with_diagnostics=True, dataset_root=self.root
        )
        self.assertEqual(len(metrics), 8)
        self.assertEqual(len(comparison), 24)
        self.assertEqual(len(list(output.glob("four_encoder_*.png"))), 5)
        self.assertEqual(len(list(output.glob("four_encoder_*.csv"))), 3)

        diagnostic_dir = output / "diagnostics"
        near = pd.read_csv(diagnostic_dir / "four_encoder_near_unity_masks.csv")
        self.assertEqual(len(near), 1)
        self.assertEqual(near.iloc[0]["encoder_stem"], "resnet50")
        self.assertTrue(bool(near.iloc[0]["identical_masks"]))
        for stem, _ in ENCODERS:
            self.assertTrue((diagnostic_dir / f"{stem}_near_unity_masks.csv").is_file())

        positives = pd.read_csv(
            diagnostic_dir / "four_encoder_positive_pair_metrics.csv"
        )
        self.assertEqual(len(positives), 24)  # 6 survey-specific IDs × 4 models
        failures = pd.read_csv(
            diagnostic_dir / "four_encoder_shared_failure_queries.csv"
        )
        # Patch ID 2 is missed by every encoder in both query directions.
        self.assertEqual(len(failures), 6)
        self.assertEqual(set(failures["pair_id"]), {2})
        self.assertEqual(
            len(list(diagnostic_dir.glob("four_encoder_positive_pair_review_*.png"))), 2
        )
        self.assertEqual(
            len(list(diagnostic_dir.glob("four_encoder_shared_failure_review_*.png"))), 1
        )

    def test_comparative_failure_selector_rejects_rank_ties(self):
        """A tie cannot silently become a binary all-model retrieval failure."""
        from src.analysis.diagnostics import select_shared_failures

        matrix = query_matrix(load_results(self.root))
        # Shared-failure diagnostics intentionally inspect SONAR only.
        # Sorted query tables begin with MASK-ONLY, so select SONAR explicitly.
        sonar_index = matrix.index[matrix["condition"] == "sonar"][0]
        matrix.loc[sonar_index, "resnet50_tie_count"] = 2
        with self.assertRaisesRegex(ValueError, "untied"):
            select_shared_failures(matrix, ENCODERS)

    def test_stale_clip_archive_rejected(self):
        np.savez_compressed(self.root / "clip_vit_b32_embeddings.npz",
                            model_architecture="ViT-B-32", weights="openai")
        with self.assertRaisesRegex(ValueError, "QuickGELU"):
            load_results(self.root)

    def test_mismatched_query_ids_rejected(self):
        path = self.root / "vit_b16_retrieval_queries.csv"
        queries = pd.read_csv(path)
        queries.loc[0, "query_patch_id"] = 999
        queries.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            load_results(self.root)


if __name__ == "__main__":
    unittest.main()
