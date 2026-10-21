"""Regression checks for cached, independent positive-pair diagnostics.

Run: python -m unittest discover -s tests -p 'test_*.py' -v
Synthetic CSV fixtures only; no sonar data or encoder weights are needed.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark_comparison import ENCODERS
from scripts.positive_pair_diagnostics import create_diagnostics, make_stratified_summary, main


class PositivePairDiagnosticsTests(unittest.TestCase):
    """Ensure all model joins and denominators match the original pair protocol."""

    def setUp(self):
        """Create two independent fake matches and four-model cached score rows."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tables = self.root / "outputs/tables"
        diag = self.root / "outputs/analysis/frozen_encoders/diagnostics"
        tables.mkdir(parents=True)
        diag.mkdir(parents=True)
        self.tables, self.diag = tables, diag

        pd.DataFrame([
            dict(pair="pass_a-pass_b", patch_id=1, distance_m=0.2,
                 heading_difference_deg=10.0, look_difference_deg=170.0,
                 incidence_difference_deg=8.0, view_class="opposite-view",
                 a_side="port", b_side="starboard"),
            dict(pair="pass_a-pass_c", patch_id=2, distance_m=0.6,
                 heading_difference_deg=4.0, look_difference_deg=4.0,
                 incidence_difference_deg=5.0, view_class="similar-view",
                 a_side="port", b_side="port"),
        ]).to_csv(tables / "positive_pair_metadata.csv", index=False)

        pd.DataFrame([
            dict(pair="pass_a-pass_b", patch_id=1, coverage_a=0.7,
                 coverage_b=0.7, joint_coverage=0.55,
                 central_joint_coverage=0.72, mask_iou=0.65,
                 max_nodata_region=0.30, requires_review=True,
                 review_reason="large-nodata-region"),
            dict(pair="pass_a-pass_c", patch_id=2, coverage_a=0.9,
                 coverage_b=0.9, joint_coverage=0.82,
                 central_joint_coverage=0.95, mask_iou=0.84,
                 max_nodata_region=0.09, requires_review=False,
                 review_reason=""),
        ]).to_csv(tables / "positive_pair_quality.csv", index=False)

        rows = []
        for stem, _ in ENCODERS:
            rows.extend([
                dict(encoder_stem=stem, survey_pair="pass_a-pass_b",
                     pair_id=1, positive_cosine=0.7, margin=-0.1,
                     recall_at_1_forward=0.0, recall_at_1_reverse=1.0),
                dict(encoder_stem=stem, survey_pair="pass_a-pass_c",
                     pair_id=2, positive_cosine=0.9, margin=0.2,
                     recall_at_1_forward=1.0, recall_at_1_reverse=0.0),
            ])
        pd.DataFrame(rows).to_csv(
            diag / "four_encoder_positive_pair_metrics.csv", index=False
        )
        pd.DataFrame([
            dict(survey_pair="pass_a-pass_b",
                 direction="pass_a->pass_b", pair_id=1, failed_models=4),
            dict(survey_pair="pass_a-pass_c",
                 direction="pass_c->pass_a", pair_id=2, failed_models=4),
        ]).to_csv(diag / "four_encoder_shared_failure_queries.csv", index=False)

    def test_join_preserves_model_metrics_and_quality(self):
        """Two pairs must generate eight model metrics without duplicate joins."""
        table, summary = create_diagnostics(self.root)
        self.assertEqual(len(table), 2)
        self.assertEqual(len(summary), 2)
        self.assertEqual(table["patch_id"].tolist(), [1, 2])
        self.assertEqual(table["total_r1_credit"].tolist(), [4.0, 4.0])
        self.assertEqual(table["shared_failure_directions"].tolist(), [1, 1])
        self.assertEqual(table["look_opposite_ge160"].tolist(), [True, False])
        self.assertEqual(table["joint_coverage_lt60pct"].tolist(), [True, False])
        for stem, _ in ENCODERS:
            self.assertIn(f"{stem}_margin", table.columns)
            self.assertEqual(table[f"{stem}_r1_credit"].tolist(), [1.0, 1.0])
        self.assertEqual(summary["n_pairs"].sum(), 2)
        self.assertEqual(summary["max_r1_credit"].sum(), 16)
        self.assertTrue((summary["mean_recall_at_1"] == 0.5).all())

    def test_missing_encoder_pair_is_rejected(self):
        """All four models must refer to exactly the same pairs."""
        path = self.diag / "four_encoder_positive_pair_metrics.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[~((frame["encoder_stem"] == "vit_b16") &
                            (frame["pair_id"] == 2))]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "inconsistent positive pair identities"):
            create_diagnostics(self.root)

    def test_shared_failure_conflicting_with_retrieval_is_rejected(self):
        """A reported all-model failure cannot have positive R@1 credit."""
        path = self.diag / "four_encoder_positive_pair_metrics.csv"
        frame = pd.read_csv(path)
        sel = ((frame["encoder_stem"] == "resnet50") &
               (frame["pair_id"] == 1))
        frame.loc[sel, "recall_at_1_forward"] = 1.0
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Shared failure conflicts"):
            create_diagnostics(self.root)

    def test_cli_writes_separate_reports(self):
        """The CLI writes only isolated analysis CSVs, not benchmark inputs."""
        before = (self.diag / "four_encoder_positive_pair_metrics.csv").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--root", str(self.root)])
        target = self.root / "outputs/analysis/positive_pair_diagnostics"
        self.assertTrue((target / "positive_pair_diagnostics.csv").is_file())
        self.assertTrue((target / "quality_group_summary.csv").is_file())
        self.assertTrue((target / "stratified_retrieval_summary.csv").is_file())
        self.assertEqual(
            before, (self.diag / "four_encoder_positive_pair_metrics.csv").read_bytes()
        )

    def test_stratified_summaries_have_correct_denominators(self):
        """Use model queries for recall and directed queries for shared failures."""
        table, _ = create_diagnostics(self.root)
        strata = make_stratified_summary(table)
        all_rows = strata.loc[(strata["stratum"] == "overall") &
                              (strata["group"] == "all")]
        self.assertEqual(len(all_rows), 1)
        all_row = all_rows.iloc[0]
        self.assertEqual(all_row["n_pairs"], 2)
        self.assertEqual(all_row["n_directional_queries"], 4)
        self.assertEqual(all_row["n_model_queries"], 16)
        self.assertEqual(all_row["r1_credit"], 8)
        self.assertEqual(all_row["recall_at_1_pct"], 50)
        self.assertEqual(all_row["shared_failure_directions"], 2)
        self.assertEqual(all_row["shared_failure_query_pct"], 50)
        self.assertEqual(all_row["zero_r1_pairs"], 0)
        self.assertEqual(
            set(strata.loc[strata["stratum"] == "audit_flag", "group"]),
            {"requires_review=True", "requires_review=False"}
        )

    def test_filtered_groups_keep_original_gallery_chance(self):
        """Filtering for coverage must not shrink the reference gallery size."""
        table, _ = create_diagnostics(self.root)
        # Add a second A-B pair to the original population; the A-B gallery is 2.
        third = table.loc[table["patch_id"] == 1].copy()
        third["patch_id"] = 3
        third["joint_coverage"] = 0.8
        third["joint_coverage_lt60pct"] = False
        third["requires_review"] = False
        third["total_r1_credit"] = 8
        third["shared_failure_directions"] = 0
        third["zero_r1_credit"] = False
        table = pd.concat([table, third], ignore_index=True)
        strata = make_stratified_summary(table)
        ab_low = strata.loc[
            (strata["stratum"] == "survey_pair_x_coverage") &
            (strata["group"] == "pass_a-pass_b | coverage<60%")
        ].iloc[0]
        self.assertEqual(ab_low["n_pairs"], 1)
        self.assertEqual(ab_low["n_model_queries"], 8)
        self.assertEqual(ab_low["chance_at_1_pct"], 50)
        full = strata.loc[strata["stratum"] == "overall"].iloc[0]
        self.assertAlmostEqual(full["chance_at_1_pct"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
