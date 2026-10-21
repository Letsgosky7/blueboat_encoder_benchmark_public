"""Check metadata joins, SONAR/MASK alignment, non-imputation and graph generation.

Run: python -m unittest discover -s tests -p 'test_*.py' -v (repository root).
Input: synthetic in-memory arrays and, where needed, temporary CSV/NPZ fixtures.
Output: unittest pass/fail assertions; temporary files are discarded after tests.
No CLI flags; these tests do not require raw BlueBoat XTF recordings.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.acquisition_condition_analysis import (
    create_analysis_table, generate, read_geometry, make_strata_table, make_delta_table,
)
from scripts.benchmark_comparison import ENCODERS, CONDITIONS, SURVEY_PAIRS


class AcquisitionAnalysisTests(unittest.TestCase):
    """Create 25 synthetic positive pairs and four encoder CSV sets for analysis tests."""
    def setUp(self):
        """Construct small synthetic inputs and isolated fixtures for these tests."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        table_dir = self.root / "outputs/tables"
        encoder_dir = self.root / "outputs/embeddings"
        table_dir.mkdir(parents=True)
        encoder_dir.mkdir(parents=True)
        rows = []
        for pair, count in zip(SURVEY_PAIRS, (4, 5, 16)):
            for index in range(count):
                rows.append(dict(
                    pair=pair, patch_id=len(rows)+1,
                    heading_difference_deg=float(5+index),
                    look_difference_deg=float(8+index),
                    a_ground_range_m=float(10+index),
                    b_ground_range_m=float(12+index),
                    a_side="port", b_side="starboard",
                    view_class="similar-view",
                    coverage_a=0.8, coverage_b=0.82,
                ))
        metadata = pd.DataFrame(rows)
        metadata.to_csv(table_dir / "positive_pair_metadata.csv", index=False)
        for stem, _ in ENCODERS:
            margins, queries = [], []
            for record in rows:
                a, b = record["pair"].split("-")
                for condition in CONDITIONS:
                    margins.append(dict(
                        survey_pair=record["pair"], pair_id=record["patch_id"],
                        condition=condition, margin=0.2 if condition == "sonar" else 0.1,
                        positive_cosine=0.9, requires_review=False,
                    ))
                    for direction in (f"{a}->{b}", f"{b}->{a}"):
                        queries.append(dict(
                            survey_pair=record["pair"], condition=condition,
                            direction=direction, query_patch_id=record["patch_id"],
                            expected_recall_at_1=0.5,
                            expected_recall_at_3=1.0, tie_count=1,
                        ))
            pd.DataFrame(margins).to_csv(
                encoder_dir / f"{stem}_matched_null_margins.csv", index=False)
            pd.DataFrame(queries).to_csv(
                encoder_dir / f"{stem}_retrieval_queries.csv", index=False)

    def test_merge_and_generate_four_figures_and_three_tables(self):
        metadata = read_geometry(self.root)
        self.assertEqual(len(metadata), 25)
        self.assertTrue((metadata["anchor_side_pair"] == "port→starboard").all())
        self.assertTrue(np.allclose(metadata["anchor_ground_range_difference_m"], 2))
        details = create_analysis_table(self.root)
        self.assertEqual(len(details), 4*2*25)
        self.assertTrue(np.allclose(details["bidirectional_recall_at_1"], 0.5))
        strata = make_strata_table(details)
        self.assertEqual(strata["n_positive_pairs"].sum(), 4*2*25*3)
        deltas = make_delta_table(details)
        self.assertEqual(len(deltas), 4*25)
        self.assertTrue(np.allclose(deltas["margin_sonar"], 0.2))
        self.assertTrue(np.allclose(deltas["margin_mask_only"], 0.1))
        self.assertTrue(np.allclose(deltas["delta_margin_sonar_minus_mask"], 0.1))

        output = self.root / "reports"
        terminal_output = io.StringIO()
        with redirect_stdout(terminal_output):
            generate(self.root, output)
        self.assertEqual(terminal_output.getvalue(), "")
        self.assertEqual(len(list(output.glob("*.png"))), 4)
        self.assertEqual(len(list(output.glob("*.csv"))), 3)
        saved = pd.read_csv(output / "acquisition_sonar_mask_margin_delta.csv")
        self.assertEqual(len(saved), len(ENCODERS) * 25)
        self.assertTrue(np.allclose(saved["delta_margin_sonar_minus_mask"], 0.1))

    def test_delta_rejects_missing_mask_condition(self):
        details = create_analysis_table(self.root)
        row_to_remove = details.index[
            (details["encoder_stem"] == "clip_vit_b32")
            & (details["condition"] == "mask-only")
        ][0]
        with self.assertRaisesRegex(ValueError, "align"):
            make_delta_table(details.drop(index=row_to_remove))

    def test_missing_geometry_cannot_be_imputed(self):
        path = self.root / "outputs/tables/positive_pair_metadata.csv"
        df = pd.read_csv(path).drop(columns="a_ground_range_m")
        df.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "a_ground_range_m"):
            read_geometry(self.root)

    def test_query_identity_mismatch_is_rejected(self):
        path = self.root / "outputs/embeddings/vit_b16_retrieval_queries.csv"
        df = pd.read_csv(path)
        df.loc[0, "query_patch_id"] = 999
        df.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "two directional queries"):
            create_analysis_table(self.root)


if __name__ == "__main__":
    unittest.main()
