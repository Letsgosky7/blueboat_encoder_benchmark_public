"""Registration stress tests with synthetic 50m patches and fake encoders."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.registration_sensitivity import (
    SHIFT_METERS, compute_registration_sensitivity, save_registration_reports,
    translate_patch,
)


class FakeFrozenEncoder:
    """Unit features at 2048D; records fresh inference without downloading weights."""

    def __init__(self):
        self.calls = 0

    def encode(self, rgb):
        self.calls += 1
        rgb = np.asarray(rgb)
        x = np.zeros(2048, dtype=np.float32)
        x[:4] = [
            float(rgb.mean()), float(rgb.std()),
            float(rgb[:, :112, :112].mean()), 1.0,
        ]
        return x / np.linalg.norm(x)


class RegistrationSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        rng = np.random.default_rng(21)
        pairs = ("pass_a-pass_b", "pass_a-pass_c", "pass_b-pass_c")
        lengths = (4, 5, 16)
        names = np.concatenate([np.repeat(p, n) for p, n in zip(pairs, lengths)])
        ids = np.arange(1, 26)
        table_dir = self.root / "outputs/tables"
        table_dir.mkdir(parents=True)
        pd.DataFrame({
            "pair": names, "patch_id": ids,
            "heading_difference_deg": np.arange(25) * 4 + 1,
            "look_difference_deg": np.arange(25) * 5 + 1,
        }).to_csv(table_dir / "positive_pair_metadata.csv", index=False)
        pd.DataFrame({
            "pair": names, "patch_id": ids,
            "joint_coverage": np.repeat(0.85, 25),
            "requires_review": [False] * 25,
        }).to_csv(table_dir / "positive_pair_quality.csv", index=False)
        # The existing NPZ schema requires exactly 25 2048-dimensional
        # positive embeddings; no pretrained model is instantiated here.
        def unit_array():
            x = rng.normal(size=(25, 2048)).astype(np.float32)
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        vector_a = unit_array()
        vector_b = unit_array()
        embedding = self.root / "outputs/embeddings/resnet50_embeddings.npz"
        embedding.parent.mkdir(parents=True)
        np.savez_compressed(
            embedding, model=np.asarray("resnet50"),
            positive_survey_pair=names,
            positive_patch_id=ids.astype(np.int64),
            positive_sonar_a=vector_a, positive_sonar_b=vector_b,
            positive_mask_only_a=vector_a, positive_mask_only_b=vector_b,
            weights=np.asarray("synthetic"),
            preprocessing=np.asarray("synthetic"),
        )
        patch_root = self.root / "outputs/patches"
        patch_root.mkdir(parents=True)
        for pair, n in zip(pairs, lengths):
            aa, bb = pair.split("-")
            pa = np.tile(np.linspace(0, 10, 200, dtype=np.float32), (n, 200, 1))
            pb = pa + 0.05
            mask = np.ones(pa.shape, dtype=bool)
            np.savez_compressed(
                patch_root / f"positive_pairs_{aa}_{bb}.npz",
                patch_id=ids[names == pair].astype(np.int64),
                patch_a=pa, patch_b=pb,
                mask_a=mask.copy(), mask_b=mask.copy(),
                grid_resolution_m=np.float64(0.25),
                patch_size_m=np.float64(50.0),
            )

    def test_positive_east_translation_never_wraps(self):
        original = np.arange(20, dtype=np.float32).reshape((4, 5))
        mask = np.ones((4, 5), dtype=bool)
        shifted, valid = translate_patch(original, mask, 2.0, 1.0)
        self.assertTrue(np.isnan(shifted[:, :2]).all())
        self.assertTrue((~valid[:, :2]).all())
        np.testing.assert_array_equal(shifted[:, 2:], original[:, :3])
        np.testing.assert_array_equal(valid[:, 2:], np.ones((4, 3), dtype=bool))

    def test_shifts_are_exact_integer_grid_cells(self):
        a = np.ones((4, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "integer"):
            translate_patch(a, a > 0, 0.15, 0.25)
        with self.assertRaisesRegex(ValueError, "entire"):
            translate_patch(a, a > 0, 5, 1)
        with self.assertRaisesRegex(ValueError, "Invalid shift"):
            translate_patch(a, a > 0, -1, 1)
        np.testing.assert_array_equal(translate_patch(a, a > 0, 0, 1)[0], a)

    def test_cached_baseline_uses_no_new_inference(self):
        scores, queries, coverage, prov = compute_registration_sensitivity(
            self.root, models=("resnet50",), shifts=(0,)
        )
        self.assertEqual(len(scores), 6)  # 3 survey pairs x 2 conditions
        self.assertEqual(len(queries), 100)  # 25 x both directions x both conditions
        self.assertEqual(len(coverage), 25)
        self.assertEqual(len(prov), 1)
        self.assertTrue(scores["status"].eq("ok").all())
        self.assertEqual(prov.iloc[0]["nonzero_shift_features"], "not_run")
        self.assertTrue(coverage["retained_b_valid_fraction"].eq(1).all())

    def test_nonzero_shift_performs_new_inference_on_b_only(self):
        encoder = FakeFrozenEncoder()
        scores, queries, coverage, prov = compute_registration_sensitivity(
            self.root, models=("resnet50",), shifts=(0, 2),
            encoders={"resnet50": encoder},
        )
        self.assertEqual(encoder.calls, 50)  # 25 B patches x 2 input conditions
        self.assertEqual(len(scores), 12)
        self.assertEqual(len(queries), 200)
        self.assertEqual(len(coverage), 50)
        shifted = coverage[coverage["shift_m"] == 2]
        self.assertTrue((shifted["shifted_b_coverage"] < 1).all())
        self.assertTrue((shifted["retained_b_valid_fraction"] < 1).all())
        self.assertTrue(scores["status"].eq("ok").all())

    def test_tested_run_is_output_isolated_and_keeps_source(self):
        path = self.root / "outputs/embeddings/resnet50_embeddings.npz"
        before = path.read_bytes()
        encoder = FakeFrozenEncoder()
        folder, metrics = save_registration_reports(
            self.root, models=("resnet50",), shifts=(0, 2),
            encoders={"resnet50": encoder}
        )
        for filename in (
            "registration_summary.csv", "registration_queries.csv",
            "registration_coverage.csv", "registration_provenance.csv",
            "resnet50_registration_sensitivity.png",
        ):
            self.assertTrue((folder / filename).is_file(), filename)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(
            (self.root / "outputs/analysis/final_evaluation/embeddings/pca").exists()
        )
        self.assertEqual(len(metrics), 12)

    def test_invalid_grid_and_missing_inputs_fail_without_output(self):
        patch = self.root / "outputs/patches/positive_pairs_pass_a_pass_b.npz"
        with np.load(patch, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        arrays["grid_resolution_m"] = np.asarray(0.5)
        np.savez_compressed(patch, **arrays)
        with self.assertRaisesRegex(ValueError, "0.25m"):
            compute_registration_sensitivity(
                self.root, models=("resnet50",), shifts=(0,)
            )
        self.assertFalse(
            (self.root / "outputs/analysis/final_evaluation/registration").exists()
        )

    def test_only_predeclared_shift_distances_are_valid(self):
        self.assertEqual(SHIFT_METERS, (0, 2, 5, 10, 20))
        with self.assertRaisesRegex(ValueError, "Shifts"):
            compute_registration_sensitivity(
                self.root, models=("resnet50",), shifts=(2, 0)
            )
        with self.assertRaisesRegex(ValueError, "Shifts"):
            compute_registration_sensitivity(
                self.root, models=("resnet50",), shifts=(0, 1)
            )


if __name__ == "__main__":
    unittest.main()
