"""Checkpoint-free diagnostic regression tests (no ImageNet download/training)."""
import unittest

import numpy as np
import pandas as pd

from src.experiments.resnet50_sonar_png.diagnostics import (
    AFFINE_NAMES, embedding_diagnostics, geographic_margins, state_deltas,
    validate_saved_manifest, verify_heldout_cosines,
)


class SplitAndParameterDiagnosticTests(unittest.TestCase):
    def test_saved_manifest_requires_matching_coordinates_and_splits(self):
        manifest = pd.DataFrame([
            {"fold": 1, "split": "train", "survey_pair": "pass_a-pass_b",
             "patch_id": 1, "center_east": 0., "center_north": 10., "patch_size_m": 50.},
            {"fold": 1, "split": "test", "survey_pair": "pass_a-pass_b",
             "patch_id": 2, "center_east": 300., "center_north": 10., "patch_size_m": 50.},
        ])
        validate_saved_manifest(manifest, manifest.iloc[::-1].reset_index(drop=True))
        tampered = manifest.copy()
        tampered.loc[0, "center_east"] += 0.1
        with self.assertRaisesRegex(ValueError, "geographic"):
            validate_saved_manifest(manifest, tampered)
        tampered = manifest.copy()
        tampered.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "splits"):
            validate_saved_manifest(manifest, tampered)

    def test_only_approved_batchnorm_affine_can_change(self):
        import torch
        original = {"bn1.weight": torch.ones(2), "bn1.running_mean": torch.zeros(2),
                    "conv1.weight": torch.ones((2, 2))}
        saved = {k: v.clone() for k, v in original.items()}
        saved["bn1.weight"][0] = 1.25
        rows = state_deltas(original, saved, allowed_changes=AFFINE_NAMES)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["max_absolute_delta"], 0.25)
        saved["bn1.running_mean"][1] = 0.1
        with self.assertRaisesRegex(ValueError, "Frozen"):
            state_deltas(original, saved, allowed_changes=AFFINE_NAMES)
        self.assertRaises(ValueError, state_deltas, original,
                          {"bn1.weight": saved["bn1.weight"]}, allowed_changes=AFFINE_NAMES)


class RetrievalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"pair": "pass_a-pass_b", "patch_id": i + 1, "east": x, "north": 0.}
            for i, x in enumerate((0., 20., 300.))
        ] + [{"pair": "pass_b-pass_c", "patch_id": 4, "east": 600., "north": 0.}]

    def test_margins_exclude_near_and_cross_group_negatives(self):
        x = np.eye(4, dtype=np.float32)
        records = geographic_margins(self.samples, x, x, buffer_m=25.)
        self.assertEqual(len(records), 8)
        byid = {r["patch_id"]: r for r in records if r["direction"] == "a_to_b"}
        self.assertEqual([byid[i]["valid_distant_negatives"] for i in (1, 2, 3, 4)],
                         [1, 1, 2, 0])
        self.assertAlmostEqual(byid[1]["positive_minus_hardest_negative"], 1.)
        self.assertTrue(np.isnan(byid[4]["positive_minus_hardest_negative"]))

    def test_embedding_drift_zero_for_identical_frozen_tuned(self):
        samples = self.samples
        f = np.zeros((4, 2048), dtype=np.float32)
        f[np.arange(4), np.arange(4)] = 1.
        drift, positives, margins = embedding_diagnostics(
            samples, f, f, f.copy(), f.copy(), buffer_m=25.)
        self.assertEqual((len(drift), len(positives), len(margins)), (8, 4, 16))
        self.assertTrue(all(v["embedding_l2_delta"] == 0 for v in drift))
        self.assertTrue(all(p["positive_cosine_delta"] == 0 for p in positives))

    def test_stored_cosines_are_a_strict_reproducibility_check(self):
        values = []
        recomputed = [{"fold": 1, "modality": "sonar", "split": "test",
                       "survey_pair": "pass_a-pass_b", "patch_id": 1,
                       "frozen_positive_cosine": 0.4,
                       "tuned_positive_cosine": 0.5}]
        for phase, score in (("frozen", .4), ("fine_tuned", .5)):
            values.append({"fold": 1, "modality": "sonar", "phase": phase,
                           "condition": "sonar", "survey_pair": "pass_a-pass_b",
                           "patch_id": 1, "positive_cosine": score})
        saved = pd.DataFrame(values)
        self.assertAlmostEqual(verify_heldout_cosines(saved, recomputed), 0.)
        saved.loc[0, "positive_cosine"] = 0.7
        with self.assertRaisesRegex(ValueError, "do not match"):
            verify_heldout_cosines(saved, recomputed)


if __name__ == "__main__":
    unittest.main()
