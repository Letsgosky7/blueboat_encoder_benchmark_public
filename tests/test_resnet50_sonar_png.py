"""BlueBoat-only cross-survey data, spatial embargo and contrastive loss tests.

All fixtures are synthetic. Never download weights or fine-tune models in tests.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.resnet50_sonar_png.augmentations import augment
from src.experiments.resnet50_sonar_png.data import (
    EXPECTED_PAIRS, load_blueboat_pairs, png_roundtrip, representation,
)
from src.experiments.resnet50_sonar_png.pipeline import run_experiment
from src.experiments.resnet50_sonar_png.splits import (
    geographically_near, plan_folds, training_indices,
)
from src.experiments.resnet50_sonar_png.training import (
    _training_batches, masked_cross_survey_info_nce,
)


class PairedBlueBoatFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        table = self.root / "outputs/tables"
        patches = self.root / "outputs/patches"
        table.mkdir(parents=True)
        patches.mkdir(parents=True)
        rows, id_num = [], 1
        for pair, count in EXPECTED_PAIRS.items():
            a, b = pair.split("-")
            pa = np.zeros((count, 200, 200), dtype=np.float32)
            pb = np.zeros_like(pa)
            ma = np.ones_like(pa, dtype=bool)
            mb = np.ones_like(pa, dtype=bool)
            ids = []
            east = []
            north = []
            for j in range(count):
                ids.append(id_num)
                # Guaranteed real, distinct cross-pass observations,
                # spatially disjoint for most samples.
                pa[j] = float(j + 1)
                pb[j] = float(j + 2)
                e = (id_num - 1) * 200.
                n = 0.
                east.append(e)
                north.append(n)
                rows.append({
                    "pair": pair, "patch_id": id_num,
                    "midpoint_east": e, "midpoint_north": n,
                    "patch_size_m": 50.,
                })
                id_num += 1
            np.savez_compressed(
                patches / f"positive_pairs_{a}_{b}.npz",
                patch_id=np.array(ids, dtype=np.int64),
                center_east=np.asarray(east), center_north=np.asarray(north),
                patch_a=pa, patch_b=pb, mask_a=ma, mask_b=mb,
                patch_size_m=np.float64(50.0),
            )
        pd.DataFrame(rows).to_csv(table / "positive_pair_metadata.csv", index=False)
        self.samples = load_blueboat_pairs(self.root)


class PairedInputTests(PairedBlueBoatFixtures):
    def test_loads_only_25_matched_cross_pass_pairs(self):
        self.assertEqual(len(self.samples), 25)
        self.assertEqual({s["pair"] for s in self.samples}, set(EXPECTED_PAIRS))
        self.assertEqual(self.samples[0]["patch_a"].shape, (200, 200))
        self.assertEqual(self.samples[0]["patch_b"].shape, (200, 200))
        self.assertFalse((self.root / "outputs/patches/patch_bank_control.npz").exists())

    def test_png_roundtrip_is_real_and_quantized(self):
        image = np.linspace(0, 1, 101, dtype=np.float32).reshape(101, 1)
        decoded, encoded = png_roundtrip(image)
        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
        np.testing.assert_array_equal(
            decoded, np.rint(image * 255).astype(np.uint8).astype(np.float32) / 255
        )
        with self.assertRaises(ValueError):
            png_roundtrip(np.array([[-0.1]], dtype=np.float32))

    def test_two_modalities_from_identical_patch(self):
        patch = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
        mask = np.ones((200, 200), dtype=bool)
        mask[10:50, 20:70] = False
        patch[~mask] = np.nan
        numeric, ma = representation(patch, mask, "sonar")
        png, mb = representation(patch, mask, "png")
        self.assertEqual(numeric.shape, (224, 224))
        np.testing.assert_array_equal(ma, mb)
        self.assertLessEqual(float(np.max(np.abs(png - numeric))), 1/510 + 1e-6)

    def test_augmentation_is_seeded_and_nontrivial(self):
        mask = np.ones((224, 224), dtype=bool)
        mask[20:40, 70:90] = False
        image = np.random.default_rng(1).uniform(0, 1, (224, 224))
        a = augment(image, mask, np.random.default_rng(5))
        b = augment(image, mask, np.random.default_rng(5))
        c = augment(image, mask, np.random.default_rng(6))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertTrue(np.isfinite(a).all())

    def test_canonical_roster_rejects_bad_metadata(self):
        path = self.root / "outputs/tables/positive_pair_metadata.csv"
        table = pd.read_csv(path).iloc[:24]
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "canonical 25"):
            load_blueboat_pairs(self.root)


class PurgedGeographicCVTests(PairedBlueBoatFixtures):
    def test_all_25_are_held_out_once_and_never_geographically_leak(self):
        manifest, audits = plan_folds(self.samples, folds=3, buffer_m=25)
        self.assertEqual(len(audits), 3)
        self.assertEqual(len(manifest), 75)  # 25 pairs x 3 folds
        self.assertEqual(int((manifest["split"] == "test").sum()), 25)
        self.assertTrue(all(a["trainable"] for a in audits))
        self.assertTrue(all(a["valid_train_negative_pair_combinations"] > 0 for a in audits))
        for fold in (1, 2, 3):
            train = training_indices(manifest, self.samples, fold, "train")
            test = training_indices(manifest, self.samples, fold, "test")
            for a in train:
                for b in test:
                    self.assertFalse(geographically_near(
                        self.samples[a], self.samples[b], margin_m=25))

    def test_nearby_training_examples_are_embargoed(self):
        self.samples[9]["east"] = self.samples[8]["east"] + 10
        manifest, audit = plan_folds(self.samples, folds=3, buffer_m=25)
        self.assertGreater(sum(a["embargoed_pairs"] for a in audit), 0)
        for fold in range(1, 4):
            train = training_indices(manifest, self.samples, fold, "train")
            test = training_indices(manifest, self.samples, fold, "test")
            self.assertTrue(all(
                not geographically_near(self.samples[a], self.samples[b], margin_m=25)
                for a in train for b in test
            ))

    def test_audit_only_never_needs_bank_or_generates_training_files(self):
        audits = run_experiment(self.root, audit_only=True)
        self.assertEqual(len(audits), 3)
        self.assertFalse(
            (self.root / "outputs/analysis/resnet50_cross_survey_cv").exists()
        )

    def test_spatial_overlap_and_invalid_fold_validation(self):
        self.assertTrue(geographically_near(
            {"east": 0, "north": 0}, {"east": 50, "north": 0}
        ))
        self.assertFalse(geographically_near(
            {"east": 0, "north": 0}, {"east": 76, "north": 0}, margin_m=25
        ))
        with self.assertRaises(ValueError):
            plan_folds(self.samples, buffer_m=-1)


class PairLossTests(unittest.TestCase):
    def test_training_batches_require_distinct_seabed_regions(self):
        samples = [{"east": e, "north": 0.} for e in (0., 20., 200., 240., 500.)]
        batches = _training_batches(samples, 4, np.random.default_rng(12), 25.)
        self.assertTrue(batches)
        self.assertTrue(all(any(
            not geographically_near(samples[i], samples[j], margin_m=25)
            for k, i in enumerate(batch) for j in batch[k + 1:]
        ) for batch in batches))

    def test_positive_is_cross_survey_and_near_negative_is_masked(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable")
        # Three nominal cross-pass pairs: pairs 0 and 1 cover nearby seabed
        # and MUST NOT become negatives; pair 2 is geographically distant.
        centers = [{"east": 0., "north": 0.},
                   {"east": 10., "north": 10.},
                   {"east": 400., "north": 0.}]
        embeddings = torch.randn(6, 12, generator=torch.Generator().manual_seed(7))
        embeddings.requires_grad_()
        loss = masked_cross_survey_info_nce(
            torch, embeddings, centers, buffer_m=25
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(embeddings.grad).all()))
        with self.assertRaisesRegex(ValueError, "No geographically distinct"):
            masked_cross_survey_info_nce(
                torch, embeddings[:4], centers[:2], buffer_m=25
            )


if __name__ == "__main__":
    unittest.main()
