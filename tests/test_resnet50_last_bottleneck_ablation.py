"""Offline regression tests for the single declared last-bottleneck ablation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments.resnet50_sonar_png.ablation import (
    FIXED_FOLLOWUP, compare_ablation, verify_ablation_setup,
)
from src.experiments.resnet50_sonar_png.pipeline import run_experiment
from src.experiments.resnet50_sonar_png.training import build_network


class AblationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.old = root / "outputs/analysis/resnet50_cross_survey_cv"
        self.old.mkdir(parents=True)
        self.new = self.old / "new_results/last_bottleneck"
        self.manifest = pd.DataFrame([
            {"fold": 1, "split": split, "survey_pair": "pass_a-pass_b",
             "patch_id": i, "center_east": float(i * 100),
             "center_north": 0., "patch_size_m": 50.}
            for i, split in ((1, "train"), (2, "test"))
        ])
        self.manifest.to_csv(self.old / "fold_manifest.csv", index=False)
        (self.old / "protocol.json").write_text(json.dumps({
            **FIXED_FOLLOWUP, "total_unique_pairs": 25,
            "training": "ImageNet ResNet50 V2; only affine BN parameters in last layer4 bottleneck",
        }), encoding="utf-8")
        self.previous = self._summary()
        self.previous.to_csv(self.old / "heldout_summary.csv", index=False)

    @staticmethod
    def _summary():
        rows = []
        for modality in ("sonar", "png"):
            for phase in ("frozen", "fine_tuned"):
                for condition in ("sonar", "mask-only"):
                    rows.append({
                        "modality": modality, "phase": phase, "condition": condition,
                        "heldout_positive_pairs": 25, "rankable_directional_queries": 48,
                        "mean_positive_cosine": .7, "recall_at_1": .25,
                        "recall_at_3": .54, "chance_recall_at_1": 1/6.,
                        "chance_recall_at_3": .5,
                    })
        return pd.DataFrame(rows)

    def _preflight(self, **kwargs):
        args = dict(folds=3, buffer_m=25.0, epochs=3, batch_size=4,
                    lr=1e-5, seed=42)
        args.update(kwargs)
        return verify_ablation_setup(self.old, self.new, self.manifest, **args)

    def test_preflight_matches_first_run_and_keeps_parent_intact(self):
        summary = self._preflight()
        self.assertEqual(len(summary), 8)
        self.assertFalse(self.new.exists())
        self.assertTrue((self.old / "protocol.json").exists())

    def test_rejects_unplanned_tuning_or_split_change(self):
        with self.assertRaisesRegex(ValueError, "predeclared"):
            self._preflight(epochs=10)
        with self.assertRaisesRegex(ValueError, "predeclared"):
            self._preflight(buffer_m=0)
        changed = self.manifest.copy()
        changed.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "splits"):
            verify_ablation_setup(self.old, self.new, changed,
                                  folds=3, buffer_m=25., epochs=3,
                                  batch_size=4, lr=1e-5, seed=42)

    def test_refuses_overwriting_an_existing_followup(self):
        self.new.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self._preflight()

    def test_matched_frozen_baseline_and_ablation_deltas(self):
        later = self.previous.copy()
        selected = (later["modality"].eq("sonar") & later["phase"].eq("fine_tuned")
                    & later["condition"].eq("sonar"))
        later.loc[selected, "recall_at_1"] = .30
        comparison = compare_ablation(self.previous, later)
        mask = (comparison["modality"].eq("sonar")
                & comparison["phase"].eq("fine_tuned")
                & comparison["condition"].eq("sonar"))
        row = comparison.loc[mask].iloc[0]
        self.assertAlmostEqual(row["delta_last_minus_bn_recall_at_1"], .05)
        self.assertEqual(len(comparison), 8)
        bad = later.copy()
        bad.loc[bad["phase"].eq("frozen"), "recall_at_1"] = 0.
        with self.assertRaisesRegex(ValueError, "Frozen baseline differs"):
            compare_ablation(self.previous, bad)
        bad = later.copy()
        bad.loc[selected, "rankable_directional_queries"] = 45
        with self.assertRaisesRegex(ValueError, "query counts"):
            compare_ablation(self.previous, bad)


class BackboneScopeTests(unittest.TestCase):
    def test_only_last_bottleneck_is_unfrozen_offline(self):
        try:
            import torch
            import torchvision.models as tv_models
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"Torch/torchvision not available: {exc}")
        original = tv_models.resnet50

        def offline_constructor(*, weights):
            self.assertIsNotNone(weights)
            return original(weights=None)

        # A real ResNet50 module, but *never* download pretrained ImageNet weights.
        with patch("torchvision.models.resnet50", side_effect=offline_constructor):
            bn, head = build_network(torch)
            old = {k for k, p in bn.named_parameters() if p.requires_grad}
            self.assertEqual(len(old), 6)
            self.assertEqual(sum(p.numel() for p in bn.parameters()
                                 if p.requires_grad), 6144)
            self.assertTrue(all(x.startswith("layer4.2.bn") for x in old))
            del bn, head

            model, head = build_network(torch, tuning_scope="last_bottleneck")
            names = {k for k, p in model.named_parameters() if p.requires_grad}
            self.assertIn("layer4.2.conv2.weight", names)
            self.assertTrue(all(x.startswith("layer4.2.") for x in names))
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(count, 4_462_592)
            self.assertEqual(sum(p.numel() for p in head.parameters()), 557440)

            with self.assertRaisesRegex(ValueError, "Unknown"):
                build_network(torch, tuning_scope="entire_resnet")


if __name__ == "__main__":
    unittest.main()
