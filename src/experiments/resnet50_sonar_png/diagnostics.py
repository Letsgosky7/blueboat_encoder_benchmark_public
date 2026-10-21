"""Read-only diagnostics of six pre-existing BlueBoat geographic-CV checkpoints.

No training, new positives, new split, or modification of frozen benchmark outputs.
Only the saved layer4[-1] state is reconstructible: earlier backbone blocks are
assumed to be frozen based on the audited training implementation, not a full
checkpoint-by-checkpoint verification of those unsaved weights.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import representation
from .splits import geographically_near, training_indices
from .training import build_network, device_for, infer_features

OUTPUT_RELATIVE = Path("outputs/analysis/resnet50_cross_survey_cv")
AFFINE_NAMES = {f"bn{i}.{key}" for i in (1, 2, 3) for key in ("weight", "bias")}


def validate_saved_manifest(expected, actual):
    """Reject silent protocol/split changes before looking at tuned checkpoints."""
    columns = ["fold", "split", "survey_pair", "patch_id", "center_east",
               "center_north", "patch_size_m"]
    if not set(columns).issubset(actual.columns) or len(actual) != len(expected):
        raise ValueError("Saved fold manifest is absent or structurally incompatible")
    a = expected[columns].sort_values(columns[:4]).reset_index(drop=True)
    b = actual[columns].sort_values(columns[:4]).reset_index(drop=True)
    if not a[columns[:4]].equals(b[columns[:4]]):
        raise ValueError("Saved fold manifest no longer matches the canonical geographic splits")
    if not np.allclose(a[columns[4:]].to_numpy(float),
                       b[columns[4:]].to_numpy(float), rtol=0, atol=1e-5):
        raise ValueError("Saved geographic coordinates/footprints have changed")


def state_deltas(reference_state, checkpoint_state, *, allowed_changes):
    """Per-tensor parameter/buffer deltas; forbid any unapproved modification."""
    if set(reference_state) != set(checkpoint_state):
        raise ValueError("Checkpoint has missing or additional state keys")
    rows = []
    for name, base in reference_state.items():
        new = checkpoint_state[name]
        if base.shape != new.shape or base.dtype != new.dtype:
            raise ValueError(f"Checkpoint tensor mismatch: {name}")
        old = base.detach().cpu().to(dtype=__import__("torch").float64).numpy()
        cur = new.detach().cpu().to(dtype=__import__("torch").float64).numpy()
        if not np.isfinite(old).all() or not np.isfinite(cur).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        delta = cur - old
        norm = float(np.linalg.norm(delta.ravel()))
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        if name not in allowed_changes and max_abs != 0:
            raise ValueError(f"Frozen parameter/BN buffer unexpectedly modified: {name}")
        rows.append({"parameter": name, "category": ("trainable" if name in allowed_changes
                      else "frozen_or_buffer"), "elements": int(delta.size),
                     "l2_delta": norm, "relative_l2_delta": norm / max(
                         float(np.linalg.norm(old.ravel())), 1e-12),
                     "max_absolute_delta": max_abs})
    return rows


def geographic_margins(samples, emb_a, emb_b, *, buffer_m):
    """Directional true-vs-best-distant-negative margins inside each survey pair.

    The eligible gallery differs by query/split: these are diagnostic margins,
    NOT the original held-out Recall@k and not directly comparable to it.
    """
    if len(samples) != len(emb_a) or emb_a.shape != emb_b.shape:
        raise ValueError("Embedding and sample counts do not match")
    if not np.isfinite(emb_a).all() or not np.isfinite(emb_b).all():
        raise ValueError("Nonfinite embeddings")
    rows = []
    for source, gallery, direction in ((emb_a, emb_b, "a_to_b"),
                                       (emb_b, emb_a, "b_to_a")):
        matrix = source @ gallery.T
        for i, sample in enumerate(samples):
            eligible = [j for j, other in enumerate(samples)
                        if i != j and sample["pair"] == other["pair"]
                        and not geographically_near(sample, other, margin_m=buffer_m)]
            pos = float(matrix[i, i])
            hardest = float(np.max(matrix[i, eligible])) if eligible else float("nan")
            rows.append({"survey_pair": sample["pair"], "patch_id": sample["patch_id"],
                         "direction": direction, "positive_cosine": pos,
                         "valid_distant_negatives": len(eligible),
                         "max_distant_negative_cosine": hardest,
                         "positive_minus_hardest_negative": pos - hardest})
    return rows


def embedding_diagnostics(samples, fa, fb, ta, tb, *, buffer_m):
    """Return per-view drift, per-positive shift, and frozen/tuned margins."""
    n = len(samples)
    if any(z.shape != (n, 2048) for z in (fa, fb, ta, tb)):
        raise ValueError("Expected N x 2048 ResNet50 embeddings")
    drift, positives, margins = [], [], []
    for side, frozen, tuned in (("a", fa, ta), ("b", fb, tb)):
        for sample, f, t in zip(samples, frozen, tuned):
            drift.append({"survey_pair": sample["pair"], "patch_id": sample["patch_id"],
                          "view": side, "frozen_tuned_embedding_cosine": float(f @ t),
                          "embedding_l2_delta": float(np.linalg.norm(t - f))})
    for i, sample in enumerate(samples):
        a = float(fa[i] @ fb[i]); b = float(ta[i] @ tb[i])
        positives.append({"survey_pair": sample["pair"], "patch_id": sample["patch_id"],
                          "frozen_positive_cosine": a, "tuned_positive_cosine": b,
                          "positive_cosine_delta": b - a})
    for phase, left, right in (("frozen", fa, fb), ("fine_tuned", ta, tb)):
        margins.extend([{"phase": phase, **row} for row in
                        geographic_margins(samples, left, right, buffer_m=buffer_m)])
    return drift, positives, margins


def verify_heldout_cosines(saved, computed, *, tolerance=2e-3):
    """Check diagnostic test embeddings against previous fold-matched CSVs."""
    old = saved.loc[saved["condition"] == "sonar"].copy()
    expected_keys = ["fold", "modality", "phase", "survey_pair", "patch_id"]
    if old.duplicated(expected_keys).any():
        raise ValueError("Duplicate original held-out cosine rows")
    observed = []
    for row in computed:
        for phase, key in (("frozen", "frozen_positive_cosine"),
                           ("fine_tuned", "tuned_positive_cosine")):
            observed.append({**{k: row[k] for k in ("fold", "modality",
                                                      "survey_pair", "patch_id")},
                             "phase": phase, "recomputed": row[key]})
    new = pd.DataFrame(observed)
    if new.duplicated(expected_keys).any() or len(old) != len(new):
        raise ValueError("Diagnostic held-out roster differs from saved results")
    joined = old.merge(new, on=expected_keys, how="outer", validate="one_to_one",
                       indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Saved held-out rows differ from diagnostic rows")
    differences = np.abs(joined["positive_cosine"] - joined["recomputed"])
    if not np.isfinite(differences).all() or (differences > tolerance).any():
        raise ValueError("Recomputed held-out cosines do not match stored results")
    return float(differences.max()) if len(differences) else 0.0


def run_diagnostics(root, pairs, manifest, *, folds, buffer_m, device="auto"):
    """Use local original data and checkpoints; write *only* diagnostics/*.csv/json."""
    root = Path(root)
    output = root / OUTPUT_RELATIVE
    protocol_file = output / "protocol.json"
    if not protocol_file.exists():
        raise FileNotFoundError(f"Original CV protocol missing: {protocol_file}")
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (protocol.get("folds") != folds
            or not np.isclose(float(protocol.get("buffer_m", -1)), buffer_m)
            or protocol.get("total_unique_pairs") != 25):
        raise ValueError("Requested diagnostic split does not match original protocol")
    validate_saved_manifest(manifest, pd.read_csv(output / "fold_manifest.csv"))
    saved_cos = pd.read_csv(output / "heldout_positive_cosines.csv")
    for required in ("fold", "modality", "phase", "condition", "survey_pair",
                     "patch_id", "positive_cosine"):
        if required not in saved_cos:
            raise ValueError(f"Held-out cosine CSV missing field {required}")
    import torch
    target = device_for(torch, device)
    parameter_rows, projection_rows, drift_rows = [], [], []
    positive_rows, margin_rows, summary_rows = [], [], []
    for fold in range(1, folds + 1):
        for modality in ("sonar", "png"):
            checkpoint_path = (output / f"fold_{fold:02d}" / modality
                               / "resnet50_finetuned.pt")
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")
            # Reproduce both pretrained weights and initial projection head.
            torch.manual_seed(int(protocol["seed"]) + fold)
            baseline, head = build_network(torch)
            tuned = copy.deepcopy(baseline)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (checkpoint.get("fold") != fold or checkpoint.get("modality") != modality
                    or checkpoint.get("epochs") != protocol["epochs"]
                    or checkpoint.get("weights") != "ResNet50_Weights.IMAGENET1K_V2"):
                raise ValueError(f"Checkpoint identity/protocol mismatch: {checkpoint_path}")
            state = checkpoint["trained_last_layer4_block_bn_affine"]
            deltas = state_deltas(baseline.layer4[-1].state_dict(), state,
                                  allowed_changes=AFFINE_NAMES)
            parameter_rows.extend({"fold": fold, "modality": modality, **r} for r in deltas)
            projection_rows.extend({"fold": fold, "modality": modality, **r} for r in
                                   state_deltas(head.state_dict(), checkpoint["projection"],
                                                allowed_changes=set(head.state_dict())))
            tuned.layer4[-1].load_state_dict(state, strict=True)
            head.load_state_dict(checkpoint["projection"], strict=True)
            expected_trainable = {"layer4.2." + key for key in AFFINE_NAMES}
            actual_trainable = {name for name, p in tuned.named_parameters()
                                if p.requires_grad}
            if actual_trainable != expected_trainable:
                raise ValueError("Unexpected backbone trainable parameter roster")
            if sum(p.numel() for p in tuned.parameters() if p.requires_grad) != 6144:
                raise ValueError("Unexpected trainable backbone parameter count")
            baseline = baseline.to(target).eval()
            tuned = tuned.to(target).eval()
            print(f"diagnose: fold {fold}/{folds}, modality {modality}, device={target}")
            for split in ("train", "test"):
                samples = [pairs[i] for i in training_indices(manifest, pairs, fold, split)]
                la, lb = [], []
                for sample in samples:
                    la.append(representation(sample["patch_a"], sample["mask_a"], modality)[0])
                    lb.append(representation(sample["patch_b"], sample["mask_b"], modality)[0])
                fa = infer_features(torch, baseline, la, target)
                fb = infer_features(torch, baseline, lb, target)
                ta = infer_features(torch, tuned, la, target)
                tb = infer_features(torch, tuned, lb, target)
                drift, positive, margins = embedding_diagnostics(
                    samples, fa, fb, ta, tb, buffer_m=buffer_m)
                labels = {"fold": fold, "modality": modality, "split": split}
                drift_rows.extend({**labels, **r} for r in drift)
                positive_rows.extend({**labels, **r} for r in positive)
                margin_rows.extend({**labels, **r} for r in margins)
                valid = [r for r in margins if r["phase"] == "fine_tuned"
                         and r["valid_distant_negatives"] > 0]
                summary_rows.append({**labels, "pairs": len(samples),
                     "views": len(drift),
                     "mean_embedding_l2_delta": float(np.mean([r["embedding_l2_delta"]
                                                                for r in drift])),
                     "mean_embedding_frozen_tuned_cosine": float(np.mean([
                         r["frozen_tuned_embedding_cosine"] for r in drift])),
                     "mean_positive_cosine_delta": float(np.mean([
                         r["positive_cosine_delta"] for r in positive])),
                     "valid_directional_margin_queries": len(valid),
                     "mean_tuned_distant_margin": (float(np.mean([
                         r["positive_minus_hardest_negative"] for r in valid]))
                         if valid else float("nan"))})
            del baseline, tuned, head
    max_error = verify_heldout_cosines(saved_cos, [r for r in positive_rows
                                                    if r["split"] == "test"])
    destination = output / "new_results" / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("backbone_parameter_deltas", parameter_rows),
                          ("projection_parameter_deltas", projection_rows),
                          ("embedding_drift", drift_rows),
                          ("crosspass_positive_shift", positive_rows),
                          ("distant_negative_margins", margin_rows),
                          ("diagnostic_summary", summary_rows)):
        pd.DataFrame(records).to_csv(destination / f"{name}.csv", index=False)
    (destination / "diagnostic_protocol.json").write_text(json.dumps({
        "kind": "read-only audit of existing 6 checkpoints",
        "source": str(output), "checkpoint_count": 2 * folds,
        "buffer_m": buffer_m, "folds": folds, "device": str(target),
        "recomputed_heldout_max_abs_error": max_error,
        "saved_heldout_tolerance": 2e-3,
        "only_reconstructible_backbone_weights": "layer4[-1] checkpoint submodule",
        "unsaved_earlier_weights": "assumed unchanged from audited source code; cannot verify from checkpoint",
        "training_vs_test": "not rank-comparable: varying gallery sizes and geographic negative counts",
        "positives": "25 nominal GPS-derived cross-pass pairs; not verified visual ground truth",
        "model_selection": "no hyperparameter search or new training", "input_comparison":
        "same preprocessed source; PNG differs primarily by 8-bit quantization",
    }, indent=2), encoding="utf-8")
    print("Diagnostic CSVs:", destination)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return pd.DataFrame(summary_rows)
