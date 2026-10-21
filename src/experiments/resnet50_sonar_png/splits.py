"""Spatially purged cross-validation of the 25 nominal positive pairs.

No image or pair can occur in both a fold's training and test sets; additionally
we embargo training pairs whose 50-m footprint lies within a configurable
buffer of ANY test footprint. Other folds may use those points for training,
but always with their OWN held-out geographic region and fresh model weights.

Folds are formed by rank along the dominant geographic axis, not by labels or
model performance. A neighboring pair excluded by the embargo is neither train
nor test in that fold. This avoids whole-component collapse on chained tracks.
"""
import numpy as np
import pandas as pd

from .data import PATCH_SIZE_M


def geographically_near(a, b, *, margin_m=0.0):
    """True if axis-aligned 50m footprints approach within margin in both axes."""
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be nonnegative")
    return (abs(float(a["east"]) - float(b["east"])) <= PATCH_SIZE_M + margin_m
            and abs(float(a["north"]) - float(b["north"])) <= PATCH_SIZE_M + margin_m)


def _spatial_order(samples):
    coords = np.asarray([(s["east"], s["north"]) for s in samples], dtype=np.float64)
    if coords.shape != (25, 2) or not np.isfinite(coords).all():
        raise ValueError("Expected 25 valid geocoded pair centers")
    centered = coords - coords.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[0] <= 1e-9:
        raise ValueError("All positive pairs share one nominal location")
    axis = vt[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projection = centered @ axis
    # Stable ordering for ties at exactly the same position.
    return sorted(range(len(samples)),
                  key=lambda i: (float(projection[i]), float(coords[i, 0]),
                                 float(coords[i, 1]), samples[i]["pair"],
                                 samples[i]["patch_id"]))


def plan_folds(samples, *, folds=3, buffer_m=25.0, min_train_pairs=4):
    """Return (manifest, audit). Run this before loading model weights."""
    if not (isinstance(folds, int) and 2 <= folds <= 5 and
            isinstance(min_train_pairs, int) and min_train_pairs >= 2):
        raise ValueError("Require 2..5 folds and at least 2 training pairs")
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Invalid geographic buffer")
    if len(samples) != 25 or len({(s["pair"], s["patch_id"]) for s in samples}) != 25:
        raise ValueError("Exactly 25 distinct canonical positive pairs required")
    order = _spatial_order(samples)
    assignments = np.empty(len(samples), dtype=np.int32)
    for rank, index in enumerate(order):
        assignments[index] = min(folds - 1, rank * folds // len(samples))
    audit, manifest = [], []
    for fold in range(folds):
        test_ids = [i for i in range(len(samples)) if assignments[i] == fold]
        train_ids, embargo_ids = [], []
        for i, s in enumerate(samples):
            if i in test_ids:
                continue
            if any(geographically_near(s, samples[t], margin_m=buffer_m)
                   for t in test_ids):
                embargo_ids.append(i)
            else:
                train_ids.append(i)
        negatives = sum(
            not geographically_near(samples[i], samples[j], margin_m=buffer_m)
            for k, i in enumerate(train_ids) for j in train_ids[k+1:]
        )
        per_pair = {name: sum(samples[i]["pair"] == name for i in test_ids)
                    for name in sorted({s["pair"] for s in samples})}
        rankable_queries = sum(2 * n for n in per_pair.values() if n >= 2)
        ok = len(train_ids) >= min_train_pairs and negatives >= 1
        audit.append({
            "fold": fold + 1, "train_pairs": len(train_ids),
            "embargoed_pairs": len(embargo_ids), "test_pairs": len(test_ids),
            "valid_train_negative_pair_combinations": int(negatives),
            "bidirectional_rankable_queries_per_condition": int(rankable_queries),
            "test_by_survey_pair": per_pair, "trainable": bool(ok),
        })
        for i, s in enumerate(samples):
            status = ("test" if i in test_ids else
                      "train" if i in train_ids else "embargo")
            manifest.append({
                "fold": fold + 1, "split": status, "survey_pair": s["pair"],
                "patch_id": s["patch_id"], "center_east": float(s["east"]),
                "center_north": float(s["north"]), "patch_size_m": PATCH_SIZE_M,
            })
        for i in train_ids:
            for j in test_ids:
                if geographically_near(samples[i], samples[j], margin_m=buffer_m):
                    raise AssertionError("Cross-validation geographic leakage")
    df = pd.DataFrame(manifest)
    if (df["split"] == "test").sum() != 25:
        raise AssertionError("Each of 25 positive pairs must be tested once")
    return df, audit


def training_indices(manifest, samples, fold, split):
    """Recover original sample order from stable (survey pair, ID) keys."""
    keys = {
        (r.survey_pair, int(r.patch_id))
        for r in manifest.loc[(manifest["fold"] == fold) &
                              (manifest["split"] == split)].itertuples(index=False)
    }
    return [i for i, s in enumerate(samples) if (s["pair"], s["patch_id"]) in keys]
