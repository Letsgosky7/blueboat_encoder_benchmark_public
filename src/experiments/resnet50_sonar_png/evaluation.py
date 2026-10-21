"""BlueBoat cross-pass evaluation for strictly held-out spatial test blocks.

Do not compare embeddings produced by different folds in the SAME retrieval
gallery: each fold has an independently trained encoder. If a held-out survey
pair offers <2 candidates, report cosine but omit uninformative Recall@k.
"""
import numpy as np

from src.analysis.retrieval import rank_queries
from src.encoders.preprocessing import prepare_sonar_rgb

from .data import representation
from .training import infer_features


def evaluate_fold(torch, model, test_pairs, modality, phase, fold, device):
    """Return per-pair cosine records and per-query rank records (one fold)."""
    if not test_pairs:
        raise ValueError("No held-out positive pairs")
    cosine_rows, query_rows = [], []
    for condition in ("sonar", "mask-only"):
        left, right = [], []
        for sample in test_pairs:
            for side, bucket in (("a", left), ("b", right)):
                patch, mask = sample[f"patch_{side}"], sample[f"mask_{side}"]
                if condition == "sonar":
                    image, _ = representation(patch, mask, modality)
                else:
                    _, valid = prepare_sonar_rgb(patch, mask)
                    image = valid.astype(np.float32)
                bucket.append(image)
        features_a = infer_features(torch, model, left, device)
        features_b = infer_features(torch, model, right, device)
        for i, sample in enumerate(test_pairs):
            cosine_rows.append({
                "fold": fold, "modality": modality, "phase": phase,
                "condition": condition, "survey_pair": sample["pair"],
                "patch_id": sample["patch_id"],
                "positive_cosine": float(np.dot(features_a[i], features_b[i])),
            })
        for pair in sorted({s["pair"] for s in test_pairs}):
            indices = [i for i, s in enumerate(test_pairs) if s["pair"] == pair]
            if len(indices) < 2:
                continue  # single-item gallery gives trivial recall 1.0
            ids = [test_pairs[i]["patch_id"] for i in indices]
            name_a, name_b = pair.split("-")
            for direction, src, tgt in (
                (f"{name_a}->{name_b}", features_a, features_b),
                (f"{name_b}->{name_a}", features_b, features_a),
            ):
                result = rank_queries(src[indices], tgt[indices], ids,
                                      survey_pair=pair, condition=condition,
                                      direction=direction)
                for row in result:
                    query_rows.append({
                        "fold": fold, "modality": modality, "phase": phase, **row,
                    })
    return cosine_rows, query_rows
