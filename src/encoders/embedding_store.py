"""Serialize frozen encoder features and verify pair scores without inference.

Input: positive-pair vectors, survey-bank vectors, checkpoint identity and CSV score records.
Output: NPZ feature archive; verification summaries or errors on inconsistent scores.
Usage: save_embeddings/verify_embeddings are imported by all four benchmarks; no CLI.
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = (("sonar", "sonar"), ("mask-only", "mask_only"))


# Model-specific archive metadata; the six feature-array keys stay unchanged.
ARCHIVE_SPECS = {
    "resnet50": (2048, "ImageNet-V2-channel-normalization", None),
    "convnext_tiny": (768, "ImageNet-V1-channel-normalization", None),
    "vit_b16": (768, "ImageNet-V1-channel-normalization", None),
    "clip_vit_b32": (512, "OpenCLIP-checkpoint-channel-normalization",
                     "ViT-B-32-quickgelu"),
}


def _archive_spec(model):
    """Reject unknown encoder identities before reading or writing an archive."""
    try:
        return ARCHIVE_SPECS[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported encoder archive: {model}") from exc


def save_embeddings(path, positives, bank_vectors, weights, device, *, model):
    """Save validated unit embeddings in the established per-model NPZ schema."""
    feature_dim, normalization, architecture = _archive_spec(model)
    if not positives or not bank_vectors:
        raise ValueError("Cannot save empty embeddings.")
    names = [(pair, patch_id) for pair, patch_id, _, _ in positives]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate positive IDs.")
    bank_keys = sorted(bank_vectors)

    arrays = {
        "positive_survey_pair": np.asarray([name[0] for name in names]),
        "positive_patch_id": np.asarray([name[1] for name in names], dtype=np.int64),
        "bank_survey": np.asarray([key[0] for key in bank_keys]),
        "bank_index": np.asarray([key[1] for key in bank_keys], dtype=np.int64),
        "model": np.asarray(model),
        **({"model_architecture": np.asarray(architecture)} if architecture else {}),
        "weights": np.asarray(str(weights)),
        "device": np.asarray(str(device)),
        "torch_version": np.asarray(version("torch")),
        "torchvision_version": np.asarray(version("torchvision")),
        "preprocessing": np.asarray(
            "prepare_sonar_rgb(224x224,percentiles=1,99.5,median-fill);"
            "mask-only=per-observation-resized-valid-mask;"
            + normalization
        ),
    }

    for condition, suffix in CONDITIONS:
        arrays[f"positive_{suffix}_a"] = np.stack(
            [left[condition] for _, _, left, _ in positives]
        ).astype(np.float32)
        arrays[f"positive_{suffix}_b"] = np.stack(
            [right[condition] for _, _, _, right in positives]
        ).astype(np.float32)
        arrays[f"bank_{suffix}"] = np.stack(
            [bank_vectors[key][condition] for key in bank_keys]
        ).astype(np.float32)

    for key in ("positive_sonar_a", "positive_sonar_b",
                "positive_mask_only_a", "positive_mask_only_b",
                "bank_sonar", "bank_mask_only"):
        vectors = arrays[key]
        if vectors.ndim != 2 or vectors.shape[1] != feature_dim:
            raise ValueError(f"{key} has invalid shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError(f"{key} has non-finite values.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError(f"{key} vectors are not L2-normalized.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def verify_embeddings(path, scores_path, null_metadata_path, *, model):
    """Check model provenance and recompute every stored positive/null cosine."""
    feature_dim, _, architecture = _archive_spec(model)
    with np.load(path, allow_pickle=False) as data:
        if str(data["model"]) != model:
            raise ValueError(f"Expected a {model} embedding archive.")
        if architecture is not None and (
            "model_architecture" not in data.files
            or str(data["model_architecture"]) != architecture
        ):
            raise ValueError(
                "Incompatible CLIP archive: rerun with QuickGELU checkpoint architecture."
            )
        for _, suffix in CONDITIONS:
            for key in (f"positive_{suffix}_a", f"positive_{suffix}_b", f"bank_{suffix}"):
                if data[key].ndim != 2 or data[key].shape[1] != feature_dim:
                    raise ValueError(f"Invalid {key} shape for {model} ({feature_dim}D).")
        positives = {}
        for i, (pair, patch_id) in enumerate(
            zip(data["positive_survey_pair"], data["positive_patch_id"])
        ):
            key = (str(pair), int(patch_id))
            if key in positives:
                raise ValueError(f"Duplicate positive ID: {key}")
            positives[key] = {
                name: (data[f"positive_{suffix}_a"][i],
                       data[f"positive_{suffix}_b"][i])
                for name, suffix in CONDITIONS
            }

        banks = {}
        for i, (survey, bank_index) in enumerate(
            zip(data["bank_survey"], data["bank_index"])
        ):
            key = (str(survey), int(bank_index))
            if key in banks:
                raise ValueError(f"Duplicate bank index: {key}")
            banks[key] = {
                name: data[f"bank_{suffix}"][i] for name, suffix in CONDITIONS
            }

    scores = pd.read_csv(scores_path)
    nulls = pd.read_csv(null_metadata_path).set_index("null_pair_id")
    if not nulls.index.is_unique:
        raise ValueError("Duplicate null pair IDs.")
    if scores.duplicated(["condition", "group", "survey_pair", "pair_id"]).any():
        raise ValueError("Duplicate score rows.")

    seen_positive, seen_null = set(), set()
    max_difference = 0.0
    for row in scores.itertuples(index=False):
        if row.condition not in dict(CONDITIONS):
            raise ValueError(f"Unexpected condition: {row.condition}")
        if row.group == "positive":
            key = (str(row.survey_pair), int(row.pair_id))
            left, right = positives[key][row.condition]
            seen_positive.add((key, row.condition))
        else:
            record = nulls.loc[int(row.pair_id)]
            pair_name = f"{record.survey_a}-{record.survey_b}"
            if str(row.group) != str(record.null_type) or row.survey_pair != pair_name:
                raise ValueError(f"Null metadata mismatch: {row.pair_id}")
            left = banks[(str(record.survey_a), int(record.bank_index_a))][row.condition]
            right = banks[(str(record.survey_b), int(record.bank_index_b))][row.condition]
            seen_null.add((int(row.pair_id), row.condition))

        difference = abs(float(np.dot(left, right)) - float(row.cosine))
        if not np.isfinite(difference) or difference > 1e-5:
            raise ValueError(f"Cosine mismatch at pair {row.pair_id}: {difference}")
        max_difference = max(max_difference, difference)

    if len(seen_positive) != len(positives) * len(CONDITIONS):
        raise ValueError("Not all positive pairs are present in the scores.")
    if len(seen_null) != len(nulls) * len(CONDITIONS):
        raise ValueError("Not all null pairs are present in the scores.")

    print(
        f"EMBEDDINGS VERIFIED: {len(positives)} positive pairs, "
        f"{len(banks)} bank patches, {len(scores)} scores; "
        f"max absolute error={max_difference:.2e}"
    )
    return max_difference
