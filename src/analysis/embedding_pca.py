"""Principal component diagnostics on frozen BlueBoat positive-pair embeddings.

All calculations use existing, verified-format per-encoder NPZ archives; no
model inference, fitted preprocessing or changes to the frozen benchmark.

One pooled PCA is fitted to 50 observations (the two sides of 25 nominal
positive pairs) for *each* encoder and input condition separately. The
reference identities come from the actual positive pair metadata and audit;
geographic matching remains unverified.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.encoders.embedding_store import ARCHIVE_SPECS


# The PCA phase owns only its required input/output paths: no generic
# evaluation status script or layout indirection is necessary.
FROZEN_ENCODERS = tuple(ARCHIVE_SPECS)
FROZEN_CONDITIONS = ("sonar", "mask-only")
PCA_OUTPUT_RELATIVE = Path("outputs/analysis/final_evaluation/embeddings/pca")
KEY = ("pair", "patch_id")
EXPECTED_PAIRS = {"pass_a-pass_b": 4, "pass_a-pass_c": 5, "pass_b-pass_c": 16}
PALETTE = {
    "pass_a-pass_b": "tab:blue",
    "pass_a-pass_c": "tab:orange",
    "pass_b-pass_c": "tab:green",
}


def fit_pca(features):
    """Return deterministic centered 2D PCA scores and explained-variance ratios.

    Uses the 50x50 sample Gram matrix instead of an expensive full SVD of a
    50x512–2048 matrix. Scores equal standard centered PCA projections,
    up to the arbitrary choice of sign for each principal component.
    Features are *not* standardized by dimension: original L2-unit geometry
    is retained. Axes are not comparable across models or conditions.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("PCA requires at least three observations with two dimensions")
    if not np.isfinite(x).all():
        raise ValueError("PCA input contains non-finite values")
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    gram = (gram + gram.T) / 2.0
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    total = float(values.sum())
    if total <= 1e-14:
        raise ValueError("PCA is undefined for effectively constant embeddings")
    coords = vectors[:, :2] * np.sqrt(values[:2])
    # PCA signs are arbitrary; fix a reproducible convention for saved figures.
    for axis in (0, 1):
        pivot = int(np.argmax(np.abs(coords[:, axis])))
        if coords[pivot, axis] < 0:
            coords[:, axis] *= -1.0
    ratios = values[:2] / total
    return coords, ratios


def read_reference_pairs(root):
    """Match audited quality to the exact 25 positive IDs, never by row order."""
    root = Path(root).expanduser().resolve()
    path_meta = root / "outputs/tables/positive_pair_metadata.csv"
    path_quality = root / "outputs/tables/positive_pair_quality.csv"
    geo = pd.read_csv(path_meta)
    quality = pd.read_csv(path_quality)
    required_geo = {*KEY, "look_difference_deg", "heading_difference_deg"}
    required_quality = {*KEY, "joint_coverage", "requires_review"}
    if not required_geo.issubset(geo.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required_geo - set(geo.columns))}")
    if not required_quality.issubset(quality.columns):
        raise ValueError(f"Missing quality columns: {sorted(required_quality - set(quality.columns))}")
    for df, name in ((geo, "metadata"), (quality, "quality")):
        if len(df) != 25 or df.duplicated(list(KEY)).any():
            raise ValueError(f"{name}: expected 25 unique positive pairs")
    if geo["pair"].value_counts().to_dict() != EXPECTED_PAIRS:
        raise ValueError("Unexpected nominal positive survey composition")
    if not set(map(tuple, geo[list(KEY)].itertuples(index=False, name=None))) == set(
        map(tuple, quality[list(KEY)].itertuples(index=False, name=None))
    ):
        raise ValueError("Metadata and audit refer to different pairs")
    if not quality["requires_review"].isin([True, False]).all():
        raise ValueError("Audit flags must be boolean")
    for data, cols in (
        (geo, ["heading_difference_deg", "look_difference_deg"]),
        (quality, ["joint_coverage"]),
    ):
        if not np.isfinite(data[cols].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite reference geometry or coverage")
    if not quality["joint_coverage"].between(0, 1).all():
        raise ValueError("Joint coverage out of bounds")
    merged = geo[list(KEY) + ["heading_difference_deg", "look_difference_deg"]].merge(
        quality[list(KEY) + ["joint_coverage", "requires_review"]],
        on=list(KEY), how="inner", validate="one_to_one",
    )
    return merged.set_index(list(KEY)).sort_index()


def read_positive_embeddings(path, model, reference_keys):
    """Validate an encoder archive and return the two sides for each condition."""
    if model not in FROZEN_ENCODERS:
        raise ValueError(f"Unknown frozen encoder: {model}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing saved encoder archive: {path}")
    dimension, _, architecture = ARCHIVE_SPECS[model]
    with np.load(path, allow_pickle=False) as data:
        if str(data["model"]) != model:
            raise ValueError(f"{model}: archive model identifier mismatch")
        if architecture is not None and (
            "model_architecture" not in data.files
            or str(data["model_architecture"]) != architecture
        ):
            raise ValueError(f"{model}: incorrect CLIP checkpoint architecture")
        pairs = np.asarray(data["positive_survey_pair"])
        raw_ids = np.asarray(data["positive_patch_id"])
        if pairs.ndim != 1 or raw_ids.ndim != 1 or len(pairs) != 25 or len(raw_ids) != 25:
            raise ValueError(f"{model}: expected 25 paired archive IDs")
        if raw_ids.dtype.kind not in "iu":
            raise ValueError(f"{model}: positive patch IDs must be integers")
        names = pairs.astype(str)
        ids = raw_ids.astype(np.int64)
        keys = [(str(p), int(i)) for p, i in zip(names, ids)]
        if len(set(keys)) != 25 or set(keys) != reference_keys:
            raise ValueError(f"{model}: archive positive IDs differ from reference metadata")
        if pd.Series(names).value_counts().to_dict() != EXPECTED_PAIRS:
            raise ValueError(f"{model}: unexpected survey pair sizes")

        vectors = {}
        for condition in FROZEN_CONDITIONS:
            suffix = condition.replace("-", "_")
            for side in ("a", "b"):
                column = f"positive_{suffix}_{side}"
                values = np.asarray(data[column], dtype=np.float64)
                if values.shape != (25, dimension) or not np.isfinite(values).all():
                    raise ValueError(f"{model}: invalid {column} shape or values")
                if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5):
                    raise ValueError(f"{model}: {column} features are not L2 unit-normalized")
                vectors[(condition, side)] = values.copy()
        provenance = {
            "model": model,
            "embedding_dimension": dimension,
            "model_architecture": str(data["model_architecture"]) if "model_architecture" in data.files else "",
            "weights": str(data["weights"]) if "weights" in data.files else "not-recorded",
            "preprocessing": str(data["preprocessing"]) if "preprocessing" in data.files else "not-recorded",
            "source_npz": str(path),
        }
    return keys, vectors, provenance


def compute_embedding_pca(root, *, models=FROZEN_ENCODERS):
    """Return four tables and plot-ready point arrays without writing any files."""
    root = Path(root).expanduser().resolve()
    model_list = tuple(models)
    if not model_list or len(set(model_list)) != len(model_list):
        raise ValueError("Expected a nonempty set of distinct encoders")
    if not set(model_list).issubset(FROZEN_ENCODERS):
        raise ValueError("At least one requested encoder is unknown")

    reference = read_reference_pairs(root)
    ref_keys = set(reference.index.to_list())
    coordinate_rows, variance_rows, pair_rows, provenance_rows = [], [], [], []
    figure_data = {}

    for model in model_list:
        archive = root / "outputs/embeddings" / f"{model}_embeddings.npz"
        keys, embeddings, provenance = read_positive_embeddings(archive, model, ref_keys)
        annotations = reference.loc[keys].reset_index()
        for condition in FROZEN_CONDITIONS:
            a, b = embeddings[(condition, "a")], embeddings[(condition, "b")]
            joined = np.concatenate([a, b], axis=0)
            coords, ratios = fit_pca(joined)
            for side, offset in (("a", 0), ("b", 25)):
                for i, (pair, patch_id) in enumerate(keys):
                    record = annotations.iloc[i]
                    coordinate_rows.append({
                        "model": model, "condition": condition,
                        "survey_pair": pair, "patch_id": patch_id, "side": side,
                        "observed_survey": pair.split("-")[0 if side == "a" else 1],
                        "pc1": float(coords[offset + i, 0]),
                        "pc2": float(coords[offset + i, 1]),
                        "joint_coverage": float(record["joint_coverage"]),
                        "requires_review": bool(record["requires_review"]),
                    })
            distances = np.linalg.norm(coords[:25] - coords[25:], axis=1)
            full_cosine = np.einsum("ij,ij->i", a, b)
            full_euclidean = np.linalg.norm(a - b, axis=1)
            for i, (pair, patch_id) in enumerate(keys):
                row = annotations.iloc[i]
                pair_rows.append({
                    "model": model, "condition": condition,
                    "survey_pair": pair, "patch_id": patch_id,
                    "cosine_full_embedding": float(full_cosine[i]),
                    "euclidean_full_embedding": float(full_euclidean[i]),
                    "euclidean_2d_pca": float(distances[i]),
                    "joint_coverage": float(row["joint_coverage"]),
                    "requires_review": bool(row["requires_review"]),
                    "heading_difference_deg": float(row["heading_difference_deg"]),
                    "look_difference_deg": float(row["look_difference_deg"]),
                })
            variance_rows.append({
                "model": model, "condition": condition,
                "n_positive_pairs": 25, "n_observations": 50,
                "embedding_dimension": int(a.shape[1]),
                "pc1_explained_variance_ratio": float(ratios[0]),
                "pc2_explained_variance_ratio": float(ratios[1]),
                "two_pc_explained_variance_ratio": float(ratios.sum()),
                "median_paired_2d_distance": float(np.median(distances)),
            })
            provenance_rows.append({"condition": condition, **provenance})
            figure_data[(model, condition)] = (coords, keys, ratios)
    return (
        pd.DataFrame(coordinate_rows),
        pd.DataFrame(variance_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(provenance_rows),
        figure_data,
    )


def _plot_pairs(path, coords, keys, ratios, *, model, condition):
    """Plot one encoder/condition, with same-ID pair segments and survey colors."""
    figure, ax = plt.subplots(figsize=(9, 7))
    for pair, color in PALETTE.items():
        indices = [i for i, (survey_pair, _) in enumerate(keys) if survey_pair == pair]
        if not indices:
            continue
        for i in indices:
            ax.plot([coords[i, 0], coords[25 + i, 0]],
                    [coords[i, 1], coords[25 + i, 1]],
                    color=color, alpha=0.35, linewidth=0.9, zorder=1)
        ax.scatter(coords[indices, 0], coords[indices, 1],
                   label=f"{pair}: first survey", color=color,
                   marker="o", s=30, alpha=0.85, zorder=3)
        later = [25 + i for i in indices]
        ax.scatter(coords[later, 0], coords[later, 1],
                   label=f"{pair}: second survey", color=color,
                   marker="^", s=34, alpha=0.85, zorder=3)
    ax.set_xlabel(f"PC1 — {100 * ratios[0]:.1f}% explained variance")
    ax.set_ylabel(f"PC2 — {100 * ratios[1]:.1f}% explained variance")
    ax.set_title(f"{model} · {condition.upper()} | 25 nominal positive pairs")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_pca_reports(root, *, models=FROZEN_ENCODERS):
    """Generate PCA CSVs and figures only in the dedicated PCA output folder."""
    coords, variance, pairs, provenance, figures = compute_embedding_pca(
        root, models=models
    )
    output = Path(root).expanduser().resolve() / PCA_OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    csvs = {
        "pca_coordinates.csv": coords,
        "pca_variance.csv": variance,
        "pca_pair_distances.csv": pairs,
        "pca_provenance.csv": provenance,
    }
    for filename, dataframe in csvs.items():
        dataframe.to_csv(output / filename, index=False)
    for (model, condition), (points, names, ratios) in figures.items():
        _plot_pairs(
            output / f"{model}_{condition.replace('-', '_')}_pca.png",
            points, names, ratios, model=model, condition=condition,
        )
    return output, variance
