"""Frozen-encoder sensitivity to controlled image-frame translations.

Runs a clearly labelled *finite-crop translation stress test*, NOT improved
geographic re-registration. Each saved 50 m B-side image and validity mask
are shifted east by 0, 2, 5, 10 or 20 metres, without cyclic wrap or
invented sonar values. Shifted-out pixels become NoData; missing neighbouring
seabed is NOT recovered. Coverage is reported because large shifts confound
representation change with valid-area loss.

Baseline 0 m uses *cached original* feature vectors exactly. Nonzero offsets
require NEW frozen-encoder inference on shifted B images, once per model,
patch and condition; side A remains the original archived gallery/query.
Rankings are recomputed in the same 4/5/16 curated positive galleries,
with tie-aware Recall@1/@3 and both query directions. No baseline NPZ or
original benchmark CSV is modified.
"""

from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.embedding_pca import (
    EXPECTED_PAIRS, FROZEN_ENCODERS, FROZEN_CONDITIONS,
    read_positive_embeddings, read_reference_pairs,
)
from src.analysis.retrieval import rank_queries
from src.benchmark.pipeline import ENCODER_CLASSES, positive_pair_path
from src.encoders.preprocessing import prepare_sonar_rgb

OUTPUT_RELATIVE = Path("outputs/analysis/final_evaluation/registration")
SHIFT_METERS = (0, 2, 5, 10, 20)
SHIFT_AXIS = "east_positive"  # equal-size north-up grids: increasing column = east
EXPECTED_GRID_RESOLUTION_M = 0.25
EXPECTED_PATCH_SIZE_M = 50.0


def translate_patch(patch, mask, east_m, resolution_m):
    """Shift a gridded observation east with NoData padding (no interpolation)."""
    p = np.asarray(patch, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if p.ndim != 2 or p.shape != m.shape or not m.any():
        raise ValueError("Expected a valid 2D sonar patch and Boolean mask")
    if not np.isfinite(p[m]).all():
        raise ValueError("Valid sonar pixels must be finite")
    if not np.isfinite(resolution_m) or resolution_m <= 0 or east_m < 0:
        raise ValueError("Invalid shift or pixel grid resolution")
    pixels = east_m / resolution_m
    if not np.isclose(pixels, round(pixels), atol=1e-9):
        raise ValueError("Shift must be an exact integer number of grid cells")
    columns = int(round(pixels))
    if columns >= p.shape[1]:
        raise ValueError("Shift would remove the entire patch width")
    result = np.full(p.shape, np.nan, dtype=np.float32)
    validity = np.zeros(m.shape, dtype=bool)
    if columns == 0:
        result[:] = p
        validity[:] = m
    else:
        result[:, columns:] = p[:, :p.shape[1] - columns]
        validity[:, columns:] = m[:, :m.shape[1] - columns]
    result[~validity] = np.nan
    return result, validity


def load_positive_patches(root, expected_keys):
    """Read original positive patch rasters and masks aligned by compound ID."""
    root = Path(root)
    records = {}
    for pair, n in EXPECTED_PAIRS.items():
        survey_a, survey_b = pair.split("-")
        source = positive_pair_path(root / "outputs/patches", survey_a, survey_b)
        if not source.is_file():
            raise FileNotFoundError(f"Missing geocoded positive patch archive: {source}")
        with np.load(source, allow_pickle=False) as data:
            ids = np.asarray(data["patch_id"])
            if ids.ndim != 1 or len(ids) != n or ids.dtype.kind not in "iu":
                raise ValueError(f"{pair}: invalid positive patch IDs")
            resolution = float(np.asarray(data["grid_resolution_m"]).item())
            width = float(np.asarray(data["patch_size_m"]).item())
            if not np.isclose(resolution, EXPECTED_GRID_RESOLUTION_M) or not np.isclose(
                width, EXPECTED_PATCH_SIZE_M
            ):
                raise ValueError(f"{pair}: unexpected geocoded 50m / 0.25m grid")
            arrays = {key: np.asarray(data[key]) for key in
                      ("patch_a", "patch_b", "mask_a", "mask_b")}
            shape = (n, int(round(width / resolution)), int(round(width / resolution)))
            if any(arr.shape != shape for arr in arrays.values()):
                raise ValueError(f"{pair}: unexpected positive patch shape")
            for i, raw_id in enumerate(ids):
                key = (pair, int(raw_id))
                if key in records:
                    raise ValueError(f"Duplicate positive patch key {key}")
                pa, pb = arrays["patch_a"][i], arrays["patch_b"][i]
                ma, mb = arrays["mask_a"][i].astype(bool), arrays["mask_b"][i].astype(bool)
                if not ma.any() or not mb.any() or not np.isfinite(pa[ma]).all() or not np.isfinite(pb[mb]).all():
                    raise ValueError(f"Invalid positive sonar data at {key}")
                records[key] = {
                    "patch_b": pb.copy(), "mask_b": mb.copy(),
                    "mask_a": ma.copy(), "resolution_m": resolution,
                }
    if set(records) != expected_keys:
        raise ValueError("Positive patch IDs do not match frozen archive/reference IDs")
    return records


def encode_shifted(encoder, patch, mask):
    """Run fresh frozen inference on sonar content and that observation's mask."""
    sonar, valid = prepare_sonar_rgb(patch, mask)
    mask_only = np.broadcast_to(valid.astype(np.float32), sonar.shape).copy()
    features = {
        "sonar": np.asarray(encoder.encode(sonar), dtype=np.float64),
        "mask-only": np.asarray(encoder.encode(mask_only), dtype=np.float64),
    }
    for name, vec in features.items():
        if vec.ndim != 1 or not np.isfinite(vec).all() or not np.isclose(
            np.linalg.norm(vec), 1.0, atol=1e-5
        ):
            raise ValueError(f"Invalid shifted {name} encoder feature")
    return features


def _new_encoder(model):
    module_name, class_name = ENCODER_CLASSES[model]
    return getattr(import_module(module_name), class_name)()


def compute_registration_sensitivity(root, *, models=FROZEN_ENCODERS,
                                     shifts=SHIFT_METERS, encoders=None):
    """Calculate per-pair, query and group summaries without writing files.

    encoders is injectable for CPU-only synthetic tests; in production the
    frozen checkpoints are loaded lazily only when a nonzero shift is present.
    """
    root = Path(root).expanduser().resolve()
    selected = tuple(models)
    shifts = tuple(shifts)
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(FROZEN_ENCODERS):
        raise ValueError("Select distinct frozen encoders")
    if not shifts or shifts[0] != 0 or tuple(sorted(set(shifts))) != shifts or not set(shifts).issubset(SHIFT_METERS):
        raise ValueError("Shifts must be sorted distinct members of (0,2,5,10,20) starting at zero")
    reference = read_reference_pairs(root)
    reference_keys = set(reference.index.to_list())
    patch_records = load_positive_patches(root, reference_keys)

    scores, queries, coverage_rows, provenance = [], [], [], []
    for model in selected:
        source = root / "outputs/embeddings" / f"{model}_embeddings.npz"
        keys, cached, orig_provenance = read_positive_embeddings(source, model, reference_keys)
        encoder = None
        if len(shifts) > 1:
            encoder = (encoders[model] if encoders is not None else _new_encoder(model))
        for meters in shifts:
            shifted = {}
            missing = []
            for pair, patch_id in keys:
                item = patch_records[(pair, patch_id)]
                p, m = translate_patch(item["patch_b"], item["mask_b"], meters,
                                       item["resolution_m"])
                overlap = float(np.mean(m & item["mask_a"]))
                fraction = float(np.mean(m))
                coverage_rows.append({
                    "model": model, "survey_pair": pair, "patch_id": patch_id,
                    "shift_m": meters, "shift_axis": SHIFT_AXIS,
                    "original_b_coverage": float(item["mask_b"].mean()),
                    "shifted_b_coverage": fraction,
                    "original_joint_coverage": float((item["mask_b"] & item["mask_a"]).mean()),
                    "shifted_joint_coverage": overlap,
                    "retained_b_valid_fraction": (
                        float(m.sum() / item["mask_b"].sum())
                    ),
                    "valid_for_inference": bool(m.any()),
                })
                if meters and not m.any():
                    missing.append((pair, patch_id))
                elif meters:
                    shifted[(pair, patch_id)] = encode_shifted(encoder, p, m)
            for condition in FROZEN_CONDITIONS:
                a = cached[(condition, "a")]
                for pair, n in EXPECTED_PAIRS.items():
                    indices = [i for i, key in enumerate(keys) if key[0] == pair]
                    if len(indices) != n:
                        raise ValueError("Inconsistent survey-pair roster")
                    # Never shorten a gallery to hide an unusable observation.
                    if any((pair, keys[i][1]) in missing for i in indices):
                        scores.append({
                            "model": model, "condition": condition, "shift_m": meters,
                            "survey_pair": pair, "n_pairs": n,
                            "status": "invalid_shifted_patch",
                            "recall_at_1": np.nan, "recall_at_3": np.nan,
                            "mean_positive_cosine": np.nan,
                        })
                        continue
                    ia = a[indices]
                    ib = (cached[(condition, "b")][indices] if meters == 0 else
                          np.stack([shifted[keys[i]][condition] for i in indices]))
                    ids = np.asarray([keys[i][1] for i in indices])
                    survey_a, survey_b = pair.split("-")
                    directed = (
                        rank_queries(ia, ib, ids, survey_pair=pair,
                                     condition=condition, direction=f"{survey_a}->{survey_b}"),
                        rank_queries(ib, ia, ids, survey_pair=pair,
                                     condition=condition, direction=f"{survey_b}->{survey_a}"),
                    )
                    both = [record for group in directed for record in group]
                    for row in both:
                        queries.append({"model": model, "shift_m": meters,
                                        "shift_axis": SHIFT_AXIS, **row})
                    scores.append({
                        "model": model, "condition": condition, "shift_m": meters,
                        "survey_pair": pair, "n_pairs": n, "status": "ok",
                        "recall_at_1": float(np.mean(
                            [row["expected_recall_at_1"] for row in both])),
                        "recall_at_3": float(np.mean(
                            [row["expected_recall_at_3"] for row in both])),
                        "mean_positive_cosine": float(np.mean(
                            np.einsum("ij,ij->i", ia, ib))),
                    })
        provenance.append({
            **orig_provenance, "shift_axis": SHIFT_AXIS,
            "shifts_m": ",".join(map(str, shifts)),
            "translation": "integer_pixel_east_no_wrap_nan_pad",
            "zero_shift_features": "original_saved_npz",
            "nonzero_shift_features": "fresh_frozen_inference" if len(shifts) > 1 else "not_run",
            "grid_resolution_m": EXPECTED_GRID_RESOLUTION_M,
            "limitation": "finite_crop_padding_confounds_missing_edges_and_geometric_shift",
        })
    return (pd.DataFrame(scores), pd.DataFrame(queries),
            pd.DataFrame(coverage_rows), pd.DataFrame(provenance))


def save_registration_reports(root, *, models=FROZEN_ENCODERS,
                              shifts=SHIFT_METERS, encoders=None):
    """Export a completely separate experiment after source validation."""
    summary, queries, coverage, provenance = compute_registration_sensitivity(
        root, models=models, shifts=shifts, encoders=encoders
    )
    output = Path(root).expanduser().resolve() / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in (
        ("registration_summary.csv", summary),
        ("registration_queries.csv", queries),
        ("registration_coverage.csv", coverage),
        ("registration_provenance.csv", provenance),
    ):
        frame.to_csv(output / filename, index=False)

    for model in models:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for condition in FROZEN_CONDITIONS:
            selected = summary[(summary["model"] == model) & (summary["condition"] == condition) &
                               (summary["status"] == "ok")]
            if selected.empty:
                continue
            # Weight each survey-pair summary by its original number of
            # nominal pairs; exclude any setting that has an invalid shift.
            groups = []
            for meters, frame in selected.groupby("shift_m", sort=True):
                groups.append({
                    "shift_m": meters,
                    "cosine": float(np.average(
                        frame["mean_positive_cosine"], weights=frame["n_pairs"])),
                    "recall": float(np.average(
                        frame["recall_at_1"], weights=frame["n_pairs"])),
                })
            grouped = pd.DataFrame(groups)
            axes[0].plot(grouped["shift_m"], grouped["cosine"], marker="o", label=condition)
            axes[1].plot(grouped["shift_m"], grouped["recall"], marker="o", label=condition)
        for ax, title, ylabel in (
            (axes[0], "Cross-pass cosine", "Mean paired cosine"),
            (axes[1], "Bidirectional Recall@1", "Expected Recall@1"),
        ):
            ax.set_title(title)
            ax.set_xlabel("Eastward pixel translation (m)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"{model}: finite 50m-crop translation, NOT corrected registration")
        fig.tight_layout()
        fig.savefig(output / f"{model}_registration_sensitivity.png", dpi=150)
        plt.close(fig)
    return output, summary
