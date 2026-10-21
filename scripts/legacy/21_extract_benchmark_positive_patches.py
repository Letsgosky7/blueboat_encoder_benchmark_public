"""LEGACY diagnostic — Produce a historical version of the geocoded 50 m curated positives.

Invocation: python scripts/legacy/21_extract_benchmark_positive_patches.py, after restoring legacy paths/prerequisites.
Input: 17_colocated_candidates_*.csv, 16_seabed_geometry_*.npz and raw XTF.
Output: 21_positive_*.npz, 21_positive_patch_metadata.csv and review PNGs.
Not used by the canonical scripts.build_dataset pipeline; standalone execution
may require correcting historical config paths or recovering upstream artifacts.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.xtf_reader import read_xtf
from src.patches.extraction import PositivePatchConfig, build_mosaic, preselect_centres


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PATCH_DIR = PROJECT_ROOT / "outputs" / "patches"
with open(PROJECT_ROOT / "configs" / "local.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

XTF_DIR = Path(config["dataset_root"]) / "SSS-XTF"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

SURVEYS = {"pass_a": "2026-08-05-08-00-1.xtf", "pass_b": "2026-08-06-07-50.xtf", "pass_c": "2026-08-06-09-10.xtf"}
PAIRS = (("pass_a", "pass_b"), ("pass_a", "pass_c"), ("pass_b", "pass_c"))
CONFIG = PositivePatchConfig()
DISPLAY_PERCENTILES = (1.0, 99.5)
GALLERY_MAX = 6
GEOMETRY_KEYS = (
    "east", "north", "heading_deg", "in_analysis_leg", "first_bottom_port", "first_bottom_stbd",
    "high_confidence_port", "high_confidence_stbd", "canonical_slant_axis_port", "canonical_slant_axis_stbd",
)


def candidate_path(a, b):
    """Resolve the earlier colocation-candidate CSV path for a survey pair."""
    return TABLE_DIR / f"17_colocated_candidates_{a}_{b}.csv"


def geometry_path(name):
    """Resolve the historical per-survey geometry archive."""
    return TABLE_DIR / f"16_seabed_geometry_{name}.npz"


def load_geometry(name):
    """Load the necessary geocoded per-ping geometry arrays."""
    path = geometry_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry: {path}")
    with np.load(path) as archive:
        missing = [key for key in GEOMETRY_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"Missing geometry arrays in {path.name}: {missing}")
        return {key: np.asarray(archive[key]) for key in GEOMETRY_KEYS}


def normalize_pair(a, b):
    """Apply shared display limits to two sonar patches."""
    values = np.concatenate((a[np.isfinite(a)], b[np.isfinite(b)]))
    if values.size == 0:
        return a.copy(), b.copy()
    lo, hi = np.percentile(values, DISPLAY_PERCENTILES)
    if hi <= lo:
        hi = lo + 1.0

    def scale(x):
        """Rescale an image for visual comparison without changing benchmark data."""
        y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        y[~np.isfinite(x)] = np.nan
        return y

    return scale(a), scale(b)


def save_gallery(a, b, metadata, patches_a, patches_b):
    """Save a labeled visual gallery for manual inspection."""
    if metadata.empty:
        return None
    ids = np.linspace(0, len(metadata) - 1, min(GALLERY_MAX, len(metadata)), dtype=int)
    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 4.1 * len(ids)), squeeze=False)
    half = CONFIG.patch_size_m / 2.0
    extent = [-half, half, -half, half]

    for plot_row, idx in enumerate(ids):
        row = metadata.iloc[idx]
        pa, pb = normalize_pair(patches_a[idx], patches_b[idx])
        for ax, patch, name, cov in ((axes[plot_row, 0], pa, a, row.coverage_a), (axes[plot_row, 1], pb, b, row.coverage_b)):
            ax.imshow(np.ma.masked_invalid(patch), cmap="gray", origin="lower", extent=extent, interpolation="nearest")
            ax.axhline(0, linewidth=0.5, alpha=0.5); ax.axvline(0, linewidth=0.5, alpha=0.5)
            ax.set_aspect("equal"); ax.set_title(f"{name.upper()} | coverage {cov:.0%}")
            ax.set_xlabel("East offset [m]"); ax.set_ylabel("North offset [m]")
        axes[plot_row, 0].text(-24, 23, f"{row.view_class} | joint={row.joint_coverage:.0%} | ΔH={row.heading_difference_deg:.0f}° | ΔL={row.look_difference_deg:.0f}°", fontsize=8, va="top")

    fig.suptitle(f"Accepted 50 m benchmark patches — {a.upper()} vs {b.upper()}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = FIGURE_DIR / f"21_benchmark_positive_patches_{a}_{b}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    """Run this historical data-generation or inspection stage."""
    all_metadata, global_patch_id = [], 0

    for name_a, name_b in PAIRS:
        csv_path = candidate_path(name_a, name_b)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing candidates: {csv_path}")
        centres = preselect_centres(pd.read_csv(csv_path), CONFIG)
        if centres.empty:
            continue

        survey_a, survey_b = read_xtf(XTF_DIR / SURVEYS[name_a]), read_xtf(XTF_DIR / SURVEYS[name_b])
        geom_a, geom_b = load_geometry(name_a), load_geometry(name_b)
        accepted_rows = []
        patches_a, patches_b, masks_a, masks_b, counts_a, counts_b = [], [], [], [], [], []

        for row in centres.itertuples(index=False):
            pa, ca, ma = build_mosaic(survey_a, geom_a, row.midpoint_east, row.midpoint_north, CONFIG)
            pb, cb, mb = build_mosaic(survey_b, geom_b, row.midpoint_east, row.midpoint_north, CONFIG)
            coverage_a, coverage_b, joint = float(ma.mean()), float(mb.mean()), float((ma & mb).mean())
            if coverage_a < CONFIG.min_coverage_each or coverage_b < CONFIG.min_coverage_each or joint < CONFIG.min_joint_coverage:
                continue

            global_patch_id += 1
            record = row._asdict()
            record.update(
                patch_id=global_patch_id, pair=f"{name_a}-{name_b}", coverage_a=coverage_a, coverage_b=coverage_b,
                joint_coverage=joint, patch_size_m=CONFIG.patch_size_m, grid_resolution_m=CONFIG.grid_resolution_m,
                representation="geocoded_log1p_mean",
            )
            accepted_rows.append(record)
            patches_a.append(pa); patches_b.append(pb); masks_a.append(ma); masks_b.append(mb); counts_a.append(ca); counts_b.append(cb)

        metadata = pd.DataFrame(accepted_rows)
        if metadata.empty:
            del survey_a, survey_b, geom_a, geom_b
            gc.collect()
            continue

        patches_a, patches_b = np.stack(patches_a).astype(np.float32), np.stack(patches_b).astype(np.float32)
        masks_a, masks_b = np.stack(masks_a).astype(bool), np.stack(masks_b).astype(bool)
        counts_a, counts_b = np.stack(counts_a).astype(np.uint32), np.stack(counts_b).astype(np.uint32)

        out_path = PATCH_DIR / f"21_positive_{name_a}_{name_b}.npz"
        np.savez_compressed(
            out_path, patch_id=metadata["patch_id"].to_numpy(np.int64), match_id=metadata["match_id"].to_numpy(np.int64),
            center_east=metadata["midpoint_east"].to_numpy(float), center_north=metadata["midpoint_north"].to_numpy(float),
            patch_a=patches_a, patch_b=patches_b, mask_a=masks_a, mask_b=masks_b, count_a=counts_a, count_b=counts_b,
            coverage_a=metadata["coverage_a"].to_numpy(float), coverage_b=metadata["coverage_b"].to_numpy(float),
            joint_coverage=metadata["joint_coverage"].to_numpy(float), heading_difference_deg=metadata["heading_difference_deg"].to_numpy(float),
            look_difference_deg=metadata["look_difference_deg"].to_numpy(float), incidence_difference_deg=metadata["incidence_difference_deg"].to_numpy(float),
            view_class=metadata["view_class"].to_numpy(str), patch_size_m=np.float64(CONFIG.patch_size_m),
            grid_resolution_m=np.float64(CONFIG.grid_resolution_m), representation=np.asarray("geocoded_log1p_mean"),
        )

        metadata["patch_npz"] = str(out_path)
        metadata["pair_local_index"] = np.arange(len(metadata), dtype=int)
        all_metadata.append(metadata)
        save_gallery(name_a, name_b, metadata, patches_a, patches_b)
        print(f"{name_a}-{name_b}: accepted={len(metadata)}/{len(centres)}, median-joint={metadata['joint_coverage'].median():.1%}")

        del survey_a, survey_b, geom_a, geom_b, patches_a, patches_b, masks_a, masks_b, counts_a, counts_b
        gc.collect()

    if not all_metadata:
        raise RuntimeError("No positive benchmark patches passed the coverage thresholds.")

    metadata = pd.concat(all_metadata, ignore_index=True)
    metadata.to_csv(TABLE_DIR / "21_positive_patch_metadata.csv", index=False)
    summary = metadata.groupby(["pair", "view_class"], dropna=False).agg(
        patches=("patch_id", "count"), median_coverage_a=("coverage_a", "median"), median_coverage_b=("coverage_b", "median"),
        median_joint_coverage=("joint_coverage", "median"), median_heading_difference_deg=("heading_difference_deg", "median"),
        median_look_difference_deg=("look_difference_deg", "median"),
    ).reset_index()
    summary.to_csv(TABLE_DIR / "21_positive_patch_summary.csv", index=False)
    print(f"Positive patches: {len(metadata)}")
    print(f"Saved: {TABLE_DIR / '21_positive_patch_metadata.csv'}")


if __name__ == "__main__":
    main()
