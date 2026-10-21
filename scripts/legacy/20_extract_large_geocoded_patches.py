"""LEGACY diagnostic — Experiment with larger metric-ground-footprint geocoded mosaics and coverage.

Invocation: python scripts/legacy/20_extract_large_geocoded_patches.py, after restoring legacy paths/prerequisites.
Input: 17_colocated_candidates_*.csv, 16_seabed_geometry_*.npz and raw XTF.
Output: 20_large_geocoded_patches_*.png/*.npz and 20_large_geocoded_patch_summary.csv.
Not used by the canonical scripts.build_dataset pipeline; standalone execution
may require correcting historical config paths or recovering upstream artifacts.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from src.data.xtf_reader import read_xtf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

XTF_DIR = Path(config["dataset_root"]) / "SSS-XTF"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SURVEYS = {
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}

PAIRS = (("pass_a", "pass_b"), ("pass_a", "pass_c"), ("pass_b", "pass_c"))

# Large-patch mitigation specified in the project proposal.
PATCH_SIZE_M = 50.0
GRID_RESOLUTION_M = 0.25
GRID_SIZE = int(round(PATCH_SIZE_M / GRID_RESOLUTION_M))

MAX_CANDIDATE_DISTANCE_M = 2.0
MAX_PATCHES_PER_PAIR = 4
MIN_CENTER_SPACING_M = 50.0
DENSITY_RADIUS_M = 30.0
MIN_GROUND_RANGE_M = 1.0

# Diagnostic rasterisation: enough density for visual validation without
# processing every sonar sample.
SOURCE_ROW_STRIDE = 4
SOURCE_SAMPLE_STRIDE = 4
SENSOR_SEARCH_EXTRA_M = 30.0
DISPLAY_PERCENTILES = (1.0, 99.5)

# Only these arrays are needed from script 16. Loading them eagerly is
# important: np.load(...)[key] on a compressed NPZ inside a tight loop would
# repeatedly decompress arrays and make the script appear stuck.
GEOMETRY_KEYS = (
    "east",
    "north",
    "heading_deg",
    "in_analysis_leg",
    "first_bottom_port",
    "first_bottom_stbd",
    "high_confidence_port",
    "high_confidence_stbd",
    "canonical_slant_axis_port",
    "canonical_slant_axis_stbd",
)


def candidate_path(a, b):
    """Resolve the earlier colocation-candidate CSV path for a survey pair."""
    return TABLE_DIR / f"17_colocated_candidates_{a}_{b}.csv"


def geometry_path(name):
    """Resolve the historical per-survey geometry archive."""
    return TABLE_DIR / f"16_seabed_geometry_{name}.npz"


def view_class(angle):
    """Categorize relative sonar viewing angle for candidate review."""
    if angle <= 20.0:
        return "similar-view"
    if angle >= 160.0:
        return "opposite-view"
    return "oblique-view"


def bool_mask(series):
    """Cast a potentially nullable metadata column to a boolean mask."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def select_centers(df):
    """Choose geographically dispersed mosaic centres from colocation candidates."""
    df = df.copy()
    df["view_class"] = df["look_difference_deg"].map(view_class)
    valid = (
        (df["distance_m"] <= MAX_CANDIDATE_DISTANCE_M)
        & (df["a_ground_range_m"] >= MIN_GROUND_RANGE_M)
        & (df["b_ground_range_m"] >= MIN_GROUND_RANGE_M)
        & bool_mask(df["both_high_confidence"])
    )
    pool = df.loc[valid].copy()
    if pool.empty:
        raise RuntimeError("No valid coarse co-location candidates from script 17.")

    xy = pool[["midpoint_east", "midpoint_north"]].to_numpy(float)
    tree = cKDTree(xy)
    pool["local_density"] = np.fromiter(
        (len(ids) for ids in tree.query_ball_point(xy, r=DENSITY_RADIUS_M)),
        dtype=int,
        count=len(pool),
    )
    pool = pool.sort_values(
        ["local_density", "distance_m", "match_id"],
        ascending=[False, True, True],
    )

    selected = []
    selected_ids = set()

    def far_enough(row, spacing):
        """Reject candidate anchors that are too close to already selected locations."""
        return not selected or min(
            np.hypot(
                row.midpoint_east - s.midpoint_east,
                row.midpoint_north - s.midpoint_north,
            )
            for s in selected
        ) >= spacing

    def add_from(subset, spacing, limit=1):
        """Add a candidate from a subset while respecting spacing and quotas."""
        added = 0
        for row in subset.itertuples(index=False):
            if row.match_id in selected_ids or not far_enough(row, spacing):
                continue
            selected.append(row)
            selected_ids.add(row.match_id)
            added += 1
            if added >= limit or len(selected) >= MAX_PATCHES_PER_PAIR:
                break

    for spacing in (MIN_CENTER_SPACING_M, 35.0, 20.0):
        for category in ("similar-view", "opposite-view", "oblique-view"):
            if len(selected) >= MAX_PATCHES_PER_PAIR:
                break
            if not any(s.view_class == category for s in selected):
                add_from(pool[pool["view_class"] == category], spacing)
        if len(selected) < MAX_PATCHES_PER_PAIR:
            add_from(pool, spacing, MAX_PATCHES_PER_PAIR - len(selected))
        if len(selected) >= MAX_PATCHES_PER_PAIR:
            break

    if not selected:
        raise RuntimeError("Unable to choose patch centres.")

    out = pd.DataFrame([r._asdict() for r in selected])
    out.insert(0, "patch_id", np.arange(1, len(out) + 1))
    return out


def load_geometry(name):
    """Load the necessary geocoded per-ping geometry arrays."""
    path = geometry_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry: {path}\nRun script 16 first.")

    # Eagerly materialise the arrays once. This fixes the severe slowdown of
    # repeatedly reading individual values from the compressed NPZ archive.
    with np.load(path) as archive:
        missing = [key for key in GEOMETRY_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"Missing geometry arrays in {path.name}: {missing}")
        return {key: np.asarray(archive[key]) for key in GEOMETRY_KEYS}


def canonical_image(survey, side):
    """Return the selected side-scan channel in canonical orientation."""
    if side == "port":
        return np.flip(survey.port, axis=1)
    if side == "starboard":
        return survey.starboard
    raise ValueError(f"Invalid side: {side}")


def build_mosaic(survey, geom, center_e, center_n):
    """Rasterize geocoded acoustic returns around a specified metric centre."""
    half = PATCH_SIZE_M / 2.0
    e_min, n_min = center_e - half, center_n - half
    sum_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    count_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint32)

    search_radius = np.sqrt(2.0) * half + SENSOR_SEARCH_EXTRA_M
    sensor_dist = np.hypot(geom["east"] - center_e, geom["north"] - center_n)
    base_rows = geom["in_analysis_leg"].astype(bool) & (sensor_dist <= search_radius)

    for side, suffix in (("port", "port"), ("starboard", "stbd")):
        rows = np.flatnonzero(base_rows & geom[f"high_confidence_{suffix}"].astype(bool))
        rows = rows[::SOURCE_ROW_STRIDE]
        if rows.size == 0:
            continue

        image = canonical_image(survey, side)
        slant_axis = geom[f"canonical_slant_axis_{suffix}"].astype(np.float64, copy=False)
        sample_idx = np.arange(0, slant_axis.size, SOURCE_SAMPLE_STRIDE, dtype=np.int32)
        slant = slant_axis[sample_idx][None, :]

        h = geom[f"first_bottom_{suffix}"][rows].astype(np.float64, copy=False)[:, None]
        finite_h = np.isfinite(h)
        valid = finite_h & (slant >= h)

        # row x sample ground-range matrix; sizes here are only a few hundred
        # rows by ~300 samples, so vectorising is both fast and memory-safe.
        ground = np.sqrt(np.maximum(slant * slant - h * h, 0.0))
        valid &= ground >= MIN_GROUND_RANGE_M

        theta = np.deg2rad(geom["heading_deg"][rows].astype(np.float64, copy=False))[:, None]
        if side == "starboard":
            de, dn = np.cos(theta), -np.sin(theta)
        else:
            de, dn = -np.cos(theta), np.sin(theta)

        east = geom["east"][rows, None] + ground * de
        north = geom["north"][rows, None] + ground * dn

        inside = (
            valid
            & (east >= e_min)
            & (east < e_min + PATCH_SIZE_M)
            & (north >= n_min)
            & (north < n_min + PATCH_SIZE_M)
        )
        if not np.any(inside):
            continue

        values = np.log1p(image[np.ix_(rows, sample_idx)].astype(np.float64))
        ix = np.floor((east - e_min) / GRID_RESOLUTION_M).astype(np.int32)
        iy = np.floor((north - n_min) / GRID_RESOLUTION_M).astype(np.int32)

        linear = (iy[inside] * GRID_SIZE + ix[inside]).astype(np.int64)
        vals = values[inside]
        n_cells = GRID_SIZE * GRID_SIZE

        sum_grid += np.bincount(linear, weights=vals, minlength=n_cells).reshape(GRID_SIZE, GRID_SIZE)
        count_grid += np.bincount(linear, minlength=n_cells).reshape(GRID_SIZE, GRID_SIZE).astype(np.uint32)

    occupied = count_grid > 0
    mosaic = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
    mosaic[occupied] = (sum_grid[occupied] / count_grid[occupied]).astype(np.float32)
    return mosaic, count_grid, float(occupied.mean())


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


def save_gallery(pair_a, pair_b, selected, mosaics_a, mosaics_b, coverages_a, coverages_b):
    """Save a labeled visual gallery for manual inspection."""
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(11, max(4.2 * n, 5.0)), squeeze=False)
    half = PATCH_SIZE_M / 2.0
    extent = [-half, half, -half, half]

    for i, row in enumerate(selected.itertuples(index=False)):
        a, b = normalize_pair(mosaics_a[i], mosaics_b[i])
        for ax, image, survey_name, coverage in (
            (axes[i, 0], a, pair_a, coverages_a[i]),
            (axes[i, 1], b, pair_b, coverages_b[i]),
        ):
            ax.imshow(
                np.ma.masked_invalid(image),
                cmap="gray",
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
            ax.axhline(0.0, linewidth=0.6, alpha=0.6)
            ax.axvline(0.0, linewidth=0.6, alpha=0.6)
            ax.set_aspect("equal")
            ax.set_xlabel("East offset [m]")
            ax.set_ylabel("North offset [m]")
            ax.set_title(f"{survey_name.upper()} | coverage {coverage:.0%}")

        axes[i, 0].text(
            -24,
            23,
            f"{row.view_class} | d={row.distance_m:.2f} m | "
            f"ΔH={row.heading_difference_deg:.0f}° | ΔL={row.look_difference_deg:.0f}°",
            fontsize=8,
            va="top",
        )

    fig.suptitle(
        f"50 m geocoded co-located patches — {pair_a.upper()} vs {pair_b.upper()}\n"
        f"Common North-up grid, {GRID_RESOLUTION_M:.2f} m/cell",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIGURE_DIR / f"20_large_geocoded_patches_{pair_a}_{pair_b}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


print("=" * 90)
print("BLUEBOAT — 50 m GEOCODED CO-LOCATED PATCHES")
print("=" * 90)
print(f"Patch                      : {PATCH_SIZE_M:.0f} m x {PATCH_SIZE_M:.0f} m")
print(f"Common grid                : {GRID_SIZE} x {GRID_SIZE} @ {GRID_RESOLUTION_M:.2f} m/cell")
print(f"Patches per survey pair    : <= {MAX_PATCHES_PER_PAIR}")
print("Source                     : script 17 coarse co-location + script 16 geometry")
print("Purpose                    : large-patch validation/extraction, not pixel-level registration")

all_rows = []

for name_a, name_b in PAIRS:
    print("\n" + "=" * 90)
    print(f"{name_a.upper()} vs {name_b.upper()}")
    print("=" * 90)

    csv = candidate_path(name_a, name_b)
    if not csv.exists():
        raise FileNotFoundError(f"Missing candidates: {csv}\nRun script 17 first.")

    candidates = pd.read_csv(csv)
    selected = select_centers(candidates)
    print(f"Selected centres           : {len(selected)}")
    print("View classes               : " + ", ".join(selected["view_class"].tolist()))

    print(f"Loading {name_a} / {name_b}...")
    survey_a = read_xtf(XTF_DIR / SURVEYS[name_a])
    survey_b = read_xtf(XTF_DIR / SURVEYS[name_b])
    geom_a = load_geometry(name_a)
    geom_b = load_geometry(name_b)

    mosaics_a, mosaics_b, counts_a, counts_b = [], [], [], []
    coverages_a, coverages_b = [], []

    for i, row in enumerate(selected.itertuples(index=False), start=1):
        print(f"  Patch {i}/{len(selected)}...", end="", flush=True)
        ma, ca, cova = build_mosaic(survey_a, geom_a, row.midpoint_east, row.midpoint_north)
        mb, cb, covb = build_mosaic(survey_b, geom_b, row.midpoint_east, row.midpoint_north)
        mosaics_a.append(ma)
        mosaics_b.append(mb)
        counts_a.append(ca)
        counts_b.append(cb)
        coverages_a.append(cova)
        coverages_b.append(covb)
        print(f" coverage {cova:.1%} / {covb:.1%}")

    mosaics_a = np.stack(mosaics_a)
    mosaics_b = np.stack(mosaics_b)
    counts_a = np.stack(counts_a)
    counts_b = np.stack(counts_b)

    npz_path = TABLE_DIR / f"20_large_geocoded_patches_{name_a}_{name_b}.npz"
    np.savez_compressed(
        npz_path,
        patch_size_m=np.float64(PATCH_SIZE_M),
        grid_resolution_m=np.float64(GRID_RESOLUTION_M),
        match_id=selected["match_id"].to_numpy(np.int64),
        center_east=selected["midpoint_east"].to_numpy(float),
        center_north=selected["midpoint_north"].to_numpy(float),
        view_class=selected["view_class"].to_numpy(str),
        heading_difference_deg=selected["heading_difference_deg"].to_numpy(float),
        look_difference_deg=selected["look_difference_deg"].to_numpy(float),
        mosaic_a=mosaics_a,
        mosaic_b=mosaics_b,
        count_a=counts_a,
        count_b=counts_b,
        coverage_a=np.asarray(coverages_a, dtype=float),
        coverage_b=np.asarray(coverages_b, dtype=float),
    )

    figure_path = save_gallery(
        name_a, name_b, selected, mosaics_a, mosaics_b, coverages_a, coverages_b
    )

    selected = selected.copy()
    selected["pair"] = f"{name_a}-{name_b}"
    selected["coverage_a"] = coverages_a
    selected["coverage_b"] = coverages_b
    selected["npz_path"] = str(npz_path)
    all_rows.append(selected)

    print(f"Median coverage            : {np.median(coverages_a):.1%} / {np.median(coverages_b):.1%}")
    print(f"Patches                    : {npz_path}")
    print(f"Figure                     : {figure_path}")

    del survey_a, survey_b, geom_a, geom_b, candidates
    del mosaics_a, mosaics_b, counts_a, counts_b
    gc.collect()

summary = pd.concat(all_rows, ignore_index=True)
summary_path = TABLE_DIR / "20_large_geocoded_patch_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 90)
print("50 m PATCH EXTRACTION COMPLETED")
print("=" * 90)
print(f"Summary                    : {summary_path}")
print("Next check                 : inspect the common North-up 50 m regions across passes.")
