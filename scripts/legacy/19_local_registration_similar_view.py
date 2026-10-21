"""LEGACY diagnostic — Diagnose small translational misregistration in similar-view candidate patch pairs.

Invocation: python scripts/legacy/19_local_registration_similar_view.py, after restoring legacy paths/prerequisites.
Input: 17_colocated_candidates_*.csv, 16_seabed_geometry_*.npz and raw XTF.
Output: 19_local_registration_*.png and 19_local_registration_results.csv; diagnostics only.
Not used by the canonical scripts.build_dataset pipeline; standalone execution
may require correcting historical config paths or recovering upstream artifacts.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from src.data.xtf_reader import read_xtf


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

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

PAIRS = (
    ("pass_a", "pass_b"),
    ("pass_a", "pass_c"),
    ("pass_b", "pass_c"),
)

# Use only easy, similar-view cases to diagnose fine registration.
MAX_CANDIDATE_DISTANCE_M = 1.0
MIN_CENTER_GROUND_RANGE_M = 4.0
N_EXAMPLES_PER_PAIR = 4
MIN_SELECTION_SPACING_M = 25.0

# A is the template; B is searched around the geometry-predicted centre.
# Odd dimensions keep the predicted centre invariant under flips.
TEMPLATE_ROWS = 401
TEMPLATE_COLS = 301
SEARCH_ROWS = 1201
SEARCH_COLS = 701

# Structural band-pass used only for registration, not for display/data storage.
SMOOTH_SIGMA = 1.2
BACKGROUND_SIGMA = 12.0
DISPLAY_PERCENTILES = (1.0, 99.5)

# Exclusion radius for estimating a second independent correlation peak.
SECOND_PEAK_EXCLUSION_ROWS = 30
SECOND_PEAK_EXCLUSION_COLS = 20

# Candidate selection progressively relaxes only if needed.
SELECTION_LEVELS = (
    # max look diff, max heading-axis diff, max incidence diff
    (5.0, 5.0, 8.0),
    (10.0, 10.0, 12.0),
    (20.0, 20.0, 15.0),
)


def candidate_path(a, b):
    """Resolve the earlier colocation-candidate CSV path for a survey pair."""
    return TABLE_DIR / f"17_colocated_candidates_{a}_{b}.csv"


def geometry_path(survey):
    """Resolve the historical per-survey geometry archive."""
    return TABLE_DIR / f"16_seabed_geometry_{survey}.npz"


def bool_mask(series):
    """Cast a potentially nullable metadata column to a boolean mask."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def heading_axis_difference_deg(values):
    """Difference between along-track axes, ignoring a 180-degree reversal."""
    d = np.asarray(values, dtype=np.float64)
    return np.minimum(d, 180.0 - d)


def normalize_side(side):
    """Normalize side labels used by CSVs and NPZ archives."""
    side = str(side).strip().lower()
    if side == "port":
        return "port"
    if side in {"starboard", "stbd"}:
        return "starboard"
    raise ValueError(f"Invalid side: {side}")


def geometry_side_suffix(side):
    """NPZ geometry uses *_port and *_stbd key suffixes."""
    return "port" if normalize_side(side) == "port" else "stbd"


def canonical_sonar(survey, side):
    """Return port or starboard samples with a consistent orientation."""
    side = normalize_side(side)
    if side == "port":
        return np.flip(survey.port, axis=1)
    return survey.starboard


def extract_exact(image, row, col, height, width):
    """Extract a fully in-bounds odd-sized patch centred at (row, col)."""
    row, col = int(row), int(col)
    hh, hw = height // 2, width // 2
    r0, r1 = row - hh, row + hh + 1
    c0, c1 = col - hw, col + hw + 1
    if r0 < 0 or c0 < 0 or r1 > image.shape[0] or c1 > image.shape[1]:
        raise ValueError("Requested patch is out of bounds.")
    return image[r0:r1, c0:c1].astype(np.float32, copy=False)


def align_second_window(window, heading_difference_deg, look_difference_deg):
    """Orient the second sonar crop relative to the reference heading/look."""
    aligned = window
    flip_ud = heading_difference_deg > 90.0
    flip_lr = look_difference_deg > 90.0
    if flip_ud:
        aligned = np.flipud(aligned)
    if flip_lr:
        aligned = np.fliplr(aligned)
    return aligned, flip_ud, flip_lr


def registration_preprocess(image):
    """Radiometry-robust structural representation for correlation only."""
    x = np.log1p(image.astype(np.float64))
    lo, hi = np.percentile(x, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip(x, lo, hi)
    fine = gaussian_filter(x, sigma=SMOOTH_SIGMA)
    background = gaussian_filter(x, sigma=BACKGROUND_SIGMA)
    return fine - background


def zncc_map(search, template):
    """Zero-mean normalized cross-correlation for every valid template position."""
    t = template.astype(np.float64, copy=False)
    s = search.astype(np.float64, copy=False)
    t0 = t - np.mean(t)
    t_energy = np.sum(t0 * t0)
    if t_energy <= 0:
        raise RuntimeError("Degenerate template: zero variance.")

    # Because sum(t0) == 0, the numerator is already invariant to local search mean.
    numerator = fftconvolve(s, t0[::-1, ::-1], mode="valid")
    kernel = np.ones(t.shape, dtype=np.float64)
    local_sum = fftconvolve(s, kernel, mode="valid")
    local_sq_sum = fftconvolve(s * s, kernel, mode="valid")
    n = float(t.size)
    local_energy = local_sq_sum - (local_sum * local_sum) / n
    denominator = np.sqrt(np.maximum(local_energy, 0.0) * t_energy)

    corr = np.full_like(numerator, -np.inf, dtype=np.float64)
    valid = denominator > 1e-12
    corr[valid] = numerator[valid] / denominator[valid]
    return np.clip(corr, -1.0, 1.0)


def best_and_second_peak(corr):
    """Estimate strongest and independent runner-up registration peaks."""
    best_flat = int(np.nanargmax(corr))
    br, bc = np.unravel_index(best_flat, corr.shape)
    best = float(corr[br, bc])

    masked = corr.copy()
    r0 = max(0, br - SECOND_PEAK_EXCLUSION_ROWS)
    r1 = min(corr.shape[0], br + SECOND_PEAK_EXCLUSION_ROWS + 1)
    c0 = max(0, bc - SECOND_PEAK_EXCLUSION_COLS)
    c1 = min(corr.shape[1], bc + SECOND_PEAK_EXCLUSION_COLS + 1)
    masked[r0:r1, c0:c1] = -np.inf
    second = float(np.max(masked)) if np.any(np.isfinite(masked)) else np.nan
    return br, bc, best, second


def far_enough(row, selected):
    """Reject candidate anchors that are too close to already selected locations."""
    if not selected:
        return True
    dx = np.asarray([row.midpoint_east - r.midpoint_east for r in selected])
    dy = np.asarray([row.midpoint_north - r.midpoint_north for r in selected])
    return bool(np.all(np.hypot(dx, dy) >= MIN_SELECTION_SPACING_M))


def select_examples(df, n_rows_a, n_cols_a, n_rows_b, n_cols_b):
    """Select well-spaced, compatible candidates for fine-registration diagnostics."""
    df = df.copy()
    df["heading_axis_difference_deg"] = heading_axis_difference_deg(df["heading_difference_deg"])

    base = (
        (df["distance_m"] <= MAX_CANDIDATE_DISTANCE_M)
        & (df["a_ground_range_m"] >= MIN_CENTER_GROUND_RANGE_M)
        & (df["b_ground_range_m"] >= MIN_CENTER_GROUND_RANGE_M)
        & bool_mask(df["both_high_confidence"])
        & (df["a_row_index"] >= TEMPLATE_ROWS // 2)
        & (df["a_row_index"] < n_rows_a - TEMPLATE_ROWS // 2)
        & (df["a_canonical_sample_index"] >= TEMPLATE_COLS // 2)
        & (df["a_canonical_sample_index"] < n_cols_a - TEMPLATE_COLS // 2)
        & (df["b_row_index"] >= SEARCH_ROWS // 2)
        & (df["b_row_index"] < n_rows_b - SEARCH_ROWS // 2)
        & (df["b_canonical_sample_index"] >= SEARCH_COLS // 2)
        & (df["b_canonical_sample_index"] < n_cols_b - SEARCH_COLS // 2)
    )

    selected = []
    selected_ids = set()
    used_level = {}

    for level, (look_max, axis_max, inc_max) in enumerate(SELECTION_LEVELS, start=1):
        pool = df.loc[
            base
            & (df["look_difference_deg"] <= look_max)
            & (df["heading_axis_difference_deg"] <= axis_max)
            & (df["incidence_difference_deg"] <= inc_max)
        ].sort_values(["distance_m", "match_id"], kind="stable")

        for row in pool.itertuples(index=False):
            if int(row.match_id) in selected_ids or not far_enough(row, selected):
                continue
            selected.append(row)
            selected_ids.add(int(row.match_id))
            used_level[int(row.match_id)] = level
            if len(selected) == N_EXAMPLES_PER_PAIR:
                break
        if len(selected) == N_EXAMPLES_PER_PAIR:
            break

    if not selected:
        raise RuntimeError("No suitable similar-view candidates for local registration.")

    out = pd.DataFrame([r._asdict() for r in selected])
    out["selection_level"] = [used_level[int(x)] for x in out["match_id"]]
    return out


def display_normalize_joint(*patches):
    """Apply one shared display range to a group of sonar windows."""
    logs = [np.log1p(p.astype(np.float64)) for p in patches]
    values = np.concatenate([p.ravel() for p in logs])
    lo, hi = np.percentile(values, DISPLAY_PERCENTILES)
    if hi <= lo:
        hi = lo + 1.0
    return [np.clip((p - lo) / (hi - lo), 0.0, 1.0) for p in logs]


def local_along_track_spacing_m(geometry, row, radius=50):
    """Estimate ground distance represented by adjacent ping rows."""
    east = geometry["east"]
    north = geometry["north"]
    r0 = max(0, int(row) - radius)
    r1 = min(len(east), int(row) + radius + 1)
    step = np.hypot(np.diff(east[r0:r1]), np.diff(north[r0:r1]))
    step = step[np.isfinite(step) & (step > 0)]
    return float(np.median(step)) if step.size else np.nan


def ground_axis_for_row(geometry, side, row):
    """Find valid horizontal seabed ranges for a given ping and sonar side."""
    suffix = geometry_side_suffix(side)
    slant = geometry[f"canonical_slant_axis_{suffix}"]
    h = float(geometry[f"first_bottom_{suffix}"][int(row)])
    ground = np.full_like(slant, np.nan, dtype=np.float64)
    valid = slant >= h
    ground[valid] = np.sqrt(np.maximum(slant[valid] ** 2 - h ** 2, 0.0))
    return ground


def local_cross_track_spacing_m(geometry, side, row, sample):
    """Estimate local per-sample ground distance across track."""
    ground = ground_axis_for_row(geometry, side, row)
    sample = int(sample)
    i0, i1 = max(0, sample - 1), min(len(ground) - 1, sample + 1)
    if i1 == i0 or not np.isfinite(ground[i0]) or not np.isfinite(ground[i1]):
        return np.nan
    return float(abs(ground[i1] - ground[i0]) / (i1 - i0))


def geocode_sample(geometry, side, row, sample):
    """Map one row/sample observation to ground UTM coordinates."""
    row, sample = int(row), int(sample)
    side = normalize_side(side)
    suffix = geometry_side_suffix(side)
    slant = geometry[f"canonical_slant_axis_{suffix}"]
    h = float(geometry[f"first_bottom_{suffix}"][row])
    r = float(slant[sample])
    if not np.isfinite(h) or r < h:
        return np.nan, np.nan
    g = float(np.sqrt(max(r * r - h * h, 0.0)))
    theta = np.deg2rad(float(geometry["heading_deg"][row]))
    if side == "starboard":
        de, dn = np.cos(theta), -np.sin(theta)
    else:
        de, dn = -np.cos(theta), np.sin(theta)
    e = float(geometry["east"][row]) + g * de
    n = float(geometry["north"][row]) + g * dn
    return e, n


def register_one(row, survey_a, survey_b, geom_b):
    """Estimate local translation and candidate fit for one matching sonar pair."""
    image_a = canonical_sonar(survey_a, row.a_side)
    image_b = canonical_sonar(survey_b, row.b_side)

    template_raw = extract_exact(
        image_a, row.a_row_index, row.a_canonical_sample_index, TEMPLATE_ROWS, TEMPLATE_COLS
    )
    search_raw_native = extract_exact(
        image_b, row.b_row_index, row.b_canonical_sample_index, SEARCH_ROWS, SEARCH_COLS
    )
    search_raw, flip_ud, flip_lr = align_second_window(
        search_raw_native, row.heading_difference_deg, row.look_difference_deg
    )

    template = registration_preprocess(template_raw)
    search = registration_preprocess(search_raw)
    corr = zncc_map(search, template)
    br, bc, best_score, second_score = best_and_second_peak(corr)

    expected_r = (SEARCH_ROWS - TEMPLATE_ROWS) // 2
    expected_c = (SEARCH_COLS - TEMPLATE_COLS) // 2
    dr_aligned = int(br - expected_r)
    dc_aligned = int(bc - expected_c)
    dr_native = -dr_aligned if flip_ud else dr_aligned
    dc_native = -dc_aligned if flip_lr else dc_aligned

    best_b_row = int(row.b_row_index + dr_native)
    best_b_sample = int(row.b_canonical_sample_index + dc_native)

    # Matched crop in the aligned search window for visual comparison.
    best_crop_raw = search_raw[br:br + TEMPLATE_ROWS, bc:bc + TEMPLATE_COLS]
    predicted_crop_raw = search_raw[
        expected_r:expected_r + TEMPLATE_ROWS,
        expected_c:expected_c + TEMPLATE_COLS,
    ]

    along_spacing = local_along_track_spacing_m(geom_b, row.b_row_index)
    cross_spacing = local_cross_track_spacing_m(
        geom_b, row.b_side, row.b_row_index, row.b_canonical_sample_index
    )
    along_shift_m = abs(dr_native) * along_spacing if np.isfinite(along_spacing) else np.nan
    cross_shift_m = abs(dc_native) * cross_spacing if np.isfinite(cross_spacing) else np.nan
    local_shift_m = (
        float(np.hypot(along_shift_m, cross_shift_m))
        if np.isfinite(along_shift_m) and np.isfinite(cross_shift_m)
        else np.nan
    )

    reg_e, reg_n = geocode_sample(geom_b, row.b_side, best_b_row, best_b_sample)
    registered_distance_to_a = (
        float(np.hypot(reg_e - row.a_east, reg_n - row.a_north))
        if np.isfinite(reg_e) and np.isfinite(reg_n)
        else np.nan
    )

    result = {
        **row._asdict(),
        "flip_ud": flip_ud,
        "flip_lr": flip_lr,
        "zncc_best": best_score,
        "zncc_second": second_score,
        "zncc_peak_margin": best_score - second_score if np.isfinite(second_score) else np.nan,
        "row_shift_aligned_px": dr_aligned,
        "sample_shift_aligned_px": dc_aligned,
        "row_shift_native_px": dr_native,
        "sample_shift_native_px": dc_native,
        "local_along_spacing_m_per_row": along_spacing,
        "local_cross_spacing_m_per_sample": cross_spacing,
        "estimated_along_shift_m": along_shift_m,
        "estimated_cross_shift_m": cross_shift_m,
        "estimated_local_shift_m": local_shift_m,
        "registered_b_row_index": best_b_row,
        "registered_b_canonical_sample_index": best_b_sample,
        "registered_b_east": reg_e,
        "registered_b_north": reg_n,
        "registered_distance_to_a_m": registered_distance_to_a,
    }

    return result, template_raw, predicted_crop_raw, best_crop_raw


def save_pair_figure(pair_results, visual_patches, survey_a, survey_b):
    """Render raw, predicted and translated crops for a diagnostic pair."""
    n = len(pair_results)
    fig, axes = plt.subplots(n, 3, figsize=(14, max(3.2 * n, 5.0)), squeeze=False)

    for i, (result, patches) in enumerate(zip(pair_results, visual_patches)):
        a_raw, b_pred_raw, b_best_raw = patches
        a_disp, pred_disp, best_disp = display_normalize_joint(a_raw, b_pred_raw, b_best_raw)

        for j, image in enumerate((a_disp, pred_disp, best_disp)):
            axes[i, j].imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
            axes[i, j].axhline(TEMPLATE_ROWS // 2, linewidth=0.6, alpha=0.6)
            axes[i, j].axvline(TEMPLATE_COLS // 2, linewidth=0.6, alpha=0.6)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        axes[i, 0].set_title(
            f"{survey_a} predicted anchor\nL{result['a_leg_number']} {str(result['a_side']).upper()} | "
            f"row {result['a_row_index']} sample {result['a_canonical_sample_index']}", fontsize=9
        )
        axes[i, 1].set_title(
            f"{survey_b} geometry-predicted crop\n"
            f"d_geo={result['distance_m']:.3f} m", fontsize=9
        )
        axes[i, 2].set_title(
            f"{survey_b} best local translation\n"
            f"ZNCC={result['zncc_best']:.2f}, shift≈{result['estimated_local_shift_m']:.1f} m", fontsize=9
        )
        axes[i, 0].set_ylabel(
            f"Δlook={result['look_difference_deg']:.1f}°\n"
            f"Δaxis={result['heading_axis_difference_deg']:.1f}°\n"
            f"Δinc={result['incidence_difference_deg']:.1f}°\n"
            f"dr={result['row_shift_native_px']} dc={result['sample_shift_native_px']}",
            fontsize=8,
        )

    fig.suptitle(
        f"Local registration diagnostic — {survey_a} vs {survey_b}\n"
        "similar-view candidates; translation search only (diagnostic, not ground truth)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIGURE_DIR / f"19_local_registration_{survey_a}_{survey_b}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

print("=" * 90)
print("BLUEBOAT — LOCAL REGISTRATION DIAGNOSTIC (SIMILAR VIEW)")
print("=" * 90)
print(f"Template                      : {TEMPLATE_ROWS} x {TEMPLATE_COLS} px")
print(f"Search window                 : {SEARCH_ROWS} x {SEARCH_COLS} px")
print(f"Examples per survey pair      : {N_EXAMPLES_PER_PAIR}")
print(f"Max geometric candidate dist. : {MAX_CANDIDATE_DISTANCE_M:.2f} m")
print("Purpose: test whether 17 is approximately correct but locally shifted.")
print("This script does NOT create ground-truth pairs and does NOT modify 16/17 geometry.")

all_results = []

for survey_name_a, survey_name_b in PAIRS:
    print("\n" + "=" * 90)
    print(f"{survey_name_a.upper()} vs {survey_name_b.upper()}")
    print("=" * 90)

    cpath = candidate_path(survey_name_a, survey_name_b)
    if not cpath.exists():
        raise FileNotFoundError(f"Missing {cpath}. Run script 17 first.")

    gpath_a = geometry_path(survey_name_a)
    gpath_b = geometry_path(survey_name_b)
    if not gpath_a.exists() or not gpath_b.exists():
        raise FileNotFoundError("Missing script-16 geometry outputs.")

    candidates = pd.read_csv(cpath)
    geom_a = np.load(gpath_a)
    geom_b = np.load(gpath_b)

    n_rows_a = len(geom_a["row_index"])
    n_rows_b = len(geom_b["row_index"])
    n_cols_a = len(geom_a["canonical_slant_axis_port"])
    n_cols_b = len(geom_b["canonical_slant_axis_port"])

    selected = select_examples(candidates, n_rows_a, n_cols_a, n_rows_b, n_cols_b)
    print(f"Selected candidates            : {len(selected)}")
    print(
        "Selection levels              : "
        + ", ".join(str(int(x)) for x in selected["selection_level"])
    )
    print(
        f"Median geometry distance       : {selected['distance_m'].median():.3f} m"
    )
    print(
        f"Median Δlook / Δaxis / Δinc    : "
        f"{selected['look_difference_deg'].median():.1f}° / "
        f"{selected['heading_axis_difference_deg'].median():.1f}° / "
        f"{selected['incidence_difference_deg'].median():.1f}°"
    )

    print(f"Loading {survey_name_a}...")
    survey_a = read_xtf(XTF_DIR / SURVEYS[survey_name_a])
    print(f"Loading {survey_name_b}...")
    survey_b = read_xtf(XTF_DIR / SURVEYS[survey_name_b])

    pair_results = []
    visual_patches = []

    for row in selected.itertuples(index=False):
        result, a_raw, b_pred_raw, b_best_raw = register_one(row, survey_a, survey_b, geom_b)
        result["pair"] = f"{survey_name_a}-{survey_name_b}"
        pair_results.append(result)
        visual_patches.append((a_raw, b_pred_raw, b_best_raw))

    figure_path = save_pair_figure(pair_results, visual_patches, survey_name_a, survey_name_b)
    pair_df = pd.DataFrame(pair_results)
    all_results.append(pair_df)

    print(f"Median best ZNCC              : {pair_df['zncc_best'].median():.3f}")
    print(f"Median local shift estimate   : {pair_df['estimated_local_shift_m'].median():.2f} m")
    print(f"Median registered A-B dist.   : {pair_df['registered_distance_to_a_m'].median():.2f} m")
    print(f"Figure                         : {figure_path}")

    del survey_a, survey_b, candidates, selected, geom_a, geom_b
    gc.collect()

results = pd.concat(all_results, ignore_index=True)
output_path = TABLE_DIR / "19_local_registration_results.csv"
results.to_csv(output_path, index=False)

print("\n" + "=" * 90)
print("LOCAL REGISTRATION DIAGNOSTIC COMPLETED")
print("=" * 90)
print(f"Results: {output_path}")
print("Inspect whether the third column (best local translation) resembles the A template better than the geometry-predicted crop.")
print("A strong, spatially coherent offset suggests coarse geometry is right but fine registration needs correction.")
print("Weak/ambiguous matches mean translation-only registration is insufficient; do not force a correction.")
