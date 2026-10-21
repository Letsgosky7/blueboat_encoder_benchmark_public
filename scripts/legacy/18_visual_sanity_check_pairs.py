"""LEGACY diagnostic — Inspect whether candidate colocated sonar patches appear consistent by eye.

Invocation: python scripts/legacy/18_visual_sanity_check_pairs.py, after restoring legacy paths/prerequisites.
Input: 17_colocated_candidates_*.csv and raw XTF recordings.
Output: 18_visual_sanity_*.png and 18_visual_sanity_selection.csv.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

XTF_DIR = Path(config["dataset_root"]) / "SSS-XTF"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

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

# Visual sanity-check only: these are not the final benchmark patch dimensions.
PATCH_ROWS = 384
PATCH_COLS = 384

# Select only very close geometric candidates for the visual check.
MAX_VISUAL_DISTANCE_M = 1.0

# Avoid patch centres too close to the nadir/water-column boundary.
MIN_CENTER_GROUND_RANGE_M = 4.0

# Two examples per viewing-geometry class, when available.
N_PER_VIEW_CLASS = 2

# Prevent the gallery from containing many almost identical neighbouring patches.
MIN_SELECTION_SPACING_M = 20.0

# Percentiles used after log1p. The same limits are used for both patches of a pair.
DISPLAY_PERCENTILES = (1.0, 99.5)


# -----------------------------------------------------------------------------
# Paths / helpers
# -----------------------------------------------------------------------------

def candidate_path(survey_a, survey_b):
    """Resolve the earlier colocation-candidate CSV path for a survey pair."""
    return TABLE_DIR / f"17_colocated_candidates_{survey_a}_{survey_b}.csv"


def view_class(look_difference_deg):
    """Categorize relative sonar viewing angle for candidate review."""
    if look_difference_deg <= 20.0:
        return "similar-view"
    if look_difference_deg >= 160.0:
        return "opposite-view"
    return "oblique-view"


def canonical_sonar(survey, side):
    """Return port or starboard samples with a consistent orientation."""
    if side == "port":
        return np.flip(survey.port, axis=1)
    if side == "starboard":
        return survey.starboard
    raise ValueError(f"Side non valido: {side}")


def extract_centered_patch(image, row, col, height=PATCH_ROWS, width=PATCH_COLS):
    """Extract a fixed-size patch, padding outside the image with NaN."""
    row = int(row)
    col = int(col)
    half_h = height // 2
    half_w = width // 2

    r0, r1 = row - half_h, row - half_h + height
    c0, c1 = col - half_w, col - half_w + width

    src_r0, src_r1 = max(0, r0), min(image.shape[0], r1)
    src_c0, src_c1 = max(0, c0), min(image.shape[1], c1)

    patch = np.full((height, width), np.nan, dtype=np.float32)
    dst_r0, dst_r1 = src_r0 - r0, src_r1 - r0
    dst_c0, dst_c1 = src_c0 - c0, src_c1 - c0

    patch[dst_r0:dst_r1, dst_c0:dst_c1] = image[src_r0:src_r1, src_c0:src_c1].astype(np.float32)
    return patch


def normalize_pair_for_display(patch_a, patch_b):
    """Joint log+percentile normalization so A/B use the same display scale."""
    log_a = np.log1p(patch_a)
    log_b = np.log1p(patch_b)
    values = np.concatenate((log_a[np.isfinite(log_a)], log_b[np.isfinite(log_b)]))

    if values.size == 0:
        return np.zeros_like(log_a), np.zeros_like(log_b)

    low, high = np.percentile(values, DISPLAY_PERCENTILES)
    if high <= low:
        high = low + 1.0

    def scale(x):
        """Rescale an image for visual comparison without changing benchmark data."""
        y = (x - low) / (high - low)
        y = np.clip(y, 0.0, 1.0)
        return np.nan_to_num(y, nan=0.0)

    return scale(log_a), scale(log_b)


def display_align_second_patch(patch_b, heading_difference_deg, look_difference_deg):
    """
    Approximate display-only alignment of local image axes.

    Vertical image axis follows vehicle direction; horizontal image axis follows
    sonar look direction. Flips are therefore useful for 180-degree sign changes.
    No rotation/interpolation is applied, and the stored data are never modified.
    """
    aligned = patch_b
    operations = []

    if heading_difference_deg > 90.0:
        aligned = np.flipud(aligned)
        operations.append("UD")

    if look_difference_deg > 90.0:
        aligned = np.fliplr(aligned)
        operations.append("LR")

    return aligned, "+".join(operations) if operations else "none"


def bool_mask(series):
    """Cast a potentially nullable metadata column to a boolean mask."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


# -----------------------------------------------------------------------------
# Candidate selection
# -----------------------------------------------------------------------------

def far_enough(row, selected_rows, minimum_spacing_m):
    """Reject candidate anchors that are too close to already selected locations."""
    if not selected_rows:
        return True
    dx = np.asarray([row.midpoint_east - s.midpoint_east for s in selected_rows])
    dy = np.asarray([row.midpoint_north - s.midpoint_north for s in selected_rows])
    return bool(np.all(np.hypot(dx, dy) >= minimum_spacing_m))


def greedy_select(pool, n, already_selected=None):
    """Select spatially separated review candidates under per-class quotas."""
    selected = [] if already_selected is None else list(already_selected)
    added = []

    # Best registration distance first; spacing then forces geographic diversity.
    pool = pool.sort_values(["distance_m", "match_id"], ascending=True)

    for row in pool.itertuples(index=False):
        if far_enough(row, selected, MIN_SELECTION_SPACING_M):
            selected.append(row)
            added.append(row)
            if len(added) == n:
                break

    return added


def select_candidates(df):
    """Choose a compact set of colocation cases for visual review."""
    df = df.copy()
    df["view_class"] = df["look_difference_deg"].map(view_class)

    valid = (
        (df["distance_m"] <= MAX_VISUAL_DISTANCE_M)
        & (df["a_ground_range_m"] >= MIN_CENTER_GROUND_RANGE_M)
        & (df["b_ground_range_m"] >= MIN_CENTER_GROUND_RANGE_M)
        & (df["a_canonical_sample_index"] >= PATCH_COLS // 2)
        & (df["b_canonical_sample_index"] >= PATCH_COLS // 2)
        & (df["a_canonical_sample_index"] < 1200 - PATCH_COLS // 2)
        & (df["b_canonical_sample_index"] < 1200 - PATCH_COLS // 2)
        & bool_mask(df["both_high_confidence"])
    )

    pool = df.loc[valid].copy()
    selected = []

    for category in ("similar-view", "oblique-view", "opposite-view"):
        category_pool = pool[pool["view_class"] == category]
        selected.extend(greedy_select(category_pool, N_PER_VIEW_CLASS, selected))

    target_total = 3 * N_PER_VIEW_CLASS
    if len(selected) < target_total:
        selected_ids = {int(r.match_id) for r in selected}
        remainder = pool[~pool["match_id"].isin(selected_ids)]
        selected.extend(greedy_select(remainder, target_total - len(selected), selected))

    if not selected:
        raise RuntimeError("Nessun candidato valido per il visual sanity-check.")

    selected_df = pd.DataFrame([r._asdict() for r in selected])
    selected_df.insert(0, "visual_rank", np.arange(1, len(selected_df) + 1))
    return selected_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def pair_patch(row, survey_a, survey_b):
    """Extract corresponding raw sonar neighborhoods from both surveys."""
    image_a = canonical_sonar(survey_a, row.a_side)
    image_b = canonical_sonar(survey_b, row.b_side)

    patch_a = extract_centered_patch(
        image_a, row.a_row_index, row.a_canonical_sample_index
    )
    patch_b = extract_centered_patch(
        image_b, row.b_row_index, row.b_canonical_sample_index
    )

    return normalize_pair_for_display(patch_a, patch_b)


def save_gallery(selected, survey_name_a, survey_name_b, survey_a, survey_b, aligned=False):
    """Save a labeled visual gallery for manual inspection."""
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(11, max(3.0 * n, 5.0)), squeeze=False)

    for i, row in enumerate(selected.itertuples(index=False)):
        patch_a, patch_b = pair_patch(row, survey_a, survey_b)
        alignment_label = "native"

        if aligned:
            patch_b, alignment_label = display_align_second_patch(
                patch_b, row.heading_difference_deg, row.look_difference_deg
            )

        axes[i, 0].imshow(patch_a, cmap="gray", aspect="auto", interpolation="nearest")
        axes[i, 1].imshow(patch_b, cmap="gray", aspect="auto", interpolation="nearest")

        for ax in axes[i]:
            ax.axhline(PATCH_ROWS / 2, linewidth=0.6, alpha=0.6)
            ax.axvline(PATCH_COLS / 2, linewidth=0.6, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[i, 0].set_title(
            f"{survey_name_a} | L{row.a_leg_number} | {row.a_side.upper()} | "
            f"row {row.a_row_index} | sample {row.a_canonical_sample_index}",
            fontsize=9,
        )
        suffix = f" | display flips: {alignment_label}" if aligned else ""
        axes[i, 1].set_title(
            f"{survey_name_b} | L{row.b_leg_number} | {row.b_side.upper()} | "
            f"row {row.b_row_index} | sample {row.b_canonical_sample_index}{suffix}",
            fontsize=9,
        )

        axes[i, 0].set_ylabel(
            f"{row.view_class}\n"
            f"d={row.distance_m:.2f} m\n"
            f"ΔH={row.heading_difference_deg:.0f}°  ΔL={row.look_difference_deg:.0f}°\n"
            f"Δinc={row.incidence_difference_deg:.0f}°",
            fontsize=8,
        )

    mode = "display-aligned" if aligned else "native orientation"
    fig.suptitle(
        f"Visual co-location sanity check — {survey_name_a} vs {survey_name_b} — {mode}\n"
        f"{PATCH_ROWS}×{PATCH_COLS} sonar pixels, joint log/percentile display normalization",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    suffix = "aligned" if aligned else "native"
    output_path = FIGURE_DIR / f"18_visual_sanity_{survey_name_a}_{survey_name_b}_{suffix}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

print("=" * 90)
print("BLUEBOAT — VISUAL SANITY CHECK OF CO-LOCATED PATCH CANDIDATES")
print("=" * 90)
print(f"Patch size                   : {PATCH_ROWS} x {PATCH_COLS} sonar pixels")
print(f"Max candidate distance       : {MAX_VISUAL_DISTANCE_M:.2f} m")
print(f"Min centre ground range      : {MIN_CENTER_GROUND_RANGE_M:.2f} m")
print(f"Examples per view class      : {N_PER_VIEW_CLASS}")
print(f"Min spatial selection spacing: {MIN_SELECTION_SPACING_M:.1f} m")
print("NOTE: display-aligned figures only flip image axes for easier inspection; raw data are unchanged.")

all_selected = []

for survey_name_a, survey_name_b in PAIRS:
    print("\n" + "=" * 90)
    print(f"{survey_name_a.upper()} vs {survey_name_b.upper()}")
    print("=" * 90)

    csv_path = candidate_path(survey_name_a, survey_name_b)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Candidates not found: {csv_path}\n"
            "Run first: python -m scripts.17_find_colocated_seabed_candidates"
        )

    candidates = pd.read_csv(csv_path)
    selected = select_candidates(candidates)
    selected.insert(1, "pair", f"{survey_name_a}-{survey_name_b}")
    all_selected.append(selected)

    print(f"Total candidates             : {len(candidates)}")
    print(f"Selected pairs           : {len(selected)}")
    print("Selected classes:")
    print(selected["view_class"].value_counts().to_string())
    print(f"Selected median distance : {selected['distance_m'].median():.3f} m")

    print(f"Loading {survey_name_a}...")
    survey_a = read_xtf(XTF_DIR / SURVEYS[survey_name_a])
    print(f"Loading {survey_name_b}...")
    survey_b = read_xtf(XTF_DIR / SURVEYS[survey_name_b])

    native_path = save_gallery(
        selected, survey_name_a, survey_name_b, survey_a, survey_b, aligned=False
    )
    aligned_path = save_gallery(
        selected, survey_name_a, survey_name_b, survey_a, survey_b, aligned=True
    )

    print(f"Native figure               : {native_path}")
    print(f"Display-aligned figure       : {aligned_path}")

    del survey_a, survey_b, candidates
    gc.collect()

selection = pd.concat(all_selected, ignore_index=True)
selection_path = TABLE_DIR / "18_visual_sanity_selection.csv"
selection.to_csv(selection_path, index=False)

print("\n" + "=" * 90)
print("VISUAL SANITY CHECK PREPARATO")
print("=" * 90)
print(f"Selection saved: {selection_path}")
print("Check, in particular, whether persistent structures/textures fall around the central cross in the two patches.")
print("Do not use these pairs as ground truth yet, until the visual check is satisfactory.")
