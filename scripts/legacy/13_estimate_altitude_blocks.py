"""LEGACY diagnostic — Estimate altitude change candidates using contiguous blocks of sonar pings.

Invocation: python scripts/legacy/13_estimate_altitude_blocks.py, with historical paths/config restored.
Input: Raw XTF samples and range axes via legacy local config.
Output: 13_altitude_candidates_*.png candidate-profile diagnostics.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from src.data.xtf_reader import read_xtf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATASET_ROOT = Path(config["dataset_root"])
XTF_DIR = DATASET_ROOT / "SSS-XTF"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}


# ============================================================
# PARAMETERS
# ============================================================

# 200 ping ≈ 10 seconds.
BLOCK_SIZE = 200

# We advance by 100 ping ≈ 5 seconds.
BLOCK_STEP = 100

SEARCH_MIN_RANGE_M = 0.4
SEARCH_MAX_RANGE_M = 10.0

SMOOTH_SIGMA = 3.0

# To be able to divide the profile into two meaningful regions,
# we require at least this much width on each side
# of the change point.
MIN_SEGMENT_WIDTH_M = 0.4


# ============================================================
# CHANGE-POINT DETECTOR
# ============================================================

def detect_change_point(
    profile,
    range_axis,
):
    """
    Find the point that best divides the profile into:

    a darker near-range zone
    a brighter far-range zone

    using a two-region segmentation. 

    Returns:
    candidate_range
    confidence
    mean_before
    mean_after
    """

    profile = np.log1p(
        profile.astype(np.float64)
    )

    profile = gaussian_filter1d(
        profile,
        sigma=SMOOTH_SIGMA,
    )

    valid_mask = (
        (range_axis >= SEARCH_MIN_RANGE_M)
        &
        (range_axis <= SEARCH_MAX_RANGE_M)
    )

    valid_indices = np.where(
        valid_mask
    )[0]

    first = int(valid_indices[0])
    last = int(valid_indices[-1])

    spacing = float(
        np.median(
            np.diff(range_axis)
        )
    )

    min_segment_samples = max(
        2,
        int(
            round(
                MIN_SEGMENT_WIDTH_M
                / spacing
            )
        ),
    )

    search_profile = profile[
        first:last + 1
    ]

    n = len(search_profile)

    if n < 2 * min_segment_samples + 1:
        raise RuntimeError(
            "Profile too short for the change-point detector."
        )

    # --------------------------------------------------------
    # SSE with a single level
    # --------------------------------------------------------

    global_mean = float(
        np.mean(search_profile)
    )

    sse_single = float(
        np.sum(
            (
                search_profile
                - global_mean
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # Cumulative sums for quickly calculating
    # the mean and SSE of the two regions.
    # --------------------------------------------------------

    cumulative = np.cumsum(
        search_profile
    )

    cumulative_sq = np.cumsum(
        search_profile ** 2
    )

    best_index = None
    best_sse = np.inf
    best_before = None
    best_after = None

    for split in range(
        min_segment_samples,
        n - min_segment_samples,
    ):

        # Left region [0, split)
        n_left = split

        sum_left = cumulative[
            split - 1
        ]

        sum_sq_left = cumulative_sq[
            split - 1
        ]

        mean_left = (
            sum_left
            / n_left
        )

        sse_left = (
            sum_sq_left
            - (
                sum_left ** 2
                / n_left
            )
        )

        # Right region [split, n)
        n_right = (
            n - split
        )

        sum_right = (
            cumulative[-1]
            - cumulative[split - 1]
        )

        sum_sq_right = (
            cumulative_sq[-1]
            - cumulative_sq[split - 1]
        )

        mean_right = (
            sum_right
            / n_right
        )

        sse_right = (
            sum_sq_right
            - (
                sum_right ** 2
                / n_right
            )
        )

        # On average, we expect:
        #
        # seabed > water column
        #
        # If that doesn’t happen, we don’t consider the candidate.
        if mean_right <= mean_left:
            continue

        total_sse = (
            sse_left
            + sse_right
        )

        if total_sse < best_sse:

            best_sse = total_sse
            best_index = split

            best_before = float(
                mean_left
            )

            best_after = float(
                mean_right
            )

    if best_index is None:

        return (
            np.nan,
            0.0,
            np.nan,
            np.nan,
        )

    # --------------------------------------------------------
    # How much does the two-region model improve upon
    # a single-level model? 
    #
    # 0 = no improvement
    # 1 = perfect separation
    # --------------------------------------------------------

    if sse_single > 0:

        confidence = (
            sse_single
            - best_sse
        ) / sse_single

    else:

        confidence = 0.0

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    global_sample_index = (
        first
        + best_index
    )

    candidate_range = float(
        range_axis[
            global_sample_index
        ]
    )

    return (
        candidate_range,
        confidence,
        best_before,
        best_after,
    )


# ============================================================
# SURVEY ANALYSIS
# ============================================================

def analyse_survey(
    survey_name,
    filename,
):
    """Aggregate candidate bottom transitions over consecutive ping blocks."""

    print("\n" + "=" * 90)
    print(
        f"{survey_name.upper()} — "
        f"{filename}"
    )
    print("=" * 90)

    survey = read_xtf(
        XTF_DIR / filename
    )

    # --------------------------------------------------------
    # Canonical orientation:
    #
    # sample 0 = near / nadir
    # --------------------------------------------------------

    port = np.flip(
        survey.port,
        axis=1,
    )

    starboard = (
        survey.starboard
    )

    n_pings = port.shape[0]
    n_samples = port.shape[1]

    port_range = float(
        survey.slant_range_port
    )

    stbd_range = float(
        survey.slant_range_starboard
    )

    # Sample centres.
    port_axis = (
        (
            np.arange(n_samples)
            + 0.5
        )
        * port_range
        / n_samples
    )

    stbd_axis = (
        (
            np.arange(n_samples)
            + 0.5
        )
        * stbd_range
        / n_samples
    )


    # ========================================================
    # LOOP OVER BLOCKS
    # ========================================================

    block_centres = []

    port_ranges = []
    stbd_ranges = []

    port_confidences = []
    stbd_confidences = []


    for start in range(
        0,
        n_pings - BLOCK_SIZE + 1,
        BLOCK_STEP,
    ):

        end = (
            start
            + BLOCK_SIZE
        )

        centre = (
            start
            + BLOCK_SIZE // 2
        )

        port_profile = np.median(
            port[
                start:end
            ],
            axis=0,
        )

        stbd_profile = np.median(
            starboard[
                start:end
            ],
            axis=0,
        )


        (
            port_candidate,
            port_confidence,
            _,
            _,
        ) = detect_change_point(
            port_profile,
            port_axis,
        )

        (
            stbd_candidate,
            stbd_confidence,
            _,
            _,
        ) = detect_change_point(
            stbd_profile,
            stbd_axis,
        )


        block_centres.append(
            centre
        )

        port_ranges.append(
            port_candidate
        )

        stbd_ranges.append(
            stbd_candidate
        )

        port_confidences.append(
            port_confidence
        )

        stbd_confidences.append(
            stbd_confidence
        )


    # ========================================================
    # ARRAY
    # ========================================================

    block_centres = np.asarray(
        block_centres
    )

    port_ranges = np.asarray(
        port_ranges,
        dtype=float,
    )

    stbd_ranges = np.asarray(
        stbd_ranges,
        dtype=float,
    )

    port_confidences = np.asarray(
        port_confidences,
        dtype=float,
    )

    stbd_confidences = np.asarray(
        stbd_confidences,
        dtype=float,
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    side_difference = np.abs(
        port_ranges
        - stbd_ranges
    )

    print(
        f"Analyzed blocks : "
        f"{len(block_centres)}"
    )

    print(
        f"PORT median range  : "
        f"{np.nanmedian(port_ranges):.3f} m"
    )

    print(
        f"STBD median range  : "
        f"{np.nanmedian(stbd_ranges):.3f} m"
    )

    print(
        f"Median side diff   : "
        f"{np.nanmedian(side_difference):.3f} m"
    )

    print(
        f"PORT median conf.  : "
        f"{np.nanmedian(port_confidences):.3f}"
    )

    print(
        f"STBD median conf.  : "
        f"{np.nanmedian(stbd_confidences):.3f}"
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
    )


    # --------------------------------------------------------
    # First-bottom candidate
    # --------------------------------------------------------

    axes[0].plot(
        block_centres,
        port_ranges,
        label="PORT candidate",
    )

    axes[0].plot(
        block_centres,
        stbd_ranges,
        label="STARBOARD candidate",
    )

    axes[0].set_ylabel(
        "Candidate slant range [m]"
    )

    axes[0].set_title(
        f"First-bottom block candidates — "
        f"{survey_name}"
    )

    axes[0].grid(
        alpha=0.3
    )

    axes[0].legend()


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    axes[1].plot(
        block_centres,
        port_confidences,
        label="PORT confidence",
    )

    axes[1].plot(
        block_centres,
        stbd_confidences,
        label="STARBOARD confidence",
    )

    axes[1].set_xlabel(
        "Ping index"
    )

    axes[1].set_ylabel(
        "Change-point confidence"
    )

    axes[1].set_ylim(
        0,
        1,
    )

    axes[1].grid(
        alpha=0.3
    )

    axes[1].legend()


    fig.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"13_altitude_candidates_{survey_name}.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Figure saved:\n"
        f"{output_path}"
    )

    plt.show()


    del survey
    del port
    del starboard

    gc.collect()


# ============================================================
# EXECUTION
# ============================================================

print("=" * 90)
print("BLUEBOAT — BLOCK FIRST-BOTTOM ESTIMATION")
print("=" * 90)


for survey_name, filename in SURVEYS.items():

    analyse_survey(
        survey_name,
        filename,
    )


print("\n" + "=" * 90)
print("BLOCK-BASED ESTIMATION COMPLETED")
print("=" * 90)