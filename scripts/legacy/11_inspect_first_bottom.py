"""LEGACY diagnostic — Visually inspect candidate first-bottom transitions in raw sonar pings.

Invocation: python scripts/legacy/11_inspect_first_bottom.py, with historical paths/config restored.
Input: Original XTF waterfalls and range axes via legacy local config.
Output: 11_first_bottom_*.png quality-inspection figures.
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


# ============================================================
# SURVEY
# ============================================================

SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}


# ============================================================
# DIAGNOSTIC PARAMETERS
# ============================================================

# We analyze four zones distributed along each acquisition.
POSITION_FRACTIONS = [
    0.20,
    0.40,
    0.60,
    0.80,
]

# For each position, we take a small block of pings
# and calculate the median profile.
#
# 100 ping per side = approximately +/- 5 seconds at 20 Hz.
PING_HALF_WINDOW = 100

# The first bottom should be located in the area near the nadir.
# We do not search across the entire 23–25 m range.
SEARCH_MIN_RANGE_M = 0.5
SEARCH_MAX_RANGE_M = 10.0

# Smoothing of the profile before calculating the gradient.
SMOOTH_SIGMA = 3.0


# ============================================================
# FIRST-BOTTOM CANDIDATE
# ============================================================

def estimate_candidate(
    profile,
    range_axis,
):
    """
    DIAGNOSTIC estimation of the first-bottom return. 

    Locate the point with the greatest increase
    in logarithmic intensity within the near-range zone. 

    This is not yet the final algorithm.
    """

    log_profile = np.log1p(
        profile.astype(np.float64)
    )

    smooth_profile = gaussian_filter1d(
        log_profile,
        sigma=SMOOTH_SIGMA,
    )

    gradient = np.gradient(
        smooth_profile,
        range_axis,
    )

    valid = (
        (range_axis >= SEARCH_MIN_RANGE_M)
        &
        (range_axis <= SEARCH_MAX_RANGE_M)
    )

    valid_indices = np.where(
        valid
    )[0]

    if len(valid_indices) == 0:
        raise RuntimeError(
            "Invalid first-bottom search range."
        )

    best_local_index = np.argmax(
        gradient[valid_indices]
    )

    best_index = valid_indices[
        best_local_index
    ]

    candidate_range = float(
        range_axis[best_index]
    )

    return (
        candidate_range,
        smooth_profile,
        gradient,
    )


# ============================================================
# ANALYSIS OF A SINGLE SURVEY
# ============================================================

def inspect_survey(
    survey_name,
    filename,
):
    """Inspect per-survey sonar geometry and save a review figure."""

    print("\n" + "=" * 90)
    print(
        f"{survey_name.upper()} — {filename}"
    )
    print("=" * 90)

    survey = read_xtf(
        XTF_DIR / filename
    )

    # --------------------------------------------------------
    #IMPORTANT:
    #
    # From the previous check, we know that:
    #
    # PORT raw:
    #     sample 0      = far range
    #     sample N - 1  = near / nadir
    #
    # STARBOARD raw:
    #     sample 0      = near / nadir
    #     sample N - 1  = far range
    #
    # For range analysis, we standardize both so that:
    #
    #     sample 0 = near
    #     sample N = far
    #
    # --------------------------------------------------------

    port_canonical = np.flip(
        survey.port,
        axis=1,
    )

    starboard_canonical = (
        survey.starboard
    )

    n_pings = (
        port_canonical.shape[0]
    )

    n_samples = (
        port_canonical.shape[1]
    )

    port_range = float(
        survey.slant_range_port
    )

    starboard_range = float(
        survey.slant_range_starboard
    )

    port_range_axis = np.linspace(
        0.0,
        port_range,
        n_samples,
    )

    starboard_range_axis = np.linspace(
        0.0,
        starboard_range,
        n_samples,
    )

    print(
        f"Ping                : {n_pings}"
    )

    print(
        f"Samples per side    : {n_samples}"
    )

    print(
        f"PORT slant range    : {port_range:.3f} m"
    )

    print(
        f"STARBOARD range     : {starboard_range:.3f} m"
    )

    print(
        f"PORT sample spacing : "
        f"{port_range / n_samples:.4f} m"
    )

    print(
        f"STBD sample spacing : "
        f"{starboard_range / n_samples:.4f} m"
    )


    # ========================================================
    # PROFILE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        len(POSITION_FRACTIONS),
        2,
        figsize=(14, 12),
    )

    altitude_candidates = []


    for row_index, fraction in enumerate(
        POSITION_FRACTIONS
    ):

        center_ping = int(
            fraction
            * (n_pings - 1)
        )

        start_ping = max(
            0,
            center_ping - PING_HALF_WINDOW,
        )

        end_ping = min(
            n_pings,
            center_ping + PING_HALF_WINDOW + 1,
        )

        # ----------------------------------------------------
        # Median profile over an acquisition period of approximately 10 seconds.
        # ----------------------------------------------------

        port_profile = np.median(
            port_canonical[
                start_ping:end_ping
            ],
            axis=0,
        )

        starboard_profile = np.median(
            starboard_canonical[
                start_ping:end_ping
            ],
            axis=0,
        )


        # ----------------------------------------------------
        # Candidate for the bottom position
        # ----------------------------------------------------

        (
            port_candidate,
            port_smooth,
            _,
        ) = estimate_candidate(
            port_profile,
            port_range_axis,
        )

        (
            stbd_candidate,
            stbd_smooth,
            _,
        ) = estimate_candidate(
            starboard_profile,
            starboard_range_axis,
        )

        mean_candidate = (
            port_candidate
            + stbd_candidate
        ) / 2.0

        altitude_candidates.append(
            mean_candidate
        )


        # ----------------------------------------------------
        # TEXT OUTPUT
        # ----------------------------------------------------

        print(
            f"\nPosition {fraction:.0%}"
        )

        print(
            f"  Ping window      : "
            f"{start_ping} → {end_ping - 1}"
        )

        print(
            f"  PORT candidate   : "
            f"{port_candidate:.3f} m"
        )

        print(
            f"  STBD candidate   : "
            f"{stbd_candidate:.3f} m"
        )

        print(
            f"  Mean candidate   : "
            f"{mean_candidate:.3f} m"
        )


        # ----------------------------------------------------
        # PORT plot
        # ----------------------------------------------------

        ax_port = axes[
            row_index,
            0,
        ]

        ax_port.plot(
            port_range_axis,
            port_smooth,
        )

        ax_port.axvline(
            port_candidate,
            linestyle="--",
        )

        ax_port.set_xlim(
            0,
            SEARCH_MAX_RANGE_M,
        )

        ax_port.set_title(
            f"{survey_name} — PORT — "
            f"{fraction:.0%}"
        )

        ax_port.set_xlabel(
            "Slant range from nadir [m]"
        )

        ax_port.set_ylabel(
            "log(1 + median intensity)"
        )

        ax_port.grid(
            alpha=0.3
        )


        # ----------------------------------------------------
        # STARBOARD plot
        # ----------------------------------------------------

        ax_stbd = axes[
            row_index,
            1,
        ]

        ax_stbd.plot(
            starboard_range_axis,
            stbd_smooth,
        )

        ax_stbd.axvline(
            stbd_candidate,
            linestyle="--",
        )

        ax_stbd.set_xlim(
            0,
            SEARCH_MAX_RANGE_M,
        )

        ax_stbd.set_title(
            f"{survey_name} — STARBOARD — "
            f"{fraction:.0%}"
        )

        ax_stbd.set_xlabel(
            "Slant range from nadir [m]"
        )

        ax_stbd.set_ylabel(
            "log(1 + median intensity)"
        )

        ax_stbd.grid(
            alpha=0.3
        )


    # ========================================================
    # SUMMARY OF PROPOSED ALTITUDE
    # ========================================================

    altitude_candidates = np.asarray(
        altitude_candidates,
        dtype=float,
    )

    print("\n" + "-" * 90)

    print(
        "CANDIDATE SUMMARY "
        "(DIAGNOSTIC ONLY)"
    )

    print(
        f"Median candidate altitude : "
        f"{np.median(altitude_candidates):.3f} m"
    )

    print(
        f"Min candidate altitude    : "
        f"{np.min(altitude_candidates):.3f} m"
    )

    print(
        f"Max candidate altitude    : "
        f"{np.max(altitude_candidates):.3f} m"
    )


    # ========================================================
    # FIGURE SAVING
    # ========================================================

    fig.suptitle(
        f"First-bottom diagnostic — "
        f"{survey_name}",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"11_first_bottom_{survey_name}.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"\nFigure saved:\n"
        f"{output_path}"
    )

    plt.show()

    del survey
    del port_canonical
    del starboard_canonical

    gc.collect()


# ============================================================
# EXECUTION
# ============================================================

print("=" * 90)
print("BLUEBOAT — FIRST-BOTTOM DIAGNOSTIC")
print("=" * 90)


for survey_name, filename in SURVEYS.items():

    inspect_survey(
        survey_name,
        filename,
    )


print("\n" + "=" * 90)
print("DIAGNOSTICS COMPLETED")
print("=" * 90)