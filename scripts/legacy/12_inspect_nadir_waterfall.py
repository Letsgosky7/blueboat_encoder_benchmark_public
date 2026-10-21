"""LEGACY diagnostic — Inspect the water-column/nadir band in sonar waterfall samples.

Invocation: python scripts/legacy/12_inspect_nadir_waterfall.py, with historical paths/config restored.
Input: Original sonar recordings via legacy local configuration.
Output: 12_nadir_waterfall_*.png figures.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import yaml

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

POSITION_FRACTIONS = [
    0.20,
    0.40,
    0.60,
    0.80,
]

# +/- 200 ping = circa +/- 10 secondi.
PING_HALF_WINDOW = 200

# We are displaying only the first 10 meters from the nadir.
MAX_RANGE_M = 10.0


# ============================================================
# ANALYSES
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

    # Canonicalization:
    #
    # sample 0 = near / nadir
    # sample N = far range

    port = np.flip(
        survey.port,
        axis=1,
    )

    starboard = survey.starboard

    n_pings = port.shape[0]
    n_samples = port.shape[1]

    port_range = float(
        survey.slant_range_port
    )

    stbd_range = float(
        survey.slant_range_starboard
    )

    port_spacing = (
        port_range
        / n_samples
    )

    stbd_spacing = (
        stbd_range
        / n_samples
    )

    port_max_sample = min(
        n_samples,
        int(
            MAX_RANGE_M
            / port_spacing
        ),
    )

    stbd_max_sample = min(
        n_samples,
        int(
            MAX_RANGE_M
            / stbd_spacing
        ),
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        len(POSITION_FRACTIONS),
        2,
        figsize=(14, 12),
    )


    for row_index, fraction in enumerate(
        POSITION_FRACTIONS
    ):

        center_ping = int(
            fraction
            * (n_pings - 1)
        )

        start_ping = max(
            0,
            center_ping
            - PING_HALF_WINDOW,
        )

        end_ping = min(
            n_pings,
            center_ping
            + PING_HALF_WINDOW
            + 1,
        )


        # ----------------------------------------------------
        # PORT
        # ----------------------------------------------------

        port_crop = port[
            start_ping:end_ping,
            :port_max_sample,
        ].astype(
            np.float32
        )

        # VISUAL normalization only.
        port_low = np.percentile(
            port_crop,
            2,
        )

        port_high = np.percentile(
            port_crop,
            98,
        )


        ax = axes[
            row_index,
            0,
        ]

        ax.imshow(
            port_crop,
            aspect="auto",
            cmap="gray",
            vmin=port_low,
            vmax=port_high,
            extent=[
                0,
                MAX_RANGE_M,
                end_ping,
                start_ping,
            ],
        )

        ax.set_title(
            f"{survey_name} — PORT — "
            f"{fraction:.0%}"
        )

        ax.set_xlabel(
            "Slant range from nadir [m]"
        )

        ax.set_ylabel(
            "Ping index"
        )


        # ----------------------------------------------------
        # STARBOARD
        # ----------------------------------------------------

        stbd_crop = starboard[
            start_ping:end_ping,
            :stbd_max_sample,
        ].astype(
            np.float32
        )

        stbd_low = np.percentile(
            stbd_crop,
            2,
        )

        stbd_high = np.percentile(
            stbd_crop,
            98,
        )


        ax = axes[
            row_index,
            1,
        ]

        ax.imshow(
            stbd_crop,
            aspect="auto",
            cmap="gray",
            vmin=stbd_low,
            vmax=stbd_high,
            extent=[
                0,
                MAX_RANGE_M,
                end_ping,
                start_ping,
            ],
        )

        ax.set_title(
            f"{survey_name} — STARBOARD — "
            f"{fraction:.0%}"
        )

        ax.set_xlabel(
            "Slant range from nadir [m]"
        )

        ax.set_ylabel(
            "Ping index"
        )


    fig.suptitle(
        f"Near-nadir waterfall — "
        f"{survey_name}",
        fontsize=14,
    )

    fig.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"12_nadir_waterfall_{survey_name}.png"
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
print("BLUEBOAT — NEAR-NADIR WATERFALL")
print("=" * 90)


for survey_name, filename in SURVEYS.items():

    inspect_survey(
        survey_name,
        filename,
    )


print("\n" + "=" * 90)
print("DIAGNOSTICS COMPLETED")
print("=" * 90)