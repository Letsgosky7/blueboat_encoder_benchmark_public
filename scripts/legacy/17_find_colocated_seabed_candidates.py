"""LEGACY diagnostic — Find cross-pass seabed anchors that are spatially close in the historical geometry.

Invocation: python scripts/legacy/17_find_colocated_seabed_candidates.py, after restoring legacy paths/prerequisites.
Input: 16_seabed_geometry_*.npz, raw survey geometry and anchor sampling configuration.
Output: 17_seabed_anchors_*.npz, 17_colocated_candidates_*.csv and colocation plots.
Not used by the canonical scripts.build_dataset pipeline; standalone execution
may require correcting historical config paths or recovering upstream artifacts.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.patches.extraction import AnchorConfig, anchor_arrays, build_anchors, build_candidate_table, summarize_control_pair, summarize_positive_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SURVEYS = ("control", "pass_a", "pass_b", "pass_c")
POSITIVE_PAIRS = (("pass_a", "pass_b"), ("pass_a", "pass_c"), ("pass_b", "pass_c"))
CONTROL_PAIRS = (("control", "pass_a"), ("control", "pass_b"), ("control", "pass_c"))
CONFIG = AnchorConfig()
PLOT_MAX_POINTS = 15000
PLOT_DISTANCE_MAX_M = 20.0


def geometry_path(name):
    """Resolve the historical per-survey geometry archive."""
    return TABLE_DIR / f"16_seabed_geometry_{name}.npz"


def anchor_path(name):
    """Resolve the historical saved seabed-anchor archive."""
    return TABLE_DIR / f"17_seabed_anchors_{name}.npz"


def candidate_path(a, b):
    """Resolve the earlier colocation-candidate CSV path for a survey pair."""
    return TABLE_DIR / f"17_colocated_candidates_{a}_{b}.csv"


def load_geometry(name):
    """Load the necessary geocoded per-ping geometry arrays."""
    path = geometry_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Geometria non trovata: {path}")
    return np.load(path)


def plot_pair(survey_a, survey_b, anchors_a, anchors_b, candidates, mutual_distances):
    """Draw two sets of anchors and their spatial correspondence candidates."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    def subsample(frame):
        """Reduce a dense diagnostic sample for a readable plot."""
        if len(frame) <= PLOT_MAX_POINTS:
            return frame
        return frame.iloc[::int(np.ceil(len(frame) / PLOT_MAX_POINTS))]

    plot_a, plot_b = subsample(anchors_a), subsample(anchors_b)
    axes[0].scatter(plot_a["east"], plot_a["north"], s=3, alpha=0.35, label=survey_a)
    axes[0].scatter(plot_b["east"], plot_b["north"], s=3, alpha=0.35, label=survey_b)
    high = candidates[candidates["both_high_confidence"]]
    if not high.empty:
        matches = subsample(high)
        axes[0].scatter(matches["midpoint_east"], matches["midpoint_north"], s=5, alpha=0.6, label="candidate midpoints")
    axes[0].set(xlabel="Easting [m]", ylabel="Northing [m]", title=f"Sparse seabed anchors and co-location candidates — {survey_a} vs {survey_b}")
    axes[0].axis("equal"); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].hist(mutual_distances[mutual_distances <= PLOT_DISTANCE_MAX_M], bins=60)
    for threshold in CONFIG.distance_thresholds_m:
        axes[1].axvline(threshold, linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set(xlabel="Mutual-nearest separation [m]", ylabel="Count", title=f"Mutual nearest-neighbour distance — {survey_a} vs {survey_b}")
    axes[1].grid(alpha=0.3)

    if not high.empty:
        axes[2].hist(high["heading_difference_deg"], bins=np.linspace(0.0, 180.0, 37), alpha=0.7, label="vehicle heading Δ")
        axes[2].hist(high["look_difference_deg"], bins=np.linspace(0.0, 180.0, 37), alpha=0.5, label="look azimuth Δ")
    axes[2].set(xlabel="Angular difference [deg]", ylabel="Count", xlim=(0.0, 180.0), title=f"Viewing-geometry distribution of candidates — {survey_a} vs {survey_b}")
    axes[2].grid(alpha=0.3); axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"17_colocation_{survey_a}_{survey_b}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run this historical data-generation or inspection stage."""
    anchors, epsg_values = {}, []

    for name in SURVEYS:
        with load_geometry(name) as geometry:
            anchors[name], epsg = build_anchors(geometry, name, CONFIG)
        np.savez_compressed(anchor_path(name), **anchor_arrays(anchors[name], epsg, CONFIG))
        epsg_values.append(epsg)

    if len(set(epsg_values)) != 1:
        raise RuntimeError("Anchor files do not use the same UTM system.")

    summary_rows = []
    for a, b in POSITIVE_PAIRS:
        candidates, distances = build_candidate_table(a, b, anchors[a], anchors[b], CONFIG)
        candidates.to_csv(candidate_path(a, b), index=False)
        plot_pair(a, b, anchors[a], anchors[b], candidates, distances)
        summary_rows.append(summarize_positive_pair(a, b, anchors[a], anchors[b], candidates, distances, CONFIG))

    for a, b in CONTROL_PAIRS:
        summary_rows.append(summarize_control_pair(a, b, anchors[a], anchors[b], CONFIG))

    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "17_colocation_summary.csv", index=False)
    counts = ", ".join(f"{a}-{b}={len(pd.read_csv(candidate_path(a, b)))}" for a, b in POSITIVE_PAIRS)
    print(f"Co-location candidates: {counts}")
    print(f"Saved: {TABLE_DIR / '17_colocation_summary.csv'}")

    del anchors
    gc.collect()


if __name__ == "__main__":
    main()
