"""LEGACY diagnostic — Review temporally tracked first-bottom estimates across the surveys.

Invocation: python scripts/legacy/15_track_first_bottom.py, with historical paths/config restored.
Input: Raw XTF pings and the shared src.geometry.seabed tracker.
Output: 15_first_bottom_track_*.npz and matching PNG diagnostic figures.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.xtf_reader import read_xtf
from src.geometry.seabed import estimate_first_bottom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
with open(PROJECT_ROOT / "configs" / "local.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

XTF_DIR = Path(config["dataset_root"]) / "SSS-XTF"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SURVEYS = {
    "control": "2026-08-04-12-59.xtf", "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf", "pass_c": "2026-08-06-09-10.xtf",
}


def save_figure(survey_name, data):
    """Persist the current per-survey first-bottom or geometry diagnostic."""
    x = data["block_centres"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(x, data["raw_port_ranges"], linewidth=0.8, alpha=0.30, label="PORT raw best")
    axes[0].plot(x, data["raw_stbd_ranges"], linewidth=0.8, alpha=0.30, label="STBD raw best")
    axes[0].plot(x, data["tracked_port"], linewidth=2, label="PORT tracked")
    axes[0].plot(x, data["tracked_stbd"], linewidth=2, label="STBD tracked")
    axes[0].set(ylabel="First-bottom slant range [m]", title=f"Dynamic first-bottom tracking — {survey_name}")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(x, data["tracked_port_scores"], label="PORT score")
    axes[1].plot(x, data["tracked_stbd_scores"], label="STBD score")
    axes[1].set(ylabel="Observation score", ylim=(0, 1)); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[2].plot(x, data["side_difference"])
    axes[2].set(xlabel="Metadata row index", ylabel="|PORT - STBD| [m]"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"15_first_bottom_track_{survey_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run this legacy diagnostic with its configured survey paths."""
    for survey_name, filename in SURVEYS.items():
        survey = read_xtf(XTF_DIR / filename)
        data = estimate_first_bottom(survey)
        np.savez_compressed(TABLE_DIR / f"15_first_bottom_track_{survey_name}.npz", **data)
        save_figure(survey_name, data)
        print(f"{survey_name}: blocks={len(data['block_centres'])}, port={np.median(data['tracked_port']):.3f} m, stbd={np.median(data['tracked_stbd']):.3f} m, side-diff={np.median(data['side_difference']):.3f} m")
        del survey, data
        gc.collect()


if __name__ == "__main__":
    main()
