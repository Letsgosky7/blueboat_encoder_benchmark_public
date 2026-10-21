"""LEGACY diagnostic — Combine selected survey legs and first-bottom tracks into seabed geometry.

Invocation: python scripts/legacy/16_prepare_seabed_geometry.py, with historical paths/config restored.
Input: Historical 09_survey_legs.csv, 15_first_bottom_track_*.npz and raw XTFs.
Output: 16_seabed_geometry_*.npz, 16_geometry_legs_used.csv and geometry figures.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.xtf_reader import read_xtf
from src.geometry.seabed import prepare_seabed_geometry


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
    row_index = data["row_index"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    axes[0].plot(row_index, data["first_bottom_port"], label="PORT first-bottom")
    axes[0].plot(row_index, data["first_bottom_stbd"], label="STBD first-bottom")
    axes[0].set(xlabel="Metadata row index", ylabel="First-bottom slant range [m]", title=f"First-bottom restricted to analysis legs — {survey_name}")
    axes[0].grid(alpha=0.3); axes[0].legend()

    s = slice(None, None, 20)
    axes[1].plot(data["east"][s], data["north"][s], linewidth=1, alpha=0.25, label="sensor track")
    axes[1].plot(np.where(data["in_analysis_leg"], data["east"], np.nan)[s], np.where(data["in_analysis_leg"], data["north"], np.nan)[s], linewidth=1.5, label="analysis legs")
    axes[1].plot(data["port_far_east"][s], data["port_far_north"][s], linewidth=1, label="PORT far edge")
    axes[1].plot(data["stbd_far_east"][s], data["stbd_far_north"][s], linewidth=1, label="STBD far edge")
    axes[1].set(xlabel="Easting [m]", ylabel="Northing [m]", title=f"Approximate corrected swath geometry — {survey_name}")
    axes[1].axis("equal"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"16_seabed_geometry_{survey_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run this legacy diagnostic with its configured survey paths."""
    legs_path = TABLE_DIR / "09_survey_legs.csv"
    if not legs_path.exists():
        raise FileNotFoundError(f"File legs non trovato: {legs_path}")
    legs = pd.read_csv(legs_path)
    required = {"survey", "leg_number", "leg_id", "start_row", "end_row", "course_deg", "line_orientation_deg", "utm_epsg"}
    missing = required - set(legs.columns)
    if missing:
        raise RuntimeError("09_survey_legs.csv non contiene: " + ", ".join(sorted(missing)))

    epsg_values = legs["utm_epsg"].dropna().astype(int).unique()
    if len(epsg_values) != 1:
        raise RuntimeError("09_survey_legs.csv deve usare un unico EPSG UTM.")
    utm_epsg = int(epsg_values[0])
    used_rows = []

    for survey_name, filename in SURVEYS.items():
        track_path = TABLE_DIR / f"15_first_bottom_track_{survey_name}.npz"
        if not track_path.exists():
            raise FileNotFoundError(f"Tracking non trovato: {track_path}")
        survey = read_xtf(XTF_DIR / filename)
        with np.load(track_path) as tracking:
            data, survey_used_rows = prepare_seabed_geometry(survey, survey_name, legs, tracking, utm_epsg)
        data["filename"] = np.asarray(filename)
        np.savez_compressed(TABLE_DIR / f"16_seabed_geometry_{survey_name}.npz", **data)
        used_rows.extend(survey_used_rows)
        save_figure(survey_name, data)
        score_port_valid, score_stbd_valid = np.isfinite(data["first_bottom_score_port"]), np.isfinite(data["first_bottom_score_stbd"])
        print(f"{survey_name}: rows={np.count_nonzero(data['in_analysis_leg'])}, port={np.nanmedian(data['first_bottom_port']):.3f} m, stbd={np.nanmedian(data['first_bottom_stbd']):.3f} m, low-score={np.mean(data['low_score_port'][score_port_valid]):.3f}/{np.mean(data['low_score_stbd'][score_stbd_valid]):.3f}")
        del survey, data
        gc.collect()

    pd.DataFrame(used_rows).to_csv(TABLE_DIR / "16_geometry_legs_used.csv", index=False)
    print(f"Saved: {TABLE_DIR / '16_geometry_legs_used.csv'}")


if __name__ == "__main__":
    main()
