"""LEGACY diagnostic — Inspect automatically detected straight navigation legs for each survey.

Invocation: python scripts/legacy/09_detect_survey_legs.py, with historical paths/config restored.
Input: Legacy XTF configuration and survey-track metadata.
Output: 09_survey_legs.csv and 09_legs_*.png.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.xtf_reader import read_xtf
from src.geometry.survey import detect_survey_legs, project_tracks_to_utm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
with open(PROJECT_ROOT / "configs" / "local.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

XTF_DIR = Path(config["dataset_root"]) / "SSS-XTF"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}


def plot_legs(survey_name, track, legs):
    """Plot candidate straight navigation legs against the full vessel track."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(track["east"].iloc[::20], track["north"].iloc[::20], linewidth=1, alpha=0.35, label="full track")

    for leg_number, leg in enumerate(legs, start=1):
        leg_track = track.iloc[leg["start_row"]:leg["end_row"] + 1]
        ax.plot(leg_track["east"].iloc[::20], leg_track["north"].iloc[::20], linewidth=2, label=f"L{leg_number} ({leg['course_deg']:.0f}°)")
        middle = len(leg_track) // 2
        ax.text(leg_track["east"].iloc[middle], leg_track["north"].iloc[middle], f"L{leg_number}")

    ax.set(xlabel="Easting [m]", ylabel="Northing [m]", title=f"Detected survey legs — {survey_name}")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"09_legs_{survey_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run this legacy diagnostic with its configured survey paths."""
    tracks = {}
    for survey_name, filename in SURVEYS.items():
        survey = read_xtf(XTF_DIR / filename)
        tracks[survey_name] = survey.metadata[["ping_index", "timestamp", "x", "y", "heading_raw"]].copy()
        del survey
        gc.collect()

    tracks, epsg = project_tracks_to_utm(tracks)
    all_legs, rows = {}, []

    for survey_name, track in tracks.items():
        legs, _ = detect_survey_legs(track)
        all_legs[survey_name] = legs
        for leg_number, leg in enumerate(legs, start=1):
            rows.append({"survey": survey_name, "filename": SURVEYS[survey_name], "leg_number": leg_number, "leg_id": f"{survey_name.upper()}_L{leg_number}", "utm_epsg": epsg, **leg})
        plot_legs(survey_name, track, legs)

    pd.DataFrame(rows).to_csv(TABLE_DIR / "09_survey_legs.csv", index=False)
    counts = ", ".join(f"{name}={len(legs)}" for name, legs in all_legs.items())
    print(f"Survey legs: {counts}")
    print(f"Saved: {TABLE_DIR / '09_survey_legs.csv'}")


if __name__ == "__main__":
    main()
