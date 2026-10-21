"""LEGACY exploratory script — Estimate nearest approach and heading differences between survey trajectories.

Invocation: python scripts/legacy/08_compute_track_overlaps.py (only if historical paths/configuration are restored).
Input: Projected survey tracks from raw XTF files; legacy config.
Output: Console nearest-track comparison; not a validated seabed match.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path
import gc

import numpy as np
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree

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
# METADATA READING
# ============================================================

tracks = {}

print("=" * 80)
print("BLUEBOAT — TRACK OVERLAP ANALYSIS")
print("=" * 80)


for name, filename in SURVEYS.items():

    print(f"\nLettura {name}: {filename}")

    survey = read_xtf(
        XTF_DIR / filename
    )

    tracks[name] = survey.metadata[
        [
            "ping_index",
            "timestamp",
            "x",
            "y",
            "heading_raw",
        ]
    ].copy()

    print(f"  Ping: {len(tracks[name])}")

    del survey
    gc.collect()


# ============================================================
# UTM SYSTEM
# ============================================================

all_lon = np.concatenate(
    [track["x"].to_numpy() for track in tracks.values()]
)

all_lat = np.concatenate(
    [track["y"].to_numpy() for track in tracks.values()]
)

reference_lon = float(np.mean(all_lon))
reference_lat = float(np.mean(all_lat))

utm_zone = int((reference_lon + 180) // 6) + 1

if reference_lat >= 0:
    epsg = 32600 + utm_zone
else:
    epsg = 32700 + utm_zone


transformer = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg}",
    always_xy=True,
)


for track in tracks.values():

    east, north = transformer.transform(
        track["x"].to_numpy(),
        track["y"].to_numpy(),
    )

    track["east"] = east
    track["north"] = north


# ============================================================
# HEADING DIFFERENCE FUNCTION
# ============================================================

def heading_difference(angle_a, angle_b):
    """
    Minimum angular difference between two headings. 

    Result between 0° and 180°.
    """

    difference = abs(angle_a - angle_b) % 360

    return min(
        difference,
        360 - difference
    )


# ============================================================
# CLOSEST APPROACH
# ============================================================

def closest_approach(name_a, name_b):
    """Compute nearest spatial approach and orientation for two survey tracks."""

    track_a = tracks[name_a]
    track_b = tracks[name_b]

    points_a = track_a[
        ["east", "north"]
    ].to_numpy()

    points_b = track_b[
        ["east", "north"]
    ].to_numpy()

    # Let us construct a spatial structure on the points of B.
    tree_b = cKDTree(points_b)

    # For each point in A, we find the nearest point in B.
    distances, indices_b = tree_b.query(
        points_a,
        k=1
    )

    # Point in A that has the absolute minimum distance.
    index_a = int(
        np.argmin(distances)
    )

    index_b = int(
        indices_b[index_a]
    )

    minimum_distance = float(
        distances[index_a]
    )

    row_a = track_a.iloc[index_a]
    row_b = track_b.iloc[index_b]

    delta_heading = heading_difference(
        float(row_a["heading_raw"]),
        float(row_b["heading_raw"]),
    )

    print("\n" + "-" * 80)
    print(f"{name_a} ↔ {name_b}")
    print("-" * 80)

    print(
        f"Closest approach : "
        f"{minimum_distance:.3f} m"
    )

    print(
        f"{name_a} ping       : "
        f"{int(row_a['ping_index'])}"
    )

    print(
        f"{name_b} ping       : "
        f"{int(row_b['ping_index'])}"
    )

    print(
        f"{name_a} heading    : "
        f"{row_a['heading_raw']:.2f}°"
    )

    print(
        f"{name_b} heading    : "
        f"{row_b['heading_raw']:.2f}°"
    )

    print(
        f"Heading difference : "
        f"{delta_heading:.2f}°"
    )

    print(
        f"{name_a} timestamp  : "
        f"{row_a['timestamp']}"
    )

    print(
        f"{name_b} timestamp  : "
        f"{row_b['timestamp']}"
    )


# ============================================================
# MAIN PAIRS
# ============================================================

print("\n" + "=" * 80)
print("CLOSEST APPROACHES")
print("=" * 80)

closest_approach(
    "pass_a",
    "pass_b"
)

closest_approach(
    "pass_a",
    "pass_c"
)

closest_approach(
    "pass_b",
    "pass_c"
)


# ============================================================
# ISOLATED CONTROL
# ============================================================

print("\n" + "=" * 80)
print("CONTROL SURVEY")
print("=" * 80)

closest_approach(
    "control",
    "pass_a"
)

closest_approach(
    "control",
    "pass_b"
)

closest_approach(
    "control",
    "pass_c"
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETED")
print("=" * 80)