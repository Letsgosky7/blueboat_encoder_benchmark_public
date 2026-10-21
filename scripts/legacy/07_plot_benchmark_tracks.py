"""LEGACY exploratory script — Plot projected vessel tracks from the survey acquisition files.

Invocation: python scripts/legacy/07_plot_benchmark_tracks.py (only if historical paths/configuration are restored).
Input: Raw XTF recordings and historical local configuration.
Output: Historical 07_benchmark_tracks.png figure.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyproj import Transformer

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
# SURVEYS USED IN THE BENCHMARK
# ============================================================

BENCHMARK_SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}


# ============================================================
# LETTURA TRAIETTORIE
# ============================================================

tracks = {}

print("=" * 80)
print("BLUEBOAT — BENCHMARK TRACKS")
print("=" * 80)


for survey_name, filename in BENCHMARK_SURVEYS.items():

    print(f"\nLettura {survey_name}: {filename}")

    survey = read_xtf(
        XTF_DIR / filename
    )

    # Per questo script ci interessano solo i metadata.
    metadata = survey.metadata[
        [
            "ping_index",
            "timestamp",
            "x",
            "y",
            "heading_raw",
        ]
    ].copy()

    tracks[survey_name] = metadata

    print(
        f"  Ping: {len(metadata)}"
    )

    # Sonar arrays are no longer needed in this script.
    del survey
    gc.collect()


# ============================================================
# UNION OF COORDINATES
# ============================================================

all_longitudes = np.concatenate(
    [
        track["x"].to_numpy()
        for track in tracks.values()
    ]
)

all_latitudes = np.concatenate(
    [
        track["y"].to_numpy()
        for track in tracks.values()
    ]
)


# ============================================================
# HEALTH CHECK
# ============================================================

if not np.all(
    (all_longitudes >= -180)
    & (all_longitudes <= 180)
):
    raise ValueError(
        "The X values ​​are not compatible with longitudes."
    )

if not np.all(
    (all_latitudes >= -90)
    & (all_latitudes <= 90)
):
    raise ValueError(
        "The Y values ​​are not compatible with latitudes."
    )


reference_lon = float(
    np.mean(all_longitudes)
)

reference_lat = float(
    np.mean(all_latitudes)
)


# ============================================================
# AUTOMATIC DETERMINATION OF THE UTM ZONE
# ============================================================

utm_zone = int(
    (reference_lon + 180) // 6
) + 1

if reference_lat >= 0:
    epsg_utm = 32600 + utm_zone
else:
    epsg_utm = 32700 + utm_zone


print("\n" + "=" * 80)
print("COORDINATE SYSTEM")
print("=" * 80)

print(
    f"Mean longitude : {reference_lon:.6f}"
)

print(
    f"Mean latitude  : {reference_lat:.6f}"
)

print(
    f"UTM zone          : {utm_zone}"
)

print(
    f"EPSG UTM          : {epsg_utm}"
)


# ============================================================
# LONG/LAT → UTM TRANSFORMATION
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg_utm}",
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
# LOCAL SYSTEM IN METERS
#
# We subtract a common origin solely to make the graph
# more readable. The distances remain unchanged.
# ============================================================

all_east = np.concatenate(
    [
        track["east"].to_numpy()
        for track in tracks.values()
    ]
)

all_north = np.concatenate(
    [
        track["north"].to_numpy()
        for track in tracks.values()
    ]
)

origin_east = float(
    all_east.min()
)

origin_north = float(
    all_north.min()
)


for track in tracks.values():

    track["east_local"] = (
        track["east"] - origin_east
    )

    track["north_local"] = (
        track["north"] - origin_north
    )


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("EXTENSION OF TRAJECTORIES")
print("=" * 80)


for survey_name, track in tracks.items():

    print(f"\n{survey_name}")

    print(
        f"  East  : "
        f"{track['east_local'].min():.2f} → "
        f"{track['east_local'].max():.2f} m"
    )

    print(
        f"  North : "
        f"{track['north_local'].min():.2f} → "
        f"{track['north_local'].max():.2f} m"
    )


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(10, 10)
)

# For the sake of the graph, we show one ping out of every 10.
# All the original data remain available.
PLOT_STRIDE = 10


for survey_name, track in tracks.items():

    plt.plot(
        track["east_local"].iloc[::PLOT_STRIDE],
        track["north_local"].iloc[::PLOT_STRIDE],
        linewidth=1,
        label=survey_name,
    )


plt.xlabel(
    "East [m]"
)

plt.ylabel(
    "North [m]"
)

plt.title(
    "BlueBoat benchmark survey tracks"
)

plt.axis("equal")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "07_benchmark_tracks.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nFigure saved in :\n{OUTPUT_PATH}"
)


plt.show()