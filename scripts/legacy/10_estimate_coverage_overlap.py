"""LEGACY diagnostic — Compare approximate side-scan swath footprints between acquisitions.

Invocation: python scripts/legacy/10_estimate_coverage_overlap.py, with historical paths/config restored.
Input: Raw XTF navigation and swath-extent assumptions, via legacy configuration.
Output: 10_approximate_coverage_overlap.png; coverage remains approximate.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
from itertools import combinations
import gc

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyproj import Transformer
from shapely.geometry import LineString

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
# PARAMETERS
# ============================================================

# The pings are at ~20 Hz.
#
# For this rough geometric analysis, let’s assume
# approximately one ping per second.
#
# It does NOT modify the original dataset.
TRACK_STRIDE = 20


# ============================================================
# INTERPRETING SURVEYS
# ============================================================

survey_data = {}

all_longitudes = []
all_latitudes = []


print("=" * 90)
print("BLUEBOAT — APPROXIMATE SONAR COVERAGE")
print("=" * 90)


for name, filename in SURVEYS.items():

    print(f"\nReading {name}: {filename}")

    survey = read_xtf(
        XTF_DIR / filename
    )

    metadata = survey.metadata[
        [
            "ping_index",
            "x",
            "y",
            "heading_raw",
            "port_slant_range",
            "starboard_slant_range",
        ]
    ].copy()

    port_range = float(
        metadata[
            "port_slant_range"
        ].median()
    )

    starboard_range = float(
        metadata[
            "starboard_slant_range"
        ].median()
    )

    print(
        f"  Ping             : "
        f"{len(metadata)}"
    )

    print(
        f"  PORT range       : "
        f"{port_range:.3f} m"
    )

    print(
        f"  STARBOARD range  : "
        f"{starboard_range:.3f} m"
    )

    all_longitudes.append(
        metadata["x"].to_numpy()
    )

    all_latitudes.append(
        metadata["y"].to_numpy()
    )

    survey_data[name] = {
        "metadata": metadata,
        "port_range": port_range,
        "starboard_range": starboard_range,
    }

    # We no longer need the sonar arrays.
    del survey
    gc.collect()


# ============================================================
# COMMON UTM SYSTEM
# ============================================================

all_longitudes = np.concatenate(
    all_longitudes
)

all_latitudes = np.concatenate(
    all_latitudes
)

reference_lon = float(
    np.mean(all_longitudes)
)

reference_lat = float(
    np.mean(all_latitudes)
)

utm_zone = int(
    (reference_lon + 180) // 6
) + 1

if reference_lat >= 0:
    epsg = 32600 + utm_zone
else:
    epsg = 32700 + utm_zone


print("\n" + "=" * 90)
print("COORDINATE SYSTEM")
print("=" * 90)

print(
    f"Reference longitude : "
    f"{reference_lon:.6f}"
)

print(
    f"Reference latitude  : "
    f"{reference_lat:.6f}"
)

print(
    f"UTM zone            : "
    f"{utm_zone}"
)

print(
    f"EPSG                : "
    f"{epsg}"
)


transformer = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg}",
    always_xy=True,
)


# ============================================================
# CREATION OF APPROXIMATE TRAJECTORIES AND FOOTPRINTS
# ============================================================

for name, data in survey_data.items():

    metadata = data["metadata"]

    east, north = transformer.transform(
        metadata["x"].to_numpy(),
        metadata["y"].to_numpy(),
    )

    metadata["east"] = east
    metadata["north"] = north

    # --------------------------------------------------------
    # Subsampling for geometry
    # --------------------------------------------------------

    sampled = (
        metadata
        .iloc[::TRACK_STRIDE]
        .copy()
    )

    coordinates = np.column_stack(
        (
            sampled["east"].to_numpy(),
            sampled["north"].to_numpy(),
        )
    )

    centerline = LineString(
        coordinates
    )

    # --------------------------------------------------------
    # APPROXIMATE FOOTPRINT
    #
    # In this dataset, PORT and STARBOARD have the same range.
    #
    # We use the maximum of the two to construct a symmetric band
    # around the trajectory.
    #
    # ATTENTION:
    # The slant range has NOT yet been corrected to ground range. 
    # This is therefore a preliminary estimate / upper bound.
    # --------------------------------------------------------

    half_width = max(
        data["port_range"],
        data["starboard_range"],
    )

    footprint = centerline.buffer(
        half_width
    )

    data["sampled"] = sampled
    data["centerline"] = centerline
    data["footprint"] = footprint
    data["half_width"] = half_width

    print(
        f"\n{name}"
    )

    print(
        f"  Footprint half-width : "
        f"{half_width:.3f} m"
    )

    print(
        f"  Track length         : "
        f"{centerline.length:.1f} m"
    )

    print(
        f"  Approx. covered area : "
        f"{footprint.area:.1f} m²"
    )


# ============================================================
# COMPARISON BETWEEN TWO SURVEYS
# ============================================================

def compare_surveys(
    name_a,
    name_b,
):
    """Approximate swath overlap between two survey navigation tracks."""

    data_a = survey_data[name_a]
    data_b = survey_data[name_b]

    line_a = data_a[
        "centerline"
    ]

    line_b = data_b[
        "centerline"
    ]

    footprint_a = data_a[
        "footprint"
    ]

    footprint_b = data_b[
        "footprint"
    ]

    # --------------------------------------------------------
    # Minimum distance between trajectories
    # --------------------------------------------------------

    track_distance = float(
        line_a.distance(
            line_b
        )
    )

    # --------------------------------------------------------
    # Intersection of the two approximated areas
    # --------------------------------------------------------

    intersection = (
        footprint_a
        .intersection(
            footprint_b
        )
    )

    intersection_area = float(
        intersection.area
    )

    area_a = float(
        footprint_a.area
    )

    area_b = float(
        footprint_b.area
    )

    if area_a > 0:
        fraction_a = (
            intersection_area
            / area_a
        )
    else:
        fraction_a = 0.0

    if area_b > 0:
        fraction_b = (
            intersection_area
            / area_b
        )
    else:
        fraction_b = 0.0

    # --------------------------------------------------------
    # Maximum distance at which two swaths
    # can begin to touch.
    # --------------------------------------------------------

    maximum_reach = (
        data_a["half_width"]
        + data_b["half_width"]
    )

    print("\n" + "-" * 90)
    print(
        f"{name_a} ↔ {name_b}"
    )
    print("-" * 90)

    print(
        f"Closest track distance : "
        f"{track_distance:.3f} m"
    )

    print(
        f"Combined sonar reach   : "
        f"{maximum_reach:.3f} m"
    )

    print(
        f"Intersection area      : "
        f"{intersection_area:.1f} m²"
    )

    print(
        f"{name_a} co-visible approx. : "
        f"{fraction_a * 100:.1f}%"
    )

    print(
        f"{name_b} co-visible approx. : "
        f"{fraction_b * 100:.1f}%"
    )

    if intersection_area > 0:

        print(
            "Possible sonar coverage overlap: YES"
        )

    else:

        print(
            "Possible sonar coverage overlap: NO"
        )

    return {
        "name_a": name_a,
        "name_b": name_b,
        "track_distance": track_distance,
        "intersection_area": intersection_area,
        "fraction_a": fraction_a,
        "fraction_b": fraction_b,
    }


# ============================================================
# ALL COUPLES
# ============================================================

print("\n" + "=" * 90)
print("PAIRWISE COVERAGE OVERLAP")
print("=" * 90)


results = []


for name_a, name_b in combinations(
    SURVEYS.keys(),
    2,
):

    result = compare_surveys(
        name_a,
        name_b,
    )

    results.append(
        result
    )


# ============================================================
# SUMMARY OF THE THREE SCIENTIFIC PAIRS
# ============================================================

print("\n" + "=" * 90)
print("MAIN REPEAT-PASS PAIRS")
print("=" * 90)


main_pairs = [
    ("pass_a", "pass_b"),
    ("pass_a", "pass_c"),
    ("pass_b", "pass_c"),
]


for name_a, name_b in main_pairs:

    result = next(
        result
        for result in results
        if (
            result["name_a"] == name_a
            and result["name_b"] == name_b
        )
        or
        (
            result["name_a"] == name_b
            and result["name_b"] == name_a
        )
    )

    print(
        f"\n{name_a} ↔ {name_b}"
    )

    print(
        f"  overlap area : "
        f"{result['intersection_area']:.1f} m²"
    )

    print(
        f"  fractions    : "
        f"{result['fraction_a'] * 100:.1f}% / "
        f"{result['fraction_b'] * 100:.1f}%"
    )


# ============================================================
# CONTROL CHECK
# ============================================================

print("\n" + "=" * 90)
print("CONTROL CHECK")
print("=" * 90)


for result in results:

    if (
        result["name_a"] == "control"
        or result["name_b"] == "control"
    ):

        print(
            f"{result['name_a']} ↔ "
            f"{result['name_b']}: "
            f"{result['intersection_area']:.1f} m²"
        )


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 10)
)


# ------------------------------------------------------------
# Local origin for the chart ONLY
# ------------------------------------------------------------

all_bounds = [
    data["footprint"].bounds
    for data in survey_data.values()
]

origin_east = min(
    bounds[0]
    for bounds in all_bounds
)

origin_north = min(
    bounds[1]
    for bounds in all_bounds
)


def plot_polygon(
    polygon,
    origin_x,
    origin_y,
    **kwargs,
):
    """
    Draw Polygon or MultiPolygon.
    """

    if polygon.geom_type == "Polygon":

        polygons = [polygon]

    elif polygon.geom_type == "MultiPolygon":

        polygons = list(
            polygon.geoms
        )

    else:

        return

    for poly in polygons:

        x, y = poly.exterior.xy

        plt.fill(
            np.asarray(x) - origin_x,
            np.asarray(y) - origin_y,
            **kwargs,
        )


for name, data in survey_data.items():

    sampled = data[
        "sampled"
    ]

    line, = plt.plot(
        sampled["east"] - origin_east,
        sampled["north"] - origin_north,
        linewidth=1.5,
        label=name,
    )

    plot_polygon(
        data["footprint"],
        origin_east,
        origin_north,
        alpha=0.12,
        color=line.get_color(),
    )


plt.xlabel(
    "East [m]"
)

plt.ylabel(
    "North [m]"
)

plt.title(
    "Approximate side-scan coverage"
)

plt.axis(
    "equal"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


OUTPUT_PATH = (
    OUTPUT_DIR
    / "10_approximate_coverage_overlap.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(
    "\nFigure saved in:"
)

print(
    OUTPUT_PATH
)

plt.show()


print("\n" + "=" * 90)
print("ANALYSIS COMPLETED")
print("=" * 90)