"""LEGACY exploratory script — Visualize raw sonar waterfalls and the recorded sensor trajectory.

Invocation: python scripts/legacy/03_visualize_xtf.py (only if historical paths/configuration are restored).
Input: Raw XTF through the legacy dataset path/configuration.
Output: Historical 03_waterfall_channel_*.png and 03_sensor_trajectory.png diagnostics.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyxtf
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATASET_ROOT = Path(config["dataset_root"])

XTF_PATH = (
    DATASET_ROOT
    / "SSS-XTF"
    / "2026-08-04-11-35.xtf"
)


# ============================================================
# XTF READING
# ============================================================

print("=" * 70)
print("BLUEBOAT XTF VISUALIZATION")
print("=" * 70)

print(f"\nFile: {XTF_PATH}")

if not XTF_PATH.exists():
    raise FileNotFoundError(
        f"The XTF file does not exist:\n{XTF_PATH}"
    )

file_header, packets = pyxtf.xtf_read(str(XTF_PATH))

sonar_packets = packets.get(
    pyxtf.XTFHeaderType.sonar,
    []
)

if not sonar_packets:
    raise RuntimeError("No sonar package found.")

print(f"Number of pings: {len(sonar_packets)}")


# ============================================================
# EXTRACTION OF THE TWO CHANNELS
# ============================================================

channel_0_rows = []
channel_1_rows = []

x_coordinates = []
y_coordinates = []
headings = []


for ping in sonar_packets:

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    x_coordinates.append(
        ping.SensorXcoordinate
    )

    y_coordinates.append(
        ping.SensorYcoordinate
    )

    headings.append(
        ping.SensorHeading
    )

    # --------------------------------------------------------
    # Sonar samples
    # --------------------------------------------------------

    if len(ping.data) >= 2:

        channel_0 = np.asarray(
            ping.data[0],
            dtype=np.float32
        )

        channel_1 = np.asarray(
            ping.data[1],
            dtype=np.float32
        )

        channel_0_rows.append(channel_0)
        channel_1_rows.append(channel_1)


# ============================================================
# DIMENSIONAL CHECK
# ============================================================

if not channel_0_rows or not channel_1_rows:
    raise RuntimeError(
        "No valid data were found for either channel."
    )

min_samples_ch0 = min(
    len(row)
    for row in channel_0_rows
)

min_samples_ch1 = min(
    len(row)
    for row in channel_1_rows
)

min_samples = min(
    min_samples_ch0,
    min_samples_ch1
)

print(f"Minimum number of samples per ping: {min_samples}")


# ============================================================
# CREATION OF WATERFALL MATRICES
#
# row    = ping
# columns  = campioni sonar
# ============================================================

channel_0_matrix = np.stack(
    [
        row[:min_samples]
        for row in channel_0_rows
    ]
)

channel_1_matrix = np.stack(
    [
        row[:min_samples]
        for row in channel_1_rows
    ]
)

print(f"Channel 0 matrix: {channel_0_matrix.shape}")
print(f"Channel 1 matrix: {channel_1_matrix.shape}")


# ============================================================
# CONVERTING METADATA TO A NUMPY ARRAY
# ============================================================

x_coordinates = np.asarray(
    x_coordinates,
    dtype=np.float64
)

y_coordinates = np.asarray(
    y_coordinates,
    dtype=np.float64
)

headings = np.asarray(
    headings,
    dtype=np.float64
)


# ============================================================
# CONTRAST FOR VISUALIZATION PURPOSES ONLY
#
# We do not modify the original data.
# We use the 1%–99% percentiles solely as display limits.
# ============================================================

ch0_vmin = np.percentile(
    channel_0_matrix,
    1
)

ch0_vmax = np.percentile(
    channel_0_matrix,
    99
)

ch1_vmin = np.percentile(
    channel_1_matrix,
    1
)

ch1_vmax = np.percentile(
    channel_1_matrix,
    99
)


# ============================================================
# FIGURE 1 — WATERFALL CHANNEL 0
# ============================================================

plt.figure(figsize=(12, 8))

plt.imshow(
    channel_0_matrix,
    cmap="gray",
    aspect="auto",
    vmin=ch0_vmin,
    vmax=ch0_vmax
)

plt.title(
    "Side-Scan Sonar Waterfall — Channel 0"
)

plt.xlabel(
    "Sample index → increasing slant range"
)

plt.ylabel(
    "Ping index → acquisition time"
)

plt.colorbar(
    label="Sonar return intensity"
)

plt.tight_layout()
plt.savefig(
    PROJECT_ROOT / "outputs" / "figures" / "03_waterfall_channel_0.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# FIGURE 2 — WATERFALL CHANNEL 1
# ============================================================

plt.figure(figsize=(12, 8))

plt.imshow(
    channel_1_matrix,
    cmap="gray",
    aspect="auto",
    vmin=ch1_vmin,
    vmax=ch1_vmax
)

plt.title(
    "Side-Scan Sonar Waterfall — Channel 1"
)

plt.xlabel(
    "Sample index → increasing slant range"
)

plt.ylabel(
    "Ping index → acquisition time"
)

plt.colorbar(
    label="Sonar return intensity"
)

plt.tight_layout()
plt.savefig(
    PROJECT_ROOT / "outputs" / "figures" / "03_waterfall_channel_1.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# FIGURE 3 — SENSOR TRAJECTORY
# ============================================================

plt.figure(figsize=(9, 9))

plt.plot(
    x_coordinates,
    y_coordinates,
    linewidth=1
)

plt.scatter(
    x_coordinates[0],
    y_coordinates[0],
    marker="o",
    label="Start"
)

plt.scatter(
    x_coordinates[-1],
    y_coordinates[-1],
    marker="x",
    label="End"
)

plt.xlabel(
    "Sensor X coordinate"
)

plt.ylabel(
    "Sensor Y coordinate"
)

plt.title(
    "Sensor trajectory from XTF navigation data"
)

plt.axis("equal")

plt.legend()

plt.tight_layout()
plt.savefig(
    PROJECT_ROOT / "outputs" / "figures" / "03_sensor_trajectory.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("STATISTICS")
print("=" * 70)

print(
    f"Channel 0 range: "
    f"{channel_0_matrix.min():.0f} - "
    f"{channel_0_matrix.max():.0f}"
)

print(
    f"Channel 1 range: "
    f"{channel_1_matrix.min():.0f} - "
    f"{channel_1_matrix.max():.0f}"
)

print(
    f"X coordinate: "
    f"{x_coordinates.min()} - "
    f"{x_coordinates.max()}"
)

print(
    f"Y coordinate: "
    f"{y_coordinates.min()} - "
    f"{y_coordinates.max()}"
)

print(
    f"Heading: "
    f"{headings.min():.2f}° - "
    f"{headings.max():.2f}°"
)

print("\nView ready.")


# ============================================================
# SHOW THE FIGURES
# ============================================================

plt.show()
