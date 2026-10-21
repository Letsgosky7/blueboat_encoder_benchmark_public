"""LEGACY exploratory script — Inspect ping-level timing, navigation and instrument metadata from an XTF recording.

Invocation: python scripts/legacy/02_inspect_metadata.py (only if historical paths/configuration are restored).
Input: Legacy local configuration and a raw XTF file.
Output: Console summaries; does not build the canonical dataset.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path

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

file_header, packets = pyxtf.xtf_read(str(XTF_PATH))

sonar_packets = packets.get(
    pyxtf.XTFHeaderType.sonar,
    []
)

if not sonar_packets:
    raise RuntimeError("No sonar package found.")


# ============================================================
# GENERAL FILE HEADER
# ============================================================

print("=" * 70)
print("FILE HEADER")
print("=" * 70)

for field in dir(file_header):

    if field.startswith("_"):
        continue

    try:
        value = getattr(file_header, field)

        if callable(value):
            continue

        print(f"{field}: {value}")

    except Exception:
        pass


# ============================================================
# HEADER OF THE TWO CHANNELS OF THE FIRST PING
# ============================================================

ping = sonar_packets[0]

print("\n" + "=" * 70)
print("PING CHANNEL HEADERS")
print("=" * 70)

for channel_index, channel_header in enumerate(ping.ping_chan_headers):

    print(f"\n--- CHANNEL {channel_index} ---")

    for field in dir(channel_header):

        if field.startswith("_"):
            continue

        try:
            value = getattr(channel_header, field)

            if callable(value):
                continue

            print(f"{field}: {value}")

        except Exception:
            pass


# ============================================================
# COMPARISON OF SOME PINGS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PINGS")
print("=" * 70)

indices = [
    0,
    len(sonar_packets) // 4,
    len(sonar_packets) // 2,
    3 * len(sonar_packets) // 4,
    len(sonar_packets) - 1,
]

for index in indices:

    ping = sonar_packets[index]

    timestamp = (
        f"{ping.Year:04d}-"
        f"{ping.Month:02d}-"
        f"{ping.Day:02d} "
        f"{ping.Hour:02d}:"
        f"{ping.Minute:02d}:"
        f"{ping.Second:02d}."
        f"{ping.HSeconds:02d}"
    )

    print(f"\nPing index: {index}")
    print(f"  PingNumber : {ping.PingNumber}")
    print(f"  Timestamp  : {timestamp}")
    print(f"  X          : {ping.SensorXcoordinate}")
    print(f"  Y          : {ping.SensorYcoordinate}")
    print(f"  Heading    : {ping.SensorHeading}")
    print(f"  Altitude   : {ping.SensorPrimaryAltitude}")
    print(f"  Sound vel. : {ping.SoundVelocity}")

    for channel_index, channel_header in enumerate(
        ping.ping_chan_headers
    ):
        print(
            f"  Channel {channel_index}: "
            f"{len(ping.data[channel_index])} samples"
        )


print("\n" + "=" * 70)
print("METADATA INSPECTION COMPLETED")
print("=" * 70)