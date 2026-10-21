"""LEGACY exploratory script — Inspect raw XTF structure, headers, packet counts and channel measurements.

Invocation: python scripts/legacy/01_inspect_xtf.py (only if historical paths/configuration are restored).
Input: Raw XTF files plus a legacy configs/local.yaml path.
Output: Terminal inspection only; no new benchmark artifacts.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path

import numpy as np
import pyxtf
import yaml


# ============================================================
# PROJECT PATHWAYS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"


# ============================================================
# READING LOCAL CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATASET_ROOT = Path(config["dataset_root"])

XTF_PATH = (
    DATASET_ROOT
    / "SSS-XTF"
    / "2026-08-04-11-35.xtf"
)


# ============================================================
# ROUTE MONITORING
# ============================================================

print("=" * 70)
print("BLUEBOAT XTF INSPECTION")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Dataset root : {DATASET_ROOT}")
print(f"XTF file     : {XTF_PATH}")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"The dataset folder does not exist:\n{DATASET_ROOT}"
    )

if not XTF_PATH.exists():
    raise FileNotFoundError(
        f"The XTF file does not exist:\n{XTF_PATH}"
    )

print("\nRoutes: OK")


# ============================================================
# READING XTF FILES
# ============================================================

print("\nXTF Reading...")

file_header, packets = pyxtf.xtf_read(str(XTF_PATH))

print("Reading completed.")


# ============================================================
# PACKAGE TYPES
# ============================================================

print("\n" + "=" * 70)
print("PACKAGE TYPES")
print("=" * 70)

for packet_type, packet_list in packets.items():
    print(f"{packet_type}: {len(packet_list)} pacchetti")


# ============================================================
# SONAR PACKAGES
# ============================================================

sonar_packets = packets.get(
    pyxtf.XTFHeaderType.sonar,
    []
)

print("\n" + "=" * 70)
print("SONAR")
print("=" * 70)

print(f"Total number of sonar pings: {len(sonar_packets)}")

if not sonar_packets:
    raise RuntimeError("No sonar package found.")


# ============================================================
# PRIMO PING
# ============================================================

ping = sonar_packets[0]

print("\n" + "=" * 70)
print("FIRST PING — AVAILABLE COURTS")
print("=" * 70)

for field in dir(ping):

    if field.startswith("_"):
        continue

    try:
        value = getattr(ping, field)

        if callable(value):
            continue

        # Evitiamo di stampare direttamente tutti i campioni sonar
        if field == "data":
            print(f"{field}: <sonar sample data>")
        else:
            print(f"{field}: {value}")

    except Exception as error:
        print(f"{field}: <reading error: {error}>")


# ============================================================
# DATI DEI CANALI
# ============================================================

print("\n" + "=" * 70)
print("FIRST PING CHANNELS")
print("=" * 70)

print(f"Number of channels present: {len(ping.data)}")

for channel_index, channel_data in enumerate(ping.data):

    samples = np.asarray(channel_data)

    print(f"\nChannel {channel_index}")
    print(f"  Number of samples : {samples.size}")
    print(f"  Data type       : {samples.dtype}")

    if samples.size > 0:
        print(f"  Min             : {samples.min()}")
        print(f"  Max             : {samples.max()}")
        print(f"  Mean           : {samples.mean():.2f}")


print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)