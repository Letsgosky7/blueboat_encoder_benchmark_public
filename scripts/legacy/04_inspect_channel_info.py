"""LEGACY exploratory script — List the declared XTF channel configurations and acquisition fields.

Invocation: python scripts/legacy/04_inspect_channel_info.py (only if historical paths/configuration are restored).
Input: Legacy configs/local.yaml and a specifically named raw XTF recording.
Output: Console-only channel inspection.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path

import pyxtf
import yaml


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


file_header, packets = pyxtf.xtf_read(str(XTF_PATH))


print("=" * 70)
print("SONAR CHANNEL INFO")
print("=" * 70)


for index, channel_info in enumerate(file_header.sonar_info):

    print(f"\n--- SONAR INFO {index} ---")

    for field in dir(channel_info):

        if field.startswith("_"):
            continue

        try:
            value = getattr(channel_info, field)

            if callable(value):
                continue

            print(f"{field}: {value}")

        except Exception:
            pass


print("\n" + "=" * 70)
print("CHANNEL TYPE LEGEND")
print("=" * 70)

print("0 = Subbottom")
print("1 = Port")
print("2 = Starboard")
print("3 = Bathymetry")