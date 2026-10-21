"""LEGACY exploratory script — Smoke-test the project XTF reader on an early example recording.

Invocation: python scripts/legacy/05_test_xtf_reader.py (only if historical paths/configuration are restored).
Input: Legacy configs/local.yaml and an early example XTF file.
Output: Console descriptions of the parsed XTFSurvey; no persistent outputs.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path

import yaml

from src.data.xtf_reader import read_xtf


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


survey = read_xtf(XTF_PATH)


print("=" * 70)
print("XTF SURVEY")
print("=" * 70)

print(f"File: {survey.path.name}")

print(
    f"PORT shape: {survey.port.shape}"
)

print(
    f"STARBOARD shape: {survey.starboard.shape}"
)

print(
    f"PORT dtype: {survey.port.dtype}"
)

print(
    f"STARBOARD dtype: {survey.starboard.dtype}"
)

print(
    f"PORT slant range: "
    f"{survey.slant_range_port} m"
)

print(
    f"STARBOARD slant range: "
    f"{survey.slant_range_starboard} m"
)


print("\n" + "=" * 70)
print("METADATA")
print("=" * 70)

print(survey.metadata.head())


print("\n" + "=" * 70)
print("METADATA INFO")
print("=" * 70)

print(survey.metadata.dtypes)