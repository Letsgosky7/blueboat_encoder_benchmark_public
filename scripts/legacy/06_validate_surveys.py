"""LEGACY exploratory script — Check whether available survey files contain consistent side-scan navigation/arrays.

Invocation: python scripts/legacy/06_validate_surveys.py (only if historical paths/configuration are restored).
Input: Four XTF recordings through the legacy configuration.
Output: Console validation summaries.
This is not part of the supported scripts.build_dataset / benchmark pipeline;
older relative-path assumptions may require adjustment before standalone use.
"""

from pathlib import Path
import gc

import yaml

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
# ALL XTF FILES IN THE DATASET
# ============================================================

SURVEYS = {
    "2026-08-04-11-35": {
        "file": "2026-08-04-11-35.xtf",
        "benchmark": False,
    },

    "2026-08-04-11-37": {
        "file": "2026-08-04-11-37.xtf",
        "benchmark": False,
    },

    "2026-08-04-12-59": {
        "file": "2026-08-04-12-59.xtf",
        "benchmark": True,
    },

    "2026-08-04-13-34": {
        "file": "2026-08-04-13-34.xtf",
        "benchmark": False,
    },

    "2026-08-05-07-56": {
        "file": "2026-08-05-07-56.xtf",
        "benchmark": False,
    },

    "2026-08-05-08-00-1": {
        "file": "2026-08-05-08-00-1.xtf",
        "benchmark": True,
    },

    "2026-08-05-09-19": {
        "file": "2026-08-05-09-19.xtf",
        "benchmark": False,
    },

    "2026-08-06-07-50": {
        "file": "2026-08-06-07-50.xtf",
        "benchmark": True,
    },

    "2026-08-06-09-10": {
        "file": "2026-08-06-09-10.xtf",
        "benchmark": True,
    },
}


# ============================================================
# VALIDATION
# ============================================================

print("=" * 90)
print("BLUEBOAT — COMPLETE XTF DATASET VALIDATION")
print("=" * 90)

print(f"\nDataset directory: {XTF_DIR}")
print(f"Number of files to validate: {len(SURVEYS)}")


successful = 0
failed = 0


for survey_name, info in SURVEYS.items():

    filename = info["file"]
    benchmark = info["benchmark"]

    path = XTF_DIR / filename

    print("\n" + "=" * 90)

    if benchmark:
        print(f"{survey_name} — PRIMARY BENCHMARK")
    else:
        print(f"{survey_name} — ADDITIONAL FILE")

    print("=" * 90)

    print(f"File: {filename}")

    # --------------------------------------------------------
    # Existence check
    # --------------------------------------------------------

    if not path.exists():
        print("ERROR: file not found.")
        failed += 1
        continue

    print(
        f"File size: "
        f"{path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    # --------------------------------------------------------
    # Survey analysis
    # --------------------------------------------------------

    try:
        survey = read_xtf(path)

    except Exception as error:

        print("\nERROR DURING READING:")
        print(error)

        failed += 1
        continue

    metadata = survey.metadata

    # --------------------------------------------------------
    # Sonar information
    # --------------------------------------------------------

    print(f"\nNumber of valid pings : {len(metadata)}")

    print(
        f"PORT               : "
        f"{survey.port.shape}, {survey.port.dtype}"
    )

    print(
        f"STARBOARD          : "
        f"{survey.starboard.shape}, {survey.starboard.dtype}"
    )

    print(
        f"Slant range PORT    : "
        f"{survey.slant_range_port} m"
    )

    print(
        f"Slant range STBD    : "
        f"{survey.slant_range_starboard} m"
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    start_time = metadata["timestamp"].iloc[0]
    end_time = metadata["timestamp"].iloc[-1]

    duration = end_time - start_time

    print(f"\nInitial timestamp  : {start_time}")
    print(f"Final timestamp    : {end_time}")
    print(f"Duration              : {duration}")

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    print(
        f"\nX min/max           : "
        f"{metadata['x'].min()} / "
        f"{metadata['x'].max()}"
    )

    print(
        f"Y min/max           : "
        f"{metadata['y'].min()} / "
        f"{metadata['y'].max()}"
    )

    print(
        f"Heading raw min/max : "
        f"{metadata['heading_raw'].min():.2f}° / "
        f"{metadata['heading_raw'].max():.2f}°"
    )

    # --------------------------------------------------------
    # Sonar parameters
    # --------------------------------------------------------

    print(
        f"\nAltitude min/max    : "
        f"{metadata['altitude'].min()} / "
        f"{metadata['altitude'].max()}"
    )

    print(
        f"Sound velocity      : "
        f"{metadata['sound_velocity'].min()} / "
        f"{metadata['sound_velocity'].max()}"
    )

    print(
        f"PORT samples        : "
        f"{metadata['port_num_samples'].min()} / "
        f"{metadata['port_num_samples'].max()}"
    )

    print(
        f"STBD samples        : "
        f"{metadata['starboard_num_samples'].min()} / "
        f"{metadata['starboard_num_samples'].max()}"
    )
    print(
        f"PORT slant range    : "
        f"{metadata['port_slant_range'].min()} / "
        f"{metadata['port_slant_range'].max()}"
    )

    print(
        f"STBD slant range    : "
        f"{metadata['starboard_slant_range'].min()} / "
        f"{metadata['starboard_slant_range'].max()}"
    )


    # --------------------------------------------------------
    # Quick checks
    # --------------------------------------------------------

    print("\nControls:")

    same_shape = (
        survey.port.shape
        == survey.starboard.shape
    )

    print(
        f"  PORT/STBD same size : "
        f"{same_shape}"
    )

    constant_port_samples = (
        metadata["port_num_samples"].nunique() == 1
    )

    constant_stbd_samples = (
        metadata["starboard_num_samples"].nunique() == 1
    )

    print(
        f"  PORT NumSamples costante    : "
        f"{constant_port_samples}"
    )

    print(
        f"  STBD NumSamples costante    : "
        f"{constant_stbd_samples}"
    )

    gps_valid = (
        metadata["x"].notna().all()
        and metadata["y"].notna().all()
    )

    print(
        f"  Coordinates present         : "
        f"{gps_valid}"
    )

    heading_valid = (
        metadata["heading_raw"].notna().all()
    )

    print(
        f"  Heading presente            : "
        f"{heading_valid}"
    )

    print("\nRESULT: OK")

    successful += 1

    # --------------------------------------------------------
    # Free up memory before loading the next XTF
    # --------------------------------------------------------

    del survey
    del metadata

    gc.collect()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"Total files    : {len(SURVEYS)}")
print(f"Validated files  : {successful}")
print(f"Failed files   : {failed}")

if failed == 0:
    print("\nAll XTF files have been read successfully.")
else:
    print(
        "\nSome files have issues. "
        "Check the previous output."
    )

print("\n" + "=" * 90)
print("VALIDATION COMPLETED")
print("=" * 90)