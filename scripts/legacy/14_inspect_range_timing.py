"""LEGACY diagnostic — Inspect ping timing and instrument range encoding in XTF headers.

Invocation: python scripts/legacy/14_inspect_range_timing.py, with historical paths/config restored.
Input: Legacy local configuration and the selected raw XTF files.
Output: Console summaries; no canonical patch outputs.
This standalone exploratory stage is superseded by scripts.build_dataset;
check legacy PROJECT_ROOT and config assumptions before using it.
"""

from pathlib import Path
import gc

import pyxtf
import yaml


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATASET_ROOT = Path(config["dataset_root"])
XTF_DIR = DATASET_ROOT / "SSS-XTF"


SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}


# ============================================================
# ANALYSES
# ============================================================

print("=" * 90)
print("BLUEBOAT — XTF RANGE / TIMING INSPECTION")
print("=" * 90)


for survey_name, filename in SURVEYS.items():

    print("\n" + "=" * 90)
    print(f"{survey_name.upper()} — {filename}")
    print("=" * 90)

    path = XTF_DIR / filename

    file_header, packets = pyxtf.xtf_read(
        str(path)
    )

    sonar_packets = packets.get(
        pyxtf.XTFHeaderType.sonar,
        [],
    )

    if not sonar_packets:
        raise RuntimeError(
            f"No sonar ping in{filename}"
        )

    n_pings = len(
        sonar_packets
    )

    indices = [
        0,
        n_pings // 4,
        n_pings // 2,
        (3 * n_pings) // 4,
        n_pings - 1,
    ]

    print(
        f"Numero ping: {n_pings}"
    )


    for ping_index in indices:

        ping = sonar_packets[
            ping_index
        ]

        sound_velocity = float(
            ping.SoundVelocity
        )

        print("\n" + "-" * 90)

        print(
            f"Ping index      : "
            f"{ping_index}"
        )

        print(
            f"Sound velocity  : "
            f"{sound_velocity:.3f} m/s"
        )


        for channel_index, channel_name in [
            (0, "PORT"),
            (1, "STARBOARD"),
        ]:

            header = ping.ping_chan_headers[
                channel_index
            ]

            slant_range = float(
                header.SlantRange
            )

            ground_range = float(
                header.GroundRange
            )

            time_delay = float(
                header.TimeDelay
            )

            time_duration = float(
                header.TimeDuration
            )

            seconds_per_ping = float(
                header.SecondsPerPing
            )

            num_samples = int(
                header.NumSamples
            )

            frequency = int(
                header.Frequency
            )

            # -----------------------------------------------
            # Time-to-range conversions
            # -----------------------------------------------

            start_range_from_time = (
                sound_velocity
                * time_delay
                / 2.0
            )

            range_span_from_time = (
                sound_velocity
                * time_duration
                / 2.0
            )

            end_range_from_time = (
                sound_velocity
                * (
                    time_delay
                    + time_duration
                )
                / 2.0
            )

            if num_samples > 0:

                spacing_from_slant = (
                    slant_range
                    / num_samples
                )

                spacing_from_time = (
                    range_span_from_time
                    / num_samples
                )

            else:

                spacing_from_slant = float("nan")
                spacing_from_time = float("nan")


            print(
                f"\n  {channel_name}"
            )

            print(
                f"    SlantRange       : "
                f"{slant_range:.6f} m"
            )

            print(
                f"    GroundRange      : "
                f"{ground_range:.6f} m"
            )

            print(
                f"    TimeDelay        : "
                f"{time_delay:.9f} s"
            )

            print(
                f"    TimeDuration     : "
                f"{time_duration:.9f} s"
            )

            print(
                f"    SecondsPerPing   : "
                f"{seconds_per_ping:.9f} s"
            )

            print(
                f"    NumSamples       : "
                f"{num_samples}"
            )

            print(
                f"    Frequency        : "
                f"{frequency}"
            )

            print(
                f"    Start range time : "
                f"{start_range_from_time:.6f} m"
            )

            print(
                f"    Range span time  : "
                f"{range_span_from_time:.6f} m"
            )

            print(
                f"    End range time   : "
                f"{end_range_from_time:.6f} m"
            )

            print(
                f"    Δr SlantRange/N  : "
                f"{spacing_from_slant:.6f} m"
            )

            print(
                f"    Δr TimeDuration/N: "
                f"{spacing_from_time:.6f} m"
            )


    del packets
    del sonar_packets
    del file_header

    gc.collect()


print("\n" + "=" * 90)
print("INSPECTION COMPLETED")
print("=" * 90)