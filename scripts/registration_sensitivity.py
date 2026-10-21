"""Controlled finite-crop registration stress test on frozen SONAR patches.

Run from the repository root (the full run reloads pretrained weights and
performs NEW frozen inference for shifts > 0):
    python -m scripts.registration_sensitivity

Quick validation of one encoder and one nonzero displacement:
    python -m scripts.registration_sensitivity --models resnet50 --shifts 0 2

No re-training; the zero-shift baseline comes directly from saved NPZ
embeddings. Results stay under outputs/analysis/final_evaluation/registration/.
This is a padded translation of a *finite* 50m crop, NOT a re-extraction from
a bigger mosaic or a quantitative correction of geographic registration.
"""

import argparse
from pathlib import Path

from src.analysis.registration_sensitivity import (
    FROZEN_ENCODERS, SHIFT_METERS, save_registration_reports
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+", choices=FROZEN_ENCODERS,
                        default=list(FROZEN_ENCODERS))
    parser.add_argument("--shifts", nargs="+", type=int, choices=SHIFT_METERS,
                        default=list(SHIFT_METERS))
    args = parser.parse_args(argv)
    print("Fresh inference is required for each nonzero shift; weights are never trained.")
    print("WARNING: finite-crop shifting replaces lost image edges with NoData.")
    output, summary = save_registration_reports(
        args.root, models=args.models, shifts=args.shifts
    )
    print("\n=== REGISTRATION TRANSLATION SENSITIVITY ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nOutputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
