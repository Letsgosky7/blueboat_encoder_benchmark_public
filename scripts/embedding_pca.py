"""Generate PCA plots and descriptive tables from saved frozen BlueBoat embeddings.

Run from repository root:
    python -m scripts.embedding_pca

Optional: python -m scripts.embedding_pca --models resnet50 vit_b16
Requires cached NPZ archives and positive-pair geometry/quality CSVs.
No pretrained inference, clustering, new dataset construction or re-ranking.
"""

import argparse
from pathlib import Path

from src.analysis.embedding_pca import FROZEN_ENCODERS, save_pca_reports


def main(argv=None):
    """Validate and process frozen positive embeddings without changing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+", choices=FROZEN_ENCODERS,
                        default=list(FROZEN_ENCODERS))
    args = parser.parse_args(argv)

    output, summary = save_pca_reports(args.root, models=args.models)
    print("\n=== FROZEN POSITIVE-PAIR PCA ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nGenerated outputs: {output}")
    print("PCA plots describe a 2D projection of nominally paired observations.")
    print("Neither proximity nor separation validates physical seabed correspondence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
