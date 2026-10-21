"""Compare cross-survey clusters in saved frozen embeddings using ARI and NMI.

Run from the repository root:
    python -m scripts.cluster_agreement

Optional:
    python -m scripts.cluster_agreement --models resnet50 vit_b16
    python -m scripts.cluster_agreement --max-clusters 3

Uses the original feature vectors, NOT PCA projections. No additional encoder inference;
uses previously computed and cached embeddings.
Writes only under outputs/analysis/final_evaluation/embeddings/clustering/.
"""

import argparse
from pathlib import Path

from src.analysis.cluster_agreement import (
    FROZEN_ENCODERS,
    MAX_CLUSTERS,
    save_cluster_reports,
)


def main(argv=None):
    """Run the isolated partition-stability audit with descriptive metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+", choices=FROZEN_ENCODERS,
                        default=list(FROZEN_ENCODERS))
    parser.add_argument("--max-clusters", type=int, default=MAX_CLUSTERS)
    args = parser.parse_args(argv)

    output, metrics = save_cluster_reports(
        args.root, models=args.models, max_clusters=args.max_clusters
    )
    print("\n=== INDEPENDENT CROSS-SURVEY CLUSTER AGREEMENT ===")
    print("Method: Ward / Euclidean on original L2-normalized frozen embeddings.")
    print("Partitions are compared only between nominally paired IDs.")
    for condition in ("sonar", "mask-only"):
        print(f"\n{condition.upper()} k=2:")
        print(metrics[(metrics["condition"] == condition) & (metrics["k"] == 2)][
            ["model", "survey_pair", "n_pairs", "ari", "nmi", "status"]
        ].round(4).to_string(index=False))
    print(f"\nEvaluated {len(metrics)} settings; "
          f"{(metrics['status'] == 'ok').sum()} with well-defined partitions.")
    print(f"Outputs: {output}")
    print("ARI/NMI measure partition stability, NOT accurate geographic localization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
