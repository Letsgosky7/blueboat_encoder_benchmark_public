"""Audit curated positive sonar patches for invalid regions and low joint coverage.

Run: python -m scripts.audit_dataset (repository root; no flags).
Input: outputs/patches/positive_pairs_*.npz from scripts.build_dataset.
Output: outputs/tables/positive_pair_quality.csv and diagnostic PNG in outputs/figures/."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.dataset_quality import audit_positive_pair
from src.analysis.dataset_visualization import plot_patch_mask_diagnostics


POSITIVE_PAIRS = (
    ("pass_a", "pass_b"),
    ("pass_a", "pass_c"),
    ("pass_b", "pass_c"),
)


def main():
    """Score positive-pair validity geometry and flag review-worthy cases without exclusions."""
    root = Path(__file__).resolve().parents[1]
    patch_dir = root / "outputs" / "patches"
    table_dir = root / "outputs" / "tables"

    table_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for survey_a, survey_b in POSITIVE_PAIRS:
        path = (
            patch_dir
            / f"positive_pairs_{survey_a}_{survey_b}.npz"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as archive:
            n_pairs = len(archive["patch_id"])

            print(
                f"Analyzing {survey_a} vs {survey_b}: "
                f"{n_pairs} pairs"
            )

            for i in range(n_pairs):
                metrics = audit_positive_pair(
                    archive["patch_a"][i],
                    archive["patch_b"][i],
                    archive["mask_a"][i],
                    archive["mask_b"][i],
                )

                records.append({
                    "patch_id": int(archive["patch_id"][i]),
                    "pair": f"{survey_a}-{survey_b}",
                    "view_class": str(archive["view_class"][i]),
                    "heading_difference_deg": float(
                        archive["heading_difference_deg"][i]
                    ),
                    "look_difference_deg": float(
                        archive["look_difference_deg"][i]
                    ),
                    "incidence_difference_deg": float(
                        archive["incidence_difference_deg"][i]
                    ),
                    **metrics,
                })

    if not records:
        raise RuntimeError("No positive pairs found.")

    # Build quality report
    df = pd.DataFrame(records)

    df["max_nodata_region"] = df[
        ["largest_nodata_a", "largest_nodata_b"]
    ].max(axis=1)

    df["min_central_coverage"] = df[
        ["central_coverage_a", "central_coverage_b"]
    ].min(axis=1)

    # Diagnostic flags: never discard pairs automatically.
    df["low_central_coverage"] = df["central_joint_coverage"] < 0.80
    df["large_nodata_region"] = df["max_nodata_region"] > 0.25
    df["requires_review"] = (
        df["low_central_coverage"] | df["large_nodata_region"]
    )
    df["review_reason"] = df.apply(
        lambda row: ";".join(
            reason
            for flag, reason in (
                (row["low_central_coverage"], "low-central-coverage"),
                (row["large_nodata_region"], "large-nodata-region"),
            )
            if flag
        ),
        axis=1,
    )

    # Save complete CSV
    output = table_dir / "positive_pair_quality.csv"
    df.to_csv(output, index=False)

    # Global summary
    print("\n=== SUMMARY ===")

    summary = df.groupby("pair").agg(
        pairs=("patch_id", "count"),
        median_joint_coverage=("joint_coverage", "median"),
        median_mask_iou=("mask_iou", "median"),
        median_central_joint=(
            "central_joint_coverage", "median"
        ),
        worst_nodata_region=("max_nodata_region", "max"),
    )

    print(summary.round(3).to_string())

    # Worst no-data regions
    print("\n=== 10 PAIRS WITH LARGEST NODATA REGIONS ===")

    worst = df.sort_values(
        ["max_nodata_region", "joint_coverage"],
        ascending=[False, True],
    ).head(10)

    print(
        worst[
            [
                "patch_id",
                "pair",
                "joint_coverage",
                "mask_iou",
                "max_nodata_region",
            ]
        ].round(3).to_string(index=False)
    )

    # Central joint coverage
    print("\n=== CENTRAL JOINT COVERAGE ===")

    central = df.sort_values("central_joint_coverage")

    print(
        central[
            [
                "patch_id",
                "pair",
                "central_joint_coverage",
                "central_mask_iou",
            ]
        ].round(3).to_string(index=False)
    )

    # Central coverage summary
    print("\n=== CENTRAL JOINT COVERAGE BY SURVEY PAIR ===")

    central_summary = df.groupby("pair")[
        "central_joint_coverage"
    ].agg(["min", "median", "max"])

    print(central_summary.round(3).to_string())

    print("\n=== QUALITY REVIEW (diagnostic flags; no exclusions) ===")
    flagged = df.loc[df["requires_review"]]
    print(f"Flagged for review: {len(flagged)} / {len(df)}")
    if not flagged.empty:
        print(
            flagged[
                [
                    "patch_id", "pair", "central_joint_coverage",
                    "max_nodata_region", "review_reason",
                ]
            ].sort_values("patch_id").round(3).to_string(index=False)
        )

    figure = plot_patch_mask_diagnostics(
        patch_dir, root / "outputs" / "figures"
    )
    print(f"Diagnostic figure saved to: {figure}")

    print(f"\nTotal positive pairs analyzed: {len(df)}")
    print(f"CSV saved to: {output}")


if __name__ == "__main__":
    main()
