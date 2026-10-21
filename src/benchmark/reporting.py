"""Shared matched-negative analysis and closed-set retrieval reporting.

Input: model-specific score CSV / embedding NPZ and common dataset metadata.
Output: margins CSV, similarity histogram, query/summary/Top-3 CSVs.
Imported by the benchmark CLI; no standalone command.
All feature extraction happens upstream; this module never runs model inference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import CONDITIONS, GROUPS
from src.analysis.retrieval import evaluate_positive_retrieval, rank_geographic_candidates


def build_matched_report(scores, null_metadata, quality):
    """Return one positive-pair row per condition with matched-null statistics."""
    positives = scores.loc[scores["group"] == "positive", [
        "condition", "survey_pair", "pair_id", "cosine",
    ]].rename(columns={"cosine": "positive_cosine"}).copy()
    negatives = scores.loc[scores["group"] == "within-site-noncolocated", [
        "condition", "survey_pair", "pair_id", "cosine",
    ]].rename(columns={"cosine": "negative_cosine"}).copy()

    metadata = null_metadata[[
        "null_pair_id", "survey_a", "survey_b", "matched_positive_patch_id",
    ]].copy()
    metadata["survey_pair"] = metadata["survey_a"] + "-" + metadata["survey_b"]
    negatives = negatives.merge(
        metadata[["null_pair_id", "survey_pair", "matched_positive_patch_id"]],
        left_on=["pair_id", "survey_pair"],
        right_on=["null_pair_id", "survey_pair"],
        how="left",
        validate="many_to_one",
    )
    if negatives["matched_positive_patch_id"].isna().any():
        raise ValueError("Some within-site negatives lack matching metadata.")

    comparisons = positives.merge(
        negatives[[
            "condition", "survey_pair", "matched_positive_patch_id",
            "negative_cosine",
        ]],
        left_on=["condition", "survey_pair", "pair_id"],
        right_on=["condition", "survey_pair", "matched_positive_patch_id"],
        how="left",
        validate="one_to_many",
    )
    if comparisons["negative_cosine"].isna().any():
        raise ValueError("Some positives have no geometry-matched negatives.")

    # Equal cosines receive half a win rather than an arbitrary binary label.
    comparisons["win_credit"] = (
        (comparisons["positive_cosine"] > comparisons["negative_cosine"]).astype(float)
        + 0.5 * (
            comparisons["positive_cosine"] == comparisons["negative_cosine"]
        ).astype(float)
    )
    keys = ["condition", "survey_pair", "pair_id"]
    match_stats = comparisons.groupby(keys).agg(
        matched_null_count=("negative_cosine", "size"),
        matched_null_mean=("negative_cosine", "mean"),
        win_fraction=("win_credit", "mean"),
    ).reset_index()
    result = positives.merge(match_stats, on=keys, validate="one_to_one")
    # Pairwise margins use the negatives explicitly matched to each positive.
    result["margin"] = result["positive_cosine"] - result["matched_null_mean"]

    flags = quality[["patch_id", "pair", "requires_review"]].copy()
    flags = flags.rename(columns={"patch_id": "pair_id", "pair": "survey_pair"})
    if flags.duplicated(["survey_pair", "pair_id"]).any():
        raise ValueError("Duplicate quality rows for a positive pair.")
    result = result.merge(
        flags, on=["survey_pair", "pair_id"], how="left", validate="many_to_one"
    )
    if result["requires_review"].isna().any():
        raise ValueError("Missing quality flags for some positive pairs.")
    return result.sort_values(["condition", "survey_pair", "pair_id"])


def plot_distributions(scores, destination):
    """Render positive and negative cosine distributions for both input conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.linspace(-1, 1.05, 42)
    for ax, condition in zip(axes, CONDITIONS):
        for group in GROUPS:
            values = scores.loc[
                (scores["condition"] == condition) & (scores["group"] == group),
                "cosine",
            ].to_numpy()
            if len(values):
                ax.hist(
                    values, bins=bins,
                    weights=np.ones(len(values)) / len(values),
                    histtype="step", linewidth=2,
                    label=f"{group} (n={len(values)})",
                )
        ax.set_title(condition.upper())
        ax.set_xlabel("Cosine similarity")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fraction of pairs per bin")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)

def analyze_results(root, model_name):
    """Generate shared margins and cosine plots without invoking an encoder."""
    output = root / "outputs/embeddings"
    scores = pd.read_csv(output / f"{model_name}_sonar_mask_pair_scores.csv")
    nulls = pd.read_csv(root / "outputs/tables/null_pair_metadata.csv")
    quality = pd.read_csv(root / "outputs/tables/positive_pair_quality.csv")

    # Retain ResNet50's original data-integrity checks, now for every model.
    required = {"group", "condition", "survey_pair", "pair_id", "cosine"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Missing score columns: {sorted(required - set(scores))}")
    if not np.isfinite(scores["cosine"]).all():
        raise ValueError("Cosine scores must be finite.")
    if scores.duplicated(["group", "condition", "survey_pair", "pair_id"]).any():
        raise ValueError("Duplicate score rows found.")

    matched = build_matched_report(scores, nulls, quality)
    matched_file = output / f"{model_name}_matched_null_margins.csv"
    matched.to_csv(matched_file, index=False)

    print("\n=== COSINE SIMILARITY BY CONDITION AND GROUP ===")
    print(scores.groupby(["condition", "group"])["cosine"].agg(
        ["count", "mean", "median"]
    ).round(4).to_string())
    print("\n=== MATCHED-NULL MARGINS BY SURVEY PAIR ===")
    print(matched.groupby(["condition", "survey_pair"]).agg(
        pairs=("pair_id", "size"),
        mean_margin=("margin", "mean"),
        mean_win_fraction=("win_fraction", "mean"),
    ).round(4).to_string())

    figure = root / "outputs/figures" / f"{model_name}_similarity_distributions.png"
    plot_distributions(scores, figure)
    print(f"Matched-null report: {matched_file}")
    print(f"Similarity figure: {figure}")
    print("Descriptive only: observations are correlated; alignment is unverified.")
    if model_name == "resnet50":
        # Keep legacy ResNet50 inspection artifacts, including flagged positives.
        # These figures are descriptive and do not affect model scores or retrieval.
        from src.analysis.diagnostics import (
            inspect_near_unity_masks,
            plot_sonar_failure_cases,
            plot_positive_pair_review,
        )

        print("\n=== BY QUALITY FLAG (diagnostic, no exclusions) ===")
        by_quality = matched.groupby(["condition", "requires_review"]).agg(
            pairs=("pair_id", "size"),
            mean_margin=("margin", "mean"),
            mean_positive_cosine=("positive_cosine", "mean"),
        )
        print(by_quality.round(4).to_string())

        nearly_identical = scores.loc[
            (scores["condition"] == "mask-only")
            & (scores["group"] != "positive")
            & (scores["cosine"] >= 0.9999),
            ["group", "survey_pair", "pair_id", "cosine"],
        ]
        print("\n=== NEAR-UNITY MASK-ONLY NULL SCORES ===")
        print(f"Count: {len(nearly_identical)}")
        if not nearly_identical.empty:
            print(nearly_identical.to_string(index=False))
        print("Note: nearly identical embeddings do not prove identical masks.")
        mask_table = inspect_near_unity_masks(nearly_identical, nulls, root)
        case_figure = plot_sonar_failure_cases(root, matched)
        review_figures = plot_positive_pair_review(root, matched, quality)
        print(f"Near-unity mask diagnostics: {mask_table}")
        print(f"Patch 9/23 figure: {case_figure}")


def report_retrieval(root, model_name):
    """Produce bidirectional Recall@k and nominal geographic Top-3 tables."""
    output = root / "outputs/embeddings"
    archive = output / f"{model_name}_embeddings.npz"
    queries, summary = evaluate_positive_retrieval(archive)
    # Top-3 geography concerns nominal patch squares, not verified localization.
    candidates = rank_geographic_candidates(
        archive, root / "outputs/tables/positive_pair_metadata.csv", top_k=3
    )

    query_file = output / f"{model_name}_retrieval_queries.csv"
    summary_file = output / f"{model_name}_retrieval_summary.csv"
    candidate_file = output / f"{model_name}_retrieval_candidates.csv"
    queries.to_csv(query_file, index=False)
    summary.to_csv(summary_file, index=False)
    candidates.to_csv(candidate_file, index=False)

    print("\n=== BIDIRECTIONAL RETRIEVAL (TIE-AWARE RECALL@1/@3) ===")
    print(summary.round(4).to_string(index=False))
    print(f"Retrieval summary: {summary_file}")
    print(f"Per-query rankings: {query_file}")
    print(f"Geographic Top-3: {candidate_file}")
    print("Exact-ID retrieval is not verified geographic localization.")
    print("The gallery contains only curated positives, not bank negatives.")
    if model_name == "resnet50":
        # Preserve the historical ResNet50 Top-1 geographic console report.
        # Its deterministic rank-1 tie-break is not the official tie-aware Recall.
        top1 = candidates.loc[candidates["candidate_rank"] == 1]
        geography = top1.groupby(["condition", "survey_pair"]).agg(
            queries=("query_patch_id", "size"),
            exact_top1=("exact_id_match", "mean"),
            center_within_50m=("center_within_50m", "mean"),
            nominal_overlap_ge_25pct=("nominal_overlap_ge_25pct", "mean"),
        )
        print("\n=== TOP-1 GEOGRAPHIC DIAGNOSTICS (BOTH DIRECTIONS) ===")
        print(geography.round(4).to_string())
        print(
            "Nominal overlap uses axis-aligned geographic patch squares, not valid sonar pixels. "
            "Top-1 tie-breaking by patch ID is diagnostic; the official Recall@k is tie-aware."
        )

