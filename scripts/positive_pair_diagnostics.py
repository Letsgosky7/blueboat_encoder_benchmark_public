"""Join frozen SONAR retrieval, geometry, and quality audits for curated pairs.

Run from the repository root:
    python -m scripts.positive_pair_diagnostics

Inputs (cached; no XTF reading, inference, or embedding recomputation):
    outputs/tables/positive_pair_metadata.csv
    outputs/tables/positive_pair_quality.csv
    outputs/analysis/frozen_encoders/diagnostics/four_encoder_positive_pair_metrics.csv
    outputs/analysis/frozen_encoders/diagnostics/four_encoder_shared_failure_queries.csv

Outputs (new directory; original benchmark files remain untouched):
    outputs/analysis/positive_pair_diagnostics/positive_pair_diagnostics.csv
    outputs/analysis/positive_pair_diagnostics/quality_group_summary.csv
    outputs/analysis/positive_pair_diagnostics/stratified_retrieval_summary.csv

Retrieval is exact-ID, closed-set, and tie-aware. Nominal co-location does
not verify true seabed correspondence. All percentages are descriptive,
and directional/model observations are correlated.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_comparison import ENCODERS


KEY = ["pair", "patch_id"]
METRICS = [
    "positive_cosine", "margin",
    "recall_at_1_forward", "recall_at_1_reverse",
]


def _load_csv(path, columns, unique_key):
    """Read one CSV and reject missing fields or duplicate identifiers."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic input: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if frame.duplicated(unique_key).any():
        raise ValueError(f"{path.name}: duplicate identifiers in {unique_key}")
    return frame


def _require_same_pairs(left, right, name):
    """Ensure no unmatched pair identity can be silently discarded."""
    left_keys = set(map(tuple, left[KEY].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[KEY].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ValueError(f"{name}: inconsistent positive pair identities")


def create_diagnostics(root):
    """Return per-pair SONAR diagnostics and a flag-stratified descriptive summary."""
    root = Path(root)
    tables = root / "outputs/tables"
    diagnostics = root / "outputs/analysis/frozen_encoders/diagnostics"

    geo = _load_csv(
        tables / "positive_pair_metadata.csv",
        KEY + ["distance_m", "heading_difference_deg", "look_difference_deg",
               "incidence_difference_deg", "view_class", "a_side", "b_side"],
        KEY,
    )
    quality = _load_csv(
        tables / "positive_pair_quality.csv",
        KEY + ["coverage_a", "coverage_b", "joint_coverage",
               "central_joint_coverage", "mask_iou", "max_nodata_region",
               "requires_review", "review_reason"],
        KEY,
    )
    scores = _load_csv(
        diagnostics / "four_encoder_positive_pair_metrics.csv",
        ["encoder_stem", "survey_pair", "pair_id"] + METRICS,
        ["encoder_stem", "survey_pair", "pair_id"],
    ).rename(columns={"survey_pair": "pair", "pair_id": "patch_id"})
    failures = _load_csv(
        diagnostics / "four_encoder_shared_failure_queries.csv",
        ["survey_pair", "direction", "pair_id", "failed_models"],
        ["survey_pair", "direction", "pair_id"],
    ).rename(columns={"survey_pair": "pair", "pair_id": "patch_id"})

    _require_same_pairs(geo, quality, "quality")
    stems = [stem for stem, _ in ENCODERS]
    if set(scores["encoder_stem"]) != set(stems):
        raise ValueError("Metrics must contain exactly the four frozen encoders")
    if not np.isfinite(geo[["distance_m", "heading_difference_deg",
                            "look_difference_deg", "incidence_difference_deg"]].to_numpy(float)).all():
        raise ValueError("Missing or non-finite geometric values")
    if not np.isfinite(quality[["joint_coverage", "central_joint_coverage",
                                "max_nodata_region"]].to_numpy(float)).all():
        raise ValueError("Missing or non-finite quality values")
    if not quality["requires_review"].isin([True, False]).all():
        raise ValueError("Invalid requires_review flags")

    table = geo[KEY + ["distance_m", "heading_difference_deg",
                        "look_difference_deg", "incidence_difference_deg",
                        "view_class", "a_side", "b_side"]].copy()
    qcols = KEY + ["coverage_a", "coverage_b", "joint_coverage",
                   "central_joint_coverage", "mask_iou", "max_nodata_region",
                   "requires_review", "review_reason"]
    table = table.merge(quality[qcols], on=KEY, how="left",
                        validate="one_to_one", indicator=True)
    if not (table["_merge"] == "both").all():
        raise ValueError("Missing quality row for at least one positive pair")
    table = table.drop(columns="_merge")
    table["review_reason"] = table["review_reason"].fillna("")

    for stem in stems:
        subset = scores.loc[scores["encoder_stem"] == stem, KEY + METRICS].copy()
        _require_same_pairs(geo, subset, stem)
        if not np.isfinite(subset[METRICS].to_numpy(dtype=float)).all():
            raise ValueError(f"{stem}: non-finite retrieval metrics")
        for direction in ("forward", "reverse"):
            credit = subset[f"recall_at_1_{direction}"]
            if not credit.between(0, 1).all():
                raise ValueError(f"{stem}: invalid Recall@1 credits")
        subset = subset.rename(columns={
            "positive_cosine": f"{stem}_cosine",
            "margin": f"{stem}_margin",
            "recall_at_1_forward": f"{stem}_r1_forward",
            "recall_at_1_reverse": f"{stem}_r1_reverse",
        })
        subset[f"{stem}_r1_credit"] = (
            subset[f"{stem}_r1_forward"] + subset[f"{stem}_r1_reverse"]
        )
        table = table.merge(subset, on=KEY, validate="one_to_one", how="left")

    r1_cols = [f"{stem}_r1_credit" for stem in stems]
    table["total_r1_credit"] = table[r1_cols].sum(axis=1)
    table["zero_r1_credit"] = table["total_r1_credit"].eq(0)
    table["look_opposite_ge160"] = table["look_difference_deg"].ge(160)
    table["joint_coverage_lt60pct"] = table["joint_coverage"].lt(0.60)

    if not set(map(tuple, failures[KEY].itertuples(index=False, name=None))).issubset(
        set(map(tuple, geo[KEY].itertuples(index=False, name=None)))
    ):
        raise ValueError("Shared failures refer to unknown positive pairs")
    if not failures["failed_models"].eq(len(stems)).all():
        raise ValueError("Shared failure rows do not cover all four encoders")
    # A shared failure must be confirmed independently in each encoder's scores.
    by_pair = table.set_index(KEY)
    for row in failures.itertuples(index=False):
        a, b = row.pair.split("-")
        direction = "forward" if row.direction == f"{a}->{b}" else (
            "reverse" if row.direction == f"{b}->{a}" else None
        )
        if direction is None:
            raise ValueError(f"Invalid query direction: {row.direction}")
        matched = by_pair.loc[(row.pair, row.patch_id)]
        if any(float(matched[f"{stem}_r1_{direction}"]) != 0.0 for stem in stems):
            raise ValueError("Shared failure conflicts with a nonzero model Recall@1 credit")

    counts = failures.groupby(KEY).size().rename("shared_failure_directions")
    table = table.join(counts, on=KEY)
    table["shared_failure_directions"] = (
        table["shared_failure_directions"].fillna(0).astype(int)
    )
    if not table["shared_failure_directions"].between(0, 2).all():
        raise ValueError("A pair cannot have more than two failed query directions")

    summary = table.groupby("requires_review", as_index=False).agg(
        n_pairs=("patch_id", "size"),
        mean_joint_coverage=("joint_coverage", "mean"),
        r1_credit=("total_r1_credit", "sum"),
        zero_r1_pairs=("zero_r1_credit", "sum"),
        shared_failure_directions=("shared_failure_directions", "sum"),
    )
    summary["max_r1_credit"] = summary["n_pairs"] * 2 * len(stems)
    summary["mean_recall_at_1"] = summary["r1_credit"] / summary["max_r1_credit"]
    summary["zero_r1_pair_fraction"] = summary["zero_r1_pairs"] / summary["n_pairs"]

    return (
        table.sort_values(["patch_id", "pair"]).reset_index(drop=True),
        summary.sort_values("requires_review").reset_index(drop=True),
    )


def make_stratified_summary(table):
    """Compare frozen SONAR retrieval by survey, coverage, audit flag and look.

    The original closed-set gallery sizes are computed from the *whole*
    positive-pair table, not from each filtered subgroup. Thus the expected
    random Recall@1 remains 1/4 for A-B, 1/5 for A-C and 1/16 for B-C
    when applied to the actual 25-pair benchmark. Each model contributes
    two directional queries per pair; these observations are correlated.
    """
    stems = [stem for stem, _ in ENCODERS]
    if table.empty or table.duplicated(KEY).any():
        raise ValueError("Expected non-empty diagnostics with unique positive pairs")
    required = set(KEY + ["joint_coverage", "requires_review", "view_class",
                          "total_r1_credit", "zero_r1_credit",
                          "shared_failure_directions", "look_opposite_ge160",
                          "joint_coverage_lt60pct"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing fields for stratification: {missing}")

    counts = table.groupby("pair").size()
    gallery_sizes = counts.to_dict()
    if len(table) == 25:
        expected = {"pass_a-pass_b": 4, "pass_a-pass_c": 5,
                    "pass_b-pass_c": 16}
        if gallery_sizes != expected:
            raise ValueError(f"Unexpected original gallery sizes: {gallery_sizes}")

    enriched = table.copy()
    enriched["original_gallery_size"] = enriched["pair"].map(gallery_sizes)
    enriched["chance_at_1"] = 1.0 / enriched["original_gallery_size"]

    subsets = [("overall", "all", enriched)]
    for pair, group in enriched.groupby("pair", sort=True):
        subsets.append(("survey_pair", pair, group))
    for value, group in enriched.groupby("requires_review", sort=True):
        subsets.append(("audit_flag", f"requires_review={bool(value)}", group))
    for value, group in enriched.groupby("joint_coverage_lt60pct", sort=True):
        label = "coverage<60%" if value else "coverage>=60%"
        subsets.append(("joint_coverage_60pct", label, group))
    for value, group in enriched.groupby("look_opposite_ge160", sort=True):
        label = "look>=160deg" if value else "look<160deg"
        subsets.append(("look_opposite_160deg", label, group))
    for value, group in enriched.groupby("view_class", sort=True):
        subsets.append(("view_class", str(value), group))
    for pair, pair_group in enriched.groupby("pair", sort=True):
        for value, group in pair_group.groupby("joint_coverage_lt60pct", sort=True):
            label = "coverage<60%" if value else "coverage>=60%"
            subsets.append(("survey_pair_x_coverage", f"{pair} | {label}", group))
        for value, group in pair_group.groupby("requires_review", sort=True):
            subsets.append((
                "survey_pair_x_audit_flag",
                f"{pair} | requires_review={bool(value)}", group,
            ))

    rows = []
    for stratum, group_name, group in subsets:
        n_pairs = len(group)
        n_directional_queries = 2 * n_pairs
        n_model_queries = len(stems) * n_directional_queries
        credit = float(group["total_r1_credit"].sum())
        chance = float(group["chance_at_1"].mean())
        shared_failures = int(group["shared_failure_directions"].sum())
        zero_pairs = int(group["zero_r1_credit"].sum())
        if not 0 <= credit <= n_model_queries:
            raise ValueError(f"Out-of-range R@1 credit in {stratum}/{group_name}")
        if not 0 <= shared_failures <= n_directional_queries:
            raise ValueError(f"Out-of-range shared failures in {stratum}/{group_name}")
        recall = credit / n_model_queries
        rows.append({
            "stratum": stratum,
            "group": group_name,
            "n_pairs": n_pairs,
            "n_directional_queries": n_directional_queries,
            "n_model_queries": n_model_queries,
            "r1_credit": credit,
            "recall_at_1_pct": 100 * recall,
            "chance_at_1_pct": 100 * chance,
            "delta_vs_chance_pp": 100 * (recall - chance),
            "zero_r1_pairs": zero_pairs,
            "zero_r1_pair_pct": 100 * zero_pairs / n_pairs,
            "shared_failure_directions": shared_failures,
            "shared_failure_query_pct": 100 * shared_failures / n_directional_queries,
            "mean_joint_coverage_pct": 100 * float(group["joint_coverage"].mean()),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    """Generate diagnostics in a dedicated output directory, without model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    table, summary = create_diagnostics(args.root)
    stratified = make_stratified_summary(table)
    output = (args.output_dir if args.output_dir is not None
              else args.root / "outputs/analysis/positive_pair_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "positive_pair_diagnostics.csv"
    summary_path = output / "quality_group_summary.csv"
    stratified_path = output / "stratified_retrieval_summary.csv"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    stratified.to_csv(stratified_path, index=False)

    display = table[["patch_id", "pair", "joint_coverage", "requires_review",
                     "look_difference_deg", "total_r1_credit",
                     "shared_failure_directions"]].copy()
    print("\n=== POSITIVE-PAIR DIAGNOSTICS ===")
    print(display.round(3).to_string(index=False))
    print("\n=== QUALITY GROUP SUMMARY (DESCRIPTIVE) ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== STRATIFIED RETRIEVAL (DESCRIPTIVE) ===")
    print(stratified.round(2).to_string(index=False))
    print(f"\nCSV saved: {table_path}\nCSV saved: {summary_path}\nCSV saved: {stratified_path}")
    print("Nominal co-location is not validated ground truth; no causal attribution.")


if __name__ == "__main__":
    main()
