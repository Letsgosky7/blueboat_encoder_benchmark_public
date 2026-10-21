"""Compare four frozen BlueBoat encoders using cached 50 m benchmark outputs.

Run from repository root:
    python -m scripts.benchmark_comparison

Options:
    python -m scripts.benchmark_comparison --input-dir outputs/embeddings --output-dir outputs/analysis/frozen_encoders
    python -m scripts.benchmark_comparison --with-diagnostics

The optional --with-diagnostics mode also reads local positive/bank patch NPZs
and quality/null metadata, adding cross-model mask audits, a common 25-pair
review and automatic shared Top-1 failure images under diagnostics/.

Writes five PNG figures, a multi-page PDF, and three CSVs. Does not run models,
modify the original results, or claim geographic localization.

Input: outputs/embeddings/{model}_*.csv plus provenance-valid CLIP NPZ.
Output: outputs/analysis/frozen_encoders/: five PNGs, one PDF and three CSVs.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ENCODERS = (
    ("resnet50", "ResNet50"),
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("vit_b16", "ViT-B/16"),
    ("clip_vit_b32", "CLIP ViT-B/32"),
)
CONDITIONS = ("sonar", "mask-only")
SURVEY_PAIRS = ("pass_a-pass_b", "pass_a-pass_c", "pass_b-pass_c")
KEY = ("condition", "survey_pair", "direction", "query_patch_id")
PAIR_KEY = ("condition", "survey_pair", "pair_id")


def _read(path, required, keys):
    """Read a cached CSV and reject missing columns or duplicate row identities."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark output: {path}")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{path.name}: duplicate keys {keys}")
    return frame


def load_results(input_dir):
    """Load and cross-check every encoder on identical query/pair identities."""
    input_dir = Path(input_dir)
    loaded = {}
    for stem, name in ENCODERS:
        query = _read(
            input_dir / f"{stem}_retrieval_queries.csv",
            (*KEY, "gallery_size", "expected_recall_at_1", "expected_recall_at_3", "tie_count"),
            KEY,
        )
        summary = _read(
            input_dir / f"{stem}_retrieval_summary.csv",
            ("condition", "survey_pair", "direction", "queries", "expected_recall_at_1", "expected_recall_at_3"),
            KEY[:3],
        )
        margins = _read(
            input_dir / f"{stem}_matched_null_margins.csv",
            (*PAIR_KEY, "margin", "requires_review"),
            PAIR_KEY,
        )
        scores = _read(
            input_dir / f"{stem}_sonar_mask_pair_scores.csv",
            ("condition", "survey_pair", "pair_id", "group", "cosine"),
            ("condition", "survey_pair", "pair_id", "group"),
        )
        for table, numeric_cols in (
            (query, ("gallery_size", "expected_recall_at_1", "expected_recall_at_3", "tie_count")),
            (margins, ("margin",)),
            (scores, ("cosine",)),
        ):
            if not np.isfinite(table[list(numeric_cols)].to_numpy(dtype=float)).all():
                raise ValueError(f"{stem}: invalid numeric values")
        if set(query["condition"]) != set(CONDITIONS):
            raise ValueError(f"{stem}: expected SONAR and MASK-ONLY queries")
        if set(margins["condition"]) != set(CONDITIONS):
            raise ValueError(f"{stem}: expected SONAR and MASK-ONLY margins")
        if set(query["survey_pair"]) != set(SURVEY_PAIRS):
            raise ValueError(f"{stem}: expected exactly the three 50 m survey pairs")
        if not (query["expected_recall_at_1"].between(0, 1).all()
                and query["expected_recall_at_3"].between(0, 1).all()):
            raise ValueError(f"{stem}: invalid recall credits")
        recovered = query.groupby(list(KEY[:3]), as_index=False).agg(
            queries=("query_patch_id", "size"),
            expected_recall_at_1=("expected_recall_at_1", "mean"),
            expected_recall_at_3=("expected_recall_at_3", "mean"),
        )
        checks = recovered.merge(summary, on=list(KEY[:3]), how="outer", validate="one_to_one",
                                 suffixes=("_calc", "_saved"), indicator=True)
        if (not (checks["_merge"] == "both").all()
                or not (checks["queries_calc"] == checks["queries_saved"]).all()
                or not np.allclose(checks["expected_recall_at_1_calc"], checks["expected_recall_at_1_saved"])
                or not np.allclose(checks["expected_recall_at_3_calc"], checks["expected_recall_at_3_saved"])):
            raise ValueError(f"{stem}: retrieval summary does not match per-query data")
        loaded[stem] = {"name": name, "query": query, "margins": margins, "scores": scores}

    clip_archive = input_dir / "clip_vit_b32_embeddings.npz"
    if not clip_archive.is_file():
        raise FileNotFoundError(f"Missing CLIP provenance archive: {clip_archive}")
    with np.load(clip_archive, allow_pickle=False) as archive:
        if ("model_architecture" not in archive.files
                or str(archive["model_architecture"]) != "ViT-B-32-quickgelu"
                or str(archive["weights"]) != "openai"):
            raise ValueError("CLIP outputs must use OpenAI ViT-B/32 QuickGELU weights; rerun CLIP")

    for table_name, keys in (("query", KEY), ("margins", PAIR_KEY),
                             ("scores", ("condition", "survey_pair", "pair_id", "group"))):
        reference = set(map(tuple, loaded[ENCODERS[0][0]][table_name][list(keys)].itertuples(index=False, name=None)))
        for stem, _ in ENCODERS[1:]:
            current = set(map(tuple, loaded[stem][table_name][list(keys)].itertuples(index=False, name=None)))
            if reference != current:
                raise ValueError(f"{stem}: {table_name} identities do not match ResNet50")
    return loaded


def aggregate_metrics(loaded):
    """Combine tie-aware query credits into model/condition recall and chance baselines."""
    rows = []
    for stem, name in ENCODERS:
        queries = loaded[stem]["query"]
        for condition in CONDITIONS:
            subset = queries.loc[queries["condition"] == condition]
            if subset.empty:
                raise ValueError(f"No queries for {stem}/{condition}")
            rows.append({
                "model": name, "model_stem": stem, "condition": condition,
                "queries": len(subset),
                "recall_at_1": float(subset["expected_recall_at_1"].mean()),
                "recall_at_3": float(subset["expected_recall_at_3"].mean()),
                "chance_at_1": float((1 / subset["gallery_size"]).mean()),
                "chance_at_3": float(np.minimum(3 / subset["gallery_size"], 1).mean()),
            })
    return pd.DataFrame(rows)


def query_matrix(loaded):
    """Align exact query identities across models for shared success/failure diagnostics."""
    base = None
    for stem, _ in ENCODERS:
        sub = loaded[stem]["query"][list(KEY) + ["expected_recall_at_1", "expected_recall_at_3", "tie_count"]].copy()
        sub = sub.rename(columns={key: f"{stem}_{key}" for key in ("expected_recall_at_1", "expected_recall_at_3", "tie_count")})
        base = sub if base is None else base.merge(sub, on=list(KEY), how="inner", validate="one_to_one")
    if len(base) != len(loaded[ENCODERS[0][0]]["query"]):
        raise ValueError("Per-query benchmark comparisons have missing rows")
    return base.sort_values(list(KEY)).reset_index(drop=True)


def _footnote(fig, text):
    """Add a small methodological limitation note to a Matplotlib figure."""
    fig.text(0.03, 0.025, text, fontsize=9, color="dimgray", va="bottom")


def figure_retrieval(metrics):
    """Compare aggregate SONAR/MASK recall against the weighted random baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    names = [name for _, name in ENCODERS]
    x = np.arange(len(names))
    for ax, (metric, chance, title) in zip(axes, (
        ("recall_at_1", "chance_at_1", "Recall@1"),
        ("recall_at_3", "chance_at_3", "Recall@3"),
    )):
        for offset, condition in ((-0.19, "sonar"), (0.19, "mask-only")):
            frame = metrics.set_index(["model", "condition"])
            values = [frame.loc[(name, condition), metric] for name in names]
            bars = ax.bar(x + offset, values, width=0.36,
                          label=condition.upper(), alpha=0.88)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.012,
                        f"{value:.0%}", ha="center", fontsize=9)
        chance_value = float(metrics[chance].iloc[0])
        ax.axhline(chance_value, linestyle="--", color="black", linewidth=1.1,
                   label=f"Chance {chance_value:.0%}")
        ax.set_xticks(x, names, rotation=15, ha="right")
        ax.set_ylim(0, max(0.52, float(metrics[metric].max()) + 0.08))
        ax.set_ylabel("Mean tie-aware recall")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Frozen encoders | bidirectional exact-ID retrieval | 50 m", fontsize=15)
    _footnote(fig, "50 queries per condition and encoder. Chance is gallery-size weighted; not verified geographic localization.")
    fig.tight_layout(rect=(0, 0.065, 1, 0.91))
    return fig


def figure_retrieval_by_pair(loaded):
    """Plot directional recall separately for A–B, A–C and B–C galleries."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
    x = np.arange(len(SURVEY_PAIRS))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (column, title, k) in zip(axes, (
        ("expected_recall_at_1", "Recall@1 by survey pair (SONAR)", 1),
        ("expected_recall_at_3", "Recall@3 by survey pair (SONAR)", 3),
    )):
        for i, (stem, name) in enumerate(ENCODERS):
            queries = loaded[stem]["query"]
            mean = queries.loc[queries["condition"] == "sonar"].groupby("survey_pair")[column].mean()
            vals = [float(mean[pair]) for pair in SURVEY_PAIRS]
            ax.bar(x + (i - 1.5) * 0.19, vals, 0.18, label=name, color=colors[i])
        reference = loaded[ENCODERS[0][0]]["query"]
        grouped = reference.loc[reference["condition"] == "sonar"].groupby("survey_pair")["gallery_size"]
        n = np.asarray([grouped.first().loc[pair] for pair in SURVEY_PAIRS], dtype=float)
        chance = np.minimum(k / n, 1.0)
        ax.plot(x, chance, "ko--", linewidth=1.1, markersize=5,
                label="Random baseline", zorder=5)
        ax.set_xticks(x, [f"{pair.split('-')[0][-1].upper()}-{pair.split('-')[1][-1].upper()} (n={int(size)})"
                          for pair, size in zip(SURVEY_PAIRS, n)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Bidirectional mean tie-aware recall")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Retrieval by survey pair | distinct gallery sizes", fontsize=14)
    _footnote(fig, "Two query directions per matched patch. Descriptive comparisons; gallery sizes differ by survey pair.")
    fig.tight_layout(rect=(0, 0.075, 1, 0.81))
    return fig


def figure_cosine_distributions(loaded):
    """Compare positive/null cosine distributions within each encoder and condition."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.3), sharey=False)
    groups = ("positive", "within-site-noncolocated", "isolated-control")
    labels = ("Positive\n(n=25)", "Within-site neg.\n(n=100)", "Control neg.\n(n=72)")
    for col, (stem, name) in enumerate(ENCODERS):
        scores = loaded[stem]["scores"]
        for row, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            subset = scores.loc[scores["condition"] == condition]
            arrays = [subset.loc[subset["group"] == grp, "cosine"].to_numpy(float) for grp in groups]
            if any(len(a) == 0 for a in arrays):
                raise ValueError(f"{stem}: missing score groups")
            ax.boxplot(arrays, showfliers=True, patch_artist=False, widths=0.57)
            ax.set_xticks((1, 2, 3), labels, fontsize=8)
            ax.set_title(f"{name}\n{condition.upper()}", fontsize=11)
            ax.set_ylabel("Cosine similarity")
            ax.grid(axis="y", alpha=0.2)
            # Put sample sizes under the categories, not over the upper whiskers.
    fig.suptitle("Cosine similarity | compare groups within each encoder (different y-axis ranges)", fontsize=14)
    _footnote(fig, "Same 25 positives / 100 within-site / 72 isolated-control negatives per condition. Samples are correlated.")
    fig.tight_layout(rect=(0, 0.065, 1, 0.93), h_pad=3.0, w_pad=2.2)
    return fig


def figure_matched_margins(loaded):
    """Show each positive-pair matched-negative margin and pair-group mean."""
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 9), sharey=True)
    conditions = ("sonar", "mask-only")
    for col, (stem, name) in enumerate(ENCODERS):
        margin_data = loaded[stem]["margins"]
        for row, condition in enumerate(conditions):
            ax = axes[row, col]
            subset = margin_data.loc[margin_data["condition"] == condition]
            means = []
            for idx, pair in enumerate(SURVEY_PAIRS):
                values = subset.loc[subset["survey_pair"] == pair, "margin"].to_numpy(float)
                if not len(values):
                    raise ValueError(f"{stem}: missing matched margins for {condition}/{pair}")
                shifts = np.linspace(-0.12, 0.12, len(values))
                ax.scatter(np.repeat(idx, len(values)) + shifts, values,
                           s=16, alpha=0.55)
                ax.plot([idx - 0.22, idx + 0.22], [values.mean()]*2, color="black", linewidth=2.1)
                means.append(float(values.mean()))
            # Put all mean margins in a dedicated box below the plot, never over points.
            # Group order matches the A-B / A-C / B-C tick labels above the box.
            mean_text = "Mean: " + "   ".join(f"{value:+.2f}" for value in means)
            ax.text(
                0.5, -0.27, mean_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.75", alpha=0.95),
                clip_on=False,
            )
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(range(3), ("A-B", "A-C", "B-C"))
            ax.set_title(f"{name}\n{condition.upper()}", fontsize=11)
            if col == 0:
                ax.set_ylabel("Positive cosine - matched-negative mean")
            ax.grid(axis="y", alpha=0.17)
    fig.suptitle("Matched-negative margins | dots = individual positive pairs; black = mean", fontsize=14)
    _footnote(fig, "25 positive pairs per condition (4 A-B, 5 A-C, 16 B-C). No independence or significance is implied.")
    fig.tight_layout(rect=(0, 0.075, 1, 0.93), h_pad=5.3, w_pad=1.5)
    return fig


def figure_shared_failures(matrix):
    """Count queries solved by zero through four encoders at exact-ID rank 1."""
    view = matrix.loc[matrix["condition"] == "sonar"].copy()
    columns = [f"{stem}_expected_recall_at_1" for stem, _ in ENCODERS]
    ties = [f"{stem}_tie_count" for stem, _ in ENCODERS]
    if not (view[ties] == 1).all().all():
        raise ValueError("SONAR shared-failure plot requires untied queries; do not bin fractional success")
    view["models_correct"] = (view[columns].to_numpy(float) > 0.999999).sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    counts = view["models_correct"].value_counts().reindex(range(5), fill_value=0)
    axes[0].bar(counts.index, counts.to_numpy())
    axes[0].set_xticks(range(5))
    axes[0].set_ylabel("Number of queries")
    axes[0].set_xlabel("Encoders retrieving the correct patch at rank 1")
    axes[0].set_title("Success overlap across all survey pairs")
    for x, value in counts.items():
        axes[0].text(x, value+0.3, str(value), ha="center", fontsize=9)
    x = np.arange(len(SURVEY_PAIRS))
    for successes in range(5):
        values = [int((view.loc[view["survey_pair"] == pair, "models_correct"] == successes).sum())
                  for pair in SURVEY_PAIRS]
        base = np.zeros(len(SURVEY_PAIRS)) if successes == 0 else sum(
            np.asarray([(view.loc[view["survey_pair"] == pair, "models_correct"] == s).sum()
                        for pair in SURVEY_PAIRS]) for s in range(successes))
        bars = axes[1].bar(x, values, bottom=base, label=f"{successes} correct")
        for bar, n in zip(bars, values):
            if n > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(n), ha="center", va="center",
                    fontsize=8 if n < 3 else 10,
                    color="white" if successes in (0, 3, 4) else "black",
                )
    totals = view.groupby("survey_pair").size().reindex(SURVEY_PAIRS)
    for xi, total in zip(x, totals):
        axes[1].text(xi, total + 0.5, str(int(total)),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, float(totals.max()) + 3)
    axes[1].set_xticks(x, ("A-B", "A-C", "B-C"))
    axes[1].set_ylabel("Queries")
    axes[1].set_title("Success overlap by acquisition pair")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Common SONAR retrieval failures across all four models", fontsize=14)
    _footnote(fig, f"{len(view)} directed SONAR queries. All-four failures: {int(counts.loc[0])}/{len(view)}. Exact-ID task, not geolocation.")
    fig.tight_layout(rect=(0, 0.065, 1, 0.82))
    return fig


def generate_report(input_dir, output_dir, *, with_diagnostics=False, dataset_root=None):
    """Write standard cached comparison; optionally inspect the original patches.

    Default behavior and original five PNG/three CSV/PDF names are unchanged.
    Patch I/O and all extra PNG/CSV generation are opt-in only.
    """
    loaded = load_results(input_dir)
    metrics = aggregate_metrics(loaded)
    matrix = query_matrix(loaded)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "four_encoder_metrics_summary.csv", index=False)
    matrix.to_csv(output / "four_encoder_query_comparison.csv", index=False)
    margins_table = pd.concat([
        loaded[stem]["margins"].assign(model=name, model_stem=stem)
        for stem, name in ENCODERS
    ], ignore_index=True).groupby(
        ["model", "model_stem", "condition", "survey_pair"], as_index=False, sort=False
    ).agg(n_pairs=("margin", "size"), mean_margin=("margin", "mean"),
          median_margin=("margin", "median"),
          positive_margin_fraction=("margin", lambda values: float((values > 0).mean())))
    margins_table.to_csv(output / "four_encoder_margin_summary.csv", index=False)
    figures = (
        ("four_encoder_retrieval", figure_retrieval(metrics)),
        ("four_encoder_retrieval_by_pair", figure_retrieval_by_pair(loaded)),
        ("four_encoder_cosine_distributions", figure_cosine_distributions(loaded)),
        ("four_encoder_matched_margins", figure_matched_margins(loaded)),
        ("four_encoder_shared_failures", figure_shared_failures(matrix)),
    )
    pdf_path = output / "four_encoder_comparison_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for stem, fig in figures:
            fig.savefig(output / f"{stem}.png", dpi=170, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    if with_diagnostics:
        from src.analysis.diagnostics import run_comparison_diagnostics

        # A custom input directory may contain cached scores from elsewhere.
        # The source dataset root is explicit rather than guessed from CSV paths.
        source_root = (Path(__file__).resolve().parents[1]
                       if dataset_root is None else Path(dataset_root))
        run_comparison_diagnostics(source_root, output, loaded, matrix, ENCODERS)

    print("All four encoders compared using cached metrics (no inference).")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"PDF report: {pdf_path}")
    print("Figures/CSVs:", output)
    return metrics, matrix


def main():
    """Parse optional input/output directories and generate the cached comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/frozen_encoders"))
    parser.add_argument(
        "--with-diagnostics", action="store_true",
        help="Also compare mask geometry and review shared SONAR failures (requires patch NPZs).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Repository/data root with outputs/patches and outputs/tables (diagnostics only).",
    )
    args = parser.parse_args()
    generate_report(args.input_dir, args.output_dir,
                    with_diagnostics=args.with_diagnostics, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
