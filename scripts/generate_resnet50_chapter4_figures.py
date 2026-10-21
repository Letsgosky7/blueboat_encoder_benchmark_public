"""Generate reproducible Chapter 4 figures from saved ResNet50 CV results.

Creates heldout_recall_comparison.png and training_loss_comparison.png.
Uses existing CSV files only; no Torch, checkpoints, inference or retraining.

From the repository root:
    python -m scripts.generate_resnet50_chapter4_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Work with CLI/headless execution, including macOS.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "outputs" / "analysis" / "resnet50_cross_survey_cv"
FIGURES_DIRNAME = "chapter4_figures"
HISTORY_COLUMNS = {"fold", "modality", "epoch", "loss", "batches", "training_pairs"}
MODALITIES = (("sonar", "Numerical SONAR"), ("png", "8-bit PNG"))
MODEL_STAGES = (
    ("Frozen", "original", "frozen"),
    ("BN-affine", "original", "fine_tuned"),
    ("Last bottleneck", "last", "fine_tuned"),
)
NEEDED_COLUMNS = {
    "modality", "phase", "condition", "heldout_positive_pairs",
    "rankable_directional_queries", "recall_at_1", "recall_at_3",
    "chance_recall_at_1", "chance_recall_at_3",
}


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    data = pd.read_csv(path)
    missing = NEEDED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    return data


def _select(data: pd.DataFrame, modality: str, phase: str) -> pd.Series:
    rows = data.loc[
        (data["modality"] == modality)
        & (data["condition"] == "sonar")
        & (data["phase"] == phase)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for modality={modality}, condition=sonar, "
            f"phase={phase}; found {len(rows)}"
        )
    return rows.iloc[0]


def collect_heldout_recall_metrics(results_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Read all plotted values from CSV and validate comparable held-out splits."""
    original = _load(results_dir / "heldout_summary.csv")
    last = _load(results_dir / "new_results" / "last_bottleneck" / "heldout_summary.csv")
    measurements: dict[str, list[dict[str, float]]] = {}

    for modality, _ in MODALITIES:
        rows = []
        for label, source, phase in MODEL_STAGES:
            entry = _select(original if source == "original" else last, modality, phase)
            values = {name: float(entry[name]) for name in (
                "heldout_positive_pairs", "rankable_directional_queries",
                "recall_at_1", "recall_at_3",
                "chance_recall_at_1", "chance_recall_at_3",
            )}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite values for {modality}/{label}")
            if not all(0 <= values[k] <= 1 for k in (
                "recall_at_1", "recall_at_3", "chance_recall_at_1", "chance_recall_at_3",
            )):
                raise ValueError(f"Recall outside [0, 1]: {modality}/{label}")
            values["label"] = label
            rows.append(values)

        for field in (
            "heldout_positive_pairs", "rankable_directional_queries",
            "chance_recall_at_1", "chance_recall_at_3",
        ):
            if not np.allclose([row[field] for row in rows], rows[0][field], atol=1e-10, rtol=0):
                raise ValueError(f"Incomparable evaluation protocol: {modality}, field {field}")
        if rows[0]["heldout_positive_pairs"] != 25 or rows[0]["rankable_directional_queries"] != 48:
            raise ValueError(f"Unexpected evaluation counts for {modality}: expected 25 pairs and 48 queries")

        # Check the saved original and follow-up frozen baselines really match.
        older = _select(original, modality, "frozen")
        newer = _select(last, modality, "frozen")
        for key in ("recall_at_1", "recall_at_3", "chance_recall_at_1", "chance_recall_at_3"):
            if not np.isclose(float(older[key]), float(newer[key]), atol=1e-10, rtol=0):
                raise ValueError(f"Frozen baseline mismatch: {modality}/{key}")
        measurements[modality] = rows
    return measurements


def plot_heldout_recall_comparison(data: dict[str, list[dict[str, float]]], destination: Path) -> None:
    """One PNG with two panels: numerical SONAR and 8-bit PNG pathways."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    labels = [stage[0] for stage in MODEL_STAGES]
    x = np.arange(len(labels))
    width = 0.34

    for ax, (modality, title) in zip(axes, MODALITIES):
        rows = data[modality]
        r1 = np.array([100 * row["recall_at_1"] for row in rows])
        r3 = np.array([100 * row["recall_at_3"] for row in rows])
        chance1 = 100 * rows[0]["chance_recall_at_1"]
        chance3 = 100 * rows[0]["chance_recall_at_3"]
        bars1 = ax.bar(x - width / 2, r1, width, label="Recall@1")
        bars3 = ax.bar(x + width / 2, r3, width, label="Recall@3")
        ax.axhline(chance1, color="0.30", linestyle="--", linewidth=1.2,
                   label=f"Chance R@1 ({chance1:.2f}%)")
        ax.axhline(chance3, color="0.55", linestyle=":", linewidth=1.6,
                   label=f"Chance R@3 ({chance3:.2f}%)")
        ax.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=9)
        ax.bar_label(bars3, fmt="%.2f%%", padding=3, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, max(75, max(r3.max(), r1.max()) + 13))
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Recall (%)")
    # Put metric bars first, then fold-matched chance levels in the legend.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_items = dict(zip(legend_labels, handles))
    recall_labels = ["Recall@1", "Recall@3"]
    chance_labels = [s for s in legend_labels if s.startswith("Chance")]
    order = recall_labels + chance_labels
    fig.legend([legend_items[label] for label in order], order,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.88),
               frameon=False, fontsize=9)
    fig.suptitle("ResNet50 — Retrieval comparison under geographic cross-validation",
                 fontsize=15, y=0.975)
    # The held-out protocol/denominators belong in the LaTeX caption.
    # Reserve most of the image height for the two scientific panels.
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.74, wspace=0.07)
    fig.savefig(destination, dpi=260, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)



def _load_training_history(path: Path) -> pd.DataFrame:
    """Read saved InfoNCE losses, requiring all three epochs per fold and modality."""
    if not path.is_file():
        raise FileNotFoundError(f"Training-loss CSV not found: {path}")
    history = pd.read_csv(path)
    missing = HISTORY_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    if history.duplicated(["fold", "modality", "epoch"]).any():
        raise ValueError(f"{path}: duplicate fold/modality/epoch rows")
    expected = {(f, m, e) for f in (1, 2, 3) for m in ("sonar", "png")
                for e in (1, 2, 3)}
    present = set(map(tuple, history[["fold", "modality", "epoch"]].itertuples(index=False, name=None)))
    if present != expected:
        raise ValueError(f"{path}: incomplete training history or unexpected configurations")
    if (not np.isfinite(history["loss"].to_numpy(dtype=float)).all()
            or (history["batches"] <= 0).any()
            or (history["training_pairs"] <= 0).any()):
        raise ValueError(f"{path}: non-finite loss values or invalid training counts")
    return history.sort_values(["fold", "modality", "epoch"])


def plot_training_loss_comparison(results_dir: Path, destination: Path) -> None:
    """Compare actual epoch-mean InfoNCE losses, not held-out performance."""
    original = _load_training_history(results_dir / "training_history.csv")
    bottleneck = _load_training_history(
        results_dir / "new_results" / "last_bottleneck" / "training_history.csv"
    )
    original_counts = original[["fold", "modality", "epoch", "training_pairs", "batches"]]
    new_counts = bottleneck[["fold", "modality", "epoch", "training_pairs", "batches"]]
    # Both variants must have identical splits and the same batch counts.
    if not original_counts.reset_index(drop=True).equals(new_counts.reset_index(drop=True)):
        raise ValueError("Training variants have different folds, training-pair counts, or batches")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for ax, (heading, history) in zip(
        axes, (("BN-affine", original), ("Last bottleneck", bottleneck))
    ):
        for fold in (1, 2, 3):
            for modality, style, marker in (("sonar", "-", "o"), ("png", "--", "s")):
                rows = history.loc[
                    (history["fold"] == fold) & (history["modality"] == modality)
                ].sort_values("epoch")
                ax.plot(
                    rows["epoch"].to_numpy(), rows["loss"].to_numpy(),
                    color=f"C{fold - 1}", linestyle=style, marker=marker,
                    linewidth=1.8, markersize=5,
                    label=f"Fold {fold} — {'SONAR' if modality == 'sonar' else 'PNG'}",
                )
        ax.set_title(heading, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 2, 3])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean InfoNCE loss per batch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.88), frameon=False, fontsize=9)
    fig.suptitle("ResNet50 — Contrastive loss over 3 epochs",
                 fontsize=15, y=0.975)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.72, wspace=0.10)
    fig.savefig(destination, dpi=260, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def generate_chapter4_figures(results_dir: Path) -> list[Path]:
    """Generate both reproducible figures in one clearly named output folder."""
    figures = results_dir / FIGURES_DIRNAME
    metrics = collect_heldout_recall_metrics(results_dir)
    recall_path = figures / "heldout_recall_comparison.png"
    loss_path = figures / "training_loss_comparison.png"
    plot_heldout_recall_comparison(metrics, recall_path)
    plot_training_loss_comparison(results_dir, loss_path)

    # Migrate only the legacy PNG made by the previous generator.
    # Leave any other locally created files in the old directory untouched.
    legacy_folder = results_dir / "figures"
    legacy_png = legacy_folder / "heldout_recall_comparison.png"
    if legacy_png.is_file() and not legacy_png.is_symlink():
        legacy_png.unlink()  # The new replacement above was saved successfully.
        try:
            legacy_folder.rmdir()  # Remove folder only if completely empty.
        except OSError:
            pass

    return [recall_path, loss_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory containing the resnet50_cross_survey_cv CSV results",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    for path in generate_chapter4_figures(results_dir):
        print(f"Generated PNG: {path}")


if __name__ == "__main__":
    main()
