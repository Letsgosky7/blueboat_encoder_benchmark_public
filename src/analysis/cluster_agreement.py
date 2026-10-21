"""Cross-survey cluster agreement on existing frozen positive-pair embeddings.

For every encoder, condition and survey pair (A-B, A-C, B-C), cluster the
first and second observation sets independently *in their original normalized
embedding space*, with Ward linkage and Euclidean distance. Compare the
two partitions over matched nominal positive IDs using adjusted Rand index
(ARI) and arithmetic-normalized mutual information (NMI).

The agreement is descriptive stability of partitions, NOT a supervised score,
accurate physical co-location, clustering accuracy, or evidence of causality.
Do not cluster 2D PCA projections. No inference or rewriting cached inputs.

Output root: outputs/analysis/final_evaluation/embeddings/clustering/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage

# Reuse existing validated frozen NPZ and positive-ID loading rather than
# duplicating their input schema. PCA itself is NEVER run in this experiment.
from src.analysis.embedding_pca import (
    EXPECTED_PAIRS,
    FROZEN_CONDITIONS,
    FROZEN_ENCODERS,
    read_positive_embeddings,
    read_reference_pairs,
)

OUTPUT_RELATIVE = Path("outputs/analysis/final_evaluation/embeddings/clustering")
MAX_CLUSTERS = 4
LINKAGE_METHOD = "ward"
LINKAGE_METRIC = "euclidean"
NMI_NORMALIZATION = "arithmetic"
SURVEY_PAIRS = tuple(EXPECTED_PAIRS)


def partition_agreement(labels_a, labels_b):
    """ARI and arithmetic NMI, invariant to independent label permutations.

    ARI: Hubert-Arabie adjustment for the expected Rand index.
    NMI: mutual information divided by the arithmetic mean of entropies.
    Both are determined from the same contingency table; cluster ID numbers
    themselves have no semantic meaning across observations.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2:
        raise ValueError("Partition agreement requires two equal label vectors of length >=2")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    contingency = np.zeros((ia.max() + 1, ib.max() + 1), dtype=np.float64)
    np.add.at(contingency, (ia, ib), 1)
    counts_a = contingency.sum(axis=1)
    counts_b = contingency.sum(axis=0)
    n = len(a)

    def choose_two(v):
        return float(np.sum(v * (v - 1.0) / 2.0))

    row_pairs = choose_two(counts_a)
    col_pairs = choose_two(counts_b)
    pair_index = choose_two(contingency)
    expected = row_pairs * col_pairs / (n * (n - 1) / 2.0)
    maximum = (row_pairs + col_pairs) / 2.0
    denominator = maximum - expected
    if np.isclose(denominator, 0.0, atol=1e-12, rtol=0):
        # Degenerate single/all-singleton partitions: the contingency table
        # alone has no adjusted denominator. Define 1 only for identical
        # equivalence relations, the same convention as sklearn.
        equal_relations = np.array_equal(
            ia[:, None] == ia[None, :],
            ib[:, None] == ib[None, :],
        )
        ari = 1.0 if equal_relations else 0.0
    else:
        ari = float((pair_index - expected) / denominator)

    probabilities = contingency / n
    p_a = counts_a / n
    p_b = counts_b / n
    product = p_a[:, None] * p_b[None, :]
    nonzero = probabilities > 0
    mutual_info = float(np.sum(
        probabilities[nonzero] * np.log(probabilities[nonzero] / product[nonzero])
    ))
    entropy_a = -float(np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0])))
    entropy_b = -float(np.sum(p_b[p_b > 0] * np.log(p_b[p_b > 0])))
    mean_entropy = (entropy_a + entropy_b) / 2.0
    nmi = 1.0 if mean_entropy <= 1e-12 else float(mutual_info / mean_entropy)
    return float(np.clip(ari, -1, 1)), float(np.clip(nmi, 0, 1))


def ward_partition(features, k):
    """Make an ID-order-deterministic Ward partition, or explicitly skip it.

    A cluster cut at a tied merge-height boundary is not uniquely specified
    by the distance geometry. Suppress ARI/NMI for that setting instead of
    silently scoring a tie-broken arbitrary partition.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        raise ValueError("Ward requires >=3 observations and at least one feature")
    if not np.isfinite(x).all():
        raise ValueError("Ward requires finite feature vectors")
    n = len(x)
    if not isinstance(k, (int, np.integer)) or k < 2 or k >= n:
        raise ValueError("Require an integer cluster count between 2 and n-1")
    unique_rows = int(np.unique(np.round(x, decimals=7), axis=0).shape[0])
    if unique_rows < k:
        return None, "insufficient_unique_vectors", unique_rows, np.nan

    tree = linkage(x, method=LINKAGE_METHOD, metric=LINKAGE_METRIC)
    heights = tree[:, 2]
    lower = float(heights[n - k - 1])
    upper = float(heights[n - k])
    gap = upper - lower
    if np.isclose(lower, upper, atol=1e-10, rtol=1e-8):
        return None, "ambiguous_tied_cut", unique_rows, gap

    labels = cut_tree(tree, n_clusters=[k]).ravel().astype(int)
    if len(np.unique(labels)) != k:
        raise ValueError("Hierarchical clustering did not return the requested k")
    return labels, "ok", unique_rows, gap


def compute_cluster_agreement(root, *, models=FROZEN_ENCODERS, max_clusters=MAX_CLUSTERS):
    """Return metrics, per-ID assignments and model provenance, with no writes."""
    root = Path(root).expanduser().resolve()
    model_list = tuple(models)
    if not model_list or len(model_list) != len(set(model_list)):
        raise ValueError("Expected a nonempty list of distinct encoders")
    if not set(model_list).issubset(FROZEN_ENCODERS):
        raise ValueError("Unknown frozen encoder")
    if not isinstance(max_clusters, int) or not 2 <= max_clusters <= MAX_CLUSTERS:
        raise ValueError(f"max_clusters must be an integer in [2, {MAX_CLUSTERS}]")

    reference = read_reference_pairs(root)
    ref_ids = set(reference.index.to_list())
    metrics, assignments, provenance_rows = [], [], []

    for model in model_list:
        archive = root / "outputs/embeddings" / f"{model}_embeddings.npz"
        keys, vectors, provenance = read_positive_embeddings(archive, model, ref_ids)
        # Sorting IDs inside each group gives a reproducible Ward tie-breaking
        # order; the two sides must ALWAYS share this exact key order.
        for condition in FROZEN_CONDITIONS:
            for survey_pair, n_expected in EXPECTED_PAIRS.items():
                indices = sorted(
                    (i for i, (pair, _) in enumerate(keys) if pair == survey_pair),
                    key=lambda i: keys[i][1],
                )
                if len(indices) != n_expected:
                    raise ValueError("Archive survey-pair membership changed")
                ids = [keys[i][1] for i in indices]
                if len(set(ids)) != n_expected:
                    raise ValueError("Duplicate nominal pair IDs in clustering")
                side_a = vectors[(condition, "a")][indices]
                side_b = vectors[(condition, "b")][indices]
                # Both clusterings use the same number of nominally paired
                # observations, but each survey has its own fitted dendrogram.
                for k in range(2, min(max_clusters, n_expected - 1) + 1):
                    a, status_a, unique_a, gap_a = ward_partition(side_a, k)
                    b, status_b, unique_b, gap_b = ward_partition(side_b, k)
                    status = "ok" if status_a == status_b == "ok" else (
                        f"a:{status_a};b:{status_b}"
                    )
                    ari, nmi = (partition_agreement(a, b) if status == "ok"
                                else (np.nan, np.nan))
                    metrics.append({
                        "model": model,
                        "condition": condition,
                        "survey_pair": survey_pair,
                        "n_pairs": n_expected,
                        "k": k,
                        "ari": ari,
                        "nmi": nmi,
                        "status": status,
                        "unique_vectors_a": unique_a,
                        "unique_vectors_b": unique_b,
                        "boundary_gap_a": gap_a,
                        "boundary_gap_b": gap_b,
                    })
                    # Store assignments ONLY when the cut is well-defined.
                    if status == "ok":
                        for i, patch_id in enumerate(ids):
                            observation = reference.loc[(survey_pair, patch_id)]
                            assignments.append({
                                "model": model, "condition": condition,
                                "survey_pair": survey_pair, "patch_id": patch_id,
                                "k": k,
                                "survey_a": survey_pair.split("-")[0],
                                "survey_b": survey_pair.split("-")[1],
                                "cluster_a": int(a[i]), "cluster_b": int(b[i]),
                                "joint_coverage": float(observation["joint_coverage"]),
                                "requires_review": bool(observation["requires_review"]),
                            })
            provenance_rows.append({
                **provenance, "condition": condition,
                "linkage": LINKAGE_METHOD, "metric": LINKAGE_METRIC,
                "nmi_average_method": NMI_NORMALIZATION,
                "max_clusters": max_clusters,
            })
    return pd.DataFrame(metrics), pd.DataFrame(assignments), pd.DataFrame(provenance_rows)


def plot_agreement_heatmaps(output, metrics):
    """Four k=2 charts: ARI/NMI × SONAR/MASK-ONLY, without pooling k values."""
    for measure in ("ari", "nmi"):
        for condition in FROZEN_CONDITIONS:
            subset = metrics[(metrics["condition"] == condition) & (metrics["k"] == 2)]
            grid = np.full((len(FROZEN_ENCODERS), len(SURVEY_PAIRS)), np.nan)
            for i, model in enumerate(FROZEN_ENCODERS):
                for j, pair in enumerate(SURVEY_PAIRS):
                    cell = subset[(subset["model"] == model) &
                                  (subset["survey_pair"] == pair)]
                    if len(cell) == 1 and cell.iloc[0]["status"] == "ok":
                        grid[i, j] = float(cell.iloc[0][measure])
            fig, ax = plt.subplots(figsize=(8.4, 5.2))
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad(color="#D4D4D4")
            bounds = (-1, 1) if measure == "ari" else (0, 1)
            im = ax.imshow(np.ma.masked_invalid(grid), vmin=bounds[0], vmax=bounds[1],
                           cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(SURVEY_PAIRS)), [p.replace("pass_", "").upper() for p in SURVEY_PAIRS])
            ax.set_yticks(range(len(FROZEN_ENCODERS)), [
                "ResNet50", "ConvNeXt-Tiny", "ViT-B/16", "CLIP ViT-B/32",
            ])
            for i in range(len(FROZEN_ENCODERS)):
                for j in range(len(SURVEY_PAIRS)):
                    value = grid[i, j]
                    text = "N/A" if np.isnan(value) else f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center",
                            color=("black" if np.isnan(value) or value < 0.6 else "white"),
                            fontsize=10, fontweight="medium")
            fig.colorbar(im, ax=ax, shrink=0.9, label=measure.upper())
            ax.set_title(f"Cross-survey {measure.upper()} · {condition.upper()} · k=2\n"
                         "Ward on original frozen embeddings; nominal IDs")
            ax.set_xlabel("Survey pair")
            fig.tight_layout()
            fig.savefig(output / f"{measure}_{condition.replace('-', '_')}_k2.png", dpi=170)
            plt.close(fig)


def save_cluster_reports(root, *, models=FROZEN_ENCODERS, max_clusters=MAX_CLUSTERS):
    """Validate cached inputs first, then write only isolated cluster outputs."""
    metrics, assignments, provenance = compute_cluster_agreement(
        root, models=models, max_clusters=max_clusters
    )
    output = Path(root).expanduser().resolve() / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "cluster_agreement.csv": metrics,
        "cluster_assignments.csv": assignments,
        "cluster_provenance.csv": provenance,
    }
    for name, table in files.items():
        table.to_csv(output / name, index=False)
    # Chart partial model selections without suggesting missing models ran.
    if set(models) == set(FROZEN_ENCODERS):
        plot_agreement_heatmaps(output, metrics)
    return output, metrics
