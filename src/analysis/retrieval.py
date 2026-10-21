"""Closed-set cross-pass retrieval on curated positive pairs.

Each survey pair defines its own query/gallery. A query's paired observation
has the same row index in the stored archive. No bank patches are added here:
their correspondence with positive queries is not annotated.

Tied similarities receive fractional expected Recall@k under uniform random
tie breaking. This avoids optimistic scores for constant mask embeddings.

Input: one encoder's NPZ archive, optionally positive metadata for geographic diagnostics.
Output: tie-aware query and summary DataFrames plus optional geographic rankings.
Usage: imported by benchmark scripts; no standalone CLI.
"""

from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = (("sonar", "sonar"), ("mask-only", "mask_only"))
TOP_K = (1, 3)
TIE_ATOL = 1e-6


def rank_queries(query, gallery, patch_ids, *, survey_pair, condition, direction):
    """Calculate tie-aware retrieval credit for one direction of one survey pair."""
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    ids = np.asarray(patch_ids)

    if (query.ndim != 2 or gallery.ndim != 2 or query.shape != gallery.shape
            or query.shape[0] < 2 or query.shape[1] == 0):
        raise ValueError("Query and gallery must have matching nonempty (N,D) shapes, N>=2.")
    if len(ids) != len(query) or len(set(map(int, ids))) != len(ids):
        raise ValueError("Expected one unique patch ID per query/gallery row.")
    if not np.isfinite(query).all() or not np.isfinite(gallery).all():
        raise ValueError("Embedding arrays must contain only finite values.")

    similarities = query @ gallery.T
    rows = []
    n = len(query)
    for i, patch_id in enumerate(ids):
        true_score = float(similarities[i, i])
        differences = similarities[i] - true_score
        better = int(np.count_nonzero(differences > TIE_ATOL))
        tied = int(np.count_nonzero(np.abs(differences) <= TIE_ATOL))
        if tied < 1 or better + tied > n:
            raise ValueError("Invalid tie group in retrieval similarities.")

        # If b gallery items outrank the true match and t items share its
        # score, its rank is uniformly distributed over [b+1, ..., b+t].
        credit = {
            k: float(np.clip((k - better) / tied, 0.0, 1.0))
            for k in TOP_K
        }
        rows.append({
            "condition": condition,
            "survey_pair": survey_pair,
            "direction": direction,
            "query_patch_id": int(patch_id),
            "gallery_size": n,
            "true_cosine": true_score,
            "strictly_better": better,
            "tie_count": tied,
            "expected_rank": better + (tied + 1) / 2,
            "expected_recall_at_1": credit[1],
            "expected_recall_at_3": credit[3],
        })
    return rows


def evaluate_positive_retrieval(embedding_archive):
    """Return (query_table, summary_table) for both directions and conditions."""
    with np.load(Path(embedding_archive), allow_pickle=False) as data:
        pairs = np.asarray(data["positive_survey_pair"]).astype(str)
        patch_ids = np.asarray(data["positive_patch_id"])
        if pairs.ndim != 1 or patch_ids.shape != pairs.shape:
            raise ValueError("Missing or invalid positive pair identifiers.")
        if len(set(zip(pairs.tolist(), map(int, patch_ids)))) != len(pairs):
            raise ValueError("Duplicate (survey pair, patch ID) combinations.")

        observations = []
        for pair in sorted(set(pairs.tolist())):
            subset = np.flatnonzero(pairs == pair)
            survey_names = pair.split("-")
            if len(survey_names) != 2:
                raise ValueError(f"Invalid survey pair label: {pair}")
            for condition, suffix in CONDITIONS:
                a = data[f"positive_{suffix}_a"][subset]
                b = data[f"positive_{suffix}_b"][subset]
                observations.extend(rank_queries(
                    a, b, patch_ids[subset],
                    survey_pair=pair, condition=condition,
                    direction=f"{survey_names[0]}->{survey_names[1]}",
                ))
                observations.extend(rank_queries(
                    b, a, patch_ids[subset],
                    survey_pair=pair, condition=condition,
                    direction=f"{survey_names[1]}->{survey_names[0]}",
                ))

    queries = pd.DataFrame(observations)
    summary = queries.groupby(
        ["condition", "survey_pair", "direction"], as_index=False
    ).agg(
        queries=("query_patch_id", "size"),
        gallery_size=("gallery_size", "first"),
        expected_recall_at_1=("expected_recall_at_1", "mean"),
        expected_recall_at_3=("expected_recall_at_3", "mean"),
        mean_expected_rank=("expected_rank", "mean"),
        tied_queries=("tie_count", lambda s: int((s > 1).sum())),
    )
    summary["chance_recall_at_1"] = 1.0 / summary["gallery_size"]
    summary["chance_recall_at_3"] = np.minimum(
        3.0 / summary["gallery_size"], 1.0
    )
    return queries, summary


def rank_geographic_candidates(embedding_archive, positive_metadata_csv, *, top_k=3):
    """Rank candidates and annotate their nominal 50-m patch footprint geometry.

    An overlap here concerns axis-aligned geographic patch squares, NOT valid
    sonar coverage and NOT independently verified seabed correspondences.
    Candidate ranks are deterministic (patch ID breaks exact-score ties);
    tie-aware expected Recall@k remains defined by rank_queries().
    """
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    metadata = pd.read_csv(positive_metadata_csv)
    fields = {"pair", "patch_id", "midpoint_east", "midpoint_north", "patch_size_m"}
    if not fields.issubset(metadata.columns):
        raise ValueError(f"Missing geographic metadata: {sorted(fields - set(metadata))}")
    if metadata.duplicated(["pair", "patch_id"]).any():
        raise ValueError("Duplicate geographic metadata for a positive pair.")
    numeric = metadata[["midpoint_east", "midpoint_north", "patch_size_m"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Geographic metadata contains non-finite values.")
    if (metadata["patch_size_m"] <= 0).any():
        raise ValueError("Patch size must be positive.")
    # The current dataset uses equal, north-aligned squares for all observations.
    if not np.allclose(metadata["patch_size_m"], metadata["patch_size_m"].iloc[0]):
        raise ValueError("This geographic diagnostic requires equal patch sizes.")
    side_m = float(metadata["patch_size_m"].iloc[0])
    positions = metadata.set_index(["pair", "patch_id"])[
        ["midpoint_east", "midpoint_north"]
    ]

    candidates = []
    with np.load(Path(embedding_archive), allow_pickle=False) as data:
        pairs = np.asarray(data["positive_survey_pair"]).astype(str)
        patch_ids = np.asarray(data["positive_patch_id"], dtype=np.int64)
        if pairs.ndim != 1 or patch_ids.shape != pairs.shape:
            raise ValueError("Missing or invalid positive pair identifiers.")
        if len(set(zip(pairs.tolist(), patch_ids.tolist()))) != len(pairs):
            raise ValueError("Duplicate positive patch identifiers in embeddings.")
        for pair in sorted(set(pairs.tolist())):
            subset = np.flatnonzero(pairs == pair)
            ids = patch_ids[subset]
            if len(ids) < 2 or len(set(ids.tolist())) != len(ids):
                raise ValueError(f"Invalid gallery for survey pair {pair}.")
            names = pair.split("-")
            if len(names) != 2:
                raise ValueError(f"Invalid survey-pair label: {pair}")
            centers = []
            for patch_id in ids:
                key = (pair, int(patch_id))
                if key not in positions.index:
                    raise ValueError(f"Missing geographic location: {key}")
                centers.append(positions.loc[key].to_numpy(dtype=float))
            centers = np.stack(centers)

            for condition, suffix in CONDITIONS:
                a = data[f"positive_{suffix}_a"][subset]
                b = data[f"positive_{suffix}_b"][subset]
                if a.ndim != 2 or a.shape != b.shape or len(a) != len(ids):
                    raise ValueError(f"Invalid embedding shapes for {pair}: {condition}")
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("Embedding arrays must be finite.")

                for query, gallery, direction in (
                    (a, b, f"{names[0]}->{names[1]}"),
                    (b, a, f"{names[1]}->{names[0]}"),
                ):
                    scores = query @ gallery.T
                    for i, patch_id in enumerate(ids):
                        # Highest cosine first; patch ID breaks exact numerical ties.
                        order = np.lexsort((ids, -scores[i]))
                        for rank, j in enumerate(order[:top_k], start=1):
                            delta = np.abs(centers[i] - centers[j])
                            intersection_m2 = float(
                                np.prod(np.maximum(0.0, side_m - delta))
                            )
                            overlap = intersection_m2 / (side_m * side_m)
                            distance = float(np.linalg.norm(delta))
                            score = float(scores[i, j])
                            candidates.append({
                                "condition": condition,
                                "survey_pair": pair,
                                "direction": direction,
                                "query_patch_id": int(patch_id),
                                "candidate_rank": rank,
                                "candidate_patch_id": int(ids[j]),
                                "cosine": score,
                                "exact_id_match": bool(j == i),
                                "center_distance_m": distance,
                                "nominal_footprint_overlap_fraction": overlap,
                                "center_within_50m": bool(distance <= 50.0),
                                "nominal_overlap_ge_25pct": bool(overlap >= 0.25),
                                "near_tied_candidates": int(np.count_nonzero(
                                    np.abs(scores[i] - score) <= TIE_ATOL
                                )),
                            })

    return pd.DataFrame.from_records(candidates)
