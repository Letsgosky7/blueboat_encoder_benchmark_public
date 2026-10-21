"""Encoder-independent SONAR/MASK-ONLY inspection and comparative visual reviews.

Inputs: cached cosine/retrieval/margin CSVs, positive/bank patch NPZs, quality
        and negative-pair metadata. No model weights or new inference.
Outputs: original per-model diagnostic reports (legacy ResNet50 preserved)
         and optional four-encoder review/near-unity/failure figures and CSVs.
Usage: src.benchmark.reporting for legacy figures; scripts.benchmark_comparison
       with --with-diagnostics for all-model reports.

These are descriptive checks, NOT independent ground truth or geolocation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import BANK_SURVEYS, POSITIVE_PAIRS
from src.encoders.preprocessing import prepare_sonar_rgb


def inspect_near_unity_masks(near_unity, null_metadata, root, *,
                             model_name="resnet50", output_dir=None):
    """Inspect near-unity MASK-ONLY null scores using actual resized masks.

    `output_dir=None` retains the original ResNet50 artifact location.
    Different model scores select different negative pairs for inspection.
    The mask geometry itself comes from the same stored bank NPZs.
    """
    null_index = null_metadata.set_index("null_pair_id")
    if not null_index.index.is_unique:
        raise ValueError("Duplicate null_pair_id values in null metadata.")
    patch_dir = root / "outputs" / "patches"
    bank_masks = {}
    for survey in BANK_SURVEYS:
        with np.load(patch_dir / f"patch_bank_{survey}.npz") as data:
            for i, bank_index in enumerate(data["bank_index"]):
                # Exactly the resized mask given to the mask-only encoder.
                _, resized_mask = prepare_sonar_rgb(data["patch"][i], data["mask"][i])
                bank_masks[(survey, int(bank_index))] = resized_mask

    records = []
    for pair in near_unity.itertuples(index=False):
        meta = null_index.loc[int(pair.pair_id)]
        if str(meta.null_type) != str(pair.group):
            raise ValueError(f"Null metadata mismatch for ID {pair.pair_id}")
        a_key = (str(meta.survey_a), int(meta.bank_index_a))
        b_key = (str(meta.survey_b), int(meta.bank_index_b))
        mask_a, mask_b = bank_masks[a_key], bank_masks[b_key]
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        records.append({
            "null_pair_id": int(pair.pair_id),
            "null_type": str(pair.group),
            "survey_pair": str(pair.survey_pair),
            "cosine": float(pair.cosine),
            "coverage_a": float(mask_a.mean()),
            "coverage_b": float(mask_b.mean()),
            "mask_iou": float(intersection / union) if union else np.nan,
            "identical_masks": bool(np.array_equal(mask_a, mask_b)),
            "both_fully_valid": bool(mask_a.all() and mask_b.all()),
        })

    report = pd.DataFrame.from_records(records, columns=[
        "null_pair_id", "null_type", "survey_pair", "cosine",
        "coverage_a", "coverage_b", "mask_iou",
        "identical_masks", "both_fully_valid",
    ])
    destination_dir = (root / "outputs" / "embeddings"
                       if output_dir is None else Path(output_dir))
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{model_name}_near_unity_masks.csv"
    report.to_csv(destination, index=False)
    print("\n=== NEAR-UNITY MASKS: ACTUAL GEOMETRY ===")
    if len(report):
        print(report.round(4).to_string(index=False))
        print(f"Identical masks: {int(report['identical_masks'].sum())}/{len(report)}")
        print(f"Both fully valid: {int(report['both_fully_valid'].sum())}/{len(report)}")
    else:
        print("No mask-only null cosine >= 0.9999")
    return destination


def plot_sonar_failure_cases(root, matched, *, model_name="resnet50"):
    """Reproduce the two historical hand-picked cases for a given encoder.

    New automatic shared failures are handled by plot_comparative_pair_review.
    This function exists to keep the original ResNet50 PNG compatible.
    """
    cases = (("pass_a", "pass_c", 9), ("pass_b", "pass_c", 23))
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 6.5), squeeze=False)
    for row, (name_a, name_b, patch_id) in enumerate(cases):
        path = root / "outputs" / "patches" / f"positive_pairs_{name_a}_{name_b}.npz"
        with np.load(path) as data:
            idx = np.flatnonzero(data["patch_id"] == patch_id)
            if len(idx) != 1:
                raise ValueError(f"Missing or repeated patch ID {patch_id} in {path}")
            i = int(idx[0])
            views = [
                prepare_sonar_rgb(data[f"patch_{side}"][i], data[f"mask_{side}"][i])
                for side in ("a", "b")
            ]

        label = f"{name_a}-{name_b}"
        values = matched.loc[
            (matched["condition"] == "sonar")
            & (matched["survey_pair"] == label)
            & (matched["pair_id"] == patch_id)
        ]
        if len(values) != 1:
            raise ValueError(f"Missing matching sonar score for {label}, ID {patch_id}")
        score = values.iloc[0]
        for col, (survey, (rgb, valid)) in enumerate(zip((name_a, name_b), views)):
            axes[row, col * 2].imshow(rgb[0], cmap="gray", vmin=0, vmax=1, origin="lower")
            axes[row, col * 2].set_title(f"{survey.upper()} sonar", fontsize=10)
            axes[row, col * 2 + 1].imshow(
                valid, cmap="gray", vmin=0, vmax=1, origin="lower"
            )
            axes[row, col * 2 + 1].set_title(f"{survey.upper()} valid mask", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].set_xlabel(
            f"ID {patch_id} | cosine={score.positive_cosine:.3f} | matched margin={score.margin:+.3f}"
        )
        axes[row, 0].xaxis.label.set_visible(True)

    fig.suptitle("Encoder-ready sonar (median-filled no-data) and masks: patch IDs 9 and 23")
    fig.tight_layout()
    path = root / "outputs" / "figures" / f"{model_name}_patch_9_23_diagnostics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_positive_pair_review(root, matched, quality, page_size=5, *,
                              model_name="resnet50"):
    """Save historical single-encoder review pages for any encoder.

    The comparative version below is preferred for new experiments, because
    it shows the shared sonar inputs only once alongside four model metrics.


    Display the encoder-ready sonar input and the independent validity mask
    for both observations. This is a diagnostic, not visual certification.
    """
    sonar_scores = matched.loc[matched["condition"] == "sonar"].copy()
    quality_rows = quality.set_index(["pair", "patch_id"])
    if not quality_rows.index.is_unique:
        raise ValueError("Duplicate quality metadata for positive pairs.")

    score_index = sonar_scores.set_index(["survey_pair", "pair_id"])
    if not score_index.index.is_unique:
        raise ValueError("Duplicate sonar scores for positive pairs.")

    examples = []
    patch_dir = root / "outputs" / "patches"
    for survey_a, survey_b in POSITIVE_PAIRS:
        pair_name = f"{survey_a}-{survey_b}"
        with np.load(patch_dir / f"positive_pairs_{survey_a}_{survey_b}.npz") as data:
            for i, patch_id in enumerate(data["patch_id"]):
                key = (pair_name, int(patch_id))
                if key not in score_index.index or key not in quality_rows.index:
                    raise ValueError(f"Missing score or quality audit for {key}.")
                score = score_index.loc[key]
                audit = quality_rows.loc[key]
                left = prepare_sonar_rgb(data["patch_a"][i], data["mask_a"][i])
                right = prepare_sonar_rgb(data["patch_b"][i], data["mask_b"][i])
                examples.append((int(patch_id), pair_name, left, right, score, audit))

    examples.sort(key=lambda x: x[0])
    if len(examples) != len(sonar_scores):
        raise ValueError("Positive pair count differs between NPZ and score CSV.")
    ids = [example[0] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated global positive patch IDs.")

    figure_dir = root / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(examples), page_size):
        batch = examples[start:start + page_size]
        fig = plt.figure(figsize=(16, 3.0 * len(batch) + 0.8))
        # A dedicated text row prevents captions from overlapping adjacent images.
        grid = fig.add_gridspec(
            nrows=2 * len(batch), ncols=4,
            height_ratios=[0.25, 2.6] * len(batch),
            left=0.04, right=0.98, bottom=0.025, top=0.945,
            hspace=0.35, wspace=0.12,
        )
        for row, (patch_id, pair, left, right, score, audit) in enumerate(batch):
            first_survey, second_survey = pair.split("-")
            inputs = (
                (left[0][0], f"{first_survey.upper()} sonar"),
                (left[1], f"{first_survey.upper()} valid mask"),
                (right[0][0], f"{second_survey.upper()} sonar"),
                (right[1], f"{second_survey.upper()} valid mask"),
            )

            flag = "REVIEW FLAG" if bool(audit["requires_review"]) else "no audit flag"
            annotation = (
                f"ID {patch_id} | {pair} | heading {float(audit['heading_difference_deg']):.1f} deg"
                f" | cosine {float(score['positive_cosine']):.3f}"
                f" | matched margin {float(score['margin']):+.3f}"
                f" | joint coverage {float(audit['joint_coverage']):.2f}"
                f" | mask IoU {float(audit['mask_iou']):.2f}"
                f" | {flag}"
            )
            header = fig.add_subplot(grid[2 * row, :])
            header.axis("off")
            header.text(0, 0.5, annotation, va="center", fontsize=9)

            for col, (img, label) in enumerate(inputs):
                ax = fig.add_subplot(grid[2 * row + 1, col])
                ax.imshow(
                    img, cmap="gray", origin="lower", vmin=0, vmax=1,
                    interpolation="nearest",
                )
                ax.set_title(label, fontsize=9, pad=4)
                ax.axis("off")

        fig.suptitle(
            "Positive-pair review: sonar and validity masks (white=valid)",
            fontsize=12, y=0.99,
        )
        destination = figure_dir / (
            f"{model_name}_positive_pair_review_{start // page_size + 1:02d}.png"
        )
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(destination)

    print("\n=== POSITIVE-PAIR VISUAL REVIEW ===")
    print(f"Included {len(examples)} / {len(sonar_scores)} positive pairs; none excluded.")
    for path in paths:
        print(f"  {path}")
    print("Visual similarity is not proof of accurate georegistration.")
    return paths


# ---------------------------------------------------------------------------
# Four-encoder diagnostics. They never alter original benchmark artifacts.
# ---------------------------------------------------------------------------

def select_shared_failures(matrix, encoders):
    """Select SONAR directional queries missed at rank 1 by ALL encoders.

    `matrix` is the exact-ID aligned query table from benchmark_comparison.
    A success/failure decision is valid only with a unique true ranking:
    ties must not be silently assigned binary correctness.
    """
    sonar = matrix.loc[matrix["condition"] == "sonar"].copy()
    credit_cols = [f"{stem}_expected_recall_at_1" for stem, _ in encoders]
    tie_cols = [f"{stem}_tie_count" for stem, _ in encoders]
    if sonar.empty or not set(credit_cols + tie_cols).issubset(sonar.columns):
        raise ValueError("Missing SONAR query credits or tie counts for diagnostics.")
    if not (sonar[tie_cols] == 1).all().all():
        raise ValueError("Shared-failure selection requires untied SONAR rankings.")
    if not np.isfinite(sonar[credit_cols].to_numpy(dtype=float)).all():
        raise ValueError("Invalid per-encoder SONAR recall credits.")
    missed = sonar.loc[(sonar[credit_cols].to_numpy(dtype=float) == 0).all(axis=1),
                       ["survey_pair", "direction", "query_patch_id"]].copy()
    missed = missed.rename(columns={"query_patch_id": "pair_id"})
    missed["failed_models"] = len(encoders)
    return missed.sort_values(["survey_pair", "pair_id", "direction"]).reset_index(drop=True)


def build_comparative_pair_metrics(loaded, encoders):
    """Align each positive patch's SONAR cosine, margin and two retrieval credits.

    Each (survey_pair, pair_id) must occur exactly once in each model's
    positive scores and matched margins, and twice in directional retrieval.
    Row-wise joins deliberately reject silently omitted or substituted pairs.
    """
    frames = []
    for stem, display_name in encoders:
        data = loaded[stem]
        scores = data["scores"].loc[
            (data["scores"]["condition"] == "sonar")
            & (data["scores"]["group"] == "positive"),
            ["survey_pair", "pair_id", "cosine"],
        ].rename(columns={"cosine": "positive_cosine"})
        margins = data["margins"].loc[
            data["margins"]["condition"] == "sonar",
            ["survey_pair", "pair_id", "margin"],
        ]
        queries = data["query"].loc[
            data["query"]["condition"] == "sonar",
            ["survey_pair", "query_patch_id", "direction", "expected_recall_at_1"],
        ]
        keys = ["survey_pair", "pair_id"]
        if (scores.duplicated(keys).any() or margins.duplicated(keys).any()
                or queries.duplicated(["survey_pair", "query_patch_id", "direction"]).any()):
            raise ValueError(f"{stem}: duplicate positive identities.")

        # A and B must each act once as query, with the same positive patch ID.
        directional = []
        for (pair, patch_id), group in queries.groupby(
            ["survey_pair", "query_patch_id"], sort=False
        ):
            a, b = pair.split("-")
            forward, backward = f"{a}->{b}", f"{b}->{a}"
            if len(group) != 2 or set(group["direction"]) != {forward, backward}:
                raise ValueError(f"{stem}: missing or invalid bidirectional query for {pair}/{patch_id}")
            values = group.set_index("direction")["expected_recall_at_1"]
            directional.append({
                "survey_pair": pair, "pair_id": int(patch_id),
                "recall_at_1_forward": float(values.loc[forward]),
                "recall_at_1_reverse": float(values.loc[backward]),
            })
        directions = pd.DataFrame.from_records(directional, columns=[
            *keys, "recall_at_1_forward", "recall_at_1_reverse",
        ])
        joined = scores.merge(margins, on=keys, how="outer",
                              validate="one_to_one", indicator=True)
        if not (joined["_merge"] == "both").all():
            raise ValueError(f"{stem}: unmatched positive score/margin pairs.")
        joined = joined.drop(columns="_merge").merge(
            directions, on=keys, how="outer", validate="one_to_one", indicator=True
        )
        if not (joined["_merge"] == "both").all():
            raise ValueError(f"{stem}: retrieval identities do not match positive scores.")
        joined = joined.drop(columns="_merge")
        numeric = ["positive_cosine", "margin", "recall_at_1_forward", "recall_at_1_reverse"]
        if not np.isfinite(joined[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"{stem}: non-finite comparison metrics.")
        joined.insert(0, "encoder", display_name)
        joined.insert(1, "encoder_stem", stem)
        frames.append(joined)

    reference = set(map(tuple, frames[0][["survey_pair", "pair_id"]].itertuples(index=False, name=None)))
    for frame in frames[1:]:
        identities = set(map(tuple, frame[["survey_pair", "pair_id"]].itertuples(index=False, name=None)))
        if identities != reference:
            raise ValueError("Comparative review is missing positive IDs for an encoder.")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["survey_pair", "pair_id", "encoder_stem"]
    ).reset_index(drop=True)


def _load_positive_views(root, selected_pairs):
    """Load each selected positive pair ONCE, independent of encoder identity."""
    selected = set(selected_pairs)
    seen = set()
    for survey_a, survey_b in POSITIVE_PAIRS:
        pair = f"{survey_a}-{survey_b}"
        archive = root / "outputs" / "patches" / f"positive_pairs_{survey_a}_{survey_b}.npz"
        with np.load(archive, allow_pickle=False) as data:
            for i, patch_id in enumerate(data["patch_id"]):
                key = (pair, int(patch_id))
                if key not in selected:
                    continue
                if key in seen:
                    raise ValueError(f"Repeated positive patch identity: {key}")
                seen.add(key)
                yield key, (
                    prepare_sonar_rgb(data["patch_a"][i], data["mask_a"][i]),
                    prepare_sonar_rgb(data["patch_b"][i], data["mask_b"][i]),
                )
    if seen != selected:
        raise ValueError(f"Selected positive patches missing from NPZ: {sorted(selected - seen)}")


def plot_comparative_pair_review(root, metrics, quality, encoders, output_dir, *,
                                 prefix, selected_pairs=None, page_size=5):
    """Draw the same sonar/mask patches once, annotating all four encoder scores.

    The full-review mode includes ALL curated positives, including audit flags.
    The failure mode includes only patches with ≥1 all-model Top-1 miss.
    Each encoder annotation reports cosine, matched margin and directional
    Recall@1 credits; the figures do not claim verified georegistration.
    """
    if page_size < 1:
        raise ValueError("page_size must be >= 1.")
    keys = ["survey_pair", "pair_id"]
    quality_index = quality.set_index(["pair", "patch_id"])
    if not quality_index.index.is_unique:
        raise ValueError("Duplicate positive-pair quality rows.")
    metric_index = metrics.set_index(keys + ["encoder_stem"])
    if not metric_index.index.is_unique:
        raise ValueError("Duplicate encoder metrics for a positive patch.")
    all_pairs = set(map(tuple, metrics[keys].itertuples(index=False, name=None)))
    selected = all_pairs if selected_pairs is None else set(selected_pairs)
    if not selected.issubset(all_pairs):
        raise ValueError("Selected failure identities are absent from model metrics.")
    for pair, patch_id in selected:
        if (pair, patch_id) not in quality_index.index:
            raise ValueError(f"Missing quality audit: {pair}/{patch_id}")
        for stem, _ in encoders:
            if (pair, patch_id, stem) not in metric_index.index:
                raise ValueError(f"Missing {stem} diagnostic metrics for {pair}/{patch_id}.")

    # A single 4-view image layout per patch avoids quadrupling the same data.
    images = sorted(_load_positive_views(Path(root), selected),
                    key=lambda item: (item[0][1], item[0][0]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(images), page_size):
        batch = images[start:start + page_size]
        fig = plt.figure(figsize=(16, 4.2 * len(batch) + 0.6))
        grid = fig.add_gridspec(
            nrows=2 * len(batch), ncols=4,
            height_ratios=[1.55, 2.65] * len(batch),
            left=0.035, right=0.98, top=0.96, bottom=0.025,
            hspace=0.32, wspace=0.10,
        )
        for row, ((pair, patch_id), (left, right)) in enumerate(batch):
            survey_a, survey_b = pair.split("-")
            audit = quality_index.loc[(pair, patch_id)]
            header = fig.add_subplot(grid[2 * row, :])
            header.axis("off")
            flagged = "REVIEW FLAG" if bool(audit["requires_review"]) else "no audit flag"
            header.text(
                0.01, 0.96,
                f"ID {patch_id} | {pair} | heading {float(audit['heading_difference_deg']):.1f}°"
                f" | joint coverage {float(audit['joint_coverage']):.2f}"
                f" | mask IoU {float(audit['mask_iou']):.2f} | {flagged}",
                fontsize=9, va="top",
            )
            for model_index, (stem, label) in enumerate(encoders):
                result = metric_index.loc[(pair, patch_id, stem)]
                header.text(
                    0.015, 0.73 - model_index * 0.19,
                    f"{label:<18} cosine={float(result['positive_cosine']):.3f}"
                    f" | matched margin={float(result['margin']):+.3f}"
                    f" | R@1 {survey_a}→{survey_b}={float(result['recall_at_1_forward']):.2f}"
                    f", {survey_b}→{survey_a}={float(result['recall_at_1_reverse']):.2f}",
                    fontsize=8.5, va="top",
                )
            views = [
                (left[0][0], f"{survey_a.upper()} SONAR"),
                (left[1], f"{survey_a.upper()} valid mask"),
                (right[0][0], f"{survey_b.upper()} SONAR"),
                (right[1], f"{survey_b.upper()} valid mask"),
            ]
            for col, (pixels, label) in enumerate(views):
                ax = fig.add_subplot(grid[2 * row + 1, col])
                ax.imshow(pixels, cmap="gray", origin="lower",
                          vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(label, fontsize=9, pad=4)
                ax.axis("off")
        fig.suptitle(
            "Four frozen encoders | curated SONAR positives and validity masks"
            if selected_pairs is None
            else "Four frozen encoders | shared Top-1 SONAR retrieval failures",
            fontsize=12, y=0.99,
        )
        destination = output_dir / f"{prefix}_{start // page_size + 1:02d}.png"
        fig.savefig(destination, dpi=145, bbox_inches="tight")
        plt.close(fig)
        paths.append(destination)
    return paths


def run_comparison_diagnostics(root, output_dir, loaded, matrix, encoders):
    """Write opt-in joint diagnostics without touching benchmark/model files.

    The shared-failure table is defined at directional-query level; its
    companion visual pages deduplicate by (survey_pair, patch_id). A patch
    can therefore appear once even if BOTH directions fail in all four models.
    """
    root = Path(root)
    output = Path(output_dir) / "diagnostics"
    tables = root / "outputs" / "tables"
    quality = pd.read_csv(tables / "positive_pair_quality.csv")
    required_quality = {"pair", "patch_id", "heading_difference_deg",
                        "joint_coverage", "mask_iou", "requires_review"}
    if not required_quality.issubset(quality.columns):
        raise ValueError(f"Missing positive quality fields: {sorted(required_quality - set(quality))}")
    nulls = pd.read_csv(tables / "null_pair_metadata.csv")
    output.mkdir(parents=True, exist_ok=True)

    near_tables = []
    for stem, display_name in encoders:
        scores = loaded[stem]["scores"]
        near = scores.loc[
            (scores["condition"] == "mask-only")
            & (scores["group"] != "positive")
            & (scores["cosine"] >= 0.9999),
            ["group", "survey_pair", "pair_id", "cosine"],
        ]
        # The same bank masks are inspected under each model's OWN near-unity
        # cosine selection; no model-dependent geometric mask is invented.
        csv_path = inspect_near_unity_masks(
            near, nulls, root, model_name=stem, output_dir=output
        )
        report = pd.read_csv(csv_path)
        report.insert(0, "encoder_stem", stem)
        report.insert(0, "encoder", display_name)
        near_tables.append(report)
    pd.concat(near_tables, ignore_index=True).to_csv(
        output / "four_encoder_near_unity_masks.csv", index=False
    )

    metrics = build_comparative_pair_metrics(loaded, encoders)
    metrics.to_csv(output / "four_encoder_positive_pair_metrics.csv", index=False)
    all_pages = plot_comparative_pair_review(
        root, metrics, quality, encoders, output,
        prefix="four_encoder_positive_pair_review",
    )

    failures = select_shared_failures(matrix, encoders)
    failures.to_csv(output / "four_encoder_shared_failure_queries.csv", index=False)
    failed_pairs = set(map(tuple, failures[["survey_pair", "pair_id"]].itertuples(
        index=False, name=None
    )))
    failure_pages = plot_comparative_pair_review(
        root, metrics, quality, encoders, output,
        prefix="four_encoder_shared_failure_review", selected_pairs=failed_pairs,
    )
    print(
        f"Additional comparative diagnostics: {output} | "
        f"near-unity masks: {sum(len(frame) for frame in near_tables)} | "
        f"all-positive pages: {len(all_pages)} | "
        f"shared-failure queries: {len(failures)} | "
        f"failure pages: {len(failure_pages)}"
    )
    print("Descriptive only: pair correspondence and nominal footprints are not independently verified.")
    return metrics, failures
