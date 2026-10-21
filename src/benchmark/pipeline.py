"""Common frozen-encoder SONAR benchmark pipeline and CLI modes.

Input: 50 m positive-pair NPZ, patch bank NPZ, negative pair metadata CSV.
Output: saved model embedding NPZ, cosine-score CSV and shared diagnostics.
Usage: imported by four scripts/<model>_benchmark.py wrappers. Supports
--analyze-only, --verify-embeddings and --retrieval-only.

SONAR and MASK-ONLY use each observation's own mask. Negative-pair bank
observations are encoded once each, not once per assigned negative pair.
"""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import BANK_SURVEYS, CONDITIONS, EXPECTED_COUNTS, POSITIVE_PAIRS
from src.encoders.preprocessing import prepare_sonar_rgb
from src.encoders.embedding_store import save_embeddings, verify_embeddings

# Only these specific encoder classes can be loaded. Model imports are lazy.
ENCODER_CLASSES = {
    "resnet50": ("src.encoders.resnet50", "FrozenResNet50"),
    "convnext_tiny": ("src.encoders.convnext_tiny", "FrozenConvNeXtTiny"),
    "vit_b16": ("src.encoders.vit_b16", "FrozenViTB16"),
    "clip_vit_b32": ("src.encoders.clip_vit_b32", "FrozenCLIPViTB32"),
}


def positive_pair_path(patch_dir, survey_a, survey_b):
    """Translate survey names into the canonical underscore-separated NPZ path."""
    return Path(patch_dir) / f"positive_pairs_{survey_a}_{survey_b}.npz"


def encode_observation(encoder, patch, mask):
    """One observation; never use a joint/pair mask as encoder input."""
    sonar_rgb, resized_valid = prepare_sonar_rgb(patch, mask)
    # MASK-ONLY removes acoustic content; the mask is per observation, not joint.
    mask_rgb = np.broadcast_to(
        resized_valid.astype(np.float32), sonar_rgb.shape
    ).copy()
    return {
        "sonar": encoder.encode(sonar_rgb),
        "mask-only": encoder.encode(mask_rgb),
    }


def append_scores(rows, left, right, *, group, pair, pair_id, heading_deg):
    """Record paired cosine scores using the survey identity and evaluation group."""
    for condition in CONDITIONS:
        score = float(np.dot(left[condition], right[condition]))
        rows.append({
            "group": group,
            "condition": condition,
            "survey_pair": pair,
            "pair_id": int(pair_id),
            "heading_difference_deg": float(heading_deg),
            "cosine": score,
        })

def evaluate_results(root, model_name, *, encoder=None, expected_counts=EXPECTED_COUNTS):
    """Extract and cache positive/bank embeddings under a shared protocol.

    The optional encoder permits synthetic tests without downloading weights.
    Standard executions enforce the original (25, 81, 172) dataset counts.
    """
    if model_name not in ENCODER_CLASSES:
        raise ValueError(f"Unknown benchmark encoder: {model_name}")
    if encoder is None:
        module_name, class_name = ENCODER_CLASSES[model_name]
        encoder = getattr(import_module(module_name), class_name)()
    patch_dir = root / "outputs/patches"
    null_path = root / "outputs/tables/null_pair_metadata.csv"
    positives = []
    bank_vectors = {}
    scores = []

    weights = getattr(encoder.weights, "name", encoder.weights)
    print(f"{model_name}: {weights} | device: {encoder.device}")
    for survey_a, survey_b in POSITIVE_PAIRS:
        pair = f"{survey_a}-{survey_b}"
        with np.load(positive_pair_path(patch_dir, survey_a, survey_b)) as data:
            for i, patch_id in enumerate(data["patch_id"]):
                left = encode_observation(
                    encoder, data["patch_a"][i], data["mask_a"][i]
                )
                right = encode_observation(
                    encoder, data["patch_b"][i], data["mask_b"][i]
                )
                append_scores(
                    scores, left, right, group="positive",
                    pair=pair, pair_id=patch_id,
                    heading_deg=data["heading_difference_deg"][i],
                )
                positives.append((pair, int(patch_id), left, right))
        print(f"  positive {pair}: done")

    # Cache each bank observation once, then reuse vectors for negative pairs.
    for survey in BANK_SURVEYS:
        with np.load(patch_dir / f"patch_bank_{survey}.npz") as data:
            for i, bank_index in enumerate(data["bank_index"]):
                key = (survey, int(bank_index))
                if key in bank_vectors:
                    raise ValueError(f"Duplicate bank key: {key}")
                bank_vectors[key] = encode_observation(
                    encoder, data["patch"][i], data["mask"][i]
                )
        print(f"  bank {survey}: done")

    nulls = pd.read_csv(null_path)
    required = {
        "null_pair_id", "null_type", "survey_a", "survey_b",
        "bank_index_a", "bank_index_b", "heading_difference_deg",
    }
    if not required.issubset(nulls.columns):
        raise ValueError(f"Missing negative-pair fields: {sorted(required - set(nulls))}")
    if nulls["null_pair_id"].duplicated().any():
        raise ValueError("Duplicate negative-pair IDs.")

    actual = len(positives), len(bank_vectors), len(nulls)
    if expected_counts is not None and actual != expected_counts:
        raise ValueError(f"Unexpected 50 m dataset counts: {actual} != {expected_counts}")

    # Negative pairs reference bank indices and require no additional inference.
    for row in nulls.itertuples(index=False):
        left = bank_vectors[(str(row.survey_a), int(row.bank_index_a))]
        right = bank_vectors[(str(row.survey_b), int(row.bank_index_b))]
        append_scores(
            scores, left, right, group=str(row.null_type),
            pair=f"{row.survey_a}-{row.survey_b}",
            pair_id=row.null_pair_id, heading_deg=row.heading_difference_deg,
        )

    scores = pd.DataFrame.from_records(scores)
    if len(scores) != 2 * (len(positives) + len(nulls)):
        raise ValueError("Unexpected number of condition-specific similarity scores.")

    output = root / "outputs/embeddings"
    output.mkdir(parents=True, exist_ok=True)
    score_file = output / f"{model_name}_sonar_mask_pair_scores.csv"
    embedding_file = output / f"{model_name}_embeddings.npz"
    scores.to_csv(score_file, index=False)
    save_embeddings(
        embedding_file, positives, bank_vectors, weights, encoder.device, model=model_name
    )
    verify_embeddings(embedding_file, score_file, null_path, model=model_name)
    print(
        f"Saved {len(positives)} positives, {len(bank_vectors)} bank observations, "
        f"{len(nulls)} negatives and {len(scores)} scores."
    )
    print(f"Embeddings: {embedding_file}")
    print(f"Pair scores: {score_file}")


def main(model_name, argv=None):
    """Dispatch inference, cached analysis, embedding verification or retrieval.

    Passing argv overrides CLI flags for tests; None reads normal sys.argv.
    All four old Python -m entry points and output names remain unchanged.
    """
    if model_name not in ENCODER_CLASSES:
        raise ValueError(f"Unknown benchmark encoder: {model_name}")
    parser = argparse.ArgumentParser(description=f"Frozen {model_name} SONAR benchmark")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-embeddings", action="store_true",
                       help="Verify archived cosine scores without inference.")
    modes.add_argument("--analyze-only", action="store_true",
                       help="Regenerate reports and retrieval from saved outputs.")
    modes.add_argument("--retrieval-only", action="store_true",
                       help="Recompute retrieval from saved vectors only.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]

    # Local import avoids requiring plotting/retrieval for archive verification.
    from src.benchmark.reporting import analyze_results, report_retrieval

    if args.verify_embeddings:
        verify_embeddings(
            root / f"outputs/embeddings/{model_name}_embeddings.npz",
            root / f"outputs/embeddings/{model_name}_sonar_mask_pair_scores.csv",
            root / "outputs/tables/null_pair_metadata.csv", model=model_name,
        )
        return
    if args.retrieval_only:
        report_retrieval(root, model_name)
        return
    if not args.analyze_only:
        evaluate_results(root, model_name)
    analyze_results(root, model_name)
    report_retrieval(root, model_name)
