"""One predeclared, exploratory ResNet50 last-bottleneck ablation.

Guardrails are deliberately checked before importing Torch or creating new
results. Previously observed held-out results are *not* used to select learning
rates, epochs, positive pairs, or model architecture.
"""
import json

import numpy as np
import pandas as pd

from .diagnostics import validate_saved_manifest


FIXED_FOLLOWUP = {
    "folds": 3, "buffer_m": 25.0, "epochs": 3,
    "batch_size_pairs": 4, "lr": 1e-5, "seed": 42,
}
METRICS = ("recall_at_1", "recall_at_3", "mean_positive_cosine",
           "chance_recall_at_1", "chance_recall_at_3")


def verify_ablation_setup(original_output, new_output, manifest, *,
                          folds, buffer_m, epochs, batch_size, lr, seed):
    """Match the first protocol and split, and refuse overwrite of any outputs."""
    requested = {"folds": folds, "buffer_m": buffer_m, "epochs": epochs,
                 "batch_size_pairs": batch_size, "lr": lr, "seed": seed}
    if requested != FIXED_FOLLOWUP:
        raise ValueError("Exploratory ablation settings are predeclared and fixed: "
                         + str(FIXED_FOLLOWUP))
    protocol_path = original_output / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Original CV protocol required: {protocol_path}")
    original_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (any(original_protocol.get(k) != v for k, v in FIXED_FOLLOWUP.items())
            or original_protocol.get("total_unique_pairs") != 25
            or "only affine BN" not in original_protocol.get("training", "")):
        raise ValueError("Original BN-affine experiment protocol is incompatible")
    validate_saved_manifest(manifest, pd.read_csv(original_output / "fold_manifest.csv"))
    previous = original_output / "heldout_summary.csv"
    if not previous.is_file():
        raise FileNotFoundError(f"Original held-out summary required: {previous}")
    if new_output.exists():
        raise FileExistsError(f"Existing exploratory ablation output: {new_output}. "
                              "Refusing to overwrite any previous run.")
    return pd.read_csv(previous)


def compare_ablation(original, followup):
    """Produce fold-matched per-condition comparison, checking frozen baseline.

    Do not treat these as a second independent held-out validation: the previous
    results on all 25 nominal pairs have already been observed.
    """
    keys = ["modality", "phase", "condition"]
    required = set(keys + list(METRICS) + ["heldout_positive_pairs",
                                           "rankable_directional_queries"])
    if not required.issubset(original.columns) or not required.issubset(followup.columns):
        raise ValueError("Original/follow-up summaries lack comparison fields")
    if original.duplicated(keys).any() or followup.duplicated(keys).any():
        raise ValueError("Duplicate summary rows in matched comparison")
    paired = original.merge(followup, on=keys, how="outer", indicator=True,
                            validate="one_to_one", suffixes=("_bn_affine",
                                                              "_last_bottleneck"))
    if len(paired) != len(original) or not paired["_merge"].eq("both").all():
        raise ValueError("Missing condition/modality/phase in the matched comparison")
    for name in ("heldout_positive_pairs", "rankable_directional_queries"):
        if not (paired[name + "_bn_affine"]
                == paired[name + "_last_bottleneck"]).all():
            raise ValueError("Changed positive roster or rankable query counts")
    frozen = paired["phase"] == "frozen"
    for metric in METRICS:
        a = paired.loc[frozen, metric + "_bn_affine"].to_numpy(float)
        b = paired.loc[frozen, metric + "_last_bottleneck"].to_numpy(float)
        if not np.allclose(a, b, rtol=0, atol=2e-3, equal_nan=True):
            raise ValueError(f"Frozen baseline differs from original protocol: {metric}")
        paired["delta_last_minus_bn_" + metric] = (
            paired[metric + "_last_bottleneck"]
            - paired[metric + "_bn_affine"])
    return paired.drop(columns=["_merge"])
