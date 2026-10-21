"""Single BlueBoat-only cross-survey fine-tuning experiment with spatial CV.

Audits are cheap and run before importing Torch/pretrained weights. Every
training run begins from the same ImageNet ResNet50 V2 weights. Original frozen
four-encoder archives, geometry, positives and negative bank are unchanged.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_blueboat_pairs
from .evaluation import evaluate_fold
from .splits import plan_folds, training_indices
from .training import build_network, device_for, train_fold

OUTPUT_RELATIVE = Path("outputs/analysis/resnet50_cross_survey_cv")


def run_experiment(root, *, audit_only=False, diagnose_only=False,
                   ablation_last_bottleneck=False, folds=3, buffer_m=25.0,
                   epochs=3, batch_size=4, lr=1e-5, seed=42, device="auto"):
    root = Path(root).expanduser().resolve()
    pairs = load_blueboat_pairs(root)
    manifest, audit = plan_folds(pairs, folds=folds, buffer_m=buffer_m)
    print("BLUEBOAT CROSS-SURVEY SPLIT AUDIT (25 original pairs, no new data)")
    for info in audit:
        print(json.dumps(info, sort_keys=True))
    if sum(bool(x) for x in (audit_only, diagnose_only, ablation_last_bottleneck)) > 1:
        raise ValueError("Choose exactly one special operation")
    if audit_only:
        return audit
    if diagnose_only:
        from .diagnostics import run_diagnostics
        return run_diagnostics(root, pairs, manifest, folds=folds,
                               buffer_m=buffer_m, device=device)

    blocked = [a["fold"] for a in audit if not a["trainable"]]
    if blocked:
        raise ValueError(
            f"Insufficient geographically separated positives/negatives in folds "
            f"{blocked}. Training stopped to prevent misleading results. "
            f"Inspect --audit-only; do not relax the leakage buffer just to force training."
        )
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("epochs>=1 and batch-size>=2 are required")
    if not np.isfinite(lr) or lr <= 0 or not isinstance(seed, int):
        raise ValueError("Invalid learning rate or seed")

    original_output = root / OUTPUT_RELATIVE
    if ablation_last_bottleneck:
        from .ablation import verify_ablation_setup
        output = original_output / "new_results" / "last_bottleneck"
        original_summary = verify_ablation_setup(
            original_output, output, manifest, folds=folds, buffer_m=buffer_m,
            epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        tuning_scope = "last_bottleneck"
    else:
        output = original_output
        tuning_scope = "bn_affine"

    import torch
    target = device_for(torch, device)
    output.mkdir(parents=True, exist_ok=not ablation_last_bottleneck)
    manifest.to_csv(output / "fold_manifest.csv", index=False)
    cosine_rows, query_rows, training_rows = [], [], []
    for audit_row in audit:
        fold = audit_row["fold"]
        train = [pairs[i] for i in training_indices(manifest, pairs, fold, "train")]
        test = [pairs[i] for i in training_indices(manifest, pairs, fold, "test")]
        print(f"fold {fold}/{folds}: {len(train)} training pairs, "
              f"{len(test)} held-out pairs, {audit_row['embargoed_pairs']} embargoed")
        for modality in ("sonar", "png"):
            # Initialize frozen and trained comparison from identical pretrained weights.
            torch.manual_seed(seed + fold)
            baseline, unused_head = build_network(torch)
            baseline = baseline.to(target).eval()
            c, q = evaluate_fold(torch, baseline, test, modality, "frozen", fold, target)
            cosine_rows.extend(c)
            query_rows.extend(q)
            del baseline, unused_head

            tuned, history = train_fold(
                torch, train, modality, fold=fold, epochs=epochs,
                batch_size=batch_size, lr=lr, seed=seed + fold,
                buffer_m=buffer_m, device=target, output=output,
                tuning_scope=tuning_scope,
            )
            training_rows.extend(history)
            c, q = evaluate_fold(torch, tuned, test, modality, "fine_tuned", fold, target)
            cosine_rows.extend(c)
            query_rows.extend(q)
            del tuned

    cos = pd.DataFrame(cosine_rows)
    ranks = pd.DataFrame(query_rows)
    cos.to_csv(output / "heldout_positive_cosines.csv", index=False)
    ranks.to_csv(output / "heldout_retrieval_queries.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "training_history.csv", index=False)
    by = ["modality", "phase", "condition"]
    summary = cos.groupby(by, as_index=False).agg(
        heldout_positive_pairs=("patch_id", "size"),
        mean_positive_cosine=("positive_cosine", "mean"),
    )
    if not ranks.empty:
        retrieval = ranks.groupby(by, as_index=False).agg(
            rankable_directional_queries=("expected_recall_at_1", "size"),
            recall_at_1=("expected_recall_at_1", "mean"),
            recall_at_3=("expected_recall_at_3", "mean"),
            chance_recall_at_1=("gallery_size", lambda x: float(np.mean(1 / x))),
            chance_recall_at_3=("gallery_size", lambda x: float(np.mean(np.minimum(3 / x, 1)))),
        )
        summary = summary.merge(retrieval, on=by, how="left")
    else:
        summary["rankable_directional_queries"] = 0
        for key in ("recall_at_1", "recall_at_3", "chance_recall_at_1",
                    "chance_recall_at_3"):
            summary[key] = float("nan")
    summary.to_csv(output / "heldout_summary.csv", index=False)
    protocol = {
        "dataset": "BlueBoat original nominal cross-survey positives only",
        "pair_counts": {"pass_a-pass_b": 4, "pass_a-pass_c": 5, "pass_b-pass_c": 16},
        "total_unique_pairs": 25, "folds": folds, "buffer_m": buffer_m,
        "audit": audit, "epochs": epochs, "batch_size_pairs": batch_size,
        "lr": lr, "seed": seed, "device": str(target),
        "training": ("ImageNet ResNet50 V2; full last layer4 bottleneck and "
                     "projection MLP updated; paired cross-survey InfoNCE"
                     if ablation_last_bottleneck else
                     "ImageNet ResNet50 V2; only affine BN parameters in last layer4 bottleneck and "
                     "projection MLP updated; paired cross-survey InfoNCE"),
        "positive_label_warning": "GPS-derived nominal pairs, no verified ground truth",
        "negative_mining": "other spatially disjoint training pairs only; nearby "
                           "false-negative candidates excluded from contrastive loss",
        "augmentation": "independent brightness, contrast, noise, <=3px translation",
        "input": "common mask-safe SONAR intensity scaling; PNG is its 8-bit "
                 "grayscale encode/decode; quantization is the only branch difference",
        "evaluation": "held-out geographic-fold gallery only; per-survey rank "
                      "requires >=2 held-out candidates; no across-fold embedding pooling",
        "limitation": "tiny correlated data; no nested validation, results exploratory; "
                      "mask/no-data and acoustic artifacts may still be shortcuts",
    }
    if ablation_last_bottleneck:
        protocol["exploratory_ablation"] = True
        protocol["adaptation_scope"] = "last layer4 bottleneck only (including convolutions)"
        protocol["comparison"] = "matched BN-affine geographic folds, already-observed held-out data"
        protocol["output_isolation"] = "new_results/last_bottleneck; existing results unchanged"
        protocol["bn_running_stats"] = "frozen by model.eval()"
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")
    if ablation_last_bottleneck:
        from .ablation import compare_ablation
        comparison = compare_ablation(original_summary, summary)
        comparison.to_csv(output / "comparison_vs_bn_affine.csv", index=False)
    print("\nHELD-OUT GEOGRAPHIC CROSS-VALIDATION RESULTS")
    print(summary.to_string(index=False))
    print("Outputs:", output)
    return summary
