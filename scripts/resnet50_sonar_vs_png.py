"""Single entry point: BlueBoat-only ResNet50 cross-survey SONAR vs PNG training.

From repository root:
  python -m scripts.resnet50_sonar_vs_png --audit-only
  python -m scripts.resnet50_sonar_vs_png --epochs 3 --folds 3
  python -m scripts.resnet50_sonar_vs_png --diagnose-only
  python -m scripts.resnet50_sonar_vs_png --ablation-last-bottleneck --device mps

Audit inspects only the canonical positive metadata and patch NPZs; no weights
downloaded and no training. The full run creates isolated CV artifacts only.
"""
import argparse
from pathlib import Path

from src.experiments.resnet50_sonar_png.pipeline import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="BlueBoat cross-survey paired contrastive ResNet50: SONAR vs PNG"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true",
                      help="Report purged geographic folds, without training")
    mode.add_argument("--diagnose-only", action="store_true",
                      help="Analyze six saved checkpoints without training or overwriting results")
    mode.add_argument("--ablation-last-bottleneck", action="store_true",
                      help="Run one locked exploratory last-bottleneck ablation in new_results")
    parser.add_argument("--folds", type=int, choices=(2, 3, 4, 5), default=3)
    parser.add_argument("--buffer-m", type=float, default=25.0,
                        help="Additional geographic test embargo beyond the 50m patch footprint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of true cross-pass pairs per contrastive batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)
    return run_experiment(
        Path(__file__).resolve().parents[1], audit_only=args.audit_only,
        diagnose_only=args.diagnose_only,
        ablation_last_bottleneck=args.ablation_last_bottleneck,
        folds=args.folds, buffer_m=args.buffer_m, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
