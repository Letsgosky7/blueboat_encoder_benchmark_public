"""Command-line entry point for the frozen ConvNeXt-Tiny SONAR benchmark.

Run from project root:
  python -m scripts.convnext_tiny_benchmark
  python -m scripts.convnext_tiny_benchmark --analyze-only
  python -m scripts.convnext_tiny_benchmark --verify-embeddings
  python -m scripts.convnext_tiny_benchmark --retrieval-only

Input: shared positive-pair, bank and metadata files (or saved outputs in
cached modes). Output: model-specific NPZ, CSV and diagnostic figures.
The implementation is shared in src.benchmark.pipeline / reporting.
"""

from src.benchmark.pipeline import main


if __name__ == "__main__":
    main("convnext_tiny")
