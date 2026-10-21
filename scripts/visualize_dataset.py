"""Produce galleries of positive and negative geocoded sonar patches.

Run: python -m scripts.visualize_dataset (repository root; no flags).
Input: locally generated patch NPZ and matching metadata CSVs.
Output: diagnostic image files under outputs/figures/ and their paths in stdout."""

from pathlib import Path

from src.analysis.dataset_visualization import visualize_dataset


def main():
    """Generate dataset galleries using the shared visualization module."""
    project_root = Path(__file__).resolve().parents[1]
    paths = visualize_dataset(project_root)
    print(f"Dataset visualization ready: {len(paths)} figures.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
