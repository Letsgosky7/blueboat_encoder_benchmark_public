"""Build the canonical XTF-derived geometry, positive patches and negative banks.

Run: python -m scripts.build_dataset (from the repository root; no flags).
Input: configs/local.yaml with dataset_root, and four .xtf files in dataset_root/SSS-XTF/.
Output: outputs/tables/*.csv/*.npz, outputs/patches/*.npz and outputs/manifest.json."""

from pathlib import Path

import yaml

from src.data.dataset_builder import DatasetBuilder


def main():
    """Read the local dataset path and invoke the complete deterministic builder."""
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "configs" / "local.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    builder = DatasetBuilder(project_root, config["dataset_root"])
    builder.build_all()


if __name__ == "__main__":
    main()
