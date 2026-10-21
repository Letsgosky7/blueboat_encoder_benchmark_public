"""LEGACY diagnostic — Construct a historical within-site and isolated-control negative-pair benchmark.

Invocation: python scripts/legacy/22_build_null_patch_dataset.py, after restoring legacy paths/prerequisites.
Input: 21_positive_patch_metadata.csv, 16_seabed_geometry_*.npz and source XTF.
Output: 22_patch_bank_*.npz, 22_null_pair_metadata.csv and summary tables.
Not used by the canonical scripts.build_dataset pipeline; standalone execution
may require correcting historical config paths or recovering upstream artifacts.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import yaml

from src.data.xtf_reader import read_xtf
from src.patches.extraction import NullPatchConfig, build_patch_bank, select_control_negatives, select_within_site_negatives


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
PATCH_DIR = PROJECT_ROOT / "outputs" / "patches"
with open(PROJECT_ROOT / "configs" / "local.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

XTF_DIR = Path(config["dataset_root"]) / "SSS-XTF"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}
POSITIVE_PAIRS = (("pass_a", "pass_b"), ("pass_a", "pass_c"), ("pass_b", "pass_c"))
CONFIG = NullPatchConfig()
REUSE_EXISTING_BANKS = True
GEOMETRY_KEYS = (
    "east", "north", "heading_deg", "in_analysis_leg", "first_bottom_port", "first_bottom_stbd",
    "high_confidence_port", "high_confidence_stbd", "canonical_slant_axis_port", "canonical_slant_axis_stbd",
)
ANCHOR_KEYS = (
    "anchor_id", "side", "east", "north", "heading_deg", "look_azimuth_deg",
    "incidence_from_vertical_deg", "high_confidence",
)


def load_geometry(name):
    """Load the necessary geocoded per-ping geometry arrays."""
    path = TABLE_DIR / f"16_seabed_geometry_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry: {path}")
    with np.load(path) as archive:
        missing = [key for key in GEOMETRY_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"Missing geometry arrays in {path.name}: {missing}")
        return {key: np.asarray(archive[key]) for key in GEOMETRY_KEYS}


def load_anchors(name):
    """Read historical anchors with valid positional fields."""
    path = TABLE_DIR / f"17_seabed_anchors_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing anchors: {path}")
    with np.load(path) as archive:
        missing = [key for key in ANCHOR_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"Missing anchor arrays in {path.name}: {missing}")
        return pd.DataFrame({key: np.asarray(archive[key]) for key in ANCHOR_KEYS})


def create_patch_bank(name):
    """Build historical non-colocated geocoded image patches for one survey."""
    survey = read_xtf(XTF_DIR / SURVEYS[name])
    metadata, arrays = build_patch_bank(survey, load_geometry(name), load_anchors(name), name, CONFIG)
    np.savez_compressed(PATCH_DIR / f"22_patch_bank_{name}.npz", **arrays)
    del survey, arrays
    gc.collect()
    return metadata


def main():
    """Run this historical data-generation or inspection stage."""
    positive_path = TABLE_DIR / "21_positive_patch_metadata.csv"
    if not positive_path.exists():
        raise FileNotFoundError(f"Missing positive metadata: {positive_path}")
    positive_meta = pd.read_csv(positive_path)

    bank_metadata_path = TABLE_DIR / "22_patch_bank_metadata.csv"
    bank_paths_exist = all((PATCH_DIR / f"22_patch_bank_{name}.npz").exists() for name in SURVEYS)

    if REUSE_EXISTING_BANKS and bank_metadata_path.exists() and bank_paths_exist:
        bank_metadata = pd.read_csv(bank_metadata_path)
        banks = {name: bank_metadata.loc[bank_metadata["survey"] == name].copy().reset_index(drop=True) for name in SURVEYS}
        empty = [name for name, bank in banks.items() if bank.empty]
        if empty:
            raise RuntimeError(f"Existing bank metadata is incomplete for: {empty}")
    else:
        banks = {name: create_patch_bank(name) for name in SURVEYS}
        bank_metadata = pd.concat(banks.values(), ignore_index=True)
        bank_metadata.to_csv(bank_metadata_path, index=False)

    all_null = []
    for a, b in POSITIVE_PAIRS:
        positives = positive_meta.loc[positive_meta["pair"] == f"{a}-{b}"].copy()
        negatives = select_within_site_negatives(a, b, banks[a], banks[b], positives, CONFIG)
        all_null.append(negatives)
        print(f"{a}-{b}: within-site negatives={len(negatives)}")

    for target in ("pass_a", "pass_b", "pass_c"):
        n_pairs = min(CONFIG.control_pairs_per_survey, len(banks["control"]) * len(banks[target]))
        negatives = select_control_negatives(target, banks["control"], banks[target], n_pairs)
        usage = negatives["bank_index_a"].value_counts()
        all_null.append(negatives)
        print(f"control-{target}: negatives={len(negatives)}, control-used={usage.size}/{len(banks['control'])}, uses={usage.min()}-{usage.max()}")

    null_meta = pd.concat(all_null, ignore_index=True)
    null_meta.insert(0, "null_pair_id", np.arange(1, len(null_meta) + 1, dtype=np.int64))
    null_meta.to_csv(TABLE_DIR / "22_null_pair_metadata.csv", index=False)

    summary = null_meta.groupby(["null_type", "survey_a", "survey_b"], dropna=False).agg(
        pairs=("null_pair_id", "count"), median_center_separation_m=("center_separation_m", "median"),
        median_heading_difference_deg=("heading_difference_deg", "median"),
        median_look_difference_deg=("look_difference_deg", "median"),
        median_incidence_difference_deg=("incidence_difference_deg", "median"),
    ).reset_index()
    summary.to_csv(TABLE_DIR / "22_null_patch_summary.csv", index=False)

    print(f"Null pairs: {len(null_meta)}")
    print(f"Saved: {TABLE_DIR / '22_null_pair_metadata.csv'}")


if __name__ == "__main__":
    main()
