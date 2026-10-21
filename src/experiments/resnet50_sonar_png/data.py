"""BlueBoat-only matched cross-survey pairs, with shared SONAR/PNG preprocessing.

Only the 25 existing quality-controlled XTF-derived pairs are used. No patch
banks, isolated anchors, external XTFs or new pair labels are introduced.
Correspondence is GPS/geometry-based and remains *nominal*, not verified GT.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.encoders.preprocessing import prepare_sonar_rgb

EXPECTED_PAIRS = {"pass_a-pass_b": 4, "pass_a-pass_c": 5, "pass_b-pass_c": 16}
PATCH_SIZE_M = 50.0
MIN_COVERAGE = 0.60
MIN_JOINT_COVERAGE = 0.50


def png_roundtrip(gray):
    """Encode/decode a genuine grayscale 8-bit PNG from the same SONAR tensor."""
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or not np.isfinite(gray).all() or np.any((gray < 0) | (gray > 1)):
        raise ValueError("Expected finite HW grayscale image in [0,1]")
    buf = io.BytesIO()
    Image.fromarray(np.rint(gray * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    encoded = buf.getvalue()
    with Image.open(io.BytesIO(encoded)) as im:
        decoded = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return decoded, encoded


def representation(patch, mask, modality):
    """Common per-observation scaling: only 8-bit quantization differs."""
    rgb, valid = prepare_sonar_rgb(patch, mask)
    if modality == "sonar":
        return rgb[0].copy(), valid
    if modality == "png":
        decoded, _ = png_roundtrip(rgb[0])
        return decoded, valid
    raise ValueError("modality must be 'sonar' or 'png'")


def load_blueboat_pairs(root):
    """Validate metadata vs the original three pair NPZs and return all 25 pairs.

    Each record is one cross-pass pair. The same canonical 50-m geocoded
    patch footprint has two survey observations, each retaining its own mask.
    The 25 pair IDs and all original coverage thresholds remain unchanged.
    """
    root = Path(root)
    meta = pd.read_csv(root / "outputs/tables/positive_pair_metadata.csv")
    required = {"pair", "patch_id", "midpoint_east", "midpoint_north", "patch_size_m"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing pair metadata fields: {sorted(required - set(meta.columns))}")
    if (len(meta) != 25 or meta.duplicated(["pair", "patch_id"]).any()
            or meta["pair"].value_counts().to_dict() != EXPECTED_PAIRS):
        raise ValueError("The canonical 25 positive-pair roster has changed")
    geo = meta[["midpoint_east", "midpoint_north", "patch_size_m"]].to_numpy(float)
    if not np.isfinite(geo).all() or not np.allclose(geo[:, 2], PATCH_SIZE_M):
        raise ValueError("Invalid nominal positive-pair geography or footprint")
    lookup = {(str(x.pair), int(x.patch_id)): x for x in meta.itertuples(index=False)}
    samples = []
    for survey_pair, expected_count in EXPECTED_PAIRS.items():
        survey_a, survey_b = survey_pair.split("-")
        path = root / "outputs/patches" / f"positive_pairs_{survey_a}_{survey_b}.npz"
        with np.load(path, allow_pickle=False) as npz:
            ids = np.asarray(npz["patch_id"])
            if len(ids) != expected_count or len(set(map(int, ids))) != len(ids):
                raise ValueError(f"{survey_pair}: incorrect pair IDs/count")
            if any(npz[key].shape != (expected_count, 200, 200)
                   for key in ("patch_a", "patch_b", "mask_a", "mask_b")):
                raise ValueError(f"{survey_pair}: invalid original patch dimensions")
            if not np.isclose(float(npz["patch_size_m"]), PATCH_SIZE_M):
                raise ValueError(f"{survey_pair}: unexpected patch size")
            for i, patch_id in enumerate(ids):
                key = (survey_pair, int(patch_id))
                if key not in lookup:
                    raise ValueError(f"Unrecognized positive ID {key}")
                row = lookup[key]
                center = np.array([float(row.midpoint_east), float(row.midpoint_north)])
                if ("center_east" in npz.files and "center_north" in npz.files and
                        not np.allclose([float(npz["center_east"][i]),
                                         float(npz["center_north"][i])], center,
                                        atol=1e-3, rtol=0)):
                    raise ValueError(f"Geographic mismatch in {key}")
                patches = (npz["patch_a"][i], npz["patch_b"][i])
                masks = (npz["mask_a"][i].astype(bool), npz["mask_b"][i].astype(bool))
                for p, m in zip(patches, masks):
                    if m.mean() + 1e-8 < MIN_COVERAGE or not np.isfinite(p[m]).all():
                        raise ValueError(f"Invalid source quality for {key}")
                if (masks[0] & masks[1]).mean() + 1e-8 < MIN_JOINT_COVERAGE:
                    raise ValueError(f"Insufficient original joint coverage for {key}")
                samples.append({
                    "pair": survey_pair, "patch_id": int(patch_id),
                    "east": center[0], "north": center[1], "patch_size_m": PATCH_SIZE_M,
                    "patch_a": patches[0].copy(), "patch_b": patches[1].copy(),
                    "mask_a": masks[0].copy(), "mask_b": masks[1].copy(),
                })
    if len(samples) != 25 or len({(s["pair"], s["patch_id"]) for s in samples}) != 25:
        raise ValueError("Incomplete or duplicated positive-pair archive")
    return samples
