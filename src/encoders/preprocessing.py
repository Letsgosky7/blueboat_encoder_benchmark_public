"""Label-free, model-agnostic preprocessing for geocoded sonar patches.

The returned RGB image is in [0, 1]; encoder-specific transforms
(resizing choices, ImageNet/CLIP channel statistics) are applied later.

Input: 2D sonar patch plus a same-shape validity mask; configurable target size and percentiles.
Output: CHW float32 RGB in [0,1] and a resized 2D boolean validity mask.
Usage: prepare_sonar_rgb is imported by all four benchmarks; no CLI.
"""

import numpy as np
from scipy.ndimage import zoom


def prepare_sonar_rgb(
    patch,
    mask,
    *,
    target_size=(224, 224),
    percentiles=(1.0, 99.5),
    nodata_fill="median",
):
    """Return (CHW float32 RGB, HW boolean mask) without changing inputs.

    Robust intensity scaling is independent for each patch (no pair leakage).
    Missing cells are filled for inference but remain identified by the mask.
    Interpolation is mask-aware to avoid blending missing values with sonar.
    """
    patch = np.asarray(patch, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if patch.ndim != 2 or patch.shape != mask.shape:
        raise ValueError("patch and mask must be matching 2D arrays")
    if not mask.any():
        raise ValueError("patch has no valid sonar samples")
    if np.any(mask & ~np.isfinite(patch)):
        raise ValueError("valid sonar cells must contain finite values")
    if len(target_size) != 2 or any(int(v) != v or v <= 0 for v in target_size):
        raise ValueError("target_size must contain two positive integers")
    low_pct, high_pct = percentiles
    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError("percentiles must be increasing within [0, 100]")
    if nodata_fill not in ("median", "midgray"):
        raise ValueError("nodata_fill must be 'median' or 'midgray'")

    lo, hi = np.percentile(patch[mask], percentiles)
    scaled = np.zeros(patch.shape, dtype=np.float32)
    if hi > lo:
        scaled[mask] = np.clip((patch[mask] - lo) / (hi - lo), 0, 1)
    else:
        scaled[mask] = 0.5

    fill = float(np.median(scaled[mask])) if nodata_fill == "median" else 0.5
    target_size = tuple(int(v) for v in target_size)

    if patch.shape == target_size:
        processed = scaled.copy()
        valid = mask.copy()
    else:
        scale = tuple(out / dim for out, dim in zip(target_size, patch.shape))
        # Divide interpolated signal by interpolated validity weights so nodata
        # cells never dilute real acoustic samples at resampling boundaries.
        weights = zoom(mask.astype(np.float32), scale, order=1)
        numerator = zoom(scaled, scale, order=1)
        processed = np.full(target_size, fill, dtype=np.float32)
        np.divide(numerator, weights, out=processed, where=weights > 1e-6)
        # Nearest-neighbor interpolation keeps the validity mask discrete.
        valid = zoom(mask.astype(np.uint8), scale, order=0).astype(bool)

    processed[~valid] = fill
    np.clip(processed, 0, 1, out=processed)
    rgb = np.broadcast_to(processed, (3, *target_size)).copy()
    return rgb, valid
