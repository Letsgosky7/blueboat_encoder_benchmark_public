"""Compute validity-mask and no-data diagnostics for paired 2D patches.

Input: two co-located patch arrays and their boolean valid-pixel masks.
Output: scalar coverage, central-region and connected no-data metrics.
Usage: imported by scripts.audit_dataset; not a CLI."""

import numpy as np
from scipy.ndimage import label


def largest_nodata_region(mask):
    """
    Fraction of the patch occupied by the largest
    connected no-data region (8-connectivity).
    """
    mask = np.asarray(mask, dtype=bool)
    nodata = ~mask

    labels, n_regions = label(
        nodata,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    if n_regions == 0:
        return 0.0

    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / mask.size)


def central_coverage(mask):
    """
    Coverage in the central 50% of each spatial dimension.
    This central region occupies 25% of the patch area.
    """
    mask = np.asarray(mask, dtype=bool)

    height, width = mask.shape

    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4

    return float(mask[y0:y1, x0:x1].mean())


def audit_positive_pair(patch_a, patch_b, mask_a, mask_b):
    """
    Compute geometric coverage and no-data quality metrics
    for a positive pair of co-located sonar patches.
    """

    patch_a = np.asarray(patch_a)
    patch_b = np.asarray(patch_b)
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    # Validate dimensions
    if (
        patch_a.ndim != 2
        or patch_a.shape != patch_b.shape
        or patch_a.shape != mask_a.shape
        or patch_a.shape != mask_b.shape
    ):
        raise ValueError("Inconsistent patch or mask dimensions.")

    # Validate finite values in valid cells
    if np.any(mask_a & ~np.isfinite(patch_a)):
        raise ValueError(
            "Patch A contains non-finite values in valid cells."
        )

    if np.any(mask_b & ~np.isfinite(patch_b)):
        raise ValueError(
            "Patch B contains non-finite values in valid cells."
        )

    # Global coverage
    coverage_a = float(mask_a.mean())
    coverage_b = float(mask_b.mean())

    joint = mask_a & mask_b
    union = mask_a | mask_b

    joint_coverage = float(joint.mean())

    mask_iou = (
        float(joint.sum() / union.sum())
        if union.any() else 0.0
    )

    # Largest connected no-data region
    largest_nodata_a = largest_nodata_region(mask_a)
    largest_nodata_b = largest_nodata_region(mask_b)

    # Individual central coverage
    central_coverage_a = central_coverage(mask_a)
    central_coverage_b = central_coverage(mask_b)

    # Central joint coverage
    height, width = mask_a.shape

    center = np.s_[
        height // 4: 3 * height // 4,
        width // 4: 3 * width // 4,
    ]

    central_a = mask_a[center]
    central_b = mask_b[center]

    central_joint = central_a & central_b
    central_union = central_a | central_b

    central_joint_coverage = float(central_joint.mean())

    central_mask_iou = (
        float(central_joint.sum() / central_union.sum())
        if central_union.any() else 0.0
    )

    # Return quality metrics
    return {
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "joint_coverage": joint_coverage,
        "mask_iou": mask_iou,
        "largest_nodata_a": largest_nodata_a,
        "largest_nodata_b": largest_nodata_b,
        "central_coverage_a": central_coverage_a,
        "central_coverage_b": central_coverage_b,
        "central_joint_coverage": central_joint_coverage,
        "central_mask_iou": central_mask_iou,
    }
