"""Render human-inspectable galleries of positive and negative geocoded patches.

Input: locally generated patch NPZ archives and dataset-metadata tables.
Output: diagnostic PNG images below outputs/figures/.
Usage: python -m scripts.visualize_dataset or scripts.audit_dataset; functions are also importable."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITIVE_PAIRS = (("pass_a", "pass_b"), ("pass_a", "pass_c"), ("pass_b", "pass_c"))
DISPLAY_PERCENTILES = (1.0, 99.5)


def _normalize_pair(a, b, mask_a=None, mask_b=None):
    """Apply consistent intensity percentiles to a pair, excluding invalid pixels."""
    a = np.asarray(a, dtype=np.float32).copy()
    b = np.asarray(b, dtype=np.float32).copy()

    if mask_a is not None:
        a[~np.asarray(mask_a, dtype=bool)] = np.nan
    if mask_b is not None:
        b[~np.asarray(mask_b, dtype=bool)] = np.nan

    valid_a = a[np.isfinite(a)]
    valid_b = b[np.isfinite(b)]
    values = np.concatenate((valid_a, valid_b))
    if values.size == 0:
        return a, b

    lo, hi = np.percentile(values, DISPLAY_PERCENTILES)
    if hi <= lo:
        hi = lo + 1.0

    def scale(x):
        y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        y[~np.isfinite(x)] = np.nan
        return y

    return scale(a), scale(b)


def _extent(patch_size_m):
    """Return a metric-coordinate plot extent for a square raster patch."""
    half = patch_size_m / 2.0
    return [-half, half, -half, half]


def _draw_patch(ax, patch, title, patch_size_m):
    """Render a geocoded patch without mistaking invalid pixels for seabed texture."""
    ax.imshow(
        np.ma.masked_invalid(patch),
        cmap="gray",
        origin="lower",
        extent=_extent(patch_size_m),
        interpolation="nearest",
    )
    ax.axhline(0, linewidth=0.5, alpha=0.5)
    ax.axvline(0, linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("East offset [m]")
    ax.set_ylabel("North offset [m]")


def _check_positive_archive(path, archive):
    """Assert that a positive-pair archive has the required aligned fields."""
    required = {
        "patch_id", "patch_a", "patch_b", "mask_a", "mask_b", "coverage_a", "coverage_b",
        "joint_coverage", "heading_difference_deg", "look_difference_deg",
        "incidence_difference_deg", "view_class", "patch_size_m", "grid_resolution_m",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"Missing arrays in {path.name}: {missing}")

    n = len(archive["patch_id"])
    for key in required.difference({"patch_size_m", "grid_resolution_m"}):
        if len(archive[key]) != n:
            raise ValueError(f"Inconsistent length for {key} in {path.name}.")

    patch_size_m = float(archive["patch_size_m"])
    grid_resolution_m = float(archive["grid_resolution_m"])
    expected_size = int(round(patch_size_m / grid_resolution_m))
    if archive["patch_a"].shape[1:] != (expected_size, expected_size):
        raise ValueError(f"Unexpected patch shape in {path.name}: {archive['patch_a'].shape[1:]}")
    if archive["patch_b"].shape[1:] != (expected_size, expected_size):
        raise ValueError(f"Unexpected patch shape in {path.name}: {archive['patch_b'].shape[1:]}")


def _positive_galleries(patch_dir, figure_dir, rows_per_page):
    """Generate review grids for the curated matching patch pairs."""
    paths = []

    for survey_a, survey_b in POSITIVE_PAIRS:
        path = patch_dir / f"positive_pairs_{survey_a}_{survey_b}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing positive dataset: {path}")

        with np.load(path) as archive:
            _check_positive_archive(path, archive)
            n = len(archive["patch_id"])
            patch_size_m = float(archive["patch_size_m"])

            for page_start in range(0, n, rows_per_page):
                page_end = min(page_start + rows_per_page, n)
                rows = page_end - page_start
                fig, axes = plt.subplots(rows, 2, figsize=(10, 4.2 * rows), squeeze=False)

                for plot_row, idx in enumerate(range(page_start, page_end)):
                    patch_a, patch_b = _normalize_pair(
                        archive["patch_a"][idx], archive["patch_b"][idx],
                        archive["mask_a"][idx], archive["mask_b"][idx],
                    )
                    coverage_a = float(archive["coverage_a"][idx])
                    coverage_b = float(archive["coverage_b"][idx])
                    patch_id = int(archive["patch_id"][idx])
                    view_class = str(archive["view_class"][idx])

                    _draw_patch(
                        axes[plot_row, 0], patch_a,
                        f"{survey_a.upper()} | patch {patch_id} | coverage {coverage_a:.0%}",
                        patch_size_m,
                    )
                    _draw_patch(
                        axes[plot_row, 1], patch_b,
                        f"{survey_b.upper()} | patch {patch_id} | coverage {coverage_b:.0%}",
                        patch_size_m,
                    )
                    axes[plot_row, 0].text(
                        -patch_size_m / 2 + 1,
                        patch_size_m / 2 - 1,
                        f"{view_class} | joint={float(archive['joint_coverage'][idx]):.0%} | "
                        f"ΔH={float(archive['heading_difference_deg'][idx]):.0f}° | "
                        f"ΔL={float(archive['look_difference_deg'][idx]):.0f}° | "
                        f"ΔI={float(archive['incidence_difference_deg'][idx]):.0f}°",
                        fontsize=8,
                        va="top",
                    )

                page = page_start // rows_per_page + 1
                fig.suptitle(
                    f"Positive benchmark pairs — {survey_a.upper()} vs {survey_b.upper()} — page {page}",
                    fontsize=12,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.98))
                output = figure_dir / f"positive_{survey_a}_{survey_b}_{page:02d}.png"
                fig.savefig(output, dpi=200, bbox_inches="tight")
                plt.close(fig)
                paths.append(output)

    return paths


def _load_patch_banks(patch_dir, surveys):
    """Load independently sampled patch-bank archives for the requested surveys."""
    banks = {}
    for survey in surveys:
        path = patch_dir / f"patch_bank_{survey}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing patch bank: {path}")
        with np.load(path) as archive:
            required = {"patch", "mask", "coverage", "patch_size_m", "grid_resolution_m"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise KeyError(f"Missing arrays in {path.name}: {missing}")
            banks[survey] = {key: np.asarray(archive[key]) for key in required}
    return banks


def _select_evenly(group, n):
    """Choose representative examples distributed across a dataset group."""
    if len(group) <= n:
        return group.reset_index(drop=True)
    indices = np.linspace(0, len(group) - 1, n, dtype=int)
    return group.iloc[indices].reset_index(drop=True)


def _null_galleries(table_dir, patch_dir, figure_dir, examples_per_group):
    """Generate positive-versus-negative galleries to audit null pair construction."""
    metadata_path = table_dir / "null_pair_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing null metadata: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {
        "null_pair_id", "bank_index_a", "bank_index_b", "coverage_a", "coverage_b",
        "center_separation_m", "heading_difference_deg", "look_difference_deg",
        "incidence_difference_deg", "null_type", "survey_a", "survey_b",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise KeyError(f"Missing columns in {metadata_path.name}: {missing}")

    banks = _load_patch_banks(patch_dir, sorted(set(metadata["survey_a"]) | set(metadata["survey_b"])))
    paths = []

    for (null_type, survey_a, survey_b), group in metadata.groupby(["null_type", "survey_a", "survey_b"], sort=True):
        selected = _select_evenly(group.sort_values("null_pair_id"), examples_per_group)
        rows = len(selected)
        fig, axes = plt.subplots(rows, 2, figsize=(10, 4.2 * rows), squeeze=False)

        for plot_row, row in enumerate(selected.itertuples(index=False)):
            index_a = int(row.bank_index_a)
            index_b = int(row.bank_index_b)
            bank_a, bank_b = banks[survey_a], banks[survey_b]

            if not 0 <= index_a < len(bank_a["patch"]):
                raise IndexError(f"Invalid bank index {index_a} for {survey_a}.")
            if not 0 <= index_b < len(bank_b["patch"]):
                raise IndexError(f"Invalid bank index {index_b} for {survey_b}.")

            patch_a, patch_b = _normalize_pair(
                bank_a["patch"][index_a], bank_b["patch"][index_b],
                bank_a["mask"][index_a], bank_b["mask"][index_b],
            )
            patch_size_m = float(bank_a["patch_size_m"])
            if not np.isclose(patch_size_m, float(bank_b["patch_size_m"])):
                raise ValueError(f"Patch-size mismatch for {survey_a}-{survey_b}.")

            _draw_patch(
                axes[plot_row, 0], patch_a,
                f"{survey_a.upper()} | bank {index_a} | coverage {float(row.coverage_a):.0%}",
                patch_size_m,
            )
            _draw_patch(
                axes[plot_row, 1], patch_b,
                f"{survey_b.upper()} | bank {index_b} | coverage {float(row.coverage_b):.0%}",
                patch_size_m,
            )
            axes[plot_row, 0].text(
                -patch_size_m / 2 + 1,
                patch_size_m / 2 - 1,
                f"null {int(row.null_pair_id)} | d={float(row.center_separation_m):.0f} m | "
                f"ΔH={float(row.heading_difference_deg):.0f}° | "
                f"ΔL={float(row.look_difference_deg):.0f}° | "
                f"ΔI={float(row.incidence_difference_deg):.0f}°",
                fontsize=8,
                va="top",
            )

        fig.suptitle(
            f"Null pairs — {null_type} — {survey_a.upper()} vs {survey_b.upper()}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        output = figure_dir / f"null_{null_type}_{survey_a}_{survey_b}.png"
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(output)

    return paths


def visualize_dataset(project_root, positive_rows_per_page=6, null_examples_per_group=6):
    """Create positive and negative galleries and return their output paths."""
    project_root = Path(project_root)
    table_dir = project_root / "outputs" / "tables"
    patch_dir = project_root / "outputs" / "patches"
    figure_dir = project_root / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = _positive_galleries(patch_dir, figure_dir, positive_rows_per_page)
    paths.extend(_null_galleries(table_dir, patch_dir, figure_dir, null_examples_per_group))
    return paths


def plot_patch_mask_diagnostics(patch_dir, figure_dir, patch_ids=(16, 20, 21, 23)):
    """Plot intensities and validity masks for selected B-C positive pairs."""
    patch_dir = Path(patch_dir)
    figure_dir = Path(figure_dir)
    source = patch_dir / "positive_pairs_pass_b_pass_c.npz"
    if not source.exists():
        raise FileNotFoundError(source)

    with np.load(source) as archive:
        index_by_id = {int(pid): i for i, pid in enumerate(archive["patch_id"])}
        selected = [int(pid) for pid in patch_ids if int(pid) in index_by_id]
        if not selected:
            selected = list(index_by_id)[:4]
        if not selected:
            raise ValueError("No B-C pairs available for mask diagnostics.")

        fig, axes = plt.subplots(
            len(selected), 4,
            figsize=(14, 3.5 * len(selected)),
            squeeze=False,
        )
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("red")

        for row, patch_id in enumerate(selected):
            idx = index_by_id[patch_id]
            patch_b = archive["patch_a"][idx]
            patch_c = archive["patch_b"][idx]
            mask_b = archive["mask_a"][idx].astype(bool)
            mask_c = archive["mask_b"][idx].astype(bool)

            values = np.concatenate((
                patch_b[mask_b & np.isfinite(patch_b)],
                patch_c[mask_c & np.isfinite(patch_c)],
            ))
            if values.size == 0:
                raise ValueError(f"No valid intensities for patch {patch_id}")

            lo, hi = np.percentile(values, DISPLAY_PERCENTILES)
            if hi <= lo:
                hi = lo + 1.0

            for col, (patch, mask, survey) in enumerate((
                (patch_b, mask_b, "B"),
                (patch_c, mask_c, "C"),
            )):
                intensity_ax = axes[row, col * 2]
                mask_ax = axes[row, col * 2 + 1]
                intensity_ax.imshow(
                    np.ma.masked_where(~mask | ~np.isfinite(patch), patch),
                    cmap=cmap,
                    vmin=lo,
                    vmax=hi,
                    origin="lower",
                )
                intensity_ax.set_title(f"{survey}{patch_id} - intensity")
                mask_ax.imshow(
                    mask, cmap="gray", vmin=0, vmax=1, origin="lower"
                )
                mask_ax.set_title(
                    f"{survey}{patch_id} - valid mask ({mask.mean():.0%})"
                )

            for ax in axes[row]:
                ax.axis("off")

    fig.suptitle(
        "PASS_B vs PASS_C - Intensity and No-Data Masks",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "patch_mask_diagnostics_BC.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
