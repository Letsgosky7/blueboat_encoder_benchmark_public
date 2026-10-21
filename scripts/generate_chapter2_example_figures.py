from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = PROJECT_ROOT / "outputs" / "patches"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_patch(patch, mask):
    patch = np.asarray(patch, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)

    patch[~mask] = np.nan
    valid = patch[np.isfinite(patch)]

    if valid.size == 0:
        return patch

    lo, hi = np.percentile(valid, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0

    patch = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
    patch[~mask] = np.nan
    return patch


def load_pair(npz_path, patch_id):
    with np.load(npz_path) as data:
        patch_ids = data["patch_id"]
        idx = int(np.where(patch_ids == patch_id)[0][0])

        result = {
            "patch_a": data["patch_a"][idx],
            "patch_b": data["patch_b"][idx],
            "mask_a": data["mask_a"][idx].astype(bool),
            "mask_b": data["mask_b"][idx].astype(bool),
            "coverage_a": float(data["coverage_a"][idx]),
            "coverage_b": float(data["coverage_b"][idx]),
            "joint_coverage": float(data["joint_coverage"][idx]),
            "heading_diff": float(data["heading_difference_deg"][idx]),
        }
    return result


def render_sonar_pairs(ab, bc, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    fig.suptitle("Georeferenced SONAR pairs | BlueBoat", fontsize=18, fontweight="bold")

    pairs = [
        ("Pass A — coverage {:.1f}%".format(ab["coverage_a"] * 100),
         normalize_patch(ab["patch_a"], ab["mask_a"])),
        ("Pass B — coverage {:.1f}%".format(ab["coverage_b"] * 100),
         normalize_patch(ab["patch_b"], ab["mask_b"])),
        ("Pass B — coverage {:.1f}%".format(bc["coverage_a"] * 100),
         normalize_patch(bc["patch_a"], bc["mask_a"])),
        ("Pass C — coverage {:.1f}%".format(bc["coverage_b"] * 100),
         normalize_patch(bc["patch_b"], bc["mask_b"])),
    ]

    sonar_cmap = plt.get_cmap("gray").copy()
    sonar_cmap.set_bad(color="0.5")

    for ax, (title, patch) in zip(axes.flat, pairs):
        ax.imshow(
            np.ma.masked_invalid(patch),
            cmap=sonar_cmap,
            origin="upper",
            interpolation="nearest"
        )
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.text(
        0.03, 0.69,
        "A–B · ID 1\nHeading Δ {:.1f}°\nJoint coverage {:.1f}%".format(
            ab["heading_diff"], ab["joint_coverage"] * 100
        ),
        rotation=90, va="center", ha="center", fontsize=12
    )

    fig.text(
        0.03, 0.24,
        "B–C · ID 10\nHeading Δ {:.1f}°\nJoint coverage {:.1f}%".format(
            bc["heading_diff"], bc["joint_coverage"] * 100
        ),
        rotation=90, va="center", ha="center", fontsize=12
    )

    fig.text(
        0.5, 0.03,
        "50 × 50 m patch · original grid 200 × 200 · 0.25 m/cell",
        ha="center", fontsize=12
    )

    plt.tight_layout(rect=(0.07, 0.06, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_masks(ab, bc, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    fig.suptitle("Validity masks | BlueBoat", fontsize=18, fontweight="bold")

    masks = [
        ("Pass A mask — coverage {:.1f}%".format(ab["coverage_a"] * 100), ab["mask_a"]),
        ("Pass B mask — coverage {:.1f}%".format(ab["coverage_b"] * 100), ab["mask_b"]),
        ("Pass B mask — coverage {:.1f}%".format(bc["coverage_a"] * 100), bc["mask_a"]),
        ("Pass C mask — coverage {:.1f}%".format(bc["coverage_b"] * 100), bc["mask_b"]),
    ]

    for ax, (title, mask) in zip(axes.flat, masks):
        img = np.where(mask, 1.0, 0.0)
        ax.imshow(
            img,
            cmap="gray",
            origin="upper",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.text(
        0.03, 0.69,
        "A–B · ID 1\nHeading Δ {:.1f}°\nJoint coverage {:.1f}%".format(
            ab["heading_diff"], ab["joint_coverage"] * 100
        ),
        rotation=90, va="center", ha="center", fontsize=12
    )

    fig.text(
        0.03, 0.24,
        "B–C · ID 10\nHeading Δ {:.1f}°\nJoint coverage {:.1f}%".format(
            bc["heading_diff"], bc["joint_coverage"] * 100
        ),
        rotation=90, va="center", ha="center", fontsize=12
    )

    fig.text(
        0.5, 0.03,
        "50 × 50 m patch · original grid 200 × 200 · 0.25 m/cell",
        ha="center", fontsize=12
    )

    plt.tight_layout(rect=(0.07, 0.06, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ab = load_pair(PATCH_DIR / "positive_pairs_pass_a_pass_b.npz", patch_id=1)
    bc = load_pair(PATCH_DIR / "positive_pairs_pass_b_pass_c.npz", patch_id=10)

    render_sonar_pairs(ab, bc, FIG_DIR / "blueboat_sonar_pairs_en.png")
    render_masks(ab, bc, FIG_DIR / "blueboat_validity_masks_en.png")

    print("Saved:")
    print(FIG_DIR / "blueboat_sonar_pairs_en.png")
    print(FIG_DIR / "blueboat_validity_masks_en.png")


if __name__ == "__main__":
    main()