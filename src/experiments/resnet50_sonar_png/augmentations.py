"""Conservative independent augmentations of real cross-survey paired views."""
import numpy as np
from scipy.ndimage import shift as image_shift

def augment(gray, valid, rng, *, max_translation=3):
    """Augment one half of an existing cross-survey pair without moving its ID.

    This mitigates exact mask-shape shortcut cues; it does not guarantee their
    absence. Do not rotate/flip arbitrarily: sonar geometry has meaning.
    """
    gray = np.asarray(gray, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if gray.shape != valid.shape or gray.ndim != 2 or not valid.any():
        raise ValueError("Invalid augmentation image/mask")
    out = gray.copy()
    available = gray[valid]
    # Use independent local-looking intensity noise instead of fixed NoData fill.
    if (~valid).any():
        out[~valid] = rng.choice(available, size=int((~valid).sum()), replace=True)
    out = (out - float(np.mean(out))) * rng.uniform(0.85, 1.15) + float(np.mean(out))
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    out = out + rng.normal(0, 0.025, size=out.shape)
    tx, ty = rng.integers(-max_translation, max_translation + 1, size=2)
    if tx or ty:
        out = image_shift(out, (ty, tx), order=1, mode="reflect", prefilter=False)
    return np.clip(out, 0, 1).astype(np.float32)
