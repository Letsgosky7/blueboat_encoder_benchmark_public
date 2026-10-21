"""Light cross-survey contrastive fine-tuning with realistic augmentation.

The positive is always patch_a vs patch_b from a DIFFERENT BlueBoat survey.
Negatives are distinct, spatially separated 50m footprints. Nearby pairs are
masked from the InfoNCE denominator to avoid teaching false negatives.
All other ResNet50 blocks and all pretrained BN statistics stay frozen.
"""
import random

import numpy as np

from .augmentations import augment
from .data import png_roundtrip, representation
from .splits import geographically_near


def device_for(torch, choice):
    if choice != "auto":
        return torch.device(choice)
    return torch.device("mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))


def build_network(torch, *, tuning_scope="bn_affine"):
    """Pretrained backbone with precisely one of two declared adaptation scopes."""
    from torch import nn
    from torchvision.models import ResNet50_Weights, resnet50
    if tuning_scope not in ("bn_affine", "last_bottleneck"):
        raise ValueError("Unknown ResNet50 tuning scope")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.requires_grad_(False)
    if tuning_scope == "bn_affine":
        # Original experiment: only 6,144 affine BN parameters.
        for name in ("bn1", "bn2", "bn3"):
            getattr(model.layer4[-1], name).requires_grad_(True)
    else:
        # Single predeclared exploratory ablation: all weights in the final
        # layer4 bottleneck, but NO preceding blocks or FC; BN running
        # statistics stay frozen by model.eval() during training.
        model.layer4[-1].requires_grad_(True)
    projection = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(),
                               nn.Linear(256, 128))
    return model, projection


def infer_features(torch, model, images, device, *, batch_size=8):
    from torch.nn import functional as F
    if not images:
        return np.empty((0, 2048), dtype=np.float32)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            gray = np.stack(images[start:start + batch_size]).astype(np.float32)
            x = torch.from_numpy(np.repeat(gray[:, None], 3, axis=1)).to(device)
            z = F.normalize(model((x - mean) / std), dim=1)
            result.append(z.cpu().numpy())
    return np.concatenate(result)


def masked_cross_survey_info_nce(torch, vectors, centers, *, buffer_m=25.0,
                                 temperature=0.2):
    """Symmetric InfoNCE: views [A1..Ab, B1..Bb], with masked near negatives.

    Only (A_i,B_i) are positives. Cross-survey samples from nearby but
    independently numbered pairs are *never* forced apart. An anchor lacking
    any valid negative is excluded from loss; whole uninformative batch raises.
    """
    from torch.nn import functional as F
    if vectors.ndim != 2 or vectors.shape[0] < 4 or vectors.shape[0] % 2:
        raise ValueError("Expected [A..., B...] vectors for >=2 pairs")
    b = vectors.shape[0] // 2
    if len(centers) != b or temperature <= 0:
        raise ValueError("Invalid centers/contrastive temperature")
    z = F.normalize(vectors, p=2, dim=1)
    logits = (z @ z.T) / temperature
    # Exclude self and close or overlapping OTHER pairs (not cross-pass positives).
    near = torch.tensor(
        [[geographically_near(centers[i % b], centers[j % b],
                              margin_m=buffer_m) and i % b != j % b
          for j in range(2 * b)] for i in range(2 * b)],
        dtype=torch.bool, device=vectors.device,
    )
    mask = near | torch.eye(2 * b, dtype=torch.bool, device=vectors.device)
    logits = logits.masked_fill(mask, -1e4)
    idx = torch.arange(2 * b, device=vectors.device)
    targets = (idx + b) % (2 * b)
    num_negatives = (~mask).sum(dim=1) - 1
    informative = num_negatives > 0
    if not bool(informative.any().item()):
        raise ValueError("No geographically distinct negatives in contrastive batch")
    return F.cross_entropy(logits[informative], targets[informative])


def _training_batches(samples, batch_size, rng, buffer_m):
    """Draw batches containing at least two disjoint training pair footprints."""
    pool = rng.permutation(len(samples)).tolist()
    result = []
    while len(pool) >= 2:
        anchor = pool.pop(0)
        distant = next((k for k, idx in enumerate(pool)
                        if not geographically_near(samples[anchor], samples[idx],
                                                   margin_m=buffer_m)), None)
        if distant is None:
            # No useful contrast for this anchor; do not fabricate a negative.
            continue
        batch = [anchor, pool.pop(distant)]
        while pool and len(batch) < batch_size:
            batch.append(pool.pop(0))
        result.append(batch)
    return result


def train_fold(torch, samples, modality, *, fold, epochs, batch_size, lr,
               seed, buffer_m, device, output, tuning_scope="bn_affine"):
    """Train new pretrained initialization for exactly one geographic fold."""
    if len(samples) < 2 or not all(key in s for s in samples
                                  for key in ("patch_a", "patch_b", "mask_a", "mask_b")):
        raise ValueError("Cross-survey paired samples required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, head = build_network(torch, tuning_scope=tuning_scope)
    model, head = model.to(device), head.to(device)
    original = [(
        representation(s["patch_a"], s["mask_a"], modality),
        representation(s["patch_b"], s["mask_b"], modality),
    ) for s in samples]
    fold_path = output / f"fold_{fold:02d}" / modality
    fold_path.mkdir(parents=True, exist_ok=True)
    if modality == "png":
        # Save exactly the PNGs decoded for training, not independently rendered figures.
        png_dir = fold_path / "train_png"
        png_dir.mkdir(parents=True, exist_ok=True)
        for sample, views in zip(samples, original):
            for side, (gray, _valid) in zip(("a", "b"), views):
                _, encoded = png_roundtrip(gray)
                name = f'{sample["pair"]}_{sample["patch_id"]:04d}_{side}.png'
                (png_dir / name).write_bytes(encoded)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad]
        + list(head.parameters()), lr=lr, weight_decay=1e-4,
    )
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    history = []
    for epoch in range(epochs):
        model.eval()  # BN running statistics unchanged; layer4[-1] still receives gradients.
        head.train()
        rng = np.random.default_rng(seed + 1009 * epoch)
        batches = _training_batches(samples, batch_size, rng, buffer_m)
        losses = []
        for chosen in batches:
            left = [augment(*original[i][0], rng) for i in chosen]
            right = [augment(*original[i][1], rng) for i in chosen]
            images = np.stack(left + right).astype(np.float32)
            x = torch.from_numpy(np.repeat(images[:, None], 3, axis=1)).to(device)
            vectors = head(model((x - mean) / std))
            loss = masked_cross_survey_info_nce(
                torch, vectors, [samples[i] for i in chosen], buffer_m=buffer_m,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError(f"Fold {fold} has no usable cross-survey contrastive batches")
        history.append({"fold": fold, "modality": modality, "epoch": epoch + 1,
                        "loss": float(np.mean(losses)), "batches": len(losses),
                        "training_pairs": len(samples)})
        print(f"fold {fold} {modality} epoch {epoch + 1}/{epochs}: "
              f"loss={np.mean(losses):.5f} ({len(losses)} batches)")
    # Preserve the original BN-affine checkpoint schema for all historical runs.
    checkpoint_key = ("trained_last_layer4_block_bn_affine" if tuning_scope == "bn_affine"
                      else "trained_last_layer4_block")
    checkpoint = {
        checkpoint_key: {
            k: v.detach().cpu() for k, v in model.layer4[-1].state_dict().items()
        },
        "projection": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "fold": fold, "modality": modality, "epochs": epochs, "seed": seed,
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "protocol": "cross-survey nominal positives, geographic negatives, purged CV",
    }
    if tuning_scope == "last_bottleneck":
        checkpoint["tuning_scope"] = tuning_scope
        checkpoint["trainable_backbone_parameter_count"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
    torch.save(checkpoint, fold_path / "resnet50_finetuned.pt")
    return model, history
