"""
eval_backbone.py — Compare backbone (512-D) vs projection head (128-D) TAR@FAR.

Run: python eval_backbone.py
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from sklearn.metrics import roc_curve
from pathlib import Path


def compute_tar_far(embeddings, labels, target_far=0.01, max_pairs=50_000):
    """TAR at a given FAR using cosine similarity between random pairs."""
    np.random.seed(42)
    n = len(labels)
    idx_a = np.random.choice(n, max_pairs, replace=True)
    idx_b = np.random.choice(n, max_pairs, replace=True)
    same  = (labels[idx_a] == labels[idx_b])
    sims  = (embeddings[idx_a] * embeddings[idx_b]).sum(axis=1)

    fpr, tpr, _ = roc_curve(same.astype(int), sims)
    idx = min(np.searchsorted(fpr, target_far), len(tpr) - 1)

    pos_mean = sims[same].mean()  if same.any()   else 0.0
    neg_mean = sims[~same].mean() if (~same).any() else 0.0
    return float(tpr[idx]), float(pos_mean), float(neg_mean)


# ── Load backbone clean embeddings (shape: N, n_aug, 512) -> take clean version
cache = Path("data/merged/embed_cache_aug.npz")
if not cache.exists():
    # Fall back to single-version cache
    cache = Path("data/merged/embed_cache.npz")
    data = np.load(cache)
    backbone_embs = data["embeddings"]  # (N, 512)
else:
    data = np.load(cache)
    embs_aug = data["embeddings"]       # (N, n_aug, 512)
    backbone_embs = embs_aug[:, 0, :]  # clean version

labels = data["labels"]

# L2-normalise backbone embeddings (backbone output may not be unit-normalised)
norms = np.linalg.norm(backbone_embs, axis=1, keepdims=True)
backbone_norm = backbone_embs / (norms + 1e-8)

print("=" * 60)
print("Backbone (512-D, no projection):")
tar, pos, neg = compute_tar_far(backbone_norm, labels)
print(f"  TAR@FAR=1%  : {tar:.4f}")
print(f"  Pos sim     : {pos:.4f}")
print(f"  Neg sim     : {neg:.4f}")
print(f"  Pos-Neg gap : {pos - neg:.4f}")
print()

# ── Load projection head and compare
ckpt_path = Path("checkpoints/finetuned_head.pt")
if ckpt_path.exists():
    from model import ArcFaceHead
    from finetune import ProjectionHead, FineTuneHead

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    num_classes = ckpt["num_classes"]

    model = FineTuneHead(num_classes)
    model.projection.load_state_dict(ckpt["projection_state"])
    model.eval()

    with torch.no_grad():
        t = torch.from_numpy(backbone_norm).float()
        projected = []
        for i in range(0, len(t), 256):
            projected.append(model.projection(t[i:i+256]).numpy())
        proj_embs = np.vstack(projected)

    print("Projection head (128-D, fine-tuned):")
    tar2, pos2, neg2 = compute_tar_far(proj_embs, labels)
    print(f"  TAR@FAR=1%  : {tar2:.4f}")
    print(f"  Pos sim     : {pos2:.4f}")
    print(f"  Neg sim     : {neg2:.4f}")
    print(f"  Pos-Neg gap : {pos2 - neg2:.4f}")
    print()

    print("=" * 60)
    if tar > tar2:
        diff = tar - tar2
        print(f"[!] Backbone BEATS projection head by {diff:.4f}")
        print("    The 512->128 compression is HURTING performance.")
        print("    Recommendation: Use backbone 512-D embeddings directly.")
    else:
        diff = tar2 - tar
        print(f"[OK] Projection head beats backbone by {diff:.4f}")

print()
print("Interpretation:")
print("  TAR@FAR=1% > 0.90  = Excellent (production ready)")
print("  TAR@FAR=1% > 0.80  = Good")
print("  TAR@FAR=1% > 0.60  = Marginal")
print("  TAR@FAR=1% < 0.60  = Poor — needs improvement")
