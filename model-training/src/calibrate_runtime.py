"""
calibrate_runtime.py

Calibrate runtime matching threshold and multi-frame voting using the
trained projection head checkpoint + aligned embedding cache.

Outputs:
  1) Recommended threshold for single-frame verification.
  2) Verification accuracy at that threshold.
  3) Top-1 identification accuracy with 1/3/5-frame averaging.

Usage:
  cd C:\\Users\\kunal\\Desktop\\NHAI\\model-training
  $env:PYTHONUTF8="1"
  python -u src/calibrate_runtime.py `
      --embed_cache data/all_indian/embed_cache_aligned.npz `
      --ckpt checkpoints/finetuned_head.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_io import load_checkpoint
from finetune import FineTuneHead, ProjectionHead  # noqa: E402


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


@torch.no_grad()
def project_embeddings(backbone_embs: np.ndarray, ckpt_path: Path) -> np.ndarray:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    num_classes = int(ckpt["num_classes"])
    model = FineTuneHead(num_classes=num_classes)
    model.projection.load_state_dict(ckpt["projection_state"])
    model.projection.eval()

    x = torch.from_numpy(backbone_embs).float()
    out = []
    bs = 1024
    for i in range(0, len(x), bs):
        out.append(model.projection(x[i : i + bs]).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def build_gallery_probe(
    embs: np.ndarray, labels: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one-shot gallery (1 sample/identity) and probe (remaining samples).
    """
    rng = np.random.default_rng(seed)
    gallery_idx = []
    probe_idx = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        pick = rng.choice(idx, size=1, replace=False)[0]
        gallery_idx.append(pick)
        probe_idx.extend([i for i in idx if i != pick])

    g_idx = np.array(gallery_idx, dtype=np.int64)
    p_idx = np.array(probe_idx, dtype=np.int64)
    return embs[g_idx], labels[g_idx], embs[p_idx], labels[p_idx]


def sweep_threshold(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    probe_embs: np.ndarray,
    probe_labels: np.ndarray,
    steps: int = 2001,
) -> Dict[str, float]:
    """
    For verification-style decision:
      predicted_match = (max_cosine >= threshold)
      true_match      = (top1_label == true_label)

    We optimize balanced accuracy over threshold sweep.
    """
    sims = probe_embs @ gallery_embs.T
    top1_idx = np.argmax(sims, axis=1)
    top1_scores = sims[np.arange(len(sims)), top1_idx]
    top1_labels = gallery_labels[top1_idx]
    gt_match = (top1_labels == probe_labels)

    thresholds = np.linspace(0.0, 1.0, steps)
    best = {
        "threshold": 0.82,
        "balanced_acc": -1.0,
        "acc": -1.0,
        "fpr": 1.0,
        "fnr": 1.0,
    }
    for t in thresholds:
        pred_match = top1_scores >= t
        tp = np.sum(pred_match & gt_match)
        tn = np.sum((~pred_match) & (~gt_match))
        fp = np.sum(pred_match & (~gt_match))
        fn = np.sum((~pred_match) & gt_match)

        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)
        acc = (tp + tn) / max(1, len(gt_match))
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        if bal > best["balanced_acc"]:
            best.update(
                {
                    "threshold": float(t),
                    "balanced_acc": float(bal),
                    "acc": float(acc),
                    "fpr": float(fpr),
                    "fnr": float(fnr),
                }
            )
    # Also report plain top-1 ID accuracy without thresholding.
    best["top1_id_acc"] = float(np.mean(top1_labels == probe_labels))
    return best


def evaluate_multiframe_top1(
    embs: np.ndarray, labels: np.ndarray, n_frames: int, seed: int = 42
) -> float:
    """
    Simulate runtime temporal averaging:
      - 1 gallery template per identity
      - average n_frames probe embeddings of same identity
      - classify by top cosine
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)

    # Build gallery
    gallery_idx = []
    pools = {}
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < (n_frames + 1):
            continue
        g = rng.choice(idx, size=1, replace=False)[0]
        gallery_idx.append(g)
        pools[lbl] = np.array([i for i in idx if i != g], dtype=np.int64)

    if not gallery_idx:
        return 0.0

    gallery_idx = np.array(gallery_idx, dtype=np.int64)
    gallery_embs = embs[gallery_idx]
    gallery_labels = labels[gallery_idx]

    # Probe episodes
    correct = 0
    total = 0
    for lbl, pool in pools.items():
        n_episodes = min(20, len(pool) // n_frames)
        if n_episodes < 1:
            continue
        for _ in range(n_episodes):
            pick = rng.choice(pool, size=n_frames, replace=False)
            probe = np.mean(embs[pick], axis=0, keepdims=True)
            probe = l2_normalize(probe)
            sims = probe @ gallery_embs.T
            pred_lbl = gallery_labels[int(np.argmax(sims))]
            correct += int(pred_lbl == lbl)
            total += 1

    return float(correct / max(1, total))


def main(args: argparse.Namespace) -> None:
    cache_path = Path(args.embed_cache)
    ckpt_path = Path(args.ckpt)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing embed cache: {cache_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    data = np.load(cache_path)
    embs = data["embeddings"]  # (N, n_aug, 512)
    labels = data["labels"]
    clean_backbone = embs[:, 0, :].astype(np.float32)  # clean aligned view
    clean_backbone = l2_normalize(clean_backbone)

    print(f"Loaded cache: {embs.shape}, classes={len(np.unique(labels))}")
    print(f"Loaded ckpt : {ckpt_path}")

    proj_embs = project_embeddings(clean_backbone, ckpt_path)
    proj_embs = l2_normalize(proj_embs)

    g_embs, g_lbls, p_embs, p_lbls = build_gallery_probe(proj_embs, labels, seed=args.seed)
    stats = sweep_threshold(g_embs, g_lbls, p_embs, p_lbls)

    top1_1f = evaluate_multiframe_top1(proj_embs, labels, n_frames=1, seed=args.seed)
    top1_3f = evaluate_multiframe_top1(proj_embs, labels, n_frames=3, seed=args.seed)
    top1_5f = evaluate_multiframe_top1(proj_embs, labels, n_frames=5, seed=args.seed)

    print("\n=== Calibration Results ===")
    print(f"Recommended threshold   : {stats['threshold']:.4f}")
    print(f"Balanced verification   : {stats['balanced_acc']:.4f}")
    print(f"Verification accuracy   : {stats['acc']:.4f}")
    print(f"False accept rate (FPR) : {stats['fpr']:.4f}")
    print(f"False reject rate (FNR) : {stats['fnr']:.4f}")
    print(f"Top-1 ID (no threshold) : {stats['top1_id_acc']:.4f}")
    print("\n=== Temporal Voting (Top-1 ID) ===")
    print(f"1-frame  : {top1_1f:.4f}")
    print(f"3-frame  : {top1_3f:.4f}")
    print(f"5-frame  : {top1_5f:.4f}")

    print("\nUse these in app:")
    print(f"  SIMILARITY_THRESHOLD = {stats['threshold']:.3f}")
    print("  VERIFICATION_WINDOW_FRAMES = 5")
    print("  MATCH_RULE: average embedding over window, then compare.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embed_cache", default="data/all_indian/embed_cache_aligned.npz")
    parser.add_argument("--ckpt", default="checkpoints/finetuned_head.pt")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

