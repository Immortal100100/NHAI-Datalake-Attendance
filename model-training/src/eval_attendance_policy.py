"""
eval_attendance_policy.py

Evaluate an attendance-style matching policy on top of the current checkpoint:
  - K-shot enrollment gallery per identity (average of K embeddings)
  - M-frame verification averaging
  - Dual decision rule: score threshold + top1-top2 margin threshold

This better reflects real app behavior than plain single-frame top-1.

Also supports open-set evaluation (unknown-person rejection):
  python -u src/eval_attendance_policy.py `
      --embed_cache data/all_indian/embed_cache_aligned.npz `
      --ckpt checkpoints/finetuned_head.pt `
      --k_enroll 5 `
      --m_frames 5 `
      --open_set `
      --known_fraction 0.8

Usage:
  cd C:\\Users\\kunal\\Desktop\\NHAI\\model-training
  $env:PYTHONUTF8="1"
  python -u src/eval_attendance_policy.py `
      --embed_cache data/all_indian/embed_cache_aligned.npz `
      --ckpt checkpoints/finetuned_head.pt `
      --k_enroll 5 `
      --m_frames 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from finetune import FineTuneHead  # noqa: E402


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


@torch.no_grad()
def project_with_ckpt(backbone_embs: np.ndarray, ckpt_path: Path) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = FineTuneHead(num_classes=int(ckpt["num_classes"]))
    model.projection.load_state_dict(ckpt["projection_state"])
    model.projection.eval()

    x = torch.from_numpy(backbone_embs).float()
    out = []
    bs = 1024
    for i in range(0, len(x), bs):
        out.append(model.projection(x[i : i + bs]).cpu().numpy())
    return l2n(np.vstack(out).astype(np.float32))


def build_gallery_probe_indices(
    labels: np.ndarray, k_enroll: int, m_frames: int, seed: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    For each label:
      - first choose k_enroll enrollment samples
      - remaining samples used for probe episodes
    """
    rng = np.random.default_rng(seed)
    gallery_idx: Dict[int, np.ndarray] = {}
    probe_idx: Dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        min_needed = k_enroll + m_frames
        if len(idx) < min_needed:
            continue
        perm = rng.permutation(idx)
        gallery_idx[int(lbl)] = perm[:k_enroll]
        probe_idx[int(lbl)] = perm[k_enroll:]
    return gallery_idx, probe_idx


def build_gallery_templates(
    embs: np.ndarray, gallery_idx: Dict[int, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    labels = sorted(gallery_idx.keys())
    templates = []
    for lbl in labels:
        t = embs[gallery_idx[lbl]].mean(axis=0, keepdims=True)
        templates.append(l2n(t)[0])
    return np.vstack(templates), np.array(labels, dtype=np.int64)


def sample_probe_episodes(
    embs: np.ndarray,
    probe_idx: Dict[int, np.ndarray],
    m_frames: int,
    seed: int,
    max_episodes_per_id: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probes = []
    gt = []
    for lbl, idx in probe_idx.items():
        if len(idx) < m_frames:
            continue
        episodes = min(max_episodes_per_id, len(idx) // m_frames)
        for _ in range(episodes):
            pick = rng.choice(idx, size=m_frames, replace=False)
            avg = embs[pick].mean(axis=0, keepdims=True)
            probes.append(l2n(avg)[0])
            gt.append(lbl)
    return np.vstack(probes), np.array(gt, dtype=np.int64)


def predict_scores(
    probe_embs: np.ndarray, gallery_embs: np.ndarray, gallery_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sims = probe_embs @ gallery_embs.T
    top1_idx = np.argmax(sims, axis=1)
    top1_score = sims[np.arange(len(sims)), top1_idx]
    top1_label = gallery_labels[top1_idx]
    # second-best for margin rule
    sims_sorted = np.sort(sims, axis=1)
    top2_score = sims_sorted[:, -2] if sims.shape[1] > 1 else np.zeros_like(top1_score)
    margin = top1_score - top2_score
    return top1_label, top1_score, margin


def sweep_threshold_and_margin(
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    pred_margin: np.ndarray,
    gt_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Optimize acceptance rule:
      accept if (score >= t_score) and (margin >= t_margin)
      if accepted => must have correct identity
      if rejected => counted wrong for closed-set attendance check-in
    """
    best = {"acc": -1.0, "t_score": 0.0, "t_margin": 0.0}

    score_grid = np.linspace(0.75, 0.999, 180)
    margin_grid = np.linspace(0.0, 0.15, 120)

    for ts in score_grid:
        score_ok = pred_scores >= ts
        for tm in margin_grid:
            accept = score_ok & (pred_margin >= tm)
            correct_accept = accept & (pred_labels == gt_labels)
            acc = np.mean(correct_accept)  # reject is considered miss in closed-set attendance
            if acc > best["acc"]:
                best = {"acc": float(acc), "t_score": float(ts), "t_margin": float(tm)}

    return best


def split_known_unknown_labels(
    labels: np.ndarray,
    k_enroll: int,
    m_frames: int,
    known_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split identities into known (in gallery) and unknown (only probes).
    Only identities with enough samples are considered.
    """
    rng = np.random.default_rng(seed)
    eligible = []
    for lbl in np.unique(labels):
        n = np.sum(labels == lbl)
        if n >= (k_enroll + m_frames):
            eligible.append(int(lbl))
    eligible = np.array(sorted(eligible), dtype=np.int64)
    if len(eligible) < 2:
        raise ValueError("Not enough eligible identities for open-set split.")

    rng.shuffle(eligible)
    n_known = int(round(len(eligible) * known_fraction))
    n_known = max(1, min(n_known, len(eligible) - 1))
    known = np.sort(eligible[:n_known])
    unknown = np.sort(eligible[n_known:])
    return known, unknown


def sweep_open_set_threshold_and_margin(
    known_pred_labels: np.ndarray,
    known_pred_scores: np.ndarray,
    known_pred_margin: np.ndarray,
    known_gt_labels: np.ndarray,
    unknown_pred_scores: np.ndarray,
    unknown_pred_margin: np.ndarray,
) -> Dict[str, float]:
    """
    Optimize open-set rule:
      accept if score>=t_score and margin>=t_margin

    Known success:
      accepted AND predicted identity == GT identity
    Unknown success:
      rejected
    """
    best = {
        "balanced_acc": -1.0,
        "t_score": 0.0,
        "t_margin": 0.0,
        "known_success": 0.0,
        "unknown_reject": 0.0,
        "unknown_accept": 1.0,
    }

    score_grid = np.linspace(0.70, 0.999, 220)
    margin_grid = np.linspace(0.0, 0.20, 160)

    for ts in score_grid:
        known_score_ok = known_pred_scores >= ts
        unknown_score_ok = unknown_pred_scores >= ts
        for tm in margin_grid:
            known_accept = known_score_ok & (known_pred_margin >= tm)
            unknown_accept = unknown_score_ok & (unknown_pred_margin >= tm)

            known_success = np.mean(known_accept & (known_pred_labels == known_gt_labels))
            unknown_reject = np.mean(~unknown_accept)
            bal = 0.5 * (known_success + unknown_reject)

            if bal > best["balanced_acc"]:
                best = {
                    "balanced_acc": float(bal),
                    "t_score": float(ts),
                    "t_margin": float(tm),
                    "known_success": float(known_success),
                    "unknown_reject": float(unknown_reject),
                    "unknown_accept": float(np.mean(unknown_accept)),
                }
    return best


def main(args: argparse.Namespace) -> None:
    cache = Path(args.embed_cache)
    ckpt = Path(args.ckpt)
    if not cache.exists():
        raise FileNotFoundError(cache)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    data = np.load(cache)
    labels = data["labels"].astype(np.int64)
    clean_backbone = data["embeddings"][:, 0, :].astype(np.float32)
    clean_backbone = l2n(clean_backbone)

    embs = project_with_ckpt(clean_backbone, ckpt)

    if not args.open_set:
        gallery_idx, probe_idx = build_gallery_probe_indices(
            labels, args.k_enroll, args.m_frames, args.seed
        )
        gallery_embs, gallery_labels = build_gallery_templates(embs, gallery_idx)
        probe_embs, gt_labels = sample_probe_episodes(
            embs, probe_idx, args.m_frames, args.seed
        )
        pred_labels, pred_scores, pred_margin = predict_scores(
            probe_embs, gallery_embs, gallery_labels
        )

        plain_top1 = float(np.mean(pred_labels == gt_labels))
        best = sweep_threshold_and_margin(pred_labels, pred_scores, pred_margin, gt_labels)

        print("=== Attendance Policy Eval (Closed-set) ===")
        print(f"k_enroll: {args.k_enroll}, m_frames: {args.m_frames}")
        print(f"usable identities: {len(gallery_labels)}")
        print(f"probe episodes: {len(gt_labels)}")
        print(f"Top-1 (no gating): {plain_top1:.4f}")
        print()
        print("Best closed-set check-in policy:")
        print(f"  score_threshold : {best['t_score']:.4f}")
        print(f"  margin_threshold: {best['t_margin']:.4f}")
        print(f"  policy_accuracy : {best['acc']:.4f}")
        print()
        print("Apply in app:")
        print(f"  SIMILARITY_THRESHOLD = {best['t_score']:.3f}")
        print(f"  TOP1_TOP2_MARGIN_MIN = {best['t_margin']:.3f}")
        print(f"  VERIFICATION_WINDOW_FRAMES = {args.m_frames}")
    else:
        known_ids, unknown_ids = split_known_unknown_labels(
            labels=labels,
            k_enroll=args.k_enroll,
            m_frames=args.m_frames,
            known_fraction=args.known_fraction,
            seed=args.seed,
        )
        known_mask = np.isin(labels, known_ids)
        unknown_mask = np.isin(labels, unknown_ids)

        known_embs = embs[known_mask]
        known_labels = labels[known_mask]
        unknown_embs = embs[unknown_mask]
        unknown_labels = labels[unknown_mask]

        gallery_idx, known_probe_idx = build_gallery_probe_indices(
            known_labels, args.k_enroll, args.m_frames, args.seed
        )
        gallery_embs, gallery_labels = build_gallery_templates(known_embs, gallery_idx)
        known_probe_embs, known_gt = sample_probe_episodes(
            known_embs, known_probe_idx, args.m_frames, args.seed
        )

        # For unknown IDs: no gallery enrollment, only probe episodes.
        unknown_probe_idx: Dict[int, np.ndarray] = {}
        for lbl in np.unique(unknown_labels):
            idx = np.where(unknown_labels == lbl)[0]
            if len(idx) >= args.m_frames:
                unknown_probe_idx[int(lbl)] = idx
        unknown_probe_embs, _ = sample_probe_episodes(
            unknown_embs, unknown_probe_idx, args.m_frames, args.seed
        )

        known_pred_lbl, known_pred_score, known_pred_margin = predict_scores(
            known_probe_embs, gallery_embs, gallery_labels
        )
        _, unknown_pred_score, unknown_pred_margin = predict_scores(
            unknown_probe_embs, gallery_embs, gallery_labels
        )

        known_top1 = float(np.mean(known_pred_lbl == known_gt))
        best = sweep_open_set_threshold_and_margin(
            known_pred_labels=known_pred_lbl,
            known_pred_scores=known_pred_score,
            known_pred_margin=known_pred_margin,
            known_gt_labels=known_gt,
            unknown_pred_scores=unknown_pred_score,
            unknown_pred_margin=unknown_pred_margin,
        )

        print("=== Attendance Policy Eval (Open-set) ===")
        print(f"k_enroll: {args.k_enroll}, m_frames: {args.m_frames}")
        print(f"known identities: {len(np.unique(known_labels))}")
        print(f"unknown identities: {len(np.unique(unknown_labels))}")
        print(f"known probe episodes: {len(known_gt)}")
        print(f"unknown probe episodes: {len(unknown_probe_embs)}")
        print(f"Known Top-1 (no gating): {known_top1:.4f}")
        print()
        print("Best open-set check-in policy:")
        print(f"  score_threshold : {best['t_score']:.4f}")
        print(f"  margin_threshold: {best['t_margin']:.4f}")
        print(f"  balanced_acc    : {best['balanced_acc']:.4f}")
        print(f"  known_success   : {best['known_success']:.4f}")
        print(f"  unknown_reject  : {best['unknown_reject']:.4f}")
        print(f"  unknown_accept  : {best['unknown_accept']:.4f}")
        print()
        print("Apply in app:")
        print(f"  SIMILARITY_THRESHOLD = {best['t_score']:.3f}")
        print(f"  TOP1_TOP2_MARGIN_MIN = {best['t_margin']:.3f}")
        print(f"  VERIFICATION_WINDOW_FRAMES = {args.m_frames}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--embed_cache", default="data/all_indian/embed_cache_aligned.npz")
    p.add_argument("--ckpt", default="checkpoints/finetuned_head.pt")
    p.add_argument("--k_enroll", type=int, default=5)
    p.add_argument("--m_frames", type=int, default=5)
    p.add_argument("--open_set", action="store_true",
                   help="Evaluate unknown-person rejection.")
    p.add_argument("--known_fraction", type=float, default=0.8,
                   help="Fraction of identities treated as known in open-set mode.")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

