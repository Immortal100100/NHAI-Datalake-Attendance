"""
train.py — Main training loop for ArcFaceMobileFaceNet.

Features:
  - AdamW optimiser + CosineAnnealingLR scheduler
  - Mixed-precision (AMP) training to fit in 6 GB VRAM
  - TAR@FAR verification metric on the validation set
  - Checkpoints best spine weights to checkpoints/best_model.pt
  - Logs per-epoch loss, accuracy, TAR@FAR to console and CSV
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress FutureWarnings from torch.amp (fired once per DataLoader worker on Windows)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from model import ArcFaceMobileFaceNet

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG: Dict = {
    # Data
    "manifest_path":   "data/processed/manifest.csv",
    "processed_dir":   "data/processed",
    "batch_size":      32,
    # Windows multiprocessing: num_workers > 0 requires __main__ guard.
    # Using 0 here for maximum compatibility; set to 4 on Linux/macOS.
    "num_workers":     0,
    "val_split":       0.15,
    "seed":            42,
    # Model
    "embedding_dim":   128,
    "arcface_margin":  0.5,
    "arcface_scale":   64.0,
    # Training
    "epochs":          30,
    "lr":              1e-3,
    "weight_decay":    5e-4,
    "lr_min":          1e-5,       # CosineAnnealingLR floor
    # Verification metric
    "tar_far_thresh":  1e-3,       # FAR @ which TAR is reported
    # Paths
    "checkpoint_dir":  "checkpoints",
    "log_csv":         "checkpoints/training_log.csv",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Verification Metric ─────────────────────────────────────────────────────

def compute_tar_at_far(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    far_target: float = CONFIG["tar_far_thresh"],
) -> Tuple[float, float]:
    """
    Compute TAR (True Acceptance Rate) at a given FAR (False Acceptance Rate).

    Strategy:
      1. Build genuine (same-identity) and impostor (different-identity) pair scores.
      2. Sweep thresholds to find the cosine distance threshold at which FAR ≤ far_target.
      3. Report TAR at that threshold.

    Returns (TAR, threshold).
    """
    n = len(labels)
    # Pairwise cosine similarity (vectorised — fast enough for val set ≤ 5k)
    sim_matrix = cosine_similarity(embeddings)

    genuine_scores:  List[float] = []
    impostor_scores: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    if not genuine_scores or not impostor_scores:
        return 0.0, 0.0

    genuine_scores  = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    thresholds      = np.linspace(-1.0, 1.0, 1000)

    best_tar   = 0.0
    best_thresh = 0.0

    for thresh in thresholds:
        far = (impostor_scores >= thresh).mean()
        tar = (genuine_scores  >= thresh).mean()
        if far <= far_target:
            if tar > best_tar:
                best_tar    = tar
                best_thresh = thresh

    return best_tar, best_thresh


# ─── Evaluation Pass ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      ArcFaceMobileFaceNet,
    val_loader: torch.utils.data.DataLoader,
    criterion:  nn.CrossEntropyLoss,
) -> Tuple[float, float, float]:
    """
    Run a full validation pass.

    Returns (val_loss, val_acc_top1, TAR@FAR).
    """
    model.eval()
    total_loss   = 0.0
    total_correct = 0
    total_samples = 0

    all_embeddings: List[np.ndarray] = []
    all_labels:     List[int]         = []

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            logits     = model(images, labels)
            loss       = criterion(logits, labels)
            embeddings = model.get_embedding(images)

        total_loss    += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)

        all_embeddings.append(embeddings.cpu().float().numpy())
        all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = total_loss / total_samples
    val_acc  = total_correct / total_samples

    # Subsample for TAR/FAR — cap at 500 to keep O(n²) pairs fast on CPU
    emb_arr  = np.vstack(all_embeddings)
    lbl_arr  = np.array(all_labels)
    if len(lbl_arr) > 500:
        idx     = np.random.choice(len(lbl_arr), 500, replace=False)
        emb_arr = emb_arr[idx]
        lbl_arr = lbl_arr[idx]

    tar, _ = compute_tar_at_far(emb_arr, lbl_arr, CONFIG["tar_far_thresh"])
    return val_loss, val_acc, tar


# ─── Training Loop ───────────────────────────────────────────────────────────

def train() -> None:
    """Full training pipeline: setup -> train -> checkpoint best model."""
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ── Data
    print("Loading data...")
    train_loader, val_loader, num_classes = build_dataloaders(
        manifest_path = CONFIG["manifest_path"],
        root_dir      = CONFIG["processed_dir"],
        batch_size    = CONFIG["batch_size"],
        num_workers   = CONFIG["num_workers"],
        val_split     = CONFIG["val_split"],
        seed          = CONFIG["seed"],
    )
    print(f"  Classes: {num_classes} | "
          f"Train batches: {len(train_loader)} | "
          f"Val batches:   {len(val_loader)}")

    # ── Model
    model = ArcFaceMobileFaceNet(
        num_classes   = num_classes,
        embedding_dim = CONFIG["embedding_dim"],
        margin        = CONFIG["arcface_margin"],
        scale         = CONFIG["arcface_scale"],
    ).to(device)

    # ── Optimiser & scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = CONFIG["epochs"],
        eta_min = CONFIG["lr_min"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler("cuda")  # AMP mixed precision

    # ── Checkpoint dir
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Log CSV header
    log_path = Path(CONFIG["log_csv"])
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,tar_at_far,lr,elapsed_s\n")

    best_tar      = -1.0
    best_epoch    = -1

    print(f"\nStarting training on {device} for {CONFIG['epochs']} epochs\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'TAR@FAR':>8} {'LR':>9}")
    print("-" * 75)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        t0           = time.time()
        train_loss   = 0.0
        train_correct = 0
        train_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(images, labels)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_samples += len(labels)

            if (batch_idx + 1) % 50 == 0:
                cur_loss = train_loss / train_samples
                cur_acc  = train_correct / train_samples
                print(f"  [{epoch}/{CONFIG['epochs']}] "
                      f"step {batch_idx+1}/{len(train_loader)} — "
                      f"loss {cur_loss:.4f} | acc {cur_acc:.3f}", end="\r")

        scheduler.step()

        train_loss_avg = train_loss / train_samples
        train_acc      = train_correct / train_samples

        val_loss, val_acc, tar = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} {train_loss_avg:>11.4f} {train_acc:>10.3f} "
              f"{val_loss:>10.4f} {val_acc:>9.3f} {tar:>8.4f} {cur_lr:>9.2e}")

        # Log to CSV
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss_avg:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},{tar:.6f},"
                f"{cur_lr:.2e},{elapsed:.1f}\n"
            )

        # Checkpoint best backbone (spine only — not the ArcFace head)
        if tar > best_tar:
            best_tar   = tar
            best_epoch = epoch
            ckpt_path  = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch":         epoch,
                "backbone_state": model.backbone.state_dict(),
                "tar_at_far":    tar,
                "val_acc":       val_acc,
                "config":        CONFIG,
            }, ckpt_path)
            print(f"  [OK] New best saved — TAR@FAR={tar:.4f} (epoch {epoch})")

    print(f"\nTraining complete. Best TAR@FAR={best_tar:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    # Required on Windows: multiprocessing spawn needs __main__ guard
    import multiprocessing
    multiprocessing.freeze_support()
    train()
