"""
finetune_e2e.py — End-to-end fine-tuning: backbone + projection head trained together.

WHY THIS BEATS CACHED-EMBEDDING APPROACH:
  Cached embeddings: backbone is frozen → head learns to map fixed 512-D vectors.
    Ceiling: ~0.88-0.90 TAR@FAR (backbone can't adapt to Indian faces)

  End-to-end: gradients flow through both head AND backbone last blocks.
    Backbone features ADAPT to Indian face characteristics.
    Target: 0.93-0.97 TAR@FAR

HOW IT WORKS:
  1. Convert ONNX backbone to PyTorch via onnx2torch
  2. Load projection head weights from previous training checkpoint
  3. Freeze all backbone layers EXCEPT last N parameter tensors
  4. Train on raw aligned images (insightface detector → aligned crop → backbone → head)
  5. Use tiny LR for backbone (1e-5), larger LR for head (1e-3)

INSTALL (one-time):
  pip install onnx2torch

COMMANDS:
  cd C:\\Users\\kunal\\Desktop\\NHAI\\model-training
  $env:PYTHONUTF8="1"

  python -u src/finetune_e2e.py `
      --manifest_path data/all_indian/manifest.csv `
      --processed_dir data/all_indian `
      --onnx_backbone exports/w600k_mbf.onnx `
      --head_ckpt     checkpoints/finetuned_head.pt `
      --epochs 40 `
      --lr_backbone 5e-6 `
      --lr_head     5e-4 `
      --batch_size 64
"""

import os, sys, warnings, argparse, time
warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import multiprocessing
multiprocessing.freeze_support()

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Tuple
from sklearn.metrics import roc_curve
import albumentations as A

# ─── Alignment constants (ArcFace canonical 112×112) ─────────────────────────
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img_bgr: np.ndarray, det_app, size: int = 112) -> np.ndarray:
    """Detect + 5-point align using InsightFace. Falls back to resize."""
    from skimage.transform import SimilarityTransform
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        faces = det_app.get(img_rgb)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            if face.kps is not None:
                tform = SimilarityTransform()
                tform.estimate(face.kps.astype(np.float32), ARCFACE_DST * (size / 112))
                M = tform.params[:2]
                return cv2.warpAffine(img_bgr, M, (size, size),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    except Exception:
        pass
    return cv2.resize(img_bgr, (size, size))


# ─── Dataset ─────────────────────────────────────────────────────────────────

TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05, p=0.8),
    A.ToGray(p=0.1),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.Affine(rotate=(-15, 15), scale=(0.85, 1.15), p=0.4),
    A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=4, p=0.15),
    A.GaussNoise(p=0.1),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 20),
                    hole_width_range=(8, 20), fill=0, p=0.15),
])


class AlignedFaceDataset(Dataset):
    """
    Loads raw images, aligns with InsightFace 5-point detector,
    applies augmentation, and normalizes for the backbone.
    """
    def __init__(self, df: pd.DataFrame, root: Path, det_app,
                 augment: bool = True):
        self.df       = df.reset_index(drop=True)
        self.root     = root
        self.det_app  = det_app
        self.augment  = augment
        self.labels   = torch.from_numpy(df["label"].values.astype(np.int64))
        self.num_classes = int(df["label"].max()) + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.root / row["file"]))
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        aligned = align_face(img, self.det_app)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        if self.augment:
            aligned_rgb = TRAIN_AUG(image=aligned_rgb)["image"]

        # InsightFace normalization → [-1, 1]
        t = aligned_rgb.astype(np.float32) / 255.0
        t = (t - 0.5) / 0.5
        t = torch.from_numpy(t.transpose(2, 0, 1))   # CHW
        return t, self.labels[idx]


# ─── Model ───────────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int,
                 margin: float = 0.35, scale: float = 32.0):
        super().__init__()
        self.scale   = scale
        self.margin  = margin
        self.weight  = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_norm = nn.functional.normalize(x)
        w_norm = nn.functional.normalize(self.weight)
        cos    = x_norm @ w_norm.T
        # Additive angular margin
        theta     = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        one_hot   = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1)
        cos_m     = torch.cos(theta + self.margin)
        logits    = self.scale * (one_hot * cos_m + (1 - one_hot) * cos)
        return logits


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, mid_dim: int = 256, out_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,  mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim, affine=True),
            nn.PReLU(mid_dim),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        return self.net(x)


# ─── Metrics ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_tar_far(model_backbone, model_proj, val_loader,
                    device, target_far: float = 0.01,
                    max_pairs: int = 30_000) -> Tuple[float, float]:
    model_backbone.eval()
    model_proj.eval()
    all_embs, all_labels = [], []
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        with autocast("cuda"):
            feat = model_backbone(imgs)
            emb  = model_proj(feat)
        all_embs.append(emb.float().cpu().numpy())
        all_labels.append(lbls.numpy())

    embs   = np.vstack(all_embs)
    labels = np.concatenate(all_labels)
    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
    embs   = embs / (norms + 1e-8)

    np.random.seed(42)
    n    = len(labels)
    idx_a = np.random.choice(n, max_pairs, replace=True)
    idx_b = np.random.choice(n, max_pairs, replace=True)
    same  = (labels[idx_a] == labels[idx_b])
    sims  = (embs[idx_a] * embs[idx_b]).sum(axis=1)

    fpr, tpr, _ = roc_curve(same.astype(int), sims)
    idx = min(np.searchsorted(fpr, target_far), len(tpr) - 1)

    pos = sims[same].mean()  if same.any()   else 0.0
    neg = sims[~same].mean() if (~same).any() else 0.0
    return float(tpr[idx]), float(pos - neg)


# ─── Training ────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── InsightFace detector
    print("Loading InsightFace detector...")
    from insightface.app import FaceAnalysis
    det_app = FaceAnalysis(name="buffalo_sc",
                           root="exports/insightface_models",
                           allowed_modules=["detection"])
    det_app.prepare(ctx_id=0 if device.type == "cuda" else -1,
                    det_size=(320, 320))
    print("[OK] Detector ready")

    # ── Convert ONNX backbone to PyTorch
    print("Converting ONNX backbone to PyTorch via onnx2torch...")
    try:
        import onnx
        import onnx2torch
        # Load model into memory first — avoids Windows temp-file permission error
        onnx_model = onnx.load(args.onnx_backbone)
        backbone   = onnx2torch.convert(onnx_model)
    except ImportError:
        print("[ERROR] onnx2torch not installed. Run: pip install onnx2torch")
        sys.exit(1)
    backbone = backbone.to(device)

    # Freeze ALL backbone layers first
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Unfreeze last N parameter tensors (robust across converted model names)
    named_params = list(backbone.named_parameters())
    n_unfreeze = max(1, min(args.unfreeze_last_tensors, len(named_params)))
    unfrozen = 0
    for name, param in named_params[-n_unfreeze:]:
        param.requires_grad_(True)
        unfrozen += param.numel()
    if unfrozen == 0:
        raise RuntimeError("No backbone params were unfrozen; cannot run end-to-end tuning.")
    print(f"[OK] Backbone loaded. Unfrozen params: {unfrozen/1e3:.1f}K "
          f"(last {n_unfreeze} tensors)")

    # ── Projection head + ArcFace
    proj = ProjectionHead(512, 256, 128).to(device)

    # Load previous checkpoint if provided
    num_classes = None
    best_tar    = -1.0

    ckpt = None
    if args.head_ckpt and Path(args.head_ckpt).exists():
        ckpt = torch.load(args.head_ckpt, map_location=device, weights_only=True)
        num_classes = ckpt["num_classes"]
        proj.load_state_dict(ckpt["projection_state"])
        best_tar = ckpt.get("tar_at_far_001", -1.0)
        print(f"[OK] Loaded head checkpoint (TAR={best_tar:.4f}, "
              f"classes={num_classes})")

    # ── Data
    df   = pd.read_csv(args.manifest_path)
    root = Path(args.processed_dir)

    if num_classes is None:
        num_classes = int(df["label"].max()) + 1

    # 90/10 split by identity to avoid leakage
    ids = df["label"].unique()
    np.random.seed(42)
    val_ids  = set(np.random.choice(ids, max(1, int(0.1 * len(ids))), replace=False))
    train_df = df[~df["label"].isin(val_ids)]
    val_df   = df[df["label"].isin(val_ids)]

    # Re-map val labels for metric (not for classifier)
    train_ds = AlignedFaceDataset(train_df, root, det_app, augment=True)
    val_ds   = AlignedFaceDataset(val_df,   root, det_app, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size // 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {num_classes}")

    # ── ArcFace head
    head = ArcFaceHead(128, num_classes, margin=0.35, scale=32.0).to(device)
    if ckpt is not None and "head_state" in ckpt:
        try:
            head.load_state_dict(ckpt["head_state"], strict=True)
            print("[OK] Warm-started ArcFace head from checkpoint")
        except Exception as e:
            print(f"[WARN] Could not load head_state, using fresh ArcFace head: {e}")

    # ── Optimizer: tiny LR for backbone, larger for head
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params     = list(proj.parameters()) + list(head.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 1e-4},
        {"params": head_params,     "lr": args.lr_head,     "weight_decay": 5e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = GradScaler("cuda")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_epoch = ckpt.get("epoch", 0) if ckpt is not None else 0
    hdr = (f"{'Epoch':>6} {'Loss':>9} {'Tr.Acc':>8} "
           f"{'TAR@1%':>9} {'Pos-Neg':>9} {'BB_LR':>10} {'HD_LR':>9}")
    print(f"\nEnd-to-end fine-tuning on {device}\n")
    print(hdr); print("-" * len(hdr))

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        proj.train()
        head.train()

        total_loss = total_correct = total_samples = 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                feat   = backbone(imgs)
                emb    = proj(feat)
                logits = head(emb, labels)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(backbone_params + head_params, 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += len(labels)

        scheduler.step()

        train_loss = total_loss  / total_samples
        train_acc  = total_correct / total_samples
        bb_lr = optimizer.param_groups[0]["lr"]
        hd_lr = optimizer.param_groups[1]["lr"]

        marker = ""
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            tar, gap = compute_tar_far(backbone, proj, val_loader, device)
            if tar > best_tar:
                best_tar   = tar
                best_epoch = epoch
                torch.save({
                    "epoch":            epoch,
                    "backbone_state":   backbone.state_dict(),
                    "projection_state": proj.state_dict(),
                    "head_state":       head.state_dict(),
                    "tar_at_far_001":   tar,
                    "pos_neg_gap":      gap,
                    "num_classes":      num_classes,
                }, ckpt_dir / "finetuned_e2e.pt")
                marker = "  <-- best"
            elapsed = time.time() - t0
            print(f"{epoch:>6} {train_loss:>9.4f} {train_acc:>8.3f} "
                  f"{tar:>9.4f} {gap:>9.4f} {bb_lr:>10.2e} {hd_lr:>9.2e}"
                  f"  [{elapsed:.0f}s]{marker}", flush=True)
        else:
            print(f"{epoch:>6} {train_loss:>9.4f} {train_acc:>8.3f} "
                  f"{'--':>9} {'--':>9} {bb_lr:>10.2e} {hd_lr:>9.2e}", flush=True)

    print(f"\nTraining complete.")
    print(f"  Best epoch: {best_epoch}  TAR@FAR=1%: {best_tar:.4f}")
    print(f"  >0.80 TAR = good  |  >0.90 = excellent  |  >0.95 = target")
    print(f"\nNext: export the e2e checkpoint")
    print(f"  python src/export_finetuned.py --e2e checkpoints/finetuned_e2e.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest_path", default="data/all_indian/manifest.csv")
    parser.add_argument("--processed_dir", default="data/all_indian")
    parser.add_argument("--onnx_backbone", default="exports/w600k_mbf.onnx")
    parser.add_argument("--head_ckpt",     default="checkpoints/finetuned_head.pt",
                        help="Previous projection head checkpoint to warm-start from")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--lr_backbone",   type=float, default=5e-6,
                        help="Very small LR for backbone unfrozen layers")
    parser.add_argument("--lr_head",       type=float, default=5e-4,
                        help="Larger LR for projection head + ArcFace")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--unfreeze_last_tensors", type=int, default=40,
                        help="How many final backbone parameter tensors to unfreeze")
    args = parser.parse_args()
    main(args)
