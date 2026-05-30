"""
finetune.py — Proper fine-tuning of InsightFace w600k_mbf backbone on Indian faces.

WHY THE PREVIOUS VERSION WAS BAD:
  - Ran backbone ONCE on clean images → same fixed 512-D vector every epoch
  - Zero augmentation diversity ever reached the training signal
  - Shallow head (Linear only) with very limited capacity
  - ArcFace margin=0.5/scale=64 tuned for 10M+ identities, terrible for 235
  - Classification accuracy is the wrong metric for face recognition

WHAT THIS VERSION FIXES:
  1. Augmented cache: backbone runs N_AUG=10 times per image with random
     augmentation → 10× diverse training signal, same speed benefit
  2. Better head: Linear(512→256)→PReLU→Dropout→Linear(256→128)→L2
  3. ArcFace margin=0.35, scale=32 — properly tuned for small-scale data
  4. SGD + momentum (standard for ArcFace / metric learning, not AdamW)
  5. Linear LR warmup + cosine decay
  6. Verification metric: TAR@FAR=0.01 (cosine similarity ROC) — the real metric
  7. PK-balanced sampler: P identities × K samples per batch

COMMANDS:
  cd model-training
  $env:PYTHONUTF8="1"

  # Step 1: Build augmented embedding cache (~30 min, runs ONCE)
  python -u src/align_and_precompute.py `
    --manifest_path data/all_indian/manifest.csv `
    --processed_dir data/all_indian `
    --cache_path    data/all_indian/embed_cache_aligned.npz `
    --n_aug 10

  # Step 2: Train head on aligned cache (~5-15 min on RTX 3060)
  python -u src/finetune.py `
    --embed_cache data/all_indian/embed_cache_aligned.npz `
    --epochs 60 --lr 5e-2 --batch_size 256

  # Step 3: Export + deploy
  python src/export_finetuned.py --step fuse
  python src/export_finetuned.py --step quantize
  conda run -n nhai-export python src/export_finetuned.py --step tflite
  python src/export_finetuned.py --step deploy
"""

import os
import sys
import math
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_io import load_checkpoint
from model import ArcFaceHead

# ─── Config ──────────────────────────────────────────────────────────────────

CFG: Dict = {
    "onnx_backbone":  "exports/w600k_mbf.onnx",
    "manifest_path":  "data/all_indian/manifest.csv",
    "processed_dir":  "data/all_indian",
    "embed_cache":    "data/all_indian/embed_cache_aligned.npz",
    "checkpoint_dir": "checkpoints",
    # Precompute
    "n_aug":          10,    # augmented versions per image
    "infer_batch":    64,    # backbone inference batch size
    # Head
    "backbone_dim":   512,
    "embed_dim":      128,
    # ArcFace — tuned for 235 classes (~15K images, ~65/class)
    "margin":         0.35,  # 0.5 is for 10M+ IDs; 0.3-0.4 for small-scale
    "scale":          32.0,  # lower scale = softer boundary = better for small data
    # Sampler
    "P":              32,    # identities per batch
    "K":              8,     # samples per identity (batch = P*K = 256)
    # Training
    "epochs":         60,
    "lr":             5e-2,  # SGD peak LR (will warmup then cosine)
    "warmup_epochs":  5,
    "weight_decay":   5e-4,
    "momentum":       0.9,
    "val_split":      0.15,
    "seed":           42,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Augmented Precompute ─────────────────────────────────────────────────────

def _aug_pipeline():
    """Albumentations pipeline applied before backbone inference."""
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        A.ToGray(p=0.1),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=0.05, p=0.4),
        A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=4, p=0.15),
        A.GaussNoise(p=0.1),
    ])


def precompute_augmented_embeddings(
    onnx_path:     str,
    manifest_path: str,
    processed_dir: str,
    cache_path:    str,
    n_aug:         int,
    infer_batch:   int = 64,
) -> None:
    """
    Run each image through the backbone N_AUG times with random augmentations.
    Saves array of shape (N_images, N_AUG, 512) to disk.

    During training, a random augmented version is picked per sample each epoch,
    giving 10× more diverse training signal at no extra training cost.
    """
    import onnxruntime as ort
    import cv2
    import pandas as pd

    if Path(cache_path).exists():
        d = np.load(cache_path)
        shape = d["embeddings"].shape
        print(f"[SKIP] Cache exists: {cache_path}  shape={shape}  "
              f"({shape[0]} images × {shape[1]} augs × {shape[2]}-D)")
        return

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = max(1, os.cpu_count() // 2)
    # Use GPU if onnxruntime-gpu is installed, otherwise fall back to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    active_provider = sess.get_providers()[0]
    print(f"Backbone inference on: {active_provider}")
    inp_name = sess.get_inputs()[0].name

    df   = pd.read_csv(manifest_path)
    root = Path(processed_dir)
    n    = len(df)

    aug = _aug_pipeline()

    all_embeds = []   # list of (n_aug, 512) arrays
    all_labels = []

    print(f"Precomputing {n_aug} augmented embeddings per image "
          f"({n} images × {n_aug} = {n * n_aug:,} total inferences)...")
    t0 = time.time()

    for img_idx, (_, row) in enumerate(df.iterrows()):
        img_bgr = cv2.imread(str(root / row["file"]))
        if img_bgr is None:
            img_bgr = np.zeros((112, 112, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (112, 112))

        aug_imgs = []
        aug_imgs.append(img_rgb.copy())                  # version 0: clean
        for _ in range(n_aug - 1):                       # versions 1..N-1: augmented
            aug_imgs.append(aug(image=img_rgb)["image"])

        batch_np = np.stack(aug_imgs, axis=0).astype(np.float32) / 255.0
        batch_np = (batch_np - 0.5) / 0.5                # InsightFace norm: [-1, 1]
        batch_np = batch_np.transpose(0, 3, 1, 2)         # NHWC -> NCHW

        embeds = sess.run(None, {inp_name: batch_np})[0]  # (n_aug, 512)
        all_embeds.append(embeds.astype(np.float32))
        all_labels.append(int(row["label"]))

        if (img_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (img_idx + 1) * (n - img_idx - 1)
            print(f"  {img_idx+1:>5}/{n}  [{100*(img_idx+1)/n:.0f}%]  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min", flush=True)

    embeddings = np.stack(all_embeds, axis=0)           # (N, n_aug, 512)
    labels     = np.array(all_labels, dtype=np.int64)   # (N,)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, labels=labels)
    size_mb = Path(cache_path).stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"[OK] Saved {cache_path}  shape={embeddings.shape}  "
          f"{size_mb:.1f} MB  ({elapsed/60:.1f} min)")


# ─── Augmented Embedding Dataset ─────────────────────────────────────────────

class AugEmbedDataset(Dataset):
    """
    Loads pre-computed augmented embeddings.
    Each __getitem__ randomly picks one of the N_AUG augmented versions,
    providing online diversity without re-running the backbone.
    """

    def __init__(
        self,
        embeddings: np.ndarray,   # (N, n_aug, 512)
        labels:     np.ndarray,   # (N,)
        is_train:   bool = True,
    ) -> None:
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels     = torch.from_numpy(labels).long()
        self.n_aug      = embeddings.shape[1]
        self.is_train   = is_train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_train:
            aug_idx = np.random.randint(self.n_aug)   # random aug version
        else:
            aug_idx = 0                                # clean version for val
        return self.embeddings[idx, aug_idx], self.labels[idx]

    @property
    def num_classes(self) -> int:
        return int(self.labels.max().item()) + 1

    def label_to_indices(self) -> Dict[int, List[int]]:
        """Return {label: [idx, ...]} mapping."""
        mapping: Dict[int, List[int]] = {}
        for i, lbl in enumerate(self.labels.tolist()):
            mapping.setdefault(lbl, []).append(i)
        return mapping


# ─── PK Balanced Sampler ─────────────────────────────────────────────────────

class PKSampler(Sampler):
    """
    Yields batches of P identities × K samples each.
    Each batch contains exactly P*K samples from P distinct identities,
    which ensures every batch has both easy and hard negatives for ArcFace.
    """

    def __init__(self, dataset: AugEmbedDataset, P: int, K: int) -> None:
        self.lbl2idx = dataset.label_to_indices()
        self.labels  = list(self.lbl2idx.keys())
        self.P       = P
        self.K       = K
        n_batches    = len(self.labels) // P
        self._length = n_batches * P * K

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        np.random.shuffle(self.labels)
        for start in range(0, len(self.labels) - self.P + 1, self.P):
            batch_labels = self.labels[start: start + self.P]
            batch_indices = []
            for lbl in batch_labels:
                indices = self.lbl2idx[lbl]
                chosen  = np.random.choice(indices, size=self.K, replace=len(indices) < self.K)
                batch_indices.extend(chosen.tolist())
            yield from batch_indices


# ─── Improved Projection Head ─────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection: 512 -> 256 -> 128 with PReLU + Dropout.

    Significantly more capacity than the original single Linear layer.
    BN uses learnable affine=True for better feature normalization.
    """

    def __init__(
        self,
        in_dim:   int = CFG["backbone_dim"],
        mid_dim:  int = 256,
        out_dim:  int = CFG["embed_dim"],
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,  mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim, affine=True),
            nn.PReLU(mid_dim),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


# ─── Full Training Model ──────────────────────────────────────────────────────

class FineTuneHead(nn.Module):
    """ProjectionHead + ArcFaceHead. Receives pre-computed backbone features."""

    def __init__(self, num_classes: int, margin: float = CFG["margin"], scale: float = CFG["scale"]) -> None:
        super().__init__()
        self.projection = ProjectionHead()
        self.head       = ArcFaceHead(
            embedding_dim = CFG["embed_dim"],
            num_classes   = num_classes,
            margin        = margin,
            scale         = scale,
        )

    def forward(
        self,
        features: torch.Tensor,
        labels:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.projection(features)
        if labels is not None:
            return self.head(emb, labels)
        return emb


# ─── Verification Metric (TAR @ FAR) ─────────────────────────────────────────

@torch.no_grad()
def compute_verification_metric(
    model: FineTuneHead,
    val_loader: DataLoader,
    target_far: float = 0.01,
    max_pairs:  int   = 20_000,
) -> Tuple[float, float]:
    """
    Compute TAR (True Accept Rate) at a given FAR (False Accept Rate)
    using cosine similarity between all pairs of validation embeddings.

    Returns (TAR@FAR, mean_pos_sim - mean_neg_sim) — higher is better.
    TAR@FAR=0.01 of >0.80 indicates a well-trained face recognition model.
    """
    model.eval()
    all_embs   = []
    all_labels = []
    for feats, lbls in val_loader:
        feats = feats.to(device)
        embs  = model.projection(feats).cpu()
        all_embs.append(embs)
        all_labels.extend(lbls.tolist())

    embs   = torch.cat(all_embs, dim=0).numpy()       # (N, 128)
    labels = np.array(all_labels)

    # Sample pairs
    np.random.seed(0)
    n = len(labels)
    idx_a = np.random.choice(n, max_pairs, replace=True)
    idx_b = np.random.choice(n, max_pairs, replace=True)
    same  = (labels[idx_a] == labels[idx_b])

    sims  = (embs[idx_a] * embs[idx_b]).sum(axis=1)   # cosine similarity (L2-normed)

    # TAR @ FAR — threshold where FPR = target_far
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(same.astype(int), sims)
    idx   = np.searchsorted(fpr, target_far)
    idx   = min(idx, len(tpr) - 1)
    tar   = float(tpr[idx])

    pos_mean = float(sims[same].mean())  if same.any()  else 0.0
    neg_mean = float(sims[~same].mean()) if (~same).any() else 0.0
    return tar, pos_mean - neg_mean


# ─── LR Schedule helpers ─────────────────────────────────────────────────────

def _warmup_cosine_lr(epoch: int, epochs: int, warmup: int) -> float:
    """Linear warmup + cosine annealing LR multiplier."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Training ─────────────────────────────────────────────────────────────────

def finetune(args) -> None:
    """Fine-tune projection head on augmented cached embeddings."""
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    cache_path = args.embed_cache
    if not Path(cache_path).exists():
        print(f"[ERROR] Cache not found: {cache_path}")
        print("  Run first:  python -u src/align_and_precompute.py "
              "--cache_path data/all_indian/embed_cache_aligned.npz --n_aug 10")
        sys.exit(1)

    data       = np.load(cache_path)
    embeddings = data["embeddings"]   # (N, n_aug, 512)
    labels_np  = data["labels"]       # (N,)
    n_aug      = embeddings.shape[1]
    print(f"Loaded cache: {embeddings.shape}  "
          f"({len(np.unique(labels_np))} classes, {n_aug} aug versions/image)")

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels_np))
    train_idx, val_idx = train_test_split(
        idx, test_size=CFG["val_split"], stratify=labels_np,
        random_state=CFG["seed"]
    )

    train_ds = AugEmbedDataset(embeddings[train_idx], labels_np[train_idx], is_train=True)
    val_ds   = AugEmbedDataset(embeddings[val_idx],   labels_np[val_idx],   is_train=False)

    P = min(args.P, train_ds.num_classes)
    K = args.K
    pk_sampler  = PKSampler(train_ds, P=P, K=K)
    train_loader = DataLoader(train_ds, batch_sampler=None,
                              batch_size=P * K, sampler=pk_sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    num_classes = train_ds.num_classes
    print(f"Classes: {num_classes} | PK={P}×{K} | "
          f"Train batches: {len(train_loader)} | Val samples: {len(val_ds)}")

    model  = FineTuneHead(num_classes=num_classes, margin=args.margin, scale=args.scale).to(device)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"Trainable params: {n_params/1e3:.1f} K")

    # SGD with momentum — standard for ArcFace / face recognition
    optimizer = optim.SGD(params, lr=args.lr, momentum=CFG["momentum"],
                          weight_decay=CFG["weight_decay"], nesterov=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = GradScaler("cuda") if device.type == "cuda" else None

    ckpt_dir = Path(CFG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_tar   = -1.0
    best_gap   = -1.0
    best_epoch = 0

    # ── Resume from checkpoint if requested
    start_epoch = 1
    resume_raw = (getattr(args, "resume", "") or "").strip()
    resume_path = Path(resume_raw) if resume_raw else None
    if resume_path is not None and resume_path.is_file():
        ckpt = load_checkpoint(resume_path, map_location=device)
        if "projection_state" not in ckpt or "head_state" not in ckpt:
            raise KeyError(
                f"{resume_path} is not a finetune.py head checkpoint.\n"
                "  Use: --resume checkpoints/finetuned_head.pt\n"
                "  (not checkpoints/best_model.pt from older training)"
            )
        model.projection.load_state_dict(ckpt["projection_state"])
        model.head.load_state_dict(ckpt["head_state"])
        best_tar   = ckpt.get("tar_at_far_001", -1.0)
        best_gap   = ckpt.get("pos_neg_gap",    -1.0)
        best_epoch = ckpt.get("epoch", 0)
        start_epoch = best_epoch + 1
        print(f"[RESUME] Loaded {resume_path}  (best so far: epoch={best_epoch}, TAR={best_tar:.4f})")
        print(f"         Continuing from epoch {start_epoch} with lr={args.lr}")

    hdr = (f"{'Epoch':>6} {'Loss':>9} {'Tr.Acc':>8} "
           f"{'TAR@1%':>9} {'Pos-Neg':>9} {'LR':>9}")
    print(f"\nTraining on {device}  |  Metric: TAR@FAR=1%\n")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ── LR schedule
        lr_mult = _warmup_cosine_lr(epoch - 1, args.epochs, CFG["warmup_epochs"])
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_mult

        # ── Train
        model.train()
        total_loss = total_correct = total_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels   = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with autocast("cuda"):
                    logits = model(features, labels)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(features, labels)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += len(labels)

        train_loss = total_loss  / total_samples
        train_acc  = total_correct / total_samples
        cur_lr     = optimizer.param_groups[0]["lr"]

        # ── Evaluate with TAR@FAR metric (every 5 epochs to save time)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            tar, gap = compute_verification_metric(model, val_loader)
            marker = ""
            if tar > best_tar:
                best_tar   = tar
                best_gap   = gap
                best_epoch = epoch
                torch.save({
                    "epoch":            int(epoch),
                    "projection_state": model.projection.state_dict(),
                    "head_state":       model.head.state_dict(),
                    "tar_at_far_001":   float(tar),
                    "pos_neg_gap":      float(gap),
                    "num_classes":      int(num_classes),
                    "n_aug":            int(n_aug),
                }, ckpt_dir / "finetuned_head.pt")
                marker = "  <-- best"
            print(f"{epoch:>6} {train_loss:>9.4f} {train_acc:>8.3f} "
                  f"{tar:>9.4f} {gap:>9.4f} {cur_lr:>9.2e}{marker}", flush=True)
        else:
            print(f"{epoch:>6} {train_loss:>9.4f} {train_acc:>8.3f} "
                  f"{'--':>9} {'--':>9} {cur_lr:>9.2e}", flush=True)

    print(f"\nTraining complete.")
    print(f"  Best epoch: {best_epoch}  TAR@FAR=1%: {best_tar:.4f}  "
          f"Pos-Neg gap: {best_gap:.4f}")
    print(f"  >0.80 TAR = good  |  >0.90 = excellent")
    print(f"\nNext: export fine-tuned model")
    print(f"  python src/export_finetuned.py --step fuse")
    print(f"  python src/export_finetuned.py --step quantize")
    print(f"  conda run -n nhai-export python src/export_finetuned.py --step tflite")
    print(f"  python src/export_finetuned.py --step deploy")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Fine-tune InsightFace backbone on Indian face data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--precompute",    action="store_true",
                        help="Run backbone N_AUG times per image and cache embeddings")
    parser.add_argument("--onnx_backbone", default=CFG["onnx_backbone"])
    parser.add_argument("--manifest_path", default=CFG["manifest_path"])
    parser.add_argument("--processed_dir", default=CFG["processed_dir"])
    parser.add_argument("--embed_cache",   default=CFG["embed_cache"])
    parser.add_argument("--n_aug",         type=int,   default=CFG["n_aug"],
                        help="Number of augmented versions per image (precompute only)")
    parser.add_argument("--epochs",        type=int,   default=CFG["epochs"])
    parser.add_argument("--lr",            type=float, default=CFG["lr"])
    parser.add_argument("--P",             type=int,   default=CFG["P"],
                        help="Identities per batch (PK sampler)")
    parser.add_argument("--K",             type=int,   default=CFG["K"],
                        help="Samples per identity per batch (PK sampler)")
    parser.add_argument("--margin",        type=float, default=CFG["margin"],
                        help="ArcFace angular margin (default 0.35)")
    parser.add_argument("--scale",         type=float, default=CFG["scale"],
                        help="ArcFace scale/temperature (default 32.0)")
    parser.add_argument("--seed",          type=int,   default=CFG["seed"])
    parser.add_argument("--resume",        default="",
                        help="Path to checkpoint to resume from (optional)")
    args = parser.parse_args()

    if args.precompute:
        precompute_augmented_embeddings(
            onnx_path     = args.onnx_backbone,
            manifest_path = args.manifest_path,
            processed_dir = args.processed_dir,
            cache_path    = args.embed_cache,
            n_aug         = args.n_aug,
            infer_batch   = CFG["infer_batch"],
        )
    else:
        finetune(args)
