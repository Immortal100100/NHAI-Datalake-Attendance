"""
align_and_precompute.py — Proper InsightFace alignment + augmented embedding cache.

WHY THIS IS NEEDED:
  The previous precompute used Haar cascade → rough bounding box → backbone.
  Result: same-identity cosine similarity = 0.21 (terrible).

  InsightFace w600k_mbf was trained with 5-point landmark alignment:
    - Detect face with RetinaFace/SCRFD
    - Extract 5 landmarks (left eye, right eye, nose, left mouth, right mouth)
    - Affine-warp face to canonical 112×112 position
  With proper alignment, same-identity cosine similarity should be >0.6.

HOW IT WORKS:
  1. Uses insightface.app.FaceAnalysis to detect + align faces properly
  2. Runs aligned face through w600k_mbf backbone (already downloaded)
  3. Applies N_AUG augmentations before detection for diversity
  4. Saves (N_images, N_AUG, 512) cache

COMMANDS:
  cd model-training
  $env:PYTHONUTF8="1"

  # Build properly aligned cache (~15-20 min on GPU)
  python -u src/align_and_precompute.py `
    --manifest_path data/all_indian/manifest.csv `
    --processed_dir data/all_indian `
    --cache_path    data/all_indian/embed_cache_aligned.npz `
    --n_aug 10

  # Then train with the new cache
  python -u src/finetune.py `
    --embed_cache data/all_indian/embed_cache_aligned.npz `
    --epochs 60 --lr 5e-2
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

import cv2
import numpy as np
from pathlib import Path


# ─── InsightFace alignment reference points ──────────────────────────────────
# Standard 112×112 crop destination landmarks (ArcFace standard)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face_5pt(img_bgr: np.ndarray, landmarks_5: np.ndarray,
                   size: int = 112) -> np.ndarray:
    """
    Affine-warp a face to the canonical 112×112 ArcFace position
    using 5 facial landmarks.
    """
    from skimage.transform import SimilarityTransform
    tform = SimilarityTransform()
    tform.estimate(landmarks_5, ARCFACE_DST * (size / 112))
    M = tform.params[:2]
    return cv2.warpAffine(img_bgr, M, (size, size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _aug_pipeline():
    """Albumentations augmentations applied AFTER alignment."""
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        A.ToGray(p=0.1),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Affine(rotate=(-10, 10), p=0.3),
        A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=4, p=0.15),
        A.GaussNoise(p=0.1),
    ])


def preprocess_for_backbone(img_rgb: np.ndarray) -> np.ndarray:
    """Normalize 112×112 RGB image for InsightFace w600k_mbf."""
    img = cv2.resize(img_rgb, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5            # → [-1, 1]
    return img.transpose(2, 0, 1)      # HWC → CHW


def main(args):
    import pandas as pd
    import onnxruntime as ort

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"[SKIP] Cache exists: {cache_path}  shape={d['embeddings'].shape}")
        return

    # ── Load InsightFace detector
    print("Loading InsightFace detector (downloads ~20MB on first run)...")
    try:
        from insightface.app import FaceAnalysis
        det_app = FaceAnalysis(
            name="buffalo_sc",            # small model: detector + no recognizer
            root=str(Path("exports/insightface_models")),
            allowed_modules=["detection"],
        )
        det_app.prepare(ctx_id=0, det_size=(320, 320))
        use_insightface = True
        print("[OK] InsightFace detector ready (5-point landmarks)")
    except Exception as e:
        print(f"[WARN] InsightFace detector failed: {e}")
        print("  Falling back to Haar cascade (lower quality)")
        use_insightface = False

    # ── Load w600k_mbf backbone
    print("Loading backbone ONNX...")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx_backbone, sess_options=so, providers=providers)
    inp_name = sess.get_inputs()[0].name
    active = sess.get_providers()[0]
    print(f"[OK] Backbone on: {active}")

    df   = pd.read_csv(args.manifest_path)
    root = Path(args.processed_dir)
    n    = len(df)

    aug       = _aug_pipeline()
    n_aug     = args.n_aug
    all_embs  = []
    all_labels = []
    skipped   = 0

    if use_insightface:
        haar = None
    else:
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    print(f"Processing {n} images × {n_aug} augmentations "
          f"({'InsightFace 5pt' if use_insightface else 'Haar cascade'} alignment)...")
    t0 = time.time()

    for img_idx, (_, row) in enumerate(df.iterrows()):
        img_bgr = cv2.imread(str(root / row["file"]))
        if img_bgr is None:
            img_bgr = np.zeros((112, 112, 3), dtype=np.uint8)

        # ── Align face
        aligned_bgr = None

        if use_insightface:
            img_rgb_det = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            try:
                faces = det_app.get(img_rgb_det)
                if faces:
                    # Pick largest detected face
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    if face.kps is not None:
                        aligned_bgr = align_face_5pt(img_bgr, face.kps.astype(np.float32))
                    else:
                        # fallback: use bbox crop
                        x1, y1, x2, y2 = face.bbox.astype(int)
                        aligned_bgr = cv2.resize(img_bgr[y1:y2, x1:x2], (112, 112))
            except Exception:
                pass

        if aligned_bgr is None:
            if haar is not None:
                gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                    pad = int(0.15 * max(w, h))
                    x1, y1 = max(0, x-pad), max(0, y-pad)
                    x2, y2 = min(img_bgr.shape[1], x+w+pad), min(img_bgr.shape[0], y+h+pad)
                    aligned_bgr = cv2.resize(img_bgr[y1:y2, x1:x2], (112, 112))
            if aligned_bgr is None:
                aligned_bgr = cv2.resize(img_bgr, (112, 112))
                skipped += 1

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # ── Generate N_AUG versions (clean + augmented)
        versions = [aligned_rgb.copy()]
        for _ in range(n_aug - 1):
            versions.append(aug(image=aligned_rgb)["image"])

        # ── Run backbone on all versions at once
        batch = np.stack([preprocess_for_backbone(v) for v in versions])
        embs  = sess.run(None, {inp_name: batch})[0]   # (n_aug, 512)
        all_embs.append(embs.astype(np.float32))
        all_labels.append(int(row["label"]))

        if (img_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (img_idx + 1) * (n - img_idx - 1)
            print(f"  {img_idx+1:>6}/{n}  [{100*(img_idx+1)/n:.0f}%]  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min  "
                  f"(no_face={skipped})", flush=True)

    embeddings = np.stack(all_embs, axis=0)          # (N, n_aug, 512)
    labels     = np.array(all_labels, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, labels=labels)
    size_mb  = cache_path.stat().st_size / 1e6
    elapsed  = time.time() - t0
    print(f"\n[OK] Saved {cache_path}")
    print(f"     shape={embeddings.shape}  size={size_mb:.1f}MB  "
          f"time={elapsed/60:.1f}min  no_face={skipped}")

    # ── Quick quality check on clean embeddings
    clean = embeddings[:, 0, :]                      # (N, 512)
    norms = np.linalg.norm(clean, axis=1, keepdims=True)
    clean_n = clean / (norms + 1e-8)

    np.random.seed(42)
    idx_a = np.random.choice(len(labels), 10_000, replace=True)
    idx_b = np.random.choice(len(labels), 10_000, replace=True)
    same  = (labels[idx_a] == labels[idx_b])
    sims  = (clean_n[idx_a] * clean_n[idx_b]).sum(axis=1)

    pos_mean = sims[same].mean()  if same.any()   else 0.0
    neg_mean = sims[~same].mean() if (~same).any() else 0.0
    print(f"\nEmbedding quality check (backbone, no projection):")
    print(f"  Pos cosine sim: {pos_mean:.4f}  (same identity pairs)")
    print(f"  Neg cosine sim: {neg_mean:.4f}  (different identity pairs)")
    print(f"  Pos-Neg gap   : {pos_mean - neg_mean:.4f}")
    if pos_mean > 0.4:
        print(f"  [OK] Alignment looks good! (target: >0.4 for same-identity)")
    else:
        print(f"  [WARN] Low same-identity similarity — check face images quality")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Proper InsightFace-aligned embedding cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx_backbone",  default="exports/w600k_mbf.onnx")
    parser.add_argument("--manifest_path",  default="data/all_indian/manifest.csv")
    parser.add_argument("--processed_dir",  default="data/all_indian")
    parser.add_argument("--cache_path",     default="data/all_indian/embed_cache_aligned.npz")
    parser.add_argument("--n_aug",          type=int, default=8)
    args = parser.parse_args()
    main(args)
