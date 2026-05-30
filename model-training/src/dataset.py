"""
dataset.py — IndianFaceDataset with demographic-balanced loading and
outdoor lighting augmentation for MobileFaceNet fine-tuning.

Supported dataset formats:
  - FairFace  — CSV columns: file, race, gender, age
  - IMFDB     — CSV columns: file, identity, lighting, pose
  - Generic   — CSV columns: file, label (integer class ID)
"""

import os
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG: Dict = {
    "raw_dir":       "data/raw",
    "processed_dir": "data/processed",
    "csv_path":      "data/processed/manifest.csv",
    "input_size":    112,
    "batch_size":    32,
    "num_workers":   4,
    "val_split":     0.15,
    "seed":          42,
    # Pixel normalisation matching ImageNet (used in most face models)
    "mean":          (0.5, 0.5, 0.5),
    "std":           (0.5, 0.5, 0.5),
}

# ─── Augmentation Pipelines ──────────────────────────────────────────────────

def build_train_transform(input_size: int = CONFIG["input_size"]) -> A.Compose:
    """Aggressive pipeline simulating harsh NHAI outdoor site conditions."""
    return A.Compose([
        # Geometry
        A.HorizontalFlip(p=0.5),
        # Albumentations 2.x: ShiftScaleRotate -> Affine
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5,
        ),
        # Outdoor lighting simulation
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.5
        ),
        # Albumentations 2.x: RandomShadow parameters renamed
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.4,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        # Sun glare / overexposure
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),
        # Camera motion / low quality devices
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        # Albumentations 2.x: ImageCompression quality param renamed
        A.ImageCompression(quality_range=(60, 95), p=0.25),
        # Final resize + normalise
        A.Resize(input_size, input_size),
        A.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ToTensorV2(),
    ])


def build_val_transform(input_size: int = CONFIG["input_size"]) -> A.Compose:
    """Deterministic validation pipeline — resize and normalise only."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ToTensorV2(),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class IndianFaceDataset(Dataset):
    """
    Balanced face dataset for Indian demographic fine-tuning.

    The manifest CSV must contain at minimum:
      - ``file``  : relative path from the processed directory
      - ``label`` : integer identity / class index (0-based, contiguous)

    Optional columns used for demographic balancing:
      - ``race``, ``age``, ``gender`` (FairFace)
      - ``lighting``, ``pose``        (IMFDB)
    """

    def __init__(
        self,
        manifest_path: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        filter_indian: bool = True,
    ) -> None:
        """Load manifest and optionally restrict to South-Asian identities."""
        self.root_dir  = Path(root_dir)
        self.transform = transform

        df = pd.read_csv(manifest_path)

        # Keep Indian / South-Asian samples when the race column is present
        if filter_indian and "race" in df.columns:
            indian_tags = {"Indian", "South Asian", "SE Asian"}
            df = df[df["race"].isin(indian_tags)].reset_index(drop=True)

        # Re-map labels to contiguous range
        unique_labels   = sorted(df["label"].unique())
        label_map       = {orig: new for new, orig in enumerate(unique_labels)}
        df["label"]     = df["label"].map(label_map)

        self.df          = df
        self.num_classes = len(unique_labels)
        self.label_col   = "label"

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load image, apply transform, return (tensor, label)."""
        row       = self.df.iloc[idx]
        img_path  = self.root_dir / row["file"]
        label     = int(row[self.label_col])

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label

    # ── Demographic balancing helpers ──────────────────────────────────────

    def get_sample_weights(self) -> List[float]:
        """
        Compute per-sample weights for WeightedRandomSampler.
        Ensures each identity class appears with equal probability per epoch.
        """
        label_counts = self.df["label"].value_counts()
        weights = self.df["label"].map(lambda l: 1.0 / label_counts[l])
        return weights.tolist()


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def build_dataloaders(
    manifest_path: str  = CONFIG["csv_path"],
    root_dir: str       = CONFIG["processed_dir"],
    batch_size: int     = CONFIG["batch_size"],
    num_workers: int    = CONFIG["num_workers"],
    val_split: float    = CONFIG["val_split"],
    seed: int           = CONFIG["seed"],
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build balanced train and validation DataLoaders.

    Returns:
        train_loader, val_loader, num_classes
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full_df = pd.read_csv(manifest_path)

    # Stratified split by identity label
    labels  = full_df["label"].values
    indices = list(range(len(full_df)))
    random.shuffle(indices)

    val_size  = int(len(indices) * val_split)
    val_idx   = indices[:val_size]
    train_idx = indices[val_size:]

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df   = full_df.iloc[val_idx].reset_index(drop=True)

    # Write temporary split manifests
    tmp_train = manifest_path.replace(".csv", "_train_split.csv")
    tmp_val   = manifest_path.replace(".csv", "_val_split.csv")
    train_df.to_csv(tmp_train, index=False)
    val_df.to_csv(tmp_val, index=False)

    train_dataset = IndianFaceDataset(
        tmp_train, root_dir, transform=build_train_transform(), filter_indian=True
    )
    val_dataset   = IndianFaceDataset(
        tmp_val, root_dir, transform=build_val_transform(), filter_indian=False
    )

    # Weighted sampler for class-balanced training batches
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.num_classes


# ─── Preprocessing Utility ───────────────────────────────────────────────────

def preprocess_raw_to_processed(
    raw_dir: str       = CONFIG["raw_dir"],
    processed_dir: str = CONFIG["processed_dir"],
    input_size: int    = CONFIG["input_size"],
) -> None:
    """
    Align and crop raw face images to 112×112 and write a manifest CSV.
    Uses OpenCV Haar cascade as a lightweight face detector for cropping.
    Run once before training.
    """
    import json

    raw_path       = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    records: List[Dict] = []
    label_map: Dict[str, int] = {}
    label_counter = 0

    for identity_dir in sorted(raw_path.iterdir()):
        if not identity_dir.is_dir():
            continue
        identity = identity_dir.name
        if identity not in label_map:
            label_map[identity] = label_counter
            label_counter += 1
        label = label_map[identity]

        out_dir = processed_path / identity
        out_dir.mkdir(exist_ok=True)

        for img_file in identity_dir.glob("*.[jp][pn]g"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                # No face detected — resize whole image as fallback
                crop = cv2.resize(img, (input_size, input_size))
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = int(0.2 * max(w, h))
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
                crop = cv2.resize(img[y1:y2, x1:x2], (input_size, input_size))

            out_name  = img_file.stem + ".jpg"
            out_rel   = str(Path(identity) / out_name)
            cv2.imwrite(str(out_dir / out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append({"file": out_rel, "label": label, "identity": identity})

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(processed_path / "manifest.csv", index=False)

    label_map_path = processed_path / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Preprocessing complete: {len(records)} images, {label_counter} identities")
    print(f"Manifest: {processed_path / 'manifest.csv'}")


# ─── Synthetic Data Generator ────────────────────────────────────────────────

def generate_synthetic_dataset(
    processed_dir:   str = CONFIG["processed_dir"],
    num_identities:  int = 200,
    images_per_id:   int = 30,
    input_size:      int = CONFIG["input_size"],
) -> None:
    """
    Generate a synthetic face dataset for pipeline smoke-testing.

    Each 'identity' gets a unique base color + texture so the model can learn
    to distinguish them. Real training accuracy will be poor (expected), but
    the full train -> export loop runs correctly and verifies GPU throughput.

    Replace with real face images (CASIA-WebFace / VGGFace2) for production.
    """
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset: {num_identities} identities x {images_per_id} images")
    rng = np.random.default_rng(42)
    records: List[Dict] = []

    for identity_id in range(num_identities):
        id_dir = processed_path / f"id_{identity_id:04d}"
        id_dir.mkdir(exist_ok=True)

        # Each identity has a unique base hue
        base_color = rng.integers(0, 255, size=3).astype(np.uint8)

        for img_idx in range(images_per_id):
            # Slightly perturb base color per image (same identity, different lighting)
            noise = rng.integers(-30, 30, size=3)
            color = np.clip(base_color.astype(int) + noise, 0, 255).astype(np.uint8)

            # Draw a simple face-like oval on a plain background
            img = np.ones((input_size, input_size, 3), dtype=np.uint8) * 40
            cx, cy = input_size // 2, input_size // 2
            cv2.ellipse(img, (cx, cy), (30, 38), 0, 0, 360, color.tolist(), -1)
            # Eyes
            eye_color = (255, 255, 255)
            cv2.circle(img, (cx - 10, cy - 5), 5, eye_color, -1)
            cv2.circle(img, (cx + 10, cy - 5), 5, eye_color, -1)
            cv2.circle(img, (cx - 10, cy - 5), 2, (0, 0, 0), -1)
            cv2.circle(img, (cx + 10, cy - 5), 2, (0, 0, 0), -1)
            # Add small random noise for variation
            img = np.clip(img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)

            fname = f"{img_idx:03d}.jpg"
            cv2.imwrite(str(id_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            records.append({
                "file":     f"id_{identity_id:04d}/{fname}",
                "label":    identity_id,
                "identity": f"id_{identity_id:04d}",
            })

        if (identity_id + 1) % 50 == 0:
            print(f"  {identity_id + 1}/{num_identities} identities generated")

    df = pd.DataFrame(records)
    df.to_csv(processed_path / "manifest.csv", index=False)
    total = num_identities * images_per_id
    print(f"Synthetic dataset ready: {total} images, {num_identities} identities")
    print(f"Manifest: {processed_path / 'manifest.csv'}")
    print("NOTE: Replace data/processed/ with real face data for production accuracy.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic dataset instead of preprocessing raw images")
    parser.add_argument("--num_identities", type=int, default=200)
    parser.add_argument("--images_per_id",  type=int, default=30)
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(
            num_identities=args.num_identities,
            images_per_id=args.images_per_id,
        )
    else:
        preprocess_raw_to_processed()
