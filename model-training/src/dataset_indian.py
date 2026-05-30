"""
dataset_indian.py — Indian face dataset parsers for fine-tuning.

Supported datasets:
  1. IMFDB   — Indian Movie Face Database (IIIT Hyderabad)
               ~34K images, 100 Indian actors, multiple shots per identity
               Download: http://cvit.iiit.ac.in/projects/IMFDB/
               Approval time: 1-2 business days

  2. Kaggle Bollywood
               ~8K images, 43 Indian actors
               Download: kaggle datasets download -d kavyasreeb/bollywood-celebrity-dataset
               Requires: Kaggle account + API key

  3. Custom   — Any folder structure: data/raw/<identity_name>/<image.jpg>

All parsers output a unified manifest CSV:
  file, label, identity
that feeds into IndianFaceDataset (same as dataset.py).

Usage:
  # IMFDB (after extracting zip to data/raw/IMFDB/)
  python src/dataset_indian.py --source imfdb --raw_dir data/raw/IMFDB

  # Kaggle Bollywood (after download)
  python src/dataset_indian.py --source kaggle --raw_dir data/raw/bollywood

  # Custom folder structure
  python src/dataset_indian.py --source custom --raw_dir data/raw/my_dataset
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG = {
    "processed_dir": "data/processed",
    "input_size":    112,
    "min_images":    5,       # drop identities with fewer images
    "max_images":    100,     # cap per identity (for balanced training)
    "jpeg_quality":  95,
}


# ─── Face Aligner (Haar cascade — lightweight, no extra deps) ─────────────────

def _align_face(img: np.ndarray, size: int, cascade) -> Optional[np.ndarray]:
    """Detect and crop the largest face; fall back to full image resize."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return cv2.resize(img, (size, size))
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(img.shape[1], x + w + pad)
    y2  = min(img.shape[0], y + h + pad)
    return cv2.resize(img[y1:y2, x1:x2], (size, size))


def _load_cascade():
    """Load OpenCV Haar cascade for face detection."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


# ─── Parser 1: Generic folder structure ──────────────────────────────────────

def parse_custom_folder(
    raw_dir:       str,
    processed_dir: str = CONFIG["processed_dir"],
    input_size:    int = CONFIG["input_size"],
    min_images:    int = CONFIG["min_images"],
    max_images:    int = CONFIG["max_images"],
) -> pd.DataFrame:
    """
    Parse any folder where each sub-directory is an identity.

    Expected layout:
      raw_dir/
        Identity_A/
          img001.jpg
          img002.jpg
        Identity_B/
          ...
    """
    raw_path   = Path(raw_dir)
    proc_path  = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    cascade    = _load_cascade()

    label_map : Dict[str, int] = {}
    label_counter = 0
    records   : List[Dict]     = []
    skipped   = 0

    identity_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    print(f"Found {len(identity_dirs)} identity folders in {raw_dir}")

    for id_dir in identity_dirs:
        identity = id_dir.name
        images   = list(id_dir.glob("*.[jJpP][pPnN][gG]"))
        images  += list(id_dir.glob("*.jpeg")) + list(id_dir.glob("*.JPEG"))

        if len(images) < min_images:
            skipped += 1
            continue

        if identity not in label_map:
            label_map[identity] = label_counter
            label_counter += 1
        label = label_map[identity]

        out_dir = proc_path / identity
        out_dir.mkdir(exist_ok=True)

        images = images[:max_images]
        for img_file in images:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            aligned = _align_face(img, input_size, cascade)
            if aligned is None:
                continue
            fname   = img_file.stem + ".jpg"
            rel     = f"{identity}/{fname}"
            cv2.imwrite(str(out_dir / fname), aligned,
                        [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]])
            records.append({"file": rel, "label": label, "identity": identity})

    df = pd.DataFrame(records)
    _save_manifest(df, proc_path, label_map)
    print(f"[OK] {len(records)} images, {label_counter} identities "
          f"({skipped} skipped for <{min_images} images)")
    return df


# ─── Parser 2: IMFDB ─────────────────────────────────────────────────────────

def parse_imfdb(
    raw_dir:       str,
    processed_dir: str = CONFIG["processed_dir"],
    input_size:    int = CONFIG["input_size"],
) -> pd.DataFrame:
    """
    Parse IMFDB dataset.

    Expected layout after extracting IMFDB.zip:
      raw_dir/
        <ActorName>/
          <MovieName>/
            <shot_type>/
              img001.jpg

    Or simpler flat layout:
      raw_dir/
        <ActorName>/
          img001.jpg
    """
    print("Parsing IMFDB dataset...")
    raw_path = Path(raw_dir)

    # Check if images are nested inside movie sub-dirs
    sample_dirs = list(raw_path.iterdir())
    has_subdirs = any(
        any(d.is_dir() for d in identity_dir.iterdir())
        for identity_dir in sample_dirs[:5]
        if identity_dir.is_dir()
    )

    if has_subdirs:
        # Flatten: collect all images recursively per identity
        print("  Detected nested IMFDB structure — flattening...")
        temp_flat = Path(raw_dir + "_flat")
        temp_flat.mkdir(exist_ok=True)

        for actor_dir in raw_path.iterdir():
            if not actor_dir.is_dir():
                continue
            actor_flat = temp_flat / actor_dir.name
            actor_flat.mkdir(exist_ok=True)
            count = 0
            for img_file in actor_dir.rglob("*.[jJpP][pPnN][gG]"):
                dst = actor_flat / f"{count:05d}.jpg"
                shutil.copy(img_file, dst)
                count += 1
        raw_dir = str(temp_flat)

    return parse_custom_folder(raw_dir, processed_dir, input_size)


# ─── Parser 3: Kaggle Bollywood ───────────────────────────────────────────────

def download_kaggle_bollywood(output_dir: str = "data/raw/bollywood") -> str:
    """
    Download Bollywood celebrity face dataset from Kaggle.

    Requires:
      1. Kaggle account at https://www.kaggle.com
      2. API key: go to Account -> Create New Token -> saves kaggle.json
      3. Place kaggle.json at C:\\Users\\<user>\\.kaggle\\kaggle.json
      4. pip install kaggle
    """
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n[!] Kaggle API key not found.")
        print("    Steps to get it:")
        print("    1. Go to https://www.kaggle.com/account")
        print("    2. Click 'Create New Token'")
        print(f"   3. Save the downloaded kaggle.json to: {kaggle_json}")
        print("    4. Re-run this script")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading Bollywood dataset to {output_dir}...")

    # Dataset: 43 Indian celebrities, ~8K images
    subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "kavyasreeb/bollywood-celebrity-dataset",
        "-p", output_dir,
        "--unzip",
    ], check=True)

    print(f"[OK] Downloaded to {output_dir}")
    return output_dir


def parse_kaggle_bollywood(
    raw_dir:       str = "data/raw/bollywood",
    processed_dir: str = CONFIG["processed_dir"],
) -> pd.DataFrame:
    """Parse Kaggle Bollywood dataset (auto-detects folder structure)."""
    print("Parsing Kaggle Bollywood dataset...")
    return parse_custom_folder(raw_dir, processed_dir)


# ─── Manifest Utilities ────────────────────────────────────────────────────────

def _save_manifest(
    df:        pd.DataFrame,
    proc_path: Path,
    label_map: Dict[str, int],
) -> None:
    """Save manifest CSV and label_map JSON."""
    df.to_csv(proc_path / "manifest.csv", index=False)
    with open(proc_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Manifest: {proc_path / 'manifest.csv'}")
    print(f"  Label map: {proc_path / 'label_map.json'}")


def print_dataset_stats(processed_dir: str = CONFIG["processed_dir"]) -> None:
    """Print summary statistics for the processed dataset."""
    mf = Path(processed_dir) / "manifest.csv"
    if not mf.exists():
        print("No manifest found. Run a parser first.")
        return
    df = pd.read_csv(mf)
    print(f"\nDataset stats:")
    print(f"  Total images    : {len(df)}")
    print(f"  Identities      : {df['label'].nunique()}")
    print(f"  Avg images/id   : {len(df) / df['label'].nunique():.1f}")
    print(f"  Min images/id   : {df.groupby('label').size().min()}")
    print(f"  Max images/id   : {df.groupby('label').size().max()}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian face dataset parser")
    parser.add_argument("--source", choices=["imfdb", "kaggle", "custom", "download_kaggle"],
                        default="custom",
                        help="Dataset source to parse")
    parser.add_argument("--raw_dir",       default="data/raw",
                        help="Path to raw dataset folder")
    parser.add_argument("--processed_dir", default=CONFIG["processed_dir"],
                        help="Output directory for processed images + manifest")
    args = parser.parse_args()

    if args.source == "imfdb":
        df = parse_imfdb(args.raw_dir, args.processed_dir)

    elif args.source == "kaggle":
        df = parse_kaggle_bollywood(args.raw_dir, args.processed_dir)

    elif args.source == "download_kaggle":
        raw = download_kaggle_bollywood(args.raw_dir)
        if raw:
            df = parse_kaggle_bollywood(raw, args.processed_dir)

    else:  # custom
        df = parse_custom_folder(args.raw_dir, args.processed_dir)

    print_dataset_stats(args.processed_dir)
