"""
parse_new_datasets.py — Parse all downloaded Kaggle Indian face datasets
and merge with existing manifest into data/all_indian/

Datasets handled:
  - bollywood_localized  (2-level nested: group_folder/identity/images)
  - bollywood_extended   (flat: identity/images)
  - south_indian         (flat: identity/images)
  - indian_celebrities   (flat: identity/images)
  + existing merged data (data/merged)

Output:
  data/all_indian/
    manifest.csv   — unified (file, label, identity) table
    <identity>/    — symlinked or copied image files

Run:
  $env:PYTHONUTF8="1"
  python -u src/parse_new_datasets.py
"""

import os, re, sys, csv, hashlib, shutil, warnings
warnings.filterwarnings("ignore")
import cv2
import pandas as pd
from pathlib import Path

IMG_EXT      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_SIDE     = 40       # discard tiny images
OUT_SIZE     = 182      # save at 182px (InsightFace aligner needs ~160+ for 5pt detection)
MAX_PER_ID   = 200      # cap per identity to keep dataset balanced
OUT_DIR      = Path("data/all_indian")
EXISTING_DIR = Path("data/merged")


def normalize_id(name: str) -> str:
    """Lowercase, remove punctuation, replace spaces with underscores."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 _]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")


def collect_identity_folders(root: Path) -> dict[str, list[Path]]:
    """
    Auto-detect flat vs. 2-level structure.
    Returns: {identity_name: [image_path, ...]}
    """
    id_map: dict[str, list[Path]] = {}

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        imgs = [f for f in child.iterdir() if f.suffix.lower() in IMG_EXT]
        if imgs:
            # Flat structure: child IS the identity folder
            nid = normalize_id(child.name)
            id_map.setdefault(nid, []).extend(imgs)
        else:
            # Possibly 2-level: child contains identity subfolders
            for sub in sorted(child.iterdir()):
                if sub.is_dir():
                    sub_imgs = [f for f in sub.iterdir() if f.suffix.lower() in IMG_EXT]
                    if sub_imgs:
                        nid = normalize_id(sub.name)
                        id_map.setdefault(nid, []).extend(sub_imgs)

    return id_map


def is_valid_image(path: Path) -> bool:
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False
        h, w = img.shape[:2]
        return min(h, w) >= MIN_SIDE
    except Exception:
        return False


def save_resized(src: Path, dst: Path) -> bool:
    try:
        img = cv2.imread(str(src))
        if img is None:
            return False
        h, w = img.shape[:2]
        if min(h, w) < MIN_SIDE:
            return False
        scale = OUT_SIZE / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception:
        return False


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect from all new Kaggle datasets
    new_sources = {
        "bollywood_localized": Path("data/downloads/bollywood_localized"),
        "bollywood_extended":  Path("data/downloads/bollywood_extended"),
        "south_indian":        Path("data/downloads/south_indian"),
        "indian_celebrities":  Path("data/downloads/indian_celebrities"),
    }

    all_ids: dict[str, list[Path]] = {}

    for src_name, src_root in new_sources.items():
        if not src_root.exists():
            print(f"[SKIP] {src_name} — not found at {src_root}")
            continue
        print(f"Scanning {src_name}...", end=" ", flush=True)
        id_map = collect_identity_folders(src_root)
        merged_count = 0
        for nid, imgs in id_map.items():
            all_ids.setdefault(nid, []).extend(imgs)
            merged_count += 1
        print(f"{merged_count} identities")

    # ── Step 2: Also add existing merged data (data/merged)
    if EXISTING_DIR.exists() and (EXISTING_DIR / "manifest.csv").exists():
        print(f"Adding existing data from {EXISTING_DIR}...", end=" ", flush=True)
        df_exist = pd.read_csv(EXISTING_DIR / "manifest.csv")
        added = 0
        for _, row in df_exist.iterrows():
            nid = normalize_id(row["identity"])
            img_path = EXISTING_DIR / row["file"]
            if img_path.exists():
                all_ids.setdefault(nid, []).append(img_path)
                added += 1
        print(f"{df_exist['identity'].nunique()} identities, {added} images")

    print(f"\nTotal unique identities before filtering: {len(all_ids)}")

    # ── Step 3: Filter, deduplicate by hash, cap per identity
    print("Filtering and deduplicating...", flush=True)
    records = []   # (normalized_id, src_path)
    kept_ids: dict[str, int] = {}

    for nid, paths in sorted(all_ids.items()):
        seen_hashes: set[str] = set()
        valid_paths: list[Path] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                h = file_hash(p)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                valid_paths.append(p)
            except Exception:
                pass

        if len(valid_paths) < 10:       # skip identities with very few images
            continue

        # Cap at MAX_PER_ID using evenly-spaced sampling
        if len(valid_paths) > MAX_PER_ID:
            step = len(valid_paths) / MAX_PER_ID
            valid_paths = [valid_paths[int(i * step)] for i in range(MAX_PER_ID)]

        kept_ids[nid] = len(valid_paths)
        for p in valid_paths:
            records.append((nid, p))

    print(f"Identities kept (>=10 images): {len(kept_ids)}")
    print(f"Total images                 : {len(records)}")

    # ── Step 4: Copy files to OUT_DIR with consistent naming
    print(f"\nCopying images to {OUT_DIR} ...", flush=True)
    manifest_rows = []
    label_map: dict[str, int] = {nid: i for i, nid in enumerate(sorted(kept_ids))}

    for idx, (nid, src) in enumerate(records):
        id_dir = OUT_DIR / nid
        id_dir.mkdir(exist_ok=True)
        dst = id_dir / f"{nid}_{idx:06d}{src.suffix.lower()}"

        if not dst.exists():
            save_resized(src, dst)

        rel = dst.relative_to(OUT_DIR)
        manifest_rows.append({
            "file":     str(rel),
            "label":    label_map[nid],
            "identity": nid,
        })

        if (idx + 1) % 2000 == 0:
            pct = 100 * (idx + 1) / len(records)
            print(f"  {idx+1:>6}/{len(records)}  [{pct:.0f}%]", flush=True)

    # ── Step 5: Write manifest
    manifest_path = OUT_DIR / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"\n[OK] Manifest saved: {manifest_path}")
    print(f"     Identities : {len(kept_ids)}")
    print(f"     Images     : {len(manifest_rows)}")

    # Distribution stats
    counts = pd.DataFrame(manifest_rows).groupby("identity").size()
    print(f"     Avg/ID     : {counts.mean():.0f}")
    print(f"     Min/ID     : {counts.min()}")
    print(f"     Max/ID     : {counts.max()}")

    print(f"\nNext step:")
    print(f"  # Delete old aligned cache and rebuild with all new data")
    print(f"  Remove-Item -Force data/all_indian/embed_cache_aligned.npz")
    print(f"  python -u src/align_and_precompute.py \\")
    print(f"      --manifest_path data/all_indian/manifest.csv \\")
    print(f"      --processed_dir data/all_indian \\")
    print(f"      --cache_path    data/all_indian/embed_cache_aligned.npz \\")
    print(f"      --n_aug 8")


if __name__ == "__main__":
    main()
