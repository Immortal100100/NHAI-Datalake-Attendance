"""
build_vggface2_manifest.py — Build manifest.csv for extracted VGGFace2 112x112.

Expected layout:
  data/downloads/vggface2_112x112/
    id_0/0.jpg
    id_1/...

Output columns: file, label, identity  (same format as all_indian/manifest.csv)

Run from model-training/:
  $env:PYTHONUTF8="1"
  python -u src/build_vggface2_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _id_sort_key(name: str) -> tuple:
    m = re.match(r"id_(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, name)


def build_manifest(root: Path, out_path: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    id_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("id_")),
        key=lambda p: _id_sort_key(p.name),
    )
    if not id_dirs:
        raise RuntimeError(f"No id_* folders under {root}")

    rows: list[tuple[str, int, str]] = []
    t0 = time.time()

    for label, id_dir in enumerate(id_dirs):
        identity = id_dir.name
        for img in sorted(id_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMG_EXT:
                continue
            rel = str(img.relative_to(root))
            rows.append((rel, label, identity))

        if (label + 1) % 500 == 0:
            print(f"  scanned {label + 1}/{len(id_dirs)} identities, {len(rows):,} images...", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "identity"])
        w.writerows(rows)

    elapsed = time.time() - t0
    print(f"[OK] Manifest: {out_path}")
    print(f"     Identities: {len(id_dirs):,}")
    print(f"     Images    : {len(rows):,}")
    print(f"     Time      : {elapsed:.1f}s")
    print()
    print("Training (from model-training/):")
    print("  python -u src/finetune.py --precompute \\")
    print("    --manifest_path data/downloads/vggface2_112x112/manifest.csv \\")
    print("    --processed_dir data/downloads/vggface2_112x112 \\")
    print("    --embed_cache   data/downloads/vggface2_112x112/embed_cache_aug.npz \\")
    print("    --n_aug 10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/downloads/vggface2_112x112",
        help="Extracted VGGFace2 folder (id_0, id_1, ...)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="manifest.csv path (default: <root>/manifest.csv)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "manifest.csv"
    build_manifest(root, out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
