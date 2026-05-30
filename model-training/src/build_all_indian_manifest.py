"""
build_all_indian_manifest.py — Rebuild manifest.csv from existing all_indian image folders.

Use when images are already under data/all_indian/<identity>/ but manifest is missing or stale.

Run from model-training/:
  $env:PYTHONUTF8="1"
  python -u src/build_all_indian_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build(root: Path, out: Path) -> None:
    id_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower(),
    )
    if not id_dirs:
        raise RuntimeError(f"No identity folders in {root}")

    rows: list[tuple[str, int, str]] = []
    for label, id_dir in enumerate(id_dirs):
        identity = id_dir.name
        for img in sorted(id_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMG_EXT:
                rows.append((str(img.relative_to(root)), label, identity))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "identity"])
        w.writerows(rows)

    print(f"[OK] {out}")
    print(f"     Identities: {len(id_dirs):,}")
    print(f"     Images    : {len(rows):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/all_indian")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "manifest.csv"
    t0 = time.time()
    build(root, out)
    print(f"     Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
