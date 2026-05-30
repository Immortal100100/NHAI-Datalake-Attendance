"""
download_datasets.py — Download Indian face datasets from Kaggle.

Run:
  $env:PYTHONUTF8="1"
  python -u src/download_datasets.py
"""

import subprocess
import sys
import os
import zipfile
import shutil
from pathlib import Path

DOWNLOAD_DIR = Path("data/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # (kaggle_ref,                                     folder_name,              size_hint)
    ("sushilyadav1998/bollywood-celeb-localized-face-dataset", "bollywood_localized",   "28 MB"),
    ("sroy93/bollywood-celeb-localized-face-dataset-extended", "bollywood_extended",    "38 MB"),
    ("gunarakulangr/south-indian-celebrity-dataset",           "south_indian",          "56 MB"),
    ("sudhanshu2198/indian-celebtities-face-recognition",      "indian_celebrities",    "282 MB"),
]


def run(cmd: list, **kw):
    print(f"\n> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, **kw, capture_output=False, text=True)
    return result.returncode


def download_and_extract(ref: str, dest_name: str, size: str):
    dest = DOWNLOAD_DIR / dest_name
    if dest.exists():
        print(f"[SKIP] {dest_name} already downloaded at {dest}")
        return True

    print(f"\n{'='*60}")
    print(f"Downloading: {ref}  (~{size})")
    print(f"{'='*60}")

    tmp = DOWNLOAD_DIR / f"_tmp_{dest_name}"
    tmp.mkdir(exist_ok=True)

    rc = run(["kaggle", "datasets", "download", "-d", ref,
              "-p", str(tmp), "--unzip"], cwd=".")
    if rc != 0:
        print(f"[ERROR] Failed to download {ref}")
        return False

    # After unzip, find the root data folder
    children = list(tmp.iterdir())
    if len(children) == 1 and children[0].is_dir():
        # Single top-level directory — use it directly
        shutil.move(str(children[0]), str(dest))
        tmp.rmdir()
    else:
        # Multiple files/dirs — keep the tmp dir as dest
        shutil.move(str(tmp), str(dest))

    print(f"[OK] Saved to {dest}")
    return True


def inspect_structure(root: Path, max_depth: int = 3, max_items: int = 5):
    """Print first few items at each level to understand structure."""
    print(f"\nStructure of {root}:")
    for depth, (dirpath, dirs, files) in enumerate(os.walk(root)):
        rel = Path(dirpath).relative_to(root)
        indent = "  " * len(rel.parts)
        dirs_shown  = sorted(dirs)[:max_items]
        files_shown = sorted(files)[:max_items]
        if len(rel.parts) >= max_depth:
            dirs.clear()
            continue
        if dirs_shown:
            print(f"{indent}{rel}/ → subdirs: {dirs_shown}{'...' if len(dirs)>max_items else ''}")
        if files_shown and len(rel.parts) == max_depth - 1:
            print(f"{indent}  files: {files_shown}{'...' if len(files)>max_items else ''}")


if __name__ == "__main__":
    success = []
    failed  = []

    for ref, name, size in DATASETS:
        ok = download_and_extract(ref, name, size)
        (success if ok else failed).append(name)

    print(f"\n{'='*60}")
    print(f"Downloaded : {success}")
    print(f"Failed     : {failed}")
    print()
    for name in success:
        dest = DOWNLOAD_DIR / name
        if dest.exists():
            inspect_structure(dest)

    print(f"\nNext step:")
    print(f"  python -u src/parse_new_datasets.py")
