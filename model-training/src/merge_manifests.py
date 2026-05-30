"""
merge_manifests.py — Combine multiple processed manifests into one.

Usage:
  python src/merge_manifests.py \
      --dirs data/processed data/processed_actors \
      --out  data/merged
"""

import argparse
import json
import shutil
from pathlib import Path
import pandas as pd


def merge(dirs: list[str], out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_records = []
    label_offset = 0
    global_label_map = {}

    for src_dir in dirs:
        src = Path(src_dir)
        mf  = src / "manifest.csv"
        lm  = src / "label_map.json"

        if not mf.exists():
            print(f"[SKIP] {src_dir} — no manifest.csv")
            continue

        df = pd.read_csv(mf)
        with open(lm) as f:
            label_map = json.load(f)

        print(f"  {src_dir}: {len(df)} images, {df.label.nunique()} identities")

        # Offset all labels to avoid collisions
        df["label"] = df["label"] + label_offset
        df["src_dir"] = str(src)

        # Copy images to merged dir (preserving identity sub-folders)
        for _, row in df.iterrows():
            src_img = src / row["file"]
            dst_img = out_path / row["file"]
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists() and src_img.exists():
                shutil.copy2(src_img, dst_img)

        # Update label map with offset
        for identity, lbl in label_map.items():
            global_label_map[identity] = lbl + label_offset

        label_offset += df["label"].nunique()
        all_records.append(df)

    merged = pd.concat(all_records, ignore_index=True)
    merged.drop(columns=["src_dir"], inplace=True)

    merged.to_csv(out_path / "manifest.csv", index=False)
    with open(out_path / "label_map.json", "w") as f:
        json.dump(global_label_map, f, indent=2)

    print(f"\nMerged: {len(merged)} images, {merged.label.nunique()} identities")
    print(f"Saved to {out_path}/manifest.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out",  default="data/merged")
    args = parser.parse_args()
    merge(args.dirs, args.out)
