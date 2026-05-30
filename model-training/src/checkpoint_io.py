"""Load local training checkpoints (PyTorch 2.6+ weights_only safe)."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """
    Load a checkpoint saved by this repo.

    Checkpoints store numpy scalars in metadata (tar_at_far_001, etc.), so
    weights_only=True may fail on PyTorch 2.6+. We fall back to weights_only=False
    for trusted local files only.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)
