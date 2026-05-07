"""Visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np

from sigma.utils.io import ensure_dir


@dataclass
class VisualizationBundle:
    """Describe assets produced by a stage and where to save them."""

    target_dir: Path
    assets: Dict[str, Any]

    def save(self) -> None:
        ensure_dir(self.target_dir)
        for name, value in self.assets.items():
            if value is None:
                continue
            path = self.target_dir / f"{name}.npy"
            np.save(path, value)


def save_placeholder_image(path: Path) -> None:
    """Write a placeholder image useful for debugging visualization flows."""
    ensure_dir(path.parent)
    dummy = np.zeros((32, 32, 3), dtype=np.uint8)
    np.save(path.with_suffix(".npy"), dummy)
