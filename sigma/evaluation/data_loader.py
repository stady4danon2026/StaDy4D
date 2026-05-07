"""Data loaders for ground truth and prediction directories.

Supports two layouts:

1. **Per-frame files** (pipeline output / legacy GT):
    root/
    ├── intrinsics.json         (optional global intrinsics)
    ├── metadata.json           (optional)
    ├── rgb/rgb_NNNN.png
    ├── depth/depth_NNNN.npy
    ├── extrinsics/extrinsic_NNNN.npy
    ├── intrinsics/intrinsic_NNNN.npy
    └── confidence/confidence_NNNN.npy   (predictions only)

2. **StaDy4D safetensors** (new raw dataset format):
    root/
    ├── metadata.json
    ├── dynamic/
    │   └── cam_XX_name/
    │       ├── rgb.mp4
    │       ├── depth.safetensors      (key "depth", [N,H,W])
    │       ├── extrinsics.safetensors (key "c2w",   [N,4,4])
    │       └── intrinsics.safetensors (key "K",     [N,3,3])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class FrameData:
    """All available data for one frame."""
    index: int
    rgb_path: Path | None = None
    rgb: np.ndarray | None = None            # (H, W, 3) uint8
    depth: np.ndarray | None = None          # (H, W)
    extrinsic: np.ndarray | None = None      # (3, 4) or (4, 4) c2w
    intrinsic: np.ndarray | None = None      # (3, 3)
    confidence: np.ndarray | None = None     # (H, W)
    mask: np.ndarray | None = None           # (H, W) uint8
    inpainted: np.ndarray | None = None      # (H, W, 3) uint8


@dataclass
class SequenceData:
    """All frames for a sequence."""
    root: Path
    frames: Dict[int, FrameData] = field(default_factory=dict)
    global_intrinsic: np.ndarray | None = None  # (3, 3)
    metadata: Dict | None = None

    @property
    def frame_indices(self) -> List[int]:
        return sorted(self.frames.keys())

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def _extract_index(filename: str) -> int | None:
    """Pull the integer index from filenames like extrinsic_0042.npy or rgb_0042.png."""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None


def _is_safetensors_camera(path: Path) -> bool:
    """Check if a directory contains safetensors data files."""
    return (path / "rgb.mp4").exists() and (path / "depth.safetensors").exists()


def _find_camera_dir(root: Path) -> Path | None:
    """Auto-detect a camera sub-directory under root (or root/dynamic|static)."""
    actual = root
    if (root / "dynamic").exists():
        actual = root / "dynamic"
    elif (root / "static").exists():
        actual = root / "static"

    # Check if actual_root itself is a camera dir
    if _is_safetensors_camera(actual):
        return actual

    # Look for cam_* or camera_* subdirectories
    for d in sorted(actual.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("cam_") or d.name.startswith("camera_"):
            if _is_safetensors_camera(d) or (d / "extrinsics").exists() or (d / "depth").exists():
                return d
    return None


def _decode_mp4_frames(mp4_path: Path) -> List[np.ndarray]:
    """Decode all frames from an mp4 file. Returns list of (H,W,3) uint8 arrays."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _load_safetensors_sequence(
    root: Path, camera_dir: Path, load_depth: bool = True, load_rgb: bool = False,
) -> SequenceData:
    """Load a sequence from StaDy4D safetensors format."""
    from safetensors.numpy import load_file

    # scene_dir is two levels up from camera_dir: scene/dynamic/cam_*/
    scene_dir = camera_dir.parent.parent
    seq = SequenceData(root=root)

    # Metadata
    for meta_candidate in [scene_dir / "metadata.json", root / "metadata.json"]:
        if meta_candidate.exists():
            with open(meta_candidate) as f:
                seq.metadata = json.load(f)
            break

    # Global intrinsics from metadata
    if seq.metadata and "intrinsic" in seq.metadata:
        intr = seq.metadata["intrinsic"]
        seq.global_intrinsic = np.array([
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1],
        ])

    # Load safetensors
    c2w_all = load_file(str(camera_dir / "extrinsics.safetensors"))["c2w"]     # (N, 4, 4)
    K_all = load_file(str(camera_dir / "intrinsics.safetensors"))["K"]         # (N, 3, 3)
    depth_all = None
    if load_depth:
        depth_all = load_file(str(camera_dir / "depth.safetensors"))["depth"]  # (N, H, W)

    # Load RGB from mp4
    rgb_frames = None
    if load_rgb:
        mp4_path = camera_dir / "rgb.mp4"
        if mp4_path.exists():
            rgb_frames = _decode_mp4_frames(mp4_path)

    n_frames = c2w_all.shape[0]

    for i in range(n_frames):
        frame = FrameData(index=i)
        frame.extrinsic = c2w_all[i].astype(np.float64)
        frame.intrinsic = K_all[i].astype(np.float64)
        if depth_all is not None:
            frame.depth = depth_all[i].astype(np.float32)
        if rgb_frames is not None and i < len(rgb_frames):
            frame.rgb = rgb_frames[i]
        seq.frames[i] = frame

    return seq


def load_sequence(root: str | Path, load_depth: bool = True, load_rgb: bool = False) -> SequenceData:
    """Load a prediction or ground-truth sequence directory.

    Auto-detects the format:
    - StaDy4D safetensors (rgb.mp4 + *.safetensors)
    - Per-frame files (rgb/*.png, depth/*.npy, etc.)

    Handles flat layout (root/depth/...), CARLA layout with static/dynamic
    sub-folders, and StaDy4D layout with an additional camera sub-directory
    (e.g. root/dynamic/cam_00_car_forward/).
    """
    root = Path(root)
    seq = SequenceData(root=root)

    # Check for dynamic/static sub-folder layout
    actual_root = root
    if (root / "dynamic").exists():
        actual_root = root / "dynamic"
    elif (root / "static").exists():
        actual_root = root / "static"

    # Auto-detect camera sub-directory
    camera_dir = None
    if not (actual_root / "extrinsics").exists() and not (actual_root / "depth").exists():
        camera_dirs = sorted(d for d in actual_root.iterdir() if d.is_dir())
        for candidate in camera_dirs:
            if _is_safetensors_camera(candidate):
                return _load_safetensors_sequence(root, candidate, load_depth, load_rgb)
            if (candidate / "extrinsics").exists() or (candidate / "depth").exists():
                actual_root = candidate
                break

    # Check if the resolved directory is safetensors format
    if _is_safetensors_camera(actual_root):
        return _load_safetensors_sequence(root, actual_root, load_depth, load_rgb)

    # --- Per-frame file loading (legacy format) ---

    # Global intrinsics
    intrinsics_json = root / "intrinsics.json"
    if intrinsics_json.exists():
        with open(intrinsics_json) as f:
            data = json.load(f)
        K = np.array([
            [data["fx"], 0, data["cx"]],
            [0, data["fy"], data["cy"]],
            [0, 0, 1],
        ])
        seq.global_intrinsic = K

    # Metadata
    meta_json = root / "metadata.json"
    if meta_json.exists():
        with open(meta_json) as f:
            seq.metadata = json.load(f)

    # Discover frames from extrinsics (most reliable)
    ext_dir = actual_root / "extrinsics"
    if ext_dir.exists():
        for p in sorted(ext_dir.glob("*.npy")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            ext = np.load(p)
            frame = _get_or_create(seq, idx)
            frame.extrinsic = ext

    # Depth
    depth_dir = actual_root / "depth"
    if load_depth and depth_dir.exists():
        for p in sorted(depth_dir.glob("*.npy")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            frame = _get_or_create(seq, idx)
            frame.depth = np.load(p)

    # Per-frame intrinsics
    intr_dir = actual_root / "intrinsics"
    if intr_dir.exists():
        for p in sorted(intr_dir.glob("*.npy")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            frame = _get_or_create(seq, idx)
            frame.intrinsic = np.load(p)

    # Confidence (predictions only)
    conf_dir = actual_root / "confidence"
    if conf_dir.exists():
        for p in sorted(conf_dir.glob("*.npy")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            frame = _get_or_create(seq, idx)
            frame.confidence = np.load(p)

    # RGB paths (load images only when load_rgb=True)
    rgb_dir = actual_root / "rgb"
    if rgb_dir.exists():
        for p in sorted(rgb_dir.glob("*.png")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            frame = _get_or_create(seq, idx)
            frame.rgb_path = p
            if load_rgb:
                img = cv2.imread(str(p))
                if img is not None:
                    frame.rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask (predictions only) — always load for point cloud evaluation
    mask_dir = actual_root / "mask"
    if mask_dir.exists():
        for p in sorted(mask_dir.glob("*.png")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            frame = _get_or_create(seq, idx)
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frame.mask = img

    # Inpainted (predictions only)
    inpainted_dir = actual_root / "inpainted"
    if load_rgb and inpainted_dir.exists():
        for p in sorted(inpainted_dir.glob("*.png")):
            idx = _extract_index(p.name)
            if idx is None:
                continue
            frame = _get_or_create(seq, idx)
            img = cv2.imread(str(p))
            if img is not None:
                frame.inpainted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fill global intrinsic for frames missing per-frame intrinsics
    if seq.global_intrinsic is not None:
        for frame in seq.frames.values():
            if frame.intrinsic is None:
                frame.intrinsic = seq.global_intrinsic.copy()

    return seq


def _get_or_create(seq: SequenceData, idx: int) -> FrameData:
    if idx not in seq.frames:
        seq.frames[idx] = FrameData(index=idx)
    return seq.frames[idx]
