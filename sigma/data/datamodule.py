"""Data module responsible for yielding consecutive frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Tuple, Dict, Any

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass
class FrameSequenceConfig:
    """Configuration describing how to load frames."""

    frames_dir: Path
    frame_stride: int = 1
    max_frames: int | None = None
    data_format: str = "dynamic"  # "dynamic" or "static"
    camera: str = ""              # Camera sub-directory, e.g. "cam_00_car_forward"
    load_metadata: bool = True


class FrameDataModule:
    """Frame iterator supporting both legacy (per-frame files) and new
    StaDy4D format (mp4 + safetensors)."""

    def __init__(self, cfg: FrameSequenceConfig) -> None:
        self.cfg = cfg
        self.num_frames = 0
        self._is_safetensors = False

        # Legacy per-frame paths
        self._frames: List[Path] = []
        self.rgb_dir: Path | None = None
        self.depth_dir: Path | None = None
        self.extrinsic_dir: Path | None = None

        # New safetensors data (loaded in setup)
        self._rgb_frames: List[np.ndarray] | None = None
        self._depths: np.ndarray | None = None       # (N, H, W)
        self._c2w: np.ndarray | None = None           # (N, 4, 4)
        self._K_all: np.ndarray | None = None         # (N, 3, 3)

        # Common
        self.intrinsics: Dict[str, Any] | None = None
        self.metadata: Dict[str, Any] | None = None

    def setup(self) -> None:
        if not self.cfg.frames_dir.exists():
            raise FileNotFoundError(f"Frames dir {self.cfg.frames_dir} not found")

        camera_dir = self._get_camera_dir()

        if (camera_dir / "rgb.mp4").exists():
            self._setup_safetensors(camera_dir)
        else:
            self._setup_legacy(camera_dir)

        if self.cfg.load_metadata:
            self._load_metadata()

    def _get_camera_dir(self) -> Path:
        """Resolve the camera data directory."""
        data_dir = self.cfg.frames_dir / self.cfg.data_format
        if self.cfg.camera:
            data_dir = data_dir / self.cfg.camera
        return data_dir

    # ------------------------------------------------------------------
    # New format: safetensors + mp4
    # ------------------------------------------------------------------

    def _setup_safetensors(self, camera_dir: Path) -> None:
        """Load data from safetensors files and mp4 video."""
        from safetensors.numpy import load_file

        self._is_safetensors = True

        # Load depth
        depth_data = load_file(str(camera_dir / "depth.safetensors"))
        self._depths = depth_data["depth"]  # (N, H, W) float16

        # Load extrinsics (c2w matrices)
        ext_data = load_file(str(camera_dir / "extrinsics.safetensors"))
        self._c2w = ext_data["c2w"]  # (N, 4, 4) float32

        # Load intrinsics (per-frame K matrices)
        intr_data = load_file(str(camera_dir / "intrinsics.safetensors"))
        self._K_all = intr_data["K"]  # (N, 3, 3) float32

        # Decode video frames
        self._rgb_frames = []
        cap = cv2.VideoCapture(str(camera_dir / "rgb.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        n_video = len(self._rgb_frames)
        n_depth = self._depths.shape[0]
        n_ext = self._c2w.shape[0]
        n_total = min(n_video, n_depth, n_ext)

        self._rgb_frames = self._rgb_frames[:n_total]
        self._depths = self._depths[:n_total]
        self._c2w = self._c2w[:n_total]
        self._K_all = self._K_all[:n_total]

        # Apply stride and max_frames
        stride = max(1, self.cfg.frame_stride)
        indices = list(range(0, n_total, stride))
        if self.cfg.max_frames:
            indices = indices[: self.cfg.max_frames]

        self._rgb_frames = [self._rgb_frames[i] for i in indices]
        self._depths = self._depths[indices]
        self._c2w = self._c2w[indices]
        self._K_all = self._K_all[indices]
        self.num_frames = len(indices)

        # Build intrinsics dict from first frame K
        K0 = self._K_all[0]
        self.intrinsics = {
            "fx": float(K0[0, 0]),
            "fy": float(K0[1, 1]),
            "cx": float(K0[0, 2]),
            "cy": float(K0[1, 2]),
        }

    # ------------------------------------------------------------------
    # Legacy format: per-frame files
    # ------------------------------------------------------------------

    def _setup_legacy(self, camera_dir: Path) -> None:
        """Load data from per-frame PNG/NPY/JSON files (backward compat)."""
        self._is_safetensors = False

        rgb_dir = camera_dir / "rgb"
        if rgb_dir.exists():
            self.rgb_dir = rgb_dir
            self.depth_dir = camera_dir / "depth"
            self.extrinsic_dir = camera_dir / "extrinsics"
        else:
            # Fallback: frames_dir is already the rgb directory
            self.rgb_dir = self.cfg.frames_dir
            self.depth_dir = None
            self.extrinsic_dir = None

        frames = sorted(self.rgb_dir.glob("rgb_0*.png"))
        if not frames:
            frames = sorted(self.rgb_dir.glob("*.png"))

        self.num_frames = len(frames)
        stride = max(1, self.cfg.frame_stride)
        self._frames = frames[::stride]
        if self.cfg.max_frames:
            self._frames = self._frames[: self.cfg.max_frames]
        self.num_frames = len(self._frames)

    def _load_metadata(self) -> None:
        """Load metadata.json from scene directory."""
        # Scene dir is frames_dir (e.g. StaDy4D/short/test/scene_T03_008)
        scene_dir = self.cfg.frames_dir
        metadata_path = scene_dir / "metadata.json"
        if metadata_path.exists() and self.metadata is None:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

            # Extract intrinsics from metadata if not already loaded
            if self.intrinsics is None and "intrinsic" in self.metadata:
                self.intrinsics = self.metadata["intrinsic"]

        # Also check for legacy intrinsics.json in camera dir
        if self.intrinsics is None:
            search_dirs = [self.cfg.frames_dir]
            fmt_dir = self.cfg.frames_dir / self.cfg.data_format
            if fmt_dir.exists():
                search_dirs.insert(0, fmt_dir)
            if self.cfg.camera:
                cam_dir = fmt_dir / self.cfg.camera
                if cam_dir.exists():
                    search_dirs.insert(0, cam_dir)

            for base_dir in search_dirs:
                intrinsics_path = base_dir / "intrinsics.json"
                if intrinsics_path.exists():
                    with open(intrinsics_path, "r") as f:
                        self.intrinsics = json.load(f)
                    break

    def teardown(self) -> None:
        self._frames = []
        self._rgb_frames = None
        self._depths = None
        self._c2w = None
        self._K_all = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_first_frame(self) -> np.ndarray:
        """Return the first frame of the sequence."""
        if self._is_safetensors:
            if not self._rgb_frames:
                raise ValueError("No frames available. Call setup() first.")
            return self._rgb_frames[0]
        if not self._frames:
            raise ValueError("No frames available. Call setup() first.")
        return self._load_image(self._frames[0])

    def load_all_frames(self) -> List[np.ndarray]:
        """Load all frames into memory as a list of RGB numpy arrays."""
        if self._is_safetensors:
            if not self._rgb_frames:
                raise ValueError("No frames available. Call setup() first.")
            return list(self._rgb_frames)
        if not self._frames:
            raise ValueError("No frames available. Call setup() first.")
        return [self._load_image(p) for p in self._frames]

    def get_frame_paths(self) -> List[Path]:
        """Return the list of frame file paths (legacy format only)."""
        return list(self._frames)

    def iter_frame_pairs(self) -> Iterable[Tuple[Path, Path]]:
        """Yield consecutive frame path pairs (legacy format only)."""
        for prev, curr in zip(self._frames, self._frames[1:]):
            yield prev, curr

    def _load_image(self, path: Path) -> np.ndarray:
        """Load an image from disk using OpenCV."""
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_frame(self, index: int) -> np.ndarray:
        """Load a single frame by index."""
        if self._is_safetensors:
            if self._rgb_frames is None or index >= len(self._rgb_frames):
                raise ValueError(f"Frame index {index} out of range")
            return self._rgb_frames[index]
        if index >= len(self._frames):
            raise ValueError(f"Frame index {index} out of range")
        return self._load_image(self._frames[index])

    def stream(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Convenience generator that yields consecutive frame pairs."""
        if self._is_safetensors:
            for i in range(len(self._rgb_frames) - 1):
                yield self._rgb_frames[i], self._rgb_frames[i + 1]
        else:
            for prev_path, curr_path in self.iter_frame_pairs():
                yield self._load_image(prev_path), self._load_image(curr_path)

    # ------------------------------------------------------------------
    # Depth / extrinsics / intrinsics access
    # ------------------------------------------------------------------

    def load_depth(self, frame_idx: int) -> np.ndarray | None:
        """Load depth map for a specific frame index."""
        if self._is_safetensors:
            if self._depths is None or frame_idx >= self._depths.shape[0]:
                return None
            return self._depths[frame_idx].astype(np.float32)

        if self.depth_dir is None or not self.depth_dir.exists():
            return None
        depth_files = sorted(self.depth_dir.glob("depth_*.png"))
        if not depth_files:
            depth_files = sorted(self.depth_dir.glob("depth_*.npy"))
        if frame_idx >= len(depth_files):
            return None
        p = depth_files[frame_idx]
        if p.suffix == ".npy":
            return np.load(p)
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    def load_extrinsic(self, frame_idx: int) -> np.ndarray | None:
        """Load c2w extrinsic matrix for a specific frame index."""
        if self._is_safetensors:
            if self._c2w is None or frame_idx >= self._c2w.shape[0]:
                return None
            return self._c2w[frame_idx].astype(np.float64)

        if self.extrinsic_dir is None or not self.extrinsic_dir.exists():
            return None
        extrinsic_files = sorted(self.extrinsic_dir.glob("extrinsic_*.json"))
        if not extrinsic_files:
            extrinsic_files = sorted(self.extrinsic_dir.glob("extrinsic_*.npy"))
        if frame_idx >= len(extrinsic_files):
            return None
        p = extrinsic_files[frame_idx]
        if p.suffix == ".npy":
            return np.load(p)
        with open(p, "r") as f:
            return json.load(f)

    def load_intrinsic(self, frame_idx: int) -> np.ndarray | None:
        """Load 3x3 intrinsic matrix for a specific frame index."""
        if self._is_safetensors:
            if self._K_all is None or frame_idx >= self._K_all.shape[0]:
                return None
            return self._K_all[frame_idx].astype(np.float64)
        return None

    def get_intrinsics(self) -> Dict[str, Any] | None:
        """Get camera intrinsics if loaded."""
        return self.intrinsics

    def get_metadata(self) -> Dict[str, Any] | None:
        """Get sequence metadata if loaded."""
        return self.metadata

    def get_data_format(self) -> str:
        """Get the current data format (dynamic/static)."""
        return self.cfg.data_format
