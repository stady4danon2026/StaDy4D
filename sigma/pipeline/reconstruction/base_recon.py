"""Baseclass for reconstruction stage implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import BaseStage, StageResult


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters for a single view.

    Attributes:
        intrinsic: Camera intrinsic matrix (3x3) containing focal lengths and principal point.
                   Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        extrinsic: Camera extrinsic matrix (4x4) containing rotation and translation.
                   Format: [[R | t], [0 | 1]] where R is 3x3 rotation, t is 3x1 translation.
        distortion: Optional distortion coefficients (k1, k2, p1, p2, k3, ...).
    """

    intrinsic: Any  # shape (3, 3), np.ndarray or torch.Tensor
    extrinsic: Any  # shape (4, 4), np.ndarray or torch.Tensor
    distortion: Any | None = None  # shape (5,) or (8,), optional


@dataclass
class SceneReconstruction:
    """Complete 3D reconstruction output for a single timestep or view.

    Attributes:
        points_3d: Reconstructed 3D point cloud in world coordinates (N, 3).
        colors: RGB color values for each point (N, 3), range [0, 255] or [0, 1].
        depth_map: Per-pixel depth map in camera coordinates (H, W).
        confidence: Confidence score for each 3D point or depth pixel (N,) or (H, W).
        camera: Camera parameters (intrinsic and extrinsic matrices).
        normals: Optional surface normals for each point (N, 3).
        metadata: Additional reconstruction metadata (e.g., reprojection error, num_inliers).
    """

    points_3d: Any  # shape (N, 3), np.ndarray or torch.Tensor
    colors: Any | None = None  # shape (N, 3)
    depth_map: Any | None = None  # shape (H, W)
    confidence: Any | None = None  # shape (N,) or (H, W)
    camera: CameraParameters | None = None
    normals: Any | None = None  # shape (N, 3)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionOutputs:
    """Encapsulate reconstruction results for current and aggregated scenes.

    Attributes:
        current_scene: 3D reconstruction for the current timestep/view.
        aggregated_scene: Fused 3D scene from all processed frames.
        per_view_cameras: Dictionary mapping timestep to camera parameters {timestep: CameraParameters}.
        global_point_cloud: Optional merged point cloud from all views (N, 3).
        global_colors: Optional colors for global point cloud (N, 3).
        reconstruction_metadata: Global metadata (e.g., total_frames, avg_confidence).
    """
    frame_idx: int | None = None
    frame_size: tuple[int, int] | None = None  # (H, W)
    depth_map: Any | None = None  # shape (H, W)
    current_scene: SceneReconstruction | None = None
    aggregated_scene: Dict[str, Any] | None = None
    per_view_cameras: Dict[int, CameraParameters] = field(default_factory=dict)
    global_point_cloud: Any | None = None  # shape (M, 3)
    global_point_color: Any | None = None  # shape (M, 3)
    point_cloud: Any | None = None  # shape (M, 3)
    point_color: Any | None = None  # shape (M, 3)
    reconstruction_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_map: Any | None = None  # shape (H, W)
    extrinsic: Any | None = None  # shape (4, 4)
    intrinsic: Any | None = None  # shape (3, 3)
    current_frame: Any | None = None  # shape (H, W, 3)


class BaseReconstructor(BaseStage):
    """Abstract 3D/4D reconstruction class."""

    name: str = "base_reconstructor"
    run_deferred: bool = False  # If True, runner skips per-frame calls and calls finalize() at the end

    def __init__(self, aggregator: Any | None = None, **_: Any) -> None:
        self.aggregator = aggregator

    def setup(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:  # pragma: no cover - placeholder
        """Reconstruct the current scene and update the aggregated state."""
        raise NotImplementedError

    def finalize(self, frame_records: Dict[int, FrameIORecord]) -> list:  # pragma: no cover - optional
        """Run deferred reconstruction on all frames at once.

        Called by the runner after all inpainting is complete when run_deferred=True.
        Returns a list of (frame_idx, StageResult) tuples.
        """
        raise NotImplementedError

    def teardown(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError


# Utility functions for reconstruction (shared across all reconstructors)
def depth_to_pointcloud(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth map to 3D point cloud using camera parameters.

    This is a shared utility function that can be used by any reconstructor
    (VGGT, DUSt3R, etc.) to back-project depth maps to 3D world coordinates.

    Args:
        depth: Depth map in camera coordinates (H, W).
        image: RGB image for color extraction (H, W, 3), range [0, 255].
        intrinsic: Camera intrinsic matrix (3, 3) with format [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        extrinsic: Camera extrinsic matrix (3, 4) with format [[R | t]] where R is rotation, t is translation.
                   This is the camera-to-world transformation [R_c2w | t_c2w].

    Returns:
        Tuple of (points_3d, colors):
            - points_3d: 3D points in world coordinates (N, 3).
            - colors: RGB colors for each point (N, 3), range [0, 1].
    """
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    # Back-project to 3D camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to (N, 3)
    points_cam = np.stack([x, y, z], axis=-1)

    # Extract R_c2w and t_c2w
    R_c2w = extrinsic[:, :3]
    t_c2w = extrinsic[:, 3]

    # Transform points: P_world = R_c2w @ P_cam + t_c2w
    points_world = (points_cam @ R_c2w.T) + t_c2w

    # Extract colors and normalize to [0, 1]
    colors = image.reshape(-1, 3)

    return points_world, colors


def estimate_normals(depth_map: np.ndarray) -> np.ndarray:
    """Estimate surface normals from depth map using gradient.

    This is a shared utility function for computing surface normals from depth.
    Can be used by any reconstructor that outputs depth maps.

    Args:
        depth_map: Depth map (H, W).

    Returns:
        Normal map (H, W, 3) with unit normal vectors, flattened to (H*W, 3).
    """
    # Compute gradients in x and y directions
    gy, gx = np.gradient(depth_map)

    # Construct normal vectors: n = (-∂z/∂x, -∂z/∂y, 1)
    normals = np.stack([-gx, -gy, np.ones_like(depth_map)], axis=-1)

    # Normalize to unit vectors
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)

    return normals.reshape(-1, 3)


def extract_frames_from_records(
    frame_records: Dict[int, FrameIORecord],
    frame_idx: int,
    max_frames: int | None = None
) -> Dict[int, np.ndarray]:
    """Extract a frames dictionary from stored :class:`FrameIORecord` instances.

    Args:
        frame_records: Mapping of frame index to stored frame artifacts.
        frame_idx: Current timestep index being processed.
        max_frames: Maximum number of frames to extract. If specified, only the most
                   recent max_frames frames (up to and including frame_idx) will be returned.
                   If None, all frames up to frame_idx are returned.

    Returns:
        Dictionary mapping timestep to RGB frames (H, W, 3), preferring inpainted
        frames when available.
    """
    frames: Dict[int, np.ndarray] = {}
    for idx, record in sorted(frame_records.items()):
        if idx > frame_idx:
            continue
        frame = record.inpainted_image if record.inpainted_image is not None else record.origin_image
        if frame is not None:
            frames[idx] = frame

    # Limit to most recent max_frames if specified
    if max_frames is not None and len(frames) > max_frames:
        # Get the most recent max_frames timesteps
        sorted_timesteps = sorted(frames.keys())
        recent_timesteps = sorted_timesteps[-max_frames:]
        frames = {idx: frames[idx] for idx in recent_timesteps}

    return frames
