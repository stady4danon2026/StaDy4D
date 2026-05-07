"""Point cloud visualization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

LOGGER = logging.getLogger(__name__)


def save_point_cloud_ply(
    filepath: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> None:
    """Save point cloud to PLY format.

    Args:
        filepath: Output PLY file path.
        points: Point cloud (N, 3).
        colors: Optional RGB colors (N, 3) in range [0, 1].
        normals: Optional surface normals (N, 3).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(points)

    # Convert colors to 0-255 range if provided
    if colors is not None:
        colors = (colors * 255).astype(np.uint8)

    # Write PLY file
    with open(filepath, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")

        f.write("end_header\n")

        # Data
        for i in range(n_points):
            x, y, z = points[i]
            f.write(f"{x} {y} {z}")

            if colors is not None:
                r, g, b = colors[i]
                f.write(f" {r} {g} {b}")

            if normals is not None:
                nx, ny, nz = normals[i]
                f.write(f" {nx} {ny} {nz}")

            f.write("\n")

def save_point_cloud_gif(
    filepath: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    num_frames: int = 12,
    fps: int = 3,
    elevation: float = 30,
    title: str = "Point Cloud",
    downsample: int | None = None,
) -> None:
    """Generate rotating point cloud visualization as animated GIF.

    Args:
        filepath: Output GIF file path.
        points: Point cloud (N, 3).
        colors: Optional RGB colors (N, 3) in range [0, 1].
        num_frames: Number of frames in rotation.
        fps: Frames per second.
        elevation: Camera elevation angle in degrees.
        title: Plot title.
        downsample: Optional downsampling factor (take every Nth point).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if points is None or len(points) == 0:
        LOGGER.warning("Empty point cloud, skipping GIF generation")
        return

    # Ensure points are numpy array
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # Validate colors
    if colors is not None:
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors)

        if colors.max() > 1.0:
            colors = colors / 255.0

    if downsample is None and len(points) > 100000:
        downsample = max(1, len(points) // 100000)

    if downsample is not None and downsample > 1:
        points = points[::downsample]
        if colors is not None:
            colors = colors[::downsample]

    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(111, projection="3d")

    # Compute point cloud bounds
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    max_range = max(ranges.max(), 0.1)  # Ensure non-zero range

    # Center point
    center = (mins + maxs) / 2

    # Set equal aspect ratio
    margin = max_range * 0.1  # 10% margin
    ax.set_xlim(center[0] - max_range / 2 - margin, center[0] + max_range / 2 + margin)
    ax.set_ylim(center[2] - max_range / 2 - margin, center[2] + max_range / 2 + margin)
    ax.set_zlim(center[1] - margin, center[1] + max_range + margin)

    ax.set_xlabel("X (Plane)")
    ax.set_ylabel("Y (Plane)")
    ax.set_zlabel("Z (Depth)")
    ax.set_title(title)

    # Initial scatter plot
    # Remap: X->X, Y->Z (up), Z->Y to make Y-axis vertical
    if colors is not None:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 2],
            -points[:, 1],
            c=colors,
            s=1,
            alpha=0.8,
            edgecolors="none",
        )
    else:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            c=points[:, 1],  # Color by Y coordinate (height)
            cmap="viridis",
            s=1,
            alpha=0.8,
            edgecolors="none",
        )

    def update(frame):
        """Update function for animation."""
        azimuth = frame * 360 / num_frames
        # Keep camera level (elev=0) and rotate around vertical axis
        ax.view_init(elev=elevation, azim=azimuth, vertical_axis='z')
        return (scatter,)

    anim = FuncAnimation(
        fig, update, frames=num_frames, interval=1000 / fps, blit=False, repeat=True
    )

    # Save as GIF
    LOGGER.info(f"Saving GIF to {filepath}")
    writer = PillowWriter(fps=fps)
    anim.save(str(filepath), writer=writer)

    plt.close(fig)


def visualize_depth_map(
    depth_map: np.ndarray,
    filepath: Path | None = None,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Depth Map",
) -> None:
    """Visualize and save depth map.

    Args:
        depth_map: Depth map (H, W).
        filepath: Optional output file path. If None, display interactively.
        colormap: Matplotlib colormap name.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(depth_map, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth", rotation=270, labelpad=15)

    if filepath is not None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        LOGGER.info(f"Saved depth map to {filepath}")
        plt.close(fig)
    else:
        plt.show()


def visualize_camera_poses(
    camera_params: dict,
    filepath: Path | None = None,
    title: str = "Camera Poses",
) -> None:
    """Visualize camera poses in 3D.

    Args:
        camera_params: Dictionary mapping timestep to CameraParameters.
        filepath: Optional output file path.
        title: Plot title.
    """
    from sigma.pipeline.reconstruction.base_recon import CameraParameters

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Extract camera positions and orientations
    positions = []
    orientations = []

    for timestep in sorted(camera_params.keys()):
        cam: CameraParameters = camera_params[timestep]

        # Camera position in world coordinates
        # Camera center: C = -R^T * t
        R = cam.extrinsic[:3, :3]
        t = cam.extrinsic[:3, 3]
        camera_center = -R.T @ t

        positions.append(camera_center)

        # Camera forward direction (negative Z axis in camera coords)
        forward = -R[:, 2]
        orientations.append(forward)

    positions = np.array(positions)
    orientations = np.array(orientations)

    # Plot camera positions
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="red",
        s=100,
        marker="o",
        label="Cameras",
    )

    # Plot camera orientations as arrows
    for i, (pos, orient) in enumerate(zip(positions, orientations)):
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            orient[0],
            orient[1],
            orient[2],
            length=0.5,
            color="blue",
            arrow_length_ratio=0.3,
        )

    # Plot trajectory
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="green",
        linewidth=2,
        alpha=0.5,
        label="Trajectory",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array(
        [
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min(),
        ]
    ).max()

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    if filepath is not None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        LOGGER.info(f"Saved camera poses to {filepath}")
        plt.close(fig)
    else:
        plt.show()


def compare_depth_maps(
    depth_maps: dict[str, np.ndarray],
    filepath: Path | None = None,
    colormap: str = "viridis",
) -> None:
    """Compare multiple depth maps side by side.

    Args:
        depth_maps: Dictionary mapping name to depth map.
        filepath: Optional output file path.
        colormap: Matplotlib colormap name.
    """
    n = len(depth_maps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    # Find global min/max for consistent scaling
    all_depths = np.concatenate([d.flatten() for d in depth_maps.values()])
    vmin, vmax = all_depths.min(), all_depths.max()

    for ax, (name, depth) in zip(axes, depth_maps.items()):
        im = ax.imshow(depth, cmap=colormap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Depth", rotation=270, labelpad=15)

    plt.tight_layout()

    if filepath is not None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        LOGGER.info(f"Saved depth map comparison to {filepath}")
        plt.close(fig)
    else:
        plt.show()