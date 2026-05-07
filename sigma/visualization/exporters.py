"""Visualization exporters for saving outputs per stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import imageio.v2 as imageio
import numpy as np

from sigma.pipeline.reconstruction.base_recon import ReconstructionOutputs
from sigma.utils.io import ensure_dir
from sigma.visualization.point_cloud import save_point_cloud_ply, save_point_cloud_gif

def export_image(array: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    imageio.imwrite(path, array)


def export_gif(frames: np.ndarray, path: Path, fps: int = 12) -> None:
    ensure_dir(path.parent)
    imageio.mimsave(path, frames, fps=fps)


def export_video(frames: np.ndarray, path: Path, fps: int = 12) -> None:
    ensure_dir(path.parent)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def export_reconstruction_frame(
    frame_idx: int,
    assets: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Export a single reconstruction frame directly into output_dir.

    Writes rgb/, depth/, depth_vis/, extrinsics/, intrinsics/, confidence/
    (and optionally composited/, dynamic_depth/) as flat subdirectories of
    output_dir — no extra nesting.

    Args:
        frame_idx: Frame index
        assets: Visualization assets from reconstruction stage
        output_dir: Run output directory (e.g., outputs/default_run/0226_1616)
    """
    depth_dir = output_dir / "depth"
    depth_vis_dir = output_dir / "depth_vis"
    extrinsics_dir = output_dir / "extrinsics"
    intrinsics_dir = output_dir / "intrinsics"
    confidence_dir = output_dir / "confidence"
    rgb_dir = output_dir / "rgb"

    for d in (depth_dir, depth_vis_dir, extrinsics_dir, intrinsics_dir, confidence_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Export depth map as .npy + normalised PNG preview
    depth_map = assets["depth_map"]
    if depth_map is not None:
        np.save(depth_dir / f"depth_{frame_idx:04d}.npy", depth_map)
        depth_vis = depth_map.copy().astype(np.float32)
        dmin, dmax = depth_vis.min(), depth_vis.max()
        if dmax > dmin:
            depth_vis = (depth_vis - dmin) / (dmax - dmin)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        export_image(depth_vis, depth_vis_dir / f"depth_{frame_idx:04d}.png")

    export_image(assets["current_frame"], rgb_dir / f"rgb_{frame_idx:04d}.png")
    np.save(extrinsics_dir / f"extrinsic_{frame_idx:04d}.npy", assets["extrinsic"])
    np.save(intrinsics_dir / f"intrinsic_{frame_idx:04d}.npy", assets["intrinsic"])
    np.save(confidence_dir / f"confidence_{frame_idx:04d}.npy", assets["confidence_map"])

    # Composited frame (moving object blended back onto scene).
    composited_frame = assets.get("composited_frame")
    if composited_frame is not None:
        composited_dir = output_dir / "composited"
        composited_dir.mkdir(parents=True, exist_ok=True)
        export_image(composited_frame, composited_dir / f"composited_{frame_idx:04d}.png")

    # Dynamic depth (car region only) for 4D rolling visualisation.
    dynamic_depth = assets.get("dynamic_depth")
    if dynamic_depth is not None:
        dynamic_depth_dir = output_dir / "dynamic_depth"
        dynamic_depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(dynamic_depth_dir / f"dynamic_depth_{frame_idx:04d}.npy", dynamic_depth)

# Per-method visualizer scale defaults.
# VGGT normalises its scene to unit scale (~42x smaller than metric).
# Pi3X local_points are approximately metric, so no scaling needed.
_METHOD_SCENE_SCALE: dict[str, float] = {
    "vggt_reconstructor": 42.0,
    "vggt_online_reconstructor": 42.0,
    "pi3_reconstructor": 1.0,
    "pi3_online_reconstructor": 1.0,
}


def finalize_reconstruction_export(
    output_dir: Path,
    frame_nums: int,
    frame_size: tuple[int, int] | None = None,
    fps: int = 30,
    reconstruction_method: str | None = None,
) -> None:
    """Create metadata.json directly in output_dir after all frames are exported.

    Args:
        output_dir: Run output directory (e.g., outputs/default_run/0226_1616)
        frame_nums: Total number of frames
        frame_size: (H, W) tuple from shape[:2]
        fps: Frames per second for metadata
        reconstruction_method: Reconstructor name (e.g. "vggt_reconstructor").
            Used to set a per-method default ``scene_scale`` for the visualizer.
    """
    scene_scale = _METHOD_SCENE_SCALE.get(reconstruction_method or "", 1.0)
    metadata = {
        "video_idx": 0,  # Default, can be overridden
        "num_frames": frame_nums,
        "fps": fps,
        "trajectory_type": "reconstruction",
        "resolution": {
            "width": frame_size[1],
            "height": frame_size[0],
        },
        "fov_deg": 70.0,  # Default FOV
        "reconstruction_method": reconstruction_method or "unknown",
        "scene_scale": scene_scale,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def dump_stage_assets(
    stage_name: str,
    assets: Dict[str, Any],
    base_dir: Path,
    save_gifs: bool = True,
    save_ply: bool = True,
) -> None:
    """Persist visualization assets for a stage.

    For reconstruction, writes only the structured per-frame files
    (depth, rgb, extrinsics, intrinsics, confidence) and returns early —
    the generic numpy dump is intentionally skipped to avoid saving large
    redundant arrays on every frame.  Call save_reconstruction_summary()
    once at the end of the pipeline to write the global PLY / GIF.

    For all other stages, a generic per-key export is performed.
    """
    if stage_name == "reconstruction":
        frame_idx = int(assets["frame_idx"])
        # base_dir = vis_dir/reconstruction/frame_XXXXX → .parent.parent = vis_dir (output_dir)
        export_reconstruction_frame(frame_idx, assets, base_dir.parent.parent)
        return  # skip generic dump — no per-frame .npy flood, no per-frame GIF

    # Generic export for motion / inpainting / etc.
    ensure_dir(base_dir)
    for key, value in assets.items():
        if value is None:
            continue
        if key in {"rgb_frame", "current_frame_uint8", "current_frame", "origin_frame"}:
            target_stem = base_dir / f"{stage_name}_{key}"
            export_image(value, target_stem.with_suffix(".png"))
            continue
        target_stem = base_dir / f"{stage_name}_{key}"
        if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim in (2, 3):
            export_image(value, target_stem.with_suffix(".png"))
            continue
        if isinstance(value, np.ndarray):
            np.save(target_stem.with_suffix(".npy"), value)


def save_reconstruction_summary(
    output_dir: Path,
    global_points: Optional[np.ndarray],
    global_colors: Optional[np.ndarray],
    save_ply: bool = True,
    save_gifs: bool = True,
) -> None:
    """Save the final accumulated global point cloud as PLY and/or GIF.

    Called once after all frames have been exported, so only a single file
    is written rather than one per frame.
    """
    if global_points is None or len(global_points) == 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_ply:
        save_point_cloud_ply(output_dir / "point_cloud.ply", global_points, colors=global_colors)
    if save_gifs:
        save_point_cloud_gif(
            output_dir / "point_cloud.gif",
            global_points,
            colors=global_colors,
            num_frames=6,
            fps=3,
            title="VGGT Demo Reconstruction",
        )