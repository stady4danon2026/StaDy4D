#!/usr/bin/env python3
"""Export reconstruction outputs to video_04 format.

This script reorganizes reconstruction outputs into video_04 structure with:
- RGB frames as PNG
- Depth maps as .npy files  
- Extrinsics as JSON
- RGB MP4 and GIF videos
- Point cloud GIF
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def collect_frames(input_dir: Path) -> List[Path]:
    """Collect all frame directories sorted by frame number.
    
    Args:
        input_dir: Root directory containing frame_XXXXX subdirectories
        
    Returns:
        Sorted list of frame directory paths
    """
    frame_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])
    return frame_dirs


def load_depth_npy(npy_path: Path) -> Optional[np.ndarray]:
    """Load depth map from .npy file.
    
    Args:
        npy_path: Path to depth .npy file
        
    Returns:
        Depth map array or None if loading fails
    """
    try:
        depth = np.load(npy_path, allow_pickle=True)
        return depth
    except Exception as e:
        print(f"Warning: Failed to load {npy_path}: {e}")
        return None


def png_to_numpy_array(png_path: Path) -> Optional[np.ndarray]:
    """Convert PNG depth visualization to numpy array.
    
    Args:
        png_path: Path to depth PNG file
        
    Returns:
        Numpy array in RGB format
    """
    try:
        img = Image.open(png_path)
        return np.array(img)
    except Exception as e:
        print(f"Warning: Failed to load {png_path}: {e}")
        return None


def load_npy_matrix(npy_path: Path) -> Optional[np.ndarray]:
    """Load a numpy matrix saved by the reconstruction stage."""
    if not npy_path.exists():
        return None
    try:
        return np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"Warning: Failed to load {npy_path}: {e}")
        return None


def export_frames(frame_dirs: List[Path], output_dir: Path, folder_name: str = "predict") -> tuple[bool, Optional[np.ndarray]]:
    """Export frames to organized directory structure.
    
    Args:
        frame_dirs: List of frame directory paths
        output_dir: Output directory
        folder_name: Name of the main folder (default: "predict")
    Returns:
        Tuple of (has_rgb, first_intrinsic) where has_rgb indicates if any RGB frames
        were exported and first_intrinsic is the first intrinsic matrix found.
    """
    # Create output directories
    predict_dir = output_dir / folder_name
    depth_dir = predict_dir / "depth"
    depth_vis_dir = predict_dir / "depth_vis"
    extrinsic_dir = predict_dir / "extrinsic"
    rgb_dir = predict_dir / "rgb"
    
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {len(frame_dirs)} frames...")
    
    has_rgb = False
    first_intrinsic: Optional[np.ndarray] = None
    last_frame_idx = len(frame_dirs) - 1
    
    for idx, frame_dir in enumerate(tqdm(frame_dirs)):
        frame_num = idx
        
        # Handle depth map: save PNG to depth_vis, npy to depth
        depth_png_src = frame_dir / "reconstruction_depth_map.png"
        if depth_png_src.exists():
            # Save visualization to depth_vis
            depth_vis_dst = depth_vis_dir / f"depth_{frame_num:04d}.png"
            shutil.copy(depth_png_src, depth_vis_dst)
            
            # Convert PNG to numpy and save as .npy in depth folder (not .png)
            depth_array = png_to_numpy_array(depth_png_src)
            if depth_array is not None:
                depth_npy_dst = depth_dir / f"depth_{frame_num:04d}.npy"
                np.save(depth_npy_dst, depth_array)
        
        # Look for actual RGB frames saved by vggt_adapter
        rgb_frame_files = sorted(frame_dir.glob("reconstruction_rgb_frame_*.png"))
        if not rgb_frame_files:
            # Newer exports save a single file: reconstruction_rgb_frame.png
            single_rgb = frame_dir / "reconstruction_rgb_frame.png"
            if single_rgb.exists():
                rgb_frame_files = [single_rgb]

        if rgb_frame_files:
            rgb_dst = rgb_dir / f"rgb_{frame_num:04d}.png"
            shutil.copy(rgb_frame_files[0], rgb_dst)
            has_rgb = True
        
        intrinsic = load_npy_matrix(frame_dir / "reconstruction_intrinsic.npy")
        if first_intrinsic is None and intrinsic is not None:
            first_intrinsic = intrinsic

        extrinsic = load_npy_matrix(frame_dir / "reconstruction_extrinsic.npy")
        pose_payload: Dict[str, Any] = {}
        if extrinsic is not None:
            pose_payload["extrinsic_matrix"] = extrinsic.tolist()
            if extrinsic.shape[1] >= 4:
                pose_payload["rotation_matrix"] = extrinsic[:, :3].tolist()
                pose_payload["translation"] = extrinsic[:, 3].tolist()
        if intrinsic is not None:
            pose_payload["intrinsic_matrix"] = intrinsic.tolist()
            pose_payload.update(
                {
                    "fx": float(intrinsic[0, 0]),
                    "fy": float(intrinsic[1, 1]),
                    "cx": float(intrinsic[0, 2]),
                    "cy": float(intrinsic[1, 2]),
                }
            )

        if not pose_payload:
            pose_payload = {
                "extrinsic_matrix": None,
                "intrinsic_matrix": None,
            }

        extrinsic_dst = extrinsic_dir / f"extrinsic_{frame_num:04d}.json"
        with open(extrinsic_dst, "w") as f:
            json.dump(pose_payload, f, indent=2)
            
        # Copy point cloud GIF from the last frame
        if idx == last_frame_idx:
            pc_gif_src = frame_dir / "reconstruction_point_cloud.gif"
            if pc_gif_src.exists():
                pc_gif_dst = output_dir / f"{folder_name}_point_cloud.gif"
                shutil.copy(pc_gif_src, pc_gif_dst)
                print(f"\\nCopied point cloud GIF")
    
    # Save the first intrinsic alongside frame exports for convenience
    if first_intrinsic is not None:
        np.save(predict_dir / "intrinsic_first.npy", first_intrinsic)

    return has_rgb, first_intrinsic




def create_rgb_videos(output_dir: Path, folder_name: str = "predict", fps: int = 30) -> None:
    """Create MP4 and GIF videos from RGB frames.
    
    Args:
        output_dir: Output directory
        folder_name: Name of the main folder containing frames
        fps: Frames per second for videos
    """
    predict_dir = output_dir / folder_name
    rgb_dir = predict_dir / "rgb"
    
    # Collect RGB frames
    rgb_frames = sorted(rgb_dir.glob("rgb_*.png"))
    
    if not rgb_frames:
        print("No RGB frames found, skipping video generation")
        return
    
    print(f"Creating RGB videos from {len(rgb_frames)} frames...")
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(str(rgb_frames[0]))
    if first_frame is None:
        print("Error reading first frame")
        return
    
    height, width = first_frame.shape[:2]
    
    # Create MP4
    mp4_path = output_dir / f"{folder_name}_rgb.mp4"
    print(f"Creating RGB MP4: {mp4_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
    
    frames_for_gif = []
    for frame_path in tqdm(rgb_frames, desc="Creating RGB MP4"):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            video_writer.write(frame)
            # Convert BGR to RGB for GIF
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_for_gif.append(Image.fromarray(frame_rgb))
    
    video_writer.release()
    
    # Create GIF (use lower fps for smaller size)
    gif_fps = min(fps, 12)
    gif_path = output_dir / f"{folder_name}_rgb.gif"
    
    if frames_for_gif:
        frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=int(1000 / gif_fps),
            loop=0
        )
    
\

def create_metadata(output_dir: Path) -> None:
    """Create metadata.json file.
    
    Args:
        output_dir: Output directory
    """
    metadata = {
        "source": "reconstruction_carla",
        "type": "predicted_reconstruction"
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def create_intrinsics(output_dir: Path, intrinsic_matrix: Optional[np.ndarray]) -> None:
    """Persist intrinsics.json file using reconstructed camera parameters when available."""
    if intrinsic_matrix is not None:
        intrinsics = {
            "fx": float(intrinsic_matrix[0, 0]),
            "fy": float(intrinsic_matrix[1, 1]),
            "cx": float(intrinsic_matrix[0, 2]),
            "cy": float(intrinsic_matrix[1, 2]),
            "matrix": intrinsic_matrix.tolist(),
        }
    else:
        intrinsics = {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 256.0,
            "cy": 256.0,
            "matrix": None,
        }
 
    intrinsics_path = output_dir / "intrinsics.json"
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Export reconstruction outputs to video_04 format")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/visuals/reconstruction",
        help="Input directory containing reconstruction frames"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/reconstruction_carla",
        help="Output directory for organized videos"
    )
    parser.add_argument(
        "--folder-name",
        type=str,
        default="predict",
        help="Name of the main folder (default: predict)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for MP4 video"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Collect frames
    frame_dirs = collect_frames(input_dir)
    if not frame_dirs:
        print(f"No frame directories found in {input_dir}")
        return
    
    print(f"Found {len(frame_dirs)} frames in {input_dir}")
    
    # Export frames to organized structure
    has_rgb, first_intrinsic = export_frames(frame_dirs, output_dir, args.folder_name)
    
    # Create RGB videos if RGB frames exist
    if has_rgb:
        create_rgb_videos(output_dir, args.folder_name, args.fps)
    
    # Create metadata and intrinsics
    create_metadata(output_dir)
    create_intrinsics(output_dir, first_intrinsic)
    
    print(f"\nExport complete! Output saved to: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"    {args.folder_name}/")
    print(f"      depth/          (*.npy files)")
    print(f"      depth_vis/      (*.png visualizations)")
    print(f"      rgb/            (*.png frames)")
    print(f"      extrinsic/      (*.json files)")
    print(f"    {args.folder_name}_rgb.mp4")
    print(f"    {args.folder_name}_rgb.gif")
    print(f"    {args.folder_name}_point_cloud.gif")
    print(f"    metadata.json")
    print(f"    intrinsics.json")


if __name__ == "__main__":
    main()
