"""
Example script to visualize and use the generated CARLA dataset
"""
import numpy as np
import cv2
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt


def load_frame(video_dir, frame_idx):
    """Load all data for a single frame"""
    video_path = Path(video_dir)

    # Load RGB
    rgb_path = video_path / f"rgb_{frame_idx:04d}.png"
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load depth
    depth_path = video_path / f"depth_{frame_idx:04d}.npy"
    depth = np.load(str(depth_path))

    # Load extrinsics
    extr_path = video_path / f"extrinsic_{frame_idx:04d}.json"
    with open(extr_path) as f:
        extrinsic = json.load(f)

    # Load intrinsics
    intr_path = video_path / "intrinsics.json"
    with open(intr_path) as f:
        intrinsic = json.load(f)

    return rgb, depth, extrinsic, intrinsic


def depth_to_pointcloud(depth, intrinsic, max_depth=50.0):
    """Convert depth map to 3D point cloud in camera coordinates"""
    h, w = depth.shape
    fx, fy = intrinsic['fx'], intrinsic['fy']
    cx, cy = intrinsic['cx'], intrinsic['cy']

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filter by max depth
    valid_mask = (depth > 0) & (depth < max_depth)

    # Back-project to 3D
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    return points, valid_mask


def visualize_single_frame(video_dir, frame_idx=0):
    """Visualize RGB, depth, and point cloud for a single frame"""
    rgb, depth, extrinsic, intrinsic = load_frame(video_dir, frame_idx)

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # RGB
    ax1 = fig.add_subplot(131)
    ax1.imshow(rgb)
    ax1.set_title(f"RGB Frame {frame_idx}")
    ax1.axis('off')

    # Depth
    ax2 = fig.add_subplot(132)
    depth_vis = np.clip(depth / 50.0, 0, 1)
    im = ax2.imshow(depth_vis, cmap='jet')
    ax2.set_title(f"Depth Map (0-50m)")
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # Point cloud (top-down view)
    ax3 = fig.add_subplot(133)
    points, valid_mask = depth_to_pointcloud(depth, intrinsic, max_depth=50.0)

    # Sample points for visualization (every 10th point)
    sampled_points = points[::10]

    ax3.scatter(sampled_points[:, 0], sampled_points[:, 2],
                c=sampled_points[:, 1], cmap='viridis', s=0.1)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Point Cloud (Top View)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print camera pose
    loc = extrinsic['location']
    rot = extrinsic['rotation']
    print(f"\nCamera Pose:")
    print(f"  Location: x={loc['x']:.2f}, y={loc['y']:.2f}, z={loc['z']:.2f}")
    print(f"  Rotation: pitch={rot['pitch']:.2f}, yaw={rot['yaw']:.2f}, roll={rot['roll']:.2f}")
    print(f"\nPoint cloud: {len(points)} points")

    return fig


def compare_static_dynamic(static_dir, dynamic_dir, frame_idx=50):
    """Compare static and dynamic scenes side by side"""
    # Load data
    rgb_s, depth_s, _, _ = load_frame(static_dir, frame_idx)
    rgb_d, depth_d, _, _ = load_frame(dynamic_dir, frame_idx)

    # Compute difference
    diff = cv2.absdiff(rgb_s, rgb_d)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    # Static RGB
    ax1 = fig.add_subplot(231)
    ax1.imshow(rgb_s)
    ax1.set_title("Static Scene - RGB")
    ax1.axis('off')

    # Dynamic RGB
    ax2 = fig.add_subplot(232)
    ax2.imshow(rgb_d)
    ax2.set_title("Dynamic Scene - RGB")
    ax2.axis('off')

    # Difference
    ax3 = fig.add_subplot(233)
    ax3.imshow(diff)
    ax3.set_title("RGB Difference")
    ax3.axis('off')

    # Static Depth
    ax4 = fig.add_subplot(234)
    im1 = ax4.imshow(depth_s, cmap='jet', vmin=0, vmax=50)
    ax4.set_title("Static Scene - Depth")
    ax4.axis('off')
    plt.colorbar(im1, ax=ax4, fraction=0.046)

    # Dynamic Depth
    ax5 = fig.add_subplot(235)
    im2 = ax5.imshow(depth_d, cmap='jet', vmin=0, vmax=50)
    ax5.set_title("Dynamic Scene - Depth")
    ax5.axis('off')
    plt.colorbar(im2, ax=ax5, fraction=0.046)

    # Depth Difference
    ax6 = fig.add_subplot(236)
    depth_diff = np.abs(depth_s - depth_d)
    im3 = ax6.imshow(depth_diff, cmap='hot', vmin=0, vmax=10)
    ax6.set_title("Depth Difference")
    ax6.axis('off')
    plt.colorbar(im3, ax=ax6, fraction=0.046)

    plt.tight_layout()
    return fig


def create_video_ffmpeg(video_dir, output_path="output_video.mp4", fps=30, pattern="rgb_%04d.png"):
    """
    Create MP4 video from frames using ffmpeg (much faster than cv2)

    Args:
        video_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
        pattern: Frame filename pattern (use %04d for frame numbers)
    """
    video_path = Path(video_dir)

    # Check if frames exist
    first_frame = video_path / pattern.replace('%04d', '0000')
    if not first_frame.exists():
        print(f"No frames found matching pattern {pattern} in {video_dir}")
        return False

    # Count frames
    if 'rgb' in pattern:
        frame_files = sorted(video_path.glob("rgb_*.png"))
    else:
        frame_files = sorted(video_path.glob("depth_vis_*.png"))

    print(f"Creating video from {len(frame_files)} frames using ffmpeg...")

    # Build ffmpeg command
    # -framerate: input framerate
    # -i: input pattern
    # -c:v libx264: use H.264 codec
    # -pix_fmt yuv420p: pixel format for compatibility
    # -crf 18: quality (lower = better, 18 is visually lossless)
    # -preset slow: encoding speed vs compression ratio
    # -y: overwrite output file
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', str(video_path / pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'medium',
        '-y',
        output_path
    ]

    try:
        # Run ffmpeg (suppress output for cleaner console)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✓ Video saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  MacOS: brew install ffmpeg")
        print("  Windows: download from https://ffmpeg.org/download.html")
        return False


def create_side_by_side_video(static_dir, dynamic_dir, output_path="comparison.mp4", fps=30):
    """
    Create side-by-side comparison video of static and dynamic scenes using ffmpeg

    Args:
        static_dir: Directory with static scene frames
        dynamic_dir: Directory with dynamic scene frames
        output_path: Output video path
        fps: Frames per second
    """
    static_path = Path(static_dir)
    dynamic_path = Path(dynamic_dir)

    print(f"Creating side-by-side comparison video using ffmpeg...")

    # Build ffmpeg command with hstack filter for side-by-side
    # Complex filter: [0:v][1:v]hstack=inputs=2[v]
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', str(static_path / 'rgb_%04d.png'),
        '-framerate', str(fps),
        '-i', str(dynamic_path / 'rgb_%04d.png'),
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'medium',
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✓ Comparison video saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg error: {e.stderr}")
        return False


def create_all_videos(base_dir="output", fps=30):
    """
    Create videos for all generated datasets using ffmpeg

    Args:
        base_dir: Base output directory
        fps: Frames per second
    """
    base_path = Path(base_dir)

    print("="*60)
    print("Creating Videos with ffmpeg")
    print("="*60)

    # Process static videos
    static_dir = base_path / "static"
    if static_dir.exists():
        print("\nStatic scene videos:")
        for video_idx in range(5):
            video_path = static_dir / f"video_{video_idx:02d}"
            if video_path.exists():
                output = f"static_video_{video_idx:02d}.mp4"
                create_video_ffmpeg(video_path, output, fps)

                # Also create depth visualization video
                depth_output = f"static_depth_{video_idx:02d}.mp4"
                create_video_ffmpeg(video_path, depth_output, fps, pattern="depth_vis_%04d.png")

    # Process dynamic videos
    dynamic_dir = base_path / "dynamic"
    if dynamic_dir.exists():
        print("\nDynamic scene videos:")
        for video_idx in range(5):
            video_path = dynamic_dir / f"video_{video_idx:02d}"
            if video_path.exists():
                output = f"dynamic_video_{video_idx:02d}.mp4"
                create_video_ffmpeg(video_path, output, fps)

                # Also create depth visualization video
                depth_output = f"dynamic_depth_{video_idx:02d}.mp4"
                create_video_ffmpeg(video_path, depth_output, fps, pattern="depth_vis_%04d.png")

    # Create side-by-side comparisons
    print("\nCreating comparison videos:")
    for video_idx in range(5):
        static_vid = static_dir / f"video_{video_idx:02d}"
        dynamic_vid = dynamic_dir / f"video_{video_idx:02d}"
        if static_vid.exists() and dynamic_vid.exists():
            output = f"comparison_video_{video_idx:02d}.mp4"
            create_side_by_side_video(static_vid, dynamic_vid, output, fps)

    print("\n" + "="*60)
    print("Video creation complete!")
    print("="*60)


if __name__ == "__main__":
    # Example 1: Visualize single frame
    print("="*60)
    print("Example 1: Visualize Single Frame")
    print("="*60)

    video_dir = "output/static/video_00"
    if Path(video_dir).exists():
        fig = visualize_single_frame(video_dir, frame_idx=50)
        plt.savefig("visualization_single_frame.png", dpi=150, bbox_inches='tight')
        print("Saved: visualization_single_frame.png")
    else:
        print(f"Directory not found: {video_dir}")
        print("Please run gen_carla.py first to generate data")

    # Example 2: Compare static vs dynamic
    print("\n" + "="*60)
    print("Example 2: Compare Static vs Dynamic")
    print("="*60)

    static_dir = "output/static/video_00"
    dynamic_dir = "output/dynamic/video_00"

    if Path(static_dir).exists() and Path(dynamic_dir).exists():
        fig = compare_static_dynamic(static_dir, dynamic_dir, frame_idx=50)
        plt.savefig("visualization_comparison.png", dpi=150, bbox_inches='tight')
        print("Saved: visualization_comparison.png")
    else:
        print("Static or dynamic directory not found")

    # Example 3: Create videos using ffmpeg
    print("\n" + "="*60)
    print("Example 3: Create MP4 Videos with ffmpeg")
    print("="*60)

    if Path("output").exists():
        create_all_videos(base_dir="output", fps=30)
    else:
        print("Output directory not found. Run gen_carla.py first.")
