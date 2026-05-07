import argparse
import sys
from pathlib import Path
import numpy as np
import imageio


def validate_input_directory(input_dir: Path) -> dict:
    """
    Validate input directory and load all required NumPy arrays.

    Args:
        input_dir: Path to directory containing the .npy files

    Returns:
        Dictionary containing loaded arrays

    Raises:
        FileNotFoundError: If input directory or required files don't exist
        ValueError: If arrays have mismatched dimensions
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Define required files
    required_files = {
        'extrinsics': 'reconstruction_all_extrinsic.npy',
        'confidence': 'reconstruction_predictions_depth_conf.npy',
        'depth': 'reconstruction_predictions_depth.npy',
        'intrinsics': 'reconstruction_all_intrinsic.npy'
    }

    # Define optional files
    optional_files = {
        'rgb': 'reconstruction_images_tensor.npy'
    }

    # Load required arrays
    arrays = {}
    for key, filename in required_files.items():
        filepath = input_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

        try:
            arrays[key] = np.load(filepath)
            print(f"Loaded {filename}: shape {arrays[key].shape}")
        except Exception as e:
            raise ValueError(f"Error loading {filepath}: {e}")

    # Load optional arrays
    for key, filename in optional_files.items():
        filepath = input_dir / filename
        if filepath.exists():
            try:
                arrays[key] = np.load(filepath)
                print(f"Loaded {filename}: shape {arrays[key].shape}")
            except Exception as e:
                print(f"Warning: Could not load optional file {filepath}: {e}")

    # Validate that all arrays have the same number of frames
    num_frames_list = [arr.shape[0] for arr in arrays.values()]
    if len(set(num_frames_list)) != 1:
        frames_info = ", ".join([f"{key}={arr.shape[0]}" for key, arr in arrays.items()])
        raise ValueError(f"Arrays have mismatched number of frames: {frames_info}")

    num_frames = num_frames_list[0]
    print(f"\nValidation successful: {num_frames} frames detected")

    return arrays


def create_output_structure(output_dir: Path) -> dict:
    """
    Create the visualization output directory structure.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary mapping data types to their output directories
    """
    # Create predict directory
    predict_dir = output_dir

    # Create subdirectories
    subdirs = {
        'confidence': predict_dir / "confidence",
        'depth': predict_dir / "depth",
        'depth_vis': predict_dir / "depth_vis",
        'extrinsics': predict_dir / "extrinsics",
        'intrinsics': predict_dir / "intrinsics",
        'rgb': predict_dir / "rgb"
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreated output directory structure at: {output_dir}")
    return subdirs


def generate_depth_visualization(depth_map: np.ndarray) -> np.ndarray:
    """
    Generate a normalized depth visualization.

    Based on sigma/visualization/exporters.py:68-73

    Args:
        depth_map: Raw depth map array

    Returns:
        Normalized depth map as uint8 array
    """
    depth_vis = depth_map.copy()

    # Handle 3D arrays (H x W x 1) by squeezing
    if depth_vis.ndim == 3 and depth_vis.shape[-1] == 1:
        depth_vis = depth_vis.squeeze(-1)

    # Min-max normalize depth map
    depth_min = depth_vis.min()
    depth_max = depth_vis.max()

    if depth_max > depth_min:
        depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
    else:
        # Handle constant depth maps
        depth_vis = np.zeros_like(depth_vis)

    depth_vis = (depth_vis * 255).astype(np.uint8)

    return depth_vis


def process_rgb_frame(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Process RGB frame to ensure it's in uint8 format.

    Based on sigma/visualization/exporters.py:86-91

    Args:
        rgb_frame: RGB frame array (H x W x 3)

    Returns:
        RGB frame as uint8 array
    """
    # Ensure it's uint8 format
    if rgb_frame.dtype != np.uint8:
        if rgb_frame.max() <= 1.0:
            # Assume normalized [0, 1] range
            rgb_frame = np.clip(rgb_frame * 255.0, 0, 255).astype(np.uint8)
        else:
            # Assume [0, 255] range but wrong dtype
            rgb_frame = rgb_frame.astype(np.uint8)

    return rgb_frame


def convert_arrays_to_frames(arrays: dict, output_dirs: dict):
    """
    Split arrays by frame and save individual files.

    Args:
        arrays: Dictionary of loaded NumPy arrays
        output_dirs: Dictionary of output directories
    """
    num_frames = arrays['depth'].shape[0]

    print(f"\nConverting {num_frames} frames...")

    for frame_idx in range(num_frames):
        # Frame numbers are 1-indexed with 4-digit zero-padding
        frame_num = frame_idx + 1

        # Save extrinsics
        extrinsic_path = output_dirs['extrinsics'] / f"extrinsic_{frame_num:04d}.npy"
        np.save(extrinsic_path, arrays['extrinsics'][frame_idx])

        # Save confidence
        confidence_path = output_dirs['confidence'] / f"confidence_{frame_num:04d}.npy"
        np.save(confidence_path, arrays['confidence'][frame_idx])

        # Save depth
        depth_path = output_dirs['depth'] / f"depth_{frame_num:04d}.npy"
        np.save(depth_path, arrays['depth'][frame_idx])

        # Save intrinsics
        intrinsic_path = output_dirs['intrinsics'] / f"intrinsic_{frame_num:04d}.npy"
        np.save(intrinsic_path, arrays['intrinsics'][frame_idx])

        # Generate and save depth visualization
        depth_vis = generate_depth_visualization(arrays['depth'][frame_idx])
        depth_vis_path = output_dirs['depth_vis'] / f"depth_{frame_num:04d}.png"
        imageio.imwrite(depth_vis_path, depth_vis)

        # Save RGB frame if available
        if 'rgb' in arrays:
            rgb_frame = process_rgb_frame(arrays['rgb'][frame_idx])
            rgb_path = output_dirs['rgb'] / f"rgb_{frame_num:04d}.png"
            imageio.imwrite(rgb_path, rgb_frame)

        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames")

    print(f"\nConversion complete! Output saved to: {output_dirs['depth'].parent.parent}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NumPy arrays to SIGMA visualization directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python script/convert_numpy_to_visualization.py --input-dir /path/to/input
  python script/convert_numpy_to_visualization.py --input-dir /path/to/input --output-dir custom/output
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to directory containing extrinsics.npy, confidence.npy, depth.npy, intrinsics.npy, and optionally rgb.npy'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/default_run',
        help='Output directory path (run output directory)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        # Validate and load input arrays
        print("=" * 60)
        print("SIGMA NumPy Array to Visualization Converter")
        print("=" * 60)
        arrays = validate_input_directory(input_dir)

        # Create output directory structure
        output_dirs = create_output_structure(output_dir)

        # Convert arrays to per-frame files
        convert_arrays_to_frames(arrays, output_dirs)

        print("\n" + "=" * 60)
        print("Conversion successful!")
        print("=" * 60)

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
