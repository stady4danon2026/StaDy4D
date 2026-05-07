"""VGGT-specific preprocessing utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def load_images_from_paths(image_paths: List[str | Path]) -> List[np.ndarray]:
    """Load images from file paths to numpy arrays.

    Args:
        image_paths: List of paths to image files (jpg, png, etc.).

    Returns:
        List of RGB images as numpy arrays (H, W, 3), range [0, 255], dtype uint8.

    Raises:
        ImportError: If PIL is not installed.
        FileNotFoundError: If any image path does not exist.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("PIL not installed. Install with: pip install pillow") from e

    images = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load and convert to RGB
        img = Image.open(path)

        # Handle alpha channel by blending onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")
        images.append(np.array(img, dtype=np.uint8))

    return images


def preprocess_frames_for_vggt(
    frames: List[np.ndarray],
    mode: str = "crop",
    target_size: int = 518,
) -> tuple[Any, List[Dict[str, Any]]]:
    """Preprocess frames for VGGT model input.

    This function applies the same preprocessing as VGGT's load_and_preprocess_images
    but operates on already-loaded numpy arrays instead of file paths.

    Args:
        frames: List of RGB frames as numpy arrays (H, W, 3), range [0, 255].
        mode: Preprocessing mode, either "crop" or "pad".
            - "crop" (default): Sets width to 518px and center crops height if needed.
            - "pad": Preserves all pixels by making largest dimension 518px
              and padding smaller dimension to square.
        target_size: Target dimension in pixels (default: 518).

    Returns:
        torch.Tensor: Batched tensor of preprocessed images (N, 3, H, W).

    Raises:
        ImportError: If torch or PIL are not installed.
        ValueError: If frames list is empty or mode is invalid.

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - Dimensions are adjusted to be divisible by 14 for model compatibility
        - When mode="crop": Width=518px, height center-cropped if > 518px
        - When mode="pad": Largest dimension=518px, smaller padded to square
    """
    try:
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
    except ImportError as e:
        raise ImportError("torch and torchvision required. Install with: pip install torch torchvision") from e

    if len(frames) == 0:
        raise ValueError("At least 1 frame is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    processed_images = []
    shapes = set()

    for frame in frames:
        # Initialize metadata for this frame
        metadata = {
            "original_size": frame.shape[:2],  # (H, W)
            "resized_size": None,
            "padded_size": None,
            "padding": (0, 0, 0, 0),  # (top, bottom, left, right)
            "crop": (0, 0),  # (top, left)
        }
        # Convert numpy array to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        img = Image.fromarray(frame)
        width, height = img.size

        # Calculate new dimensions
        if mode == "pad":
            # Make largest dimension target_size while maintaining aspect ratio
            if width >= height:
                pad_width = resize_width = target_size
                resize_height = int(height * (resize_width / width))
                pad_height = math.ceil(height * (resize_width / width) / 14) * 14
            else:
                pad_height = resize_height = target_size
                resize_width = int(width * (resize_height / height))
                pad_width = math.ceil(width * (resize_height / height) / 14) * 14
        else:  # mode == "crop"
            # Set width to target_size
            resize_width = target_size
            resize_height = round(height * (resize_width / width) / 14) * 14

        # Resize
        img = TF.resize(TF.to_tensor(img), [resize_height, resize_width], interpolation=Image.Resampling.BICUBIC)
        metadata["resized_size"] = (resize_height, resize_width)  # (H, W)

        # Center crop height if larger than target_size (crop mode only)
        crop_top = 0
        if mode == "crop" and resize_height > target_size:
            crop_top = (resize_height - target_size) // 2
            img = img[:, crop_top : crop_top + target_size, :]
            metadata["crop"] = (crop_top, 0)
            metadata["resized_size"] = (target_size, resize_width)
        # Pad to make square (pad mode)
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        if mode == "pad":
            h_padding = pad_height - img.shape[1]
            w_padding = pad_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
                metadata["padding"] = (pad_top, -pad_bottom, pad_left, -pad_right)

        metadata["padded_size"] = (img.shape[1], img.shape[2])  # (H, W) after all processing
        shapes.add((img.shape[1], img.shape[2]))
        processed_images.append(img)

    # Stack into batch
    images_tensor = torch.stack(processed_images)

    # Ensure correct shape for single image
    if len(frames) == 1 and images_tensor.dim() == 3:
        images_tensor = images_tensor.unsqueeze(0)

    return images_tensor, metadata


def load_and_preprocess_images(
    image_paths: List[str | Path],
    mode: str = "crop",
    target_size: int = 518,
) -> Any:
    """Load images from paths and preprocess for VGGT model.

    This is a convenience function combining load_images_from_paths
    and preprocess_frames_for_vggt. Useful for testing.

    Args:
        image_paths: List of paths to image files.
        mode: Preprocessing mode ("crop" or "pad").
        target_size: Target dimension in pixels (default: 518).

    Returns:
        torch.Tensor: Batched tensor of preprocessed images (N, 3, H, W).
    """
    frames = load_images_from_paths(image_paths)
    return preprocess_frames_for_vggt(frames, mode=mode, target_size=target_size)


def preprocess_frame_dict_for_vggt(
    frames_dict: Dict[int, np.ndarray],
    timesteps: List[int],
    mode: str = "pad",
    target_size: int = 518,
    device: str = "cpu",
) -> tuple[Any, List[int], Dict[int, Dict[str, Any]]]:
    """Preprocess a dictionary of frames for VGGT model.

    Args:
        frames_dict: Dictionary mapping timestep to RGB frames (H, W, 3).
        mode: Preprocessing mode ("crop" or "pad").
        target_size: Target dimension in pixels (default: 518).
        device: Device to move tensors to.

    Returns:
        Tuple of (images_tensor, timesteps, images_info):
            - images_tensor: Batched tensor (N, 3, H, W)
            - timesteps: Sorted list of timesteps corresponding to batch dimension
            - images_info: Dict mapping timestep to preprocessing metadata
    """
    frames_list = [frames_dict[t] for t in timesteps]
    images_tensor, images_info = preprocess_frames_for_vggt(frames_list, mode=mode, target_size=target_size)
    images_tensor = images_tensor.to(device)
    return images_tensor, images_info


def rescale_vggt_prediction(predictions, images_info: List[Dict[str, Any]]):
    """Rescale VGGT model predictions back to original image sizes.

    Args:
        predictions: Model predictions tensor (N, C, H, W).
        images_info: List of metadata dictionaries for each image.

    Returns:
        List of rescaled prediction tensors for each image.
    """
    top, bottom, left, right = [pad if pad != 0 else None for pad in images_info["padding"]]
    size = images_info["original_size"]

    uppad_depth = predictions["depth"][:, :, top:bottom, left:right, 0]
    predictions["depth"] = F.interpolate(uppad_depth, size, mode="bilinear")

    uppad_depth_conf = predictions["depth_conf"][:, :, top:bottom, left:right]
    predictions["depth_conf"] = F.interpolate(uppad_depth_conf, size, mode="bilinear")
    return predictions


def rescale_vggt_intrinsic(intrinsics, images_info: List[Dict[str, Any]]):
    """Rescale VGGT intrinsic matrices back to original image sizes.

    Args:
        intrinsics: Model intrinsic matrices tensor (N, 3, 3).
        images_info: List of metadata dictionaries for each image.

    Returns:
        List of rescaled intrinsic matrices for each image.
    """
    original_size, resized_size = images_info["original_size"], images_info["resized_size"]
    top, _, left, _ = images_info["padding"]

    intrinsics[..., 0, 2] -= left
    intrinsics[..., 1, 2] -= top

    intrinsics[..., 0, [0, 2]] *= original_size[1] / resized_size[1]
    intrinsics[..., 1, [1, 2]] *= original_size[0] / resized_size[0]
    return intrinsics
