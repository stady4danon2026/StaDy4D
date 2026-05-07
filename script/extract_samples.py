#!/usr/bin/env python3
"""Extract RGB + depth at {0%, 25%, 50%, 75%, 100%} from a StaDy4D camera.

Usage:
    python script/extract_samples.py StaDy4D/Test/scene_T03_000/static/cam_06_round_building

    # Custom output dir:
    python script/extract_samples.py StaDy4D/Test/scene_T03_000/static/cam_06_round_building -o my_samples

    # Custom percentiles:
    python script/extract_samples.py ... --pcts 0 10 50 90 100
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from safetensors.numpy import load_file


def load_cam(cam_dir: Path):
    """Load RGB frames and depth from a StaDy4D camera dir."""
    rgb_path = cam_dir / "rgb.mp4"
    depth_path = cam_dir / "depth.safetensors"

    if not rgb_path.exists():
        raise FileNotFoundError(f"No rgb.mp4 in {cam_dir}")
    if not depth_path.exists():
        raise FileNotFoundError(f"No depth.safetensors in {cam_dir}")

    rgbs = iio.imread(str(rgb_path), plugin="pyav")  # (N, H, W, 3)
    depth_data = load_file(str(depth_path))
    depths = depth_data["depth"]  # (N, H, W) float16

    return np.asarray(rgbs), depths


def save_depth_vis(depth: np.ndarray, path: Path):
    """Save a depth map as a turbo-colorized PNG (frame only, no axes/colorbar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    d = depth.astype(np.float32).copy()
    d[d <= 0] = 0
    d[d >= 1000] = 0
    valid = d[d > 0]
    vmin = float(np.percentile(valid, 10)) if len(valid) > 0 else 0.0
    vmax = float(np.percentile(valid, 90)) if len(valid) > 0 else 1.0
    normed = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = (cm.inferno(normed)[:, :, :3] * 255).astype(np.uint8)
    # Black out invalid pixels
    colored[d <= 0] = 0
    iio.imwrite(str(path), colored)


def parse_cam_path(cam_dir: Path):
    """Extract short label parts from path like scene_T03_000 / cam_06_round_building."""
    parts = cam_dir.parts
    # Find scene and camera
    scene = cam = data_type = None
    for i, p in enumerate(parts):
        if p.startswith("scene_"):
            scene = p
        if p in ("static", "dynamic"):
            data_type = p
        if p.startswith("cam_") or p.startswith("camera_"):
            cam = p

    if scene is None or cam is None or data_type is None:
        return None, None, None

    # scene_T03_000 -> 03_000
    m = re.match(r"scene_T(\d+)_(\d+)", scene)
    scene_short = f"{m.group(1)}_{m.group(2)}" if m else scene

    # cam_06_round_building -> 06
    m2 = re.match(r"cam(?:era)?_(\d+)", cam)
    cam_short = m2.group(1) if m2 else cam

    tag = "sta" if data_type == "static" else "dyn"
    return scene_short, cam_short, tag


def main():
    parser = argparse.ArgumentParser(description="Extract sample RGB + depth frames")
    parser.add_argument("cam_dir", type=Path, help="Path to a StaDy4D camera dir")
    parser.add_argument("-o", "--output", type=Path, default=Path("samples"),
                        help="Output directory (default: samples/)")
    parser.add_argument("--pcts", type=int, nargs="+", default=[0, 25, 50, 75, 100],
                        help="Percentiles to extract (default: 0 25 50 75 100)")
    args = parser.parse_args()

    cam_dir = args.cam_dir.resolve()

    # Derive the sibling (static ↔ dynamic) path
    parts = list(cam_dir.parts)
    try:
        dt_idx = next(i for i, p in enumerate(parts) if p in ("static", "dynamic"))
    except StopIteration:
        print("Error: path must contain 'static' or 'dynamic'")
        sys.exit(1)

    other_type = "dynamic" if parts[dt_idx] == "static" else "static"
    sibling_parts = parts.copy()
    sibling_parts[dt_idx] = other_type
    sibling_dir = Path("/".join(sibling_parts))

    # Load both
    pairs = []
    for d in (cam_dir, sibling_dir):
        if not d.exists():
            print(f"Warning: {d} not found, skipping")
            continue
        rgbs, depths = load_cam(d)
        scene_short, cam_short, tag = parse_cam_path(d)
        pairs.append((d, rgbs, depths, scene_short, cam_short, tag))

    if not pairs:
        print("No data loaded.")
        sys.exit(1)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, rgbs, depths, scene_short, cam_short, tag in pairs:
        n_frames = len(rgbs)
        for pct in args.pcts:
            idx = min(int(round(pct / 100 * (n_frames - 1))), n_frames - 1)
            prefix = f"{scene_short}_{cam_short}_{tag}_{pct:02d}"

            # RGB
            iio.imwrite(str(out_dir / f"{prefix}_rgb.png"), rgbs[idx])

            # Depth (turbo colorized, no axes/colorbar)
            save_depth_vis(depths[idx], out_dir / f"{prefix}_depth.png")

            print(f"  {prefix}  frame {idx}/{n_frames-1}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
