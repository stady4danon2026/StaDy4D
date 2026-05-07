#!/usr/bin/env python3
"""Render a side-by-side comparison video: OLD (SDXL) vs NEW (multi-view fill).

Layout per frame (2x3):
    [origin RGB | mask overlay | GT depth]
    [OLD depth  | NEW depth    | |NEW-GT| - |OLD-GT|]

All depth maps share a fixed colormap range (computed from GT) for fair comparison.
Pred depths are median-aligned to GT scale per-frame.

Output: comparison_vis/comparison.mp4 (and .gif if specified).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from safetensors.numpy import load_file


def _decode_video(mp4_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(mp4_path))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return np.stack(frames, axis=0)


def _resize(arr: np.ndarray, H: int, W: int, nearest: bool = False) -> np.ndarray:
    if arr.shape[:2] == (H, W):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (W, H), interpolation=interp)


def _scale_to_gt(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    valid = (gt > 0.1) & (gt < 200) & (pred > 0)
    if int(valid.sum()) < 100:
        return pred.copy(), 1.0
    s = float(np.median(gt[valid] / pred[valid]))
    return pred * s, s


def render_frame(
    rgb: np.ndarray,
    mask: np.ndarray,
    gt_depth: np.ndarray,
    old_depth: np.ndarray,
    new_depth: np.ndarray,
    vmin: float,
    vmax: float,
    frame_idx: int,
) -> np.ndarray:
    """Render one comparison frame to an RGB numpy array."""
    fig = plt.figure(figsize=(15, 7), dpi=100)
    gs = GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb); ax.set_title("origin RGB", fontsize=11); ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rgb); ax.imshow(mask, alpha=0.45, cmap="Reds")
    ax.set_title("mask (red = dynamic object)", fontsize=11); ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(gt_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title("GT depth", fontsize=11); ax.axis("off")

    diff_old = np.abs(old_depth - gt_depth)
    diff_new = np.abs(new_depth - gt_depth)
    mask_bool = mask > 0
    valid_mask = mask_bool & (gt_depth > 0.1)
    mae_old = float(np.median(diff_old[valid_mask])) if valid_mask.any() else float("nan")
    mae_new = float(np.median(diff_new[valid_mask])) if valid_mask.any() else float("nan")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(old_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(f"OLD (SDXL)  in-mask MAE = {mae_old:.2f} m", fontsize=11, color="#aa3a3a")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(new_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(f"NEW (multi-view fill)  in-mask MAE = {mae_new:.2f} m", fontsize=11, color="#2a7a3a")
    ax.axis("off")

    diff_show = diff_new - diff_old
    absmax = max(2.0, float(np.percentile(np.abs(diff_show), 95)))
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(diff_show, cmap="RdBu_r", vmin=-absmax, vmax=absmax)
    ax.set_title("|NEW-GT| - |OLD-GT|   blue = NEW better", fontsize=11)
    ax.axis("off")

    fig.suptitle(f"frame {frame_idx:03d}   scene_T03_008/cam_00_car_forward", fontsize=12)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="StaDy4D/short/test/scene_T03_008/static/cam_00_car_forward")
    parser.add_argument("--old", default="outputs/_eval_batch/scene_T03_008_OLD_sdxl")
    parser.add_argument("--new", default="outputs/_eval_batch/scene_T03_008_NEW_fill")
    parser.add_argument("--out", default="comparison_vis/comparison.mp4")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--also_gif", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene)
    old_root = Path(args.old)
    new_root = Path(args.new)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_depth = load_file(str(scene_root / "depth.safetensors"))["depth"]
    rgb_video = _decode_video(scene_root / "rgb.mp4")

    n_old = len(sorted((old_root / "depth").glob("*.npy")))
    n_new = len(sorted((new_root / "depth").glob("*.npy")))
    n_frames = min(len(gt_depth), len(rgb_video), n_old, n_new)
    print(f"Rendering {n_frames} frames  (gt={len(gt_depth)}, rgb={len(rgb_video)}, "
          f"old={n_old}, new={n_new})")

    H_gt, W_gt = gt_depth.shape[1:]
    finite_gt = gt_depth[gt_depth > 0]
    vmin = 1.0
    vmax = float(np.percentile(finite_gt, 95)) if finite_gt.size else 50.0

    writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p")
    gif_frames = [] if args.also_gif else None

    in_mask_old, in_mask_new = [], []
    for idx in range(n_frames):
        rgb = _resize(rgb_video[idx], H_gt, W_gt)
        gt = gt_depth[idx].astype(np.float32)
        old = np.load(old_root / "depth" / f"depth_{idx:04d}.npy").astype(np.float32)
        new = np.load(new_root / "depth" / f"depth_{idx:04d}.npy").astype(np.float32)
        old = _resize(old, H_gt, W_gt, nearest=True)
        new = _resize(new, H_gt, W_gt, nearest=True)
        old_aligned, _ = _scale_to_gt(old, gt)
        new_aligned, _ = _scale_to_gt(new, gt)

        mask_path = old_root / "mask" / f"mask_{idx:04d}.png"
        if mask_path.exists():
            mask = np.array(imageio.imread(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = _resize(mask, H_gt, W_gt, nearest=True)
            mask_bool = mask > 127
        else:
            mask_bool = np.zeros((H_gt, W_gt), dtype=bool)

        # Frame-level in-mask metric for the running summary.
        valid = mask_bool & (gt > 0.1)
        if valid.any():
            in_mask_old.append(float(np.median(np.abs(old_aligned[valid] - gt[valid]))))
            in_mask_new.append(float(np.median(np.abs(new_aligned[valid] - gt[valid]))))

        img = render_frame(
            rgb=rgb,
            mask=mask_bool.astype(np.uint8),
            gt_depth=gt,
            old_depth=old_aligned,
            new_depth=new_aligned,
            vmin=vmin, vmax=vmax,
            frame_idx=idx,
        )
        writer.append_data(img)
        if gif_frames is not None:
            gif_frames.append(img)
        if (idx + 1) % 5 == 0 or idx == n_frames - 1:
            print(f"  {idx+1}/{n_frames}  in-mask MAE: OLD={in_mask_old[-1]:.2f} m  "
                  f"NEW={in_mask_new[-1]:.2f} m")

    writer.close()
    print(f"\nSaved {out_path}  ({n_frames} frames @ {args.fps} fps)")

    if gif_frames is not None:
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), gif_frames, fps=args.fps, loop=0)
        print(f"Saved {gif_path}")

    print()
    print(f"=== In-mask depth MAE (median across {len(in_mask_old)} frames) ===")
    print(f"  OLD (SDXL)         : {np.median(in_mask_old):.3f} m")
    print(f"  NEW (multi-view)   : {np.median(in_mask_new):.3f} m")
    print(f"  Improvement        : {np.median(in_mask_old)/max(np.median(in_mask_new),1e-6):.1f}x")


if __name__ == "__main__":
    main()
