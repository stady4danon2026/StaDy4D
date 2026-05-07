#!/usr/bin/env python3
"""Side-by-side comparison: multi-view fill vs SDXL inpaint vs GT depth.

Loads the two pipeline outputs and the StaDy4D GT, then renders a 2x3 grid
per frame: [origin RGB | mask | GT depth] / [SDXL depth | NEW depth | depth diff].
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file


def _decode_video(mp4_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(mp4_path))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return np.stack(frames, axis=0)


def main() -> None:
    scene_root = Path("StaDy4D/short/test/scene_T03_008/static/cam_00_car_forward")
    old_root = Path("outputs/_eval_batch/scene_T03_008_OLD_sdxl")
    new_root = Path("outputs/_eval_batch/scene_T03_008_NEW_fill")
    out_dir = Path("comparison_vis")
    out_dir.mkdir(exist_ok=True)

    # GT depth + RGB
    gt_depth = load_file(scene_root / "depth.safetensors")["depth"]  # (N, H, W)
    rgb_video = _decode_video(scene_root / "rgb.mp4")                 # (N, H, W, 3)

    n_frames = min(len(gt_depth), len(rgb_video), 30)

    # Compute consistent depth scale per frame for visualization (using GT range).
    sampled = [0, 7, 14, 21, 29]
    sampled = [i for i in sampled if i < n_frames]

    diff_old_total, diff_new_total = [], []

    for idx in sampled:
        rgb = rgb_video[idx]
        H, W = rgb.shape[:2]

        # Load preds
        old_depth = np.load(old_root / "depth" / f"depth_{idx:04d}.npy").astype(np.float32)
        new_depth = np.load(new_root / "depth" / f"depth_{idx:04d}.npy").astype(np.float32)
        mask_path = old_root / "mask" / f"mask_{idx:04d}.png"
        mask = np.array(imageio.imread(mask_path)) if mask_path.exists() else np.zeros((H, W), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = mask > 127

        gt = gt_depth[idx].astype(np.float32)

        # Resize preds to GT size if needed
        if old_depth.shape != gt.shape:
            import cv2
            old_depth = cv2.resize(old_depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        if new_depth.shape != gt.shape:
            import cv2
            new_depth = cv2.resize(new_depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask_bool.shape != gt.shape:
            import cv2
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

        # Median scale-align preds to GT (Pi3X is up to scale).
        valid = (gt > 0.1) & (gt < 200) & (old_depth > 0)
        if valid.sum() > 100:
            scale_old = float(np.median(gt[valid] / old_depth[valid]))
            old_aligned = old_depth * scale_old
        else:
            old_aligned = old_depth.copy()

        valid = (gt > 0.1) & (gt < 200) & (new_depth > 0)
        if valid.sum() > 100:
            scale_new = float(np.median(gt[valid] / new_depth[valid]))
            new_aligned = new_depth * scale_new
        else:
            new_aligned = new_depth.copy()

        vmin, vmax = 1.0, np.percentile(gt[gt > 0], 95) if (gt > 0).any() else 50.0

        diff_old = np.abs(old_aligned - gt)
        diff_new = np.abs(new_aligned - gt)
        diff_old_in_mask = float(np.median(diff_old[mask_bool & (gt > 0.1)])) if mask_bool.any() else float("nan")
        diff_new_in_mask = float(np.median(diff_new[mask_bool & (gt > 0.1)])) if mask_bool.any() else float("nan")
        diff_old_total.append(diff_old_in_mask)
        diff_new_total.append(diff_new_in_mask)

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        axes[0, 0].imshow(rgb); axes[0, 0].set_title(f"Frame {idx}: origin RGB"); axes[0, 0].axis("off")
        axes[0, 1].imshow(rgb); axes[0, 1].imshow(mask_bool, alpha=0.5, cmap="Reds"); axes[0, 1].set_title("mask"); axes[0, 1].axis("off")
        axes[0, 2].imshow(gt, cmap="turbo", vmin=vmin, vmax=vmax); axes[0, 2].set_title("GT depth"); axes[0, 2].axis("off")
        axes[1, 0].imshow(old_aligned, cmap="turbo", vmin=vmin, vmax=vmax); axes[1, 0].set_title(f"OLD (SDXL) — mask-region MAE: {diff_old_in_mask:.2f}m"); axes[1, 0].axis("off")
        axes[1, 1].imshow(new_aligned, cmap="turbo", vmin=vmin, vmax=vmax); axes[1, 1].set_title(f"NEW (fill) — mask-region MAE: {diff_new_in_mask:.2f}m"); axes[1, 1].axis("off")
        # Diff visualization (NEW - OLD): negative = new is closer to GT
        diff_show = diff_new - diff_old
        absmax = max(1.0, float(np.percentile(np.abs(diff_show), 95)))
        axes[1, 2].imshow(diff_show, cmap="RdBu_r", vmin=-absmax, vmax=absmax); axes[1, 2].set_title("|NEW-GT| - |OLD-GT|  (blue = NEW better)"); axes[1, 2].axis("off")

        fig.suptitle(f"scene_T03_008/cam_00_car_forward — frame {idx}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{idx:04d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"frame {idx:04d}: saved.  mask-region MAE  OLD={diff_old_in_mask:.2f}m  NEW={diff_new_in_mask:.2f}m")

    # Aggregate
    print()
    print(f"=== Per-mask depth MAE across {len(sampled)} sampled frames ===")
    print(f"  OLD (SDXL)        : median = {np.nanmedian(diff_old_total):.3f} m")
    print(f"  NEW (multi-view)  : median = {np.nanmedian(diff_new_total):.3f} m")
    print()
    print(f"Saved {len(sampled)} comparison frames to {out_dir}/")


if __name__ == "__main__":
    main()
