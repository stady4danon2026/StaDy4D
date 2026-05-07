#!/usr/bin/env python3
"""Render NEW depth vs GT depth for the worst-abs_rel failures.

Picks frame at center of sequence, shows: RGB | mask | GT depth | NEW depth | abs error.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file


FAILURES = [
    ("scene_T10_029", "cam_01_car_forward", 0.127, 3.181),
    ("scene_T10_033", "cam_00_car_forward", 0.341, 2.395),
    ("scene_T10_044", "cam_00_car_forward", 0.284, 2.195),
]


def median_align(pred, gt):
    valid = (gt > 0.1) & (gt < 200) & (pred > 0)
    if int(valid.sum()) < 100:
        return pred
    s = float(np.median(gt[valid] / pred[valid]))
    return pred * s


def main() -> None:
    out = Path("comparison_vis/failures")
    out.mkdir(parents=True, exist_ok=True)

    for scene, cam, old_abs, new_abs in FAILURES:
        gt_root = Path(f"StaDy4D/short/test/{scene}/static/{cam}")
        pred_root = Path(f"outputs/_eval_batch/{scene}_{cam}")
        if not pred_root.is_dir():
            print(f"skip {scene}/{cam} — no pred dir")
            continue

        gt_d_all = load_file(str(gt_root / "depth.safetensors"))["depth"]
        rgb_all = imageio.get_reader(str(gt_root / "rgb.mp4"))
        rgb_frames = [np.asarray(f) for f in rgb_all]; rgb_all.close()

        n = min(len(gt_d_all), len(rgb_frames), len(list((pred_root / "depth").glob("*.npy"))))
        # Pick a representative frame (center, or where mask is largest)
        idx = n // 2

        rgb = rgb_frames[idx]
        H, W = rgb.shape[:2]
        gt = gt_d_all[idx].astype(np.float32)
        pred = np.load(pred_root / "depth" / f"depth_{idx:04d}.npy").astype(np.float32)
        if pred.shape != (H, W):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
        mp = pred_root / "mask" / f"mask_{idx:04d}.png"
        mask = (np.array(imageio.imread(mp)) > 127) if mp.exists() else np.zeros((H, W), bool)
        if mask.ndim == 3: mask = mask[..., 0]
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        pred_aligned = median_align(pred, gt)
        finite_gt = gt[gt > 0]
        vmin, vmax = 1.0, float(np.percentile(finite_gt, 95)) if finite_gt.size else 50.0
        diff = np.abs(pred_aligned - gt) / np.maximum(gt, 1e-3)

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
        axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
        axes[1].imshow(rgb); axes[1].imshow(mask, alpha=0.5, cmap="Reds")
        axes[1].set_title("dynamic mask"); axes[1].axis("off")
        axes[2].imshow(gt, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[2].set_title(f"GT depth (mean={np.mean(gt[gt>0]):.1f}m)"); axes[2].axis("off")
        axes[3].imshow(pred_aligned, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[3].set_title(f"NEW pred (median-aligned)"); axes[3].axis("off")
        im = axes[4].imshow(np.clip(diff, 0, 2), cmap="hot", vmin=0, vmax=2)
        axes[4].set_title(f"|pred-GT|/GT  (clipped at 2)"); axes[4].axis("off")
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

        fig.suptitle(f"{scene}/{cam}  frame {idx}  •  abs_rel: OLD={old_abs:.3f}  →  NEW={new_abs:.3f}",
                     fontsize=12)
        fig.tight_layout()
        path = out / f"{scene}_{cam}_frame{idx:03d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")

        # Also dump a depth histogram so we see the scale issue
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(gt[gt > 0].ravel(), bins=80, range=(0, 200), alpha=0.6, label="GT", color="green")
        ax.hist(pred_aligned[pred_aligned > 0].ravel(), bins=80, range=(0, 200), alpha=0.6, label="NEW pred", color="red")
        ax.set_xlabel("depth (m)"); ax.set_ylabel("pixel count"); ax.legend()
        ax.set_title(f"{scene}/{cam} frame {idx}  depth histogram")
        fig.tight_layout()
        fig.savefig(out / f"{scene}_{cam}_frame{idx:03d}_hist.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
