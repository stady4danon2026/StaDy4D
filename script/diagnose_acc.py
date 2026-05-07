#!/usr/bin/env python3
"""Diagnose where NEW's Acc regression comes from.

For one pair, compute per-pred-point distance to nearest GT point, classify
each pred point as 'in-mask filled' vs 'static observed' vs 'mask-edge dilated',
and report Acc per category. Renders a heat map showing which image pixels
contribute outliers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file
from scipy.spatial import cKDTree


def load_seq(root: Path, kind: str = "pred"):
    if kind == "pred":
        depth = sorted((root / "depth").glob("*.npy"))
        ext = sorted((root / "extrinsics").glob("*.npy"))
        intr = sorted((root / "intrinsics").glob("*.npy"))
        masks = sorted((root / "mask").glob("*.png")) if (root / "mask").exists() else []
        return depth, ext, intr, masks
    raise ValueError(kind)


def back_project(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    v, u = np.indices((H, W))
    valid = (depth > 0.1) & (depth < 200) & np.isfinite(depth)
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = depth[valid].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
    if c2w.shape == (3, 4):
        c2w_full = np.eye(4)
        c2w_full[:3, :] = c2w
        c2w = c2w_full
    pts_w = (c2w @ pts_h.T).T[:, :3]
    pixel_idx = np.stack([v.astype(int), u.astype(int)], axis=1)
    return pts_w, pixel_idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="outputs/_eval_batch/scene_T03_008_NEW_fill")
    p.add_argument("--gt-scene", default="StaDy4D/short/test/scene_T03_008/static/cam_00_car_forward")
    p.add_argument("--out", default="comparison_vis/acc_diagnostic.png")
    p.add_argument("--max-points", type=int, default=200000)
    args = p.parse_args()

    pred_root = Path(args.pred)
    gt_scene = Path(args.gt_scene)

    # --- Load GT cloud (back-project depth.safetensors with extrinsics)
    gt_depth = load_file(str(gt_scene / "depth.safetensors"))["depth"]
    gt_c2w = load_file(str(gt_scene / "extrinsics.safetensors"))["c2w"]
    gt_K = load_file(str(gt_scene / "intrinsics.safetensors"))["K"]

    n_pred = len(sorted((pred_root / "depth").glob("*.npy")))
    n = min(n_pred, len(gt_depth))
    print(f"Frames: {n}")

    gt_pts_all = []
    for i in range(n):
        gt_d = gt_depth[i].astype(np.float32)
        # GT uses OpenGL convention -- but for pointcloud distance only the
        # relative geometry matters; use the same convention as evaluator.
        pts, _ = back_project(gt_d, gt_K[i], gt_c2w[i])
        gt_pts_all.append(pts)
    gt_pts = np.concatenate(gt_pts_all, axis=0)
    if len(gt_pts) > args.max_points:
        sel = np.random.choice(len(gt_pts), args.max_points, replace=False)
        gt_pts = gt_pts[sel]
    print(f"GT cloud: {len(gt_pts):,} points")

    # --- Build kd-tree on GT
    tree = cKDTree(gt_pts)

    # --- Load pred per-frame, classify each pred point
    H, W = gt_depth.shape[1:]
    cat_dist = {"static_obs": [], "in_mask_fill": [], "edge_dilated": []}
    pixel_outlier_heatmap = np.zeros((H, W), dtype=np.float32)
    pixel_count = np.zeros((H, W), dtype=np.int32)

    for i in range(n):
        depth = np.load(pred_root / "depth" / f"depth_{i:04d}.npy").astype(np.float32)
        K = np.load(pred_root / "intrinsics" / f"intrinsic_{i:04d}.npy")
        c2w = np.load(pred_root / "extrinsics" / f"extrinsic_{i:04d}.npy")

        # Resize pred to GT resolution if needed (Pi3X internally resizes).
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            K = K.copy()
            K[0, 0] *= W / depth.shape[1]; K[0, 2] *= W / depth.shape[1]
            K[1, 1] *= H / depth.shape[0]; K[1, 2] *= H / depth.shape[0]

        mask_path = pred_root / "mask" / f"mask_{i:04d}.png"
        mask = (np.array(imageio.imread(mask_path)) > 127).astype(np.uint8) if mask_path.exists() else np.zeros((H, W), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_orig = mask.astype(bool)
        mask_dilated = cv2.dilate(mask, np.ones((7, 7), np.uint8)).astype(bool)
        edge = mask_dilated & ~mask_orig

        pts, pix = back_project(depth, K, c2w)
        if len(pts) == 0:
            continue
        d, _ = tree.query(pts, k=1)

        for (vy, ux), dist in zip(pix, d):
            pixel_outlier_heatmap[vy, ux] += dist
            pixel_count[vy, ux] += 1

        for cat, sel in (
            ("in_mask_fill", mask_orig[pix[:, 0], pix[:, 1]]),
            ("edge_dilated", edge[pix[:, 0], pix[:, 1]]),
            ("static_obs", ~mask_dilated[pix[:, 0], pix[:, 1]]),
        ):
            cat_dist[cat].extend(d[sel].tolist())

    print()
    print("=== Per-category Acc (mean distance pred → nearest GT) ===")
    for cat, d in cat_dist.items():
        d = np.asarray(d)
        if len(d) == 0:
            continue
        print(f"  {cat:<14}  n={len(d):>9,}   mean={d.mean():>7.3f} m   "
              f"median={np.median(d):>7.3f} m   p90={np.quantile(d, 0.9):>7.3f} m")

    # --- Heatmap
    avg = np.divide(pixel_outlier_heatmap, np.maximum(pixel_count, 1))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    rgb = imageio.imread(pred_root / "rgb" / "rgb_0014.png")
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    ax[0].imshow(rgb); ax[0].set_title("frame 14 RGB"); ax[0].axis("off")
    im = ax[1].imshow(avg, cmap="hot", vmax=np.percentile(avg[avg > 0], 95) if (avg > 0).any() else 1.0)
    ax[1].set_title("avg distance pred → nearest GT (per pixel)")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], label="distance (m)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved heatmap: {args.out}")


if __name__ == "__main__":
    main()
