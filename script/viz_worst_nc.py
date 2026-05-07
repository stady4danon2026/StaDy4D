#!/usr/bin/env python3
"""Render diagnostic viz for SIGMA worst-NC scenes.

For each scene-cam, samples 3 frames and shows:
  RGB | pred depth | GT depth | abs depth diff | mask | composited
Helps diagnose where Pi3 geometry / SIGMA mask fails.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import safetensors.torch as st


SCENES = [
    ("scene_T10_048", "cam_00_car_forward"),
    ("scene_T03_034", "cam_01_car_forward"),
    ("scene_T07_001", "cam_02_car_forward"),
    ("scene_T07_021", "cam_00_car_forward"),
    ("scene_T03_020", "cam_01_car_forward"),
]


def colorize_depth(d, vmin=None, vmax=None, mask_invalid=True):
    d = d.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0.1) & (d < 200)
    if not valid.any():
        return np.zeros((*d.shape, 3), np.uint8)
    if vmin is None: vmin = float(np.percentile(d[valid], 5))
    if vmax is None: vmax = float(np.percentile(d[valid], 95))
    n = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    n_u8 = (n * 255).astype(np.uint8)
    cm = cv2.applyColorMap(n_u8, cv2.COLORMAP_TURBO)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    if mask_invalid:
        cm[~valid] = 0
    return cm


def bar(text, w, h=22):
    b = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(b, text, (8, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, default=Path("outputs/sigma_worst_nc"))
    ap.add_argument("--gt-root", type=Path, default=Path("StaDy4D/short/test"))
    ap.add_argument("--out", type=Path, default=Path("outputs/sigma_worst_nc_viz"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene, cam in SCENES:
        pred = args.pred_root / scene / cam
        gt = args.gt_root / scene / "dynamic" / cam
        if not pred.exists():
            print(f"skip {scene}/{cam}: pred missing"); continue

        # Load frames
        r = imageio.get_reader(str(gt / "rgb.mp4"))
        frames = np.stack([np.asarray(f) for f in r]); r.close()
        N, H, W = frames.shape[:3]
        gt_depth_all = st.load_file(str(gt / "depth.safetensors"))["depth"].numpy()[:N]

        # Compute global median scale (pred * scale ≈ GT) excluding sky
        scales = []
        for i in range(N):
            pd = np.load(pred / "depth" / f"depth_{i:04d}.npy").astype(np.float32)
            gd = gt_depth_all[i].astype(np.float32)
            v = np.isfinite(pd) & np.isfinite(gd) & (pd > 0.5) & (pd < 80) & (gd > 0.5) & (gd < 80)
            if v.sum() < 100: continue
            scales.append(float(np.median(gd[v]) / np.median(pd[v])))
        scale = float(np.median(scales)) if scales else 1.0
        print(f"  {scene}/{cam}: scale (gt/pred) = {scale:.3f} (no-sky)")

        sample_idx = [N // 6, N // 2, 5 * N // 6]
        rows = []
        for fi in sample_idx:
            rgb = frames[fi]
            pred_d_raw = np.load(pred / "depth" / f"depth_{fi:04d}.npy").astype(np.float32)
            pred_d = pred_d_raw * scale
            gt_d = gt_depth_all[fi].astype(np.float32)
            mask = cv2.imread(str(pred / "mask" / f"mask_{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
            comp = cv2.imread(str(pred / "composited" / f"comp_{fi:04d}.png"))
            comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)

            # exclude sky (GT > 80m as proxy) and invalid pixels for viz/diff
            valid = np.isfinite(gt_d) & (gt_d > 0.5) & (gt_d < 80) & np.isfinite(pred_d) & (pred_d > 0.5) & (pred_d < 80)
            vmin = float(np.percentile(gt_d[valid], 5)) if valid.any() else 1
            vmax = float(np.percentile(gt_d[valid], 95)) if valid.any() else 50

            pred_d_disp = pred_d.copy(); pred_d_disp[~valid] = np.nan
            gt_d_disp = gt_d.copy(); gt_d_disp[~valid] = np.nan
            pred_color = colorize_depth(pred_d_disp, vmin=vmin, vmax=vmax)
            gt_color = colorize_depth(gt_d_disp, vmin=vmin, vmax=vmax)

            diff = np.zeros_like(pred_d)
            diff[valid] = np.abs(pred_d - gt_d)[valid]
            diff_color = colorize_depth(diff, vmin=0, vmax=float(np.percentile(diff[valid], 95)) if valid.any() else 5)

            mask_rgb = np.stack([mask] * 3, axis=-1) if mask is not None else np.zeros_like(rgb)
            if mask_rgb.shape[:2] != (H, W):
                mask_rgb = cv2.resize(mask_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

            label = f"f{fi}  scale={scale:.2f}  range={vmin:.1f}-{vmax:.1f}m  abs_rel={(diff[valid]/np.abs(gt_d[valid])).mean()*100:.1f}%"
            row = np.concatenate([rgb, pred_color, gt_color, diff_color, mask_rgb, comp], axis=1)
            rows.append(np.concatenate([bar(label, row.shape[1]), row], axis=0))

        grid = np.concatenate(rows, axis=0)
        title = np.full((30, grid.shape[1], 3), 60, np.uint8)
        cv2.putText(title, f"{scene}/{cam}  |  RGB | pred depth | GT depth | |diff| | mask | composited",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        grid = np.concatenate([title, grid], axis=0)
        out_p = args.out / f"{scene}__{cam}.png"
        cv2.imwrite(str(out_p), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_p}")


if __name__ == "__main__":
    main()
