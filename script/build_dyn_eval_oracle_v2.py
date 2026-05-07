#!/usr/bin/env python3
"""Cleaner GT: ``|dyn_depth - sta_depth| > thr_d  AND  |dyn_rgb - sta_rgb| > thr_rgb``.

The depth-only GT had a major issue: when static and dynamic passes use slightly
different camera poses (handheld pedestrian / drone / orbit cams), the entire
depth map differs even where nothing actually moved. RGB-diff picks up pose
mismatch as thin edges (1-2px on object boundaries), but a real moving object
also produces high RGB-diff over a wide region.

Combined criterion:
  - high depth-diff   (> thr_d meters)
  - AND high RGB-diff (> thr_rgb across an erosion-resistant region)
gives clean motion masks even on ego-moving cameras.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("oracle_v2")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle_v2"))
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=[
        "cam_00_car_forward", "cam_01_car_forward", "cam_02_car_forward",
        "cam_03_drone_forward", "cam_04_orbit_building", "cam_05_orbit_crossroad",
        "cam_06_cctv", "cam_07_pedestrian", "cam_08_pedestrian",
    ])
    p.add_argument("--thr-depth", type=float, default=0.1, help="meters")
    p.add_argument("--thr-rgb", type=float, default=0.04,
                   help="L1 mean per-channel diff in [0,1] (4% of full range)")
    p.add_argument("--max-depth", type=float, default=900.0)
    p.add_argument("--erosion", type=int, default=3,
                   help="Erode mask before re-dilating; removes pose-mismatch hairlines.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    for sd in sorted(args.root.iterdir()):
        if not sd.is_dir() or not any(t in sd.name for t in args.towns):
            continue
        for cam in args.cameras:
            sd_d = sd / "static" / cam / "depth.safetensors"
            dd_d = sd / "dynamic" / cam / "depth.safetensors"
            sd_r = sd / "static" / cam / "rgb.mp4"
            dd_r = sd / "dynamic" / cam / "rgb.mp4"
            if all(p.exists() for p in (sd_d, dd_d, sd_r, dd_r)):
                pairs.append((sd.name, cam))
    LOGGER.info("Building oracle_v2 for %d (scene, cam) pairs", len(pairs))

    summary = []
    t_start = time.time()
    for k, (scene, cam) in enumerate(pairs):
        out_dir = args.out / scene / cam / "mask"
        if out_dir.exists() and len(list(out_dir.glob("mask_*.png"))) > 0:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            d_sta = st.load_file(args.root / scene / "static"  / cam / "depth.safetensors")["depth"].float()
            d_dyn = st.load_file(args.root / scene / "dynamic" / cam / "depth.safetensors")["depth"].float()
            r1 = imageio.get_reader(str(args.root / scene / "static"  / cam / "rgb.mp4"))
            rgb_sta = np.stack([np.asarray(f) for f in r1]); r1.close()
            r2 = imageio.get_reader(str(args.root / scene / "dynamic" / cam / "rgb.mp4"))
            rgb_dyn = np.stack([np.asarray(f) for f in r2]); r2.close()
        except Exception as e:
            LOGGER.warning("[%d/%d] %s/%s — load fail: %s", k + 1, len(pairs), scene, cam, e)
            continue

        # depth diff
        diff_d = (d_dyn - d_sta).abs()
        valid_d = (d_sta < args.max_depth) & (d_dyn < args.max_depth)
        m_depth = (diff_d > args.thr_depth) & valid_d
        m_depth = m_depth.numpy().astype(np.uint8)

        # rgb diff: |dyn - sta| / 255 → mean over channels per pixel
        N, H, W = m_depth.shape
        if rgb_dyn.shape[0] != N or rgb_sta.shape[0] != N:
            n_min = min(rgb_dyn.shape[0], rgb_sta.shape[0], N)
            m_depth = m_depth[:n_min]; rgb_dyn = rgb_dyn[:n_min]; rgb_sta = rgb_sta[:n_min]
            N = n_min
        diff_rgb = np.abs(rgb_dyn.astype(np.int16) - rgb_sta.astype(np.int16)).mean(axis=-1) / 255.0
        m_rgb = (diff_rgb > args.thr_rgb).astype(np.uint8)

        # AND, then erode to kill thin edges from pose mismatch, then dilate back
        mask = (m_depth & m_rgb).astype(np.uint8)
        if args.erosion > 0:
            kernel = np.ones((args.erosion, args.erosion), np.uint8)
            mask = np.stack([cv2.erode(mask[i], kernel) for i in range(N)])
            mask = np.stack([cv2.dilate(mask[i], kernel) for i in range(N)])

        for i in range(N):
            cv2.imwrite(str(out_dir / f"mask_{i:04d}.png"), mask[i] * 255)

        summary.append({
            "scene": scene, "camera": cam, "n_frames": int(N),
            "pos_frac_depth": float(m_depth.mean()),
            "pos_frac_rgb": float(m_rgb.mean()),
            "pos_frac_combined": float(mask.mean()),
        })
        if (k + 1) % 100 == 0 or (k + 1) == len(pairs):
            LOGGER.info("[%d/%d] cumulative=%.1fs", k + 1, len(pairs), time.time() - t_start)

    with (args.out / "oracle_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    LOGGER.info("Done. %d scenes in %.1fmin.", len(summary), (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
