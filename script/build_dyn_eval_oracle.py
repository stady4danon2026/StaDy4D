#!/usr/bin/env python3
"""Build held-out dynamic-mask ground truth from paired static/dynamic depth.

For each (scene, camera), the StaDy4D dataset stores two depth-map passes:
  - static/<cam>/depth.safetensors  → scene without dynamic actors
  - dynamic/<cam>/depth.safetensors → same scene with actors

A pixel is "dynamic" iff the two depths disagree by > ``--thr`` meters
(default 0.1 m). This is exact ground truth — no segmentation model in the
loop, no pseudo-labels.

Saves masks at the camera's native resolution to:
    out/<scene>/<camera>/mask/mask_NNNN.png
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import safetensors.torch as st
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("oracle")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle"))
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=[
        "cam_00_car_forward", "cam_01_car_forward", "cam_02_car_forward",
        "cam_03_drone_forward", "cam_04_orbit_building", "cam_05_orbit_crossroad",
        "cam_06_cctv", "cam_07_pedestrian", "cam_08_pedestrian",
    ])
    p.add_argument("--thr", type=float, default=0.1,
                   help="|dynamic - static| > thr meters → dynamic pixel.")
    p.add_argument("--max-depth", type=float, default=900.0,
                   help="Pixels with either depth >= this (sky/inf) treated as static.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    for sd in sorted(args.root.iterdir()):
        if not sd.is_dir() or not any(t in sd.name for t in args.towns):
            continue
        for cam in args.cameras:
            sd_cam = sd / "static" / cam / "depth.safetensors"
            dd_cam = sd / "dynamic" / cam / "depth.safetensors"
            if sd_cam.exists() and dd_cam.exists():
                pairs.append((sd.name, cam))
    LOGGER.info("Building oracle for %d (scene, cam) pairs", len(pairs))

    summary = []
    t_start = time.time()
    for k, (scene, cam) in enumerate(pairs):
        out_dir = args.out / scene / cam / "mask"
        if out_dir.exists() and len(list(out_dir.glob("mask_*.png"))) > 0:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        sd_path = args.root / scene / "static" / cam / "depth.safetensors"
        dd_path = args.root / scene / "dynamic" / cam / "depth.safetensors"
        try:
            d_sta = st.load_file(sd_path)["depth"].float()
            d_dyn = st.load_file(dd_path)["depth"].float()
        except Exception as e:
            LOGGER.warning("[%d/%d] %s/%s — depth read fail: %s", k + 1, len(pairs), scene, cam, e)
            continue

        diff = (d_dyn - d_sta).abs()
        valid = (d_dyn < args.max_depth) & (d_sta < args.max_depth)
        mask = ((diff > args.thr) & valid).numpy().astype(np.uint8)

        for i in range(mask.shape[0]):
            cv2.imwrite(str(out_dir / f"mask_{i:04d}.png"), mask[i] * 255)

        summary.append({
            "scene": scene, "camera": cam,
            "n_frames": int(mask.shape[0]),
            "pos_frac": float(mask.mean()),
        })
        if (k + 1) % 100 == 0 or (k + 1) == len(pairs):
            LOGGER.info(
                "[%d/%d] cumulative=%.1fs",
                k + 1, len(pairs), time.time() - t_start,
            )

    with (args.out / "oracle_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    LOGGER.info("Done. %d scenes in %.1fmin.", len(summary), (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
