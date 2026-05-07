#!/usr/bin/env python3
"""SAM3 text-only sweep — no GSAM, no TTT, no head filter.

Per scene:
  1. Load all N frames.
  2. SAM3 forward on each frame with text prompt.
  3. Filter candidates by ``--score-thr``, union, save mask.

Output:
    out/<scene>/<cam>/mask/mask_NNNN.png
    out/summary.json
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
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("sam3_only")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=[
        "cam_00_car_forward", "cam_01_car_forward", "cam_02_car_forward",
        "cam_03_drone_forward", "cam_04_orbit_building", "cam_05_orbit_crossroad",
        "cam_06_cctv", "cam_07_pedestrian", "cam_08_pedestrian",
    ])
    p.add_argument("--out", type=Path, default=Path("<DATA_ROOT>/sigma_outputs/sam3_only"))
    p.add_argument("--sam3-checkpoint", default="facebook/sam3")
    p.add_argument("--text", default="car. truck. bus. motorcycle. bicycle. pedestrian.")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, Path]] = []
    for sd in sorted(args.root.iterdir()):
        if not sd.is_dir() or not any(t in sd.name for t in args.towns):
            continue
        dyn = sd / "dynamic"
        if not dyn.is_dir():
            continue
        for cam in args.cameras:
            rgb = dyn / cam / "rgb.mp4"
            if rgb.exists():
                pairs.append((f"{sd.name}/{cam}", rgb))
    if args.limit:
        pairs = pairs[: args.limit]
    LOGGER.info("Discovered %d (scene, cam) pairs", len(pairs))

    from transformers import Sam3Processor, Sam3Model
    LOGGER.info("Loading SAM3 (%s)", args.sam3_checkpoint)
    processor = Sam3Processor.from_pretrained(args.sam3_checkpoint)
    model = Sam3Model.from_pretrained(args.sam3_checkpoint).to(args.device).eval()
    for prm in model.parameters(): prm.requires_grad_(False)

    summary = []
    t_start = time.time()
    for k, (key, rgb_path) in enumerate(pairs):
        scene_name, cam_name = key.split("/")
        out_dir = args.out / scene_name / cam_name
        mask_dir = out_dir / "mask"
        if mask_dir.exists() and len(list(mask_dir.glob("mask_*.png"))) > 0:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            r = imageio.get_reader(str(rgb_path))
            frames = np.stack([np.asarray(f) for i, f in enumerate(r) if args.max_frames is None or i < args.max_frames])
            r.close()
        except Exception as e:
            LOGGER.warning("[%d/%d] %s — load fail: %s", k + 1, len(pairs), key, e)
            continue
        N, H, W = frames.shape[:3]

        t0 = time.time()
        out_masks = np.zeros((N, H, W), dtype=np.uint8)
        for i in range(N):
            pil = Image.fromarray(frames[i])
            inp = processor(images=pil, text=args.text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                so = model(**inp)
            scores = so.pred_logits[0].sigmoid().cpu().numpy()
            mks_lr = (so.pred_masks[0].sigmoid() > 0.5).cpu().numpy()
            keep = scores > args.score_thr
            if keep.sum() == 0:
                cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), np.zeros((H, W), dtype=np.uint8))
                continue
            mks_lr = mks_lr[keep]
            cand = np.zeros_like(out_masks[i], dtype=bool)
            for c in range(mks_lr.shape[0]):
                m = cv2.resize(mks_lr[c].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                cand |= m
            out_masks[i] = cand.astype(np.uint8)
            cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), out_masks[i] * 255)

        elapsed = time.time() - t0
        rec = {
            "scene": scene_name, "camera": cam_name, "n_frames": int(N),
            "elapsed_s": round(elapsed, 2),
            "global_pos_frac": float(out_masks.mean()),
        }
        summary.append(rec)
        eta = (time.time() - t_start) / (k + 1) * (len(pairs) - k - 1)
        LOGGER.info(
            "[%d/%d] %s done in %.1fs  pos=%.4f  ETA=%.1fmin",
            k + 1, len(pairs), key, elapsed, rec["global_pos_frac"], eta / 60,
        )
        with (args.out / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

    LOGGER.info("All done. Total: %.1fmin across %d scenes.",
                (time.time() - t_start) / 60, len(summary))


if __name__ == "__main__":
    main()
