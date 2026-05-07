#!/usr/bin/env python3
"""Batch run pure photometric TTT on StaDy4D test scenes.

For each (scene, camera) pair:
  1. Load origin frames from rgb.mp4.
  2. Pi3 forward (frozen) → conf_decoder features captured via hook.
  3. Photometric pseudo-labels from Pi3 depth/poses.
  4. Train a fresh DynHead for ``--ttt-steps`` steps.
  5. Predict masks; save to ``out/<scene>/<camera>/mask/mask_NNNN.png``.

The Pi3 backbone is loaded once and reused across scenes — only the
DynHead is re-initialized per scene (TTT is per-scene by definition).

Optionally computes per-scene IoU vs existing GSAM masks if discoverable.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("pi3_dyn_ttt_batch")


def _load_video(p: Path, max_frames: int | None) -> np.ndarray:
    r = imageio.get_reader(str(p))
    out = []
    for i, f in enumerate(r):
        if max_frames is not None and i >= max_frames:
            break
        out.append(np.asarray(f))
    r.close()
    return np.stack(out, axis=0)


def _try_iou_vs_gsam(scene: str, masks: np.ndarray, gsam_root: Path | None) -> float | None:
    if gsam_root is None:
        return None
    candidates = list(gsam_root.glob(f"{scene}*"))
    if not candidates:
        return None
    mask_dir = candidates[0] / "mask"
    if not mask_dir.is_dir():
        return None
    gms = []
    for i in range(masks.shape[0]):
        f = mask_dir / f"mask_{i:04d}.png"
        if not f.exists():
            return None
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != masks[i].shape:
            m = cv2.resize(m, (masks[i].shape[1], masks[i].shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        gms.append((m > 127).astype(np.uint8))
    gms = np.stack(gms)
    inter = int((masks & gms).sum())
    union = int((masks | gms).sum())
    return inter / max(union, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="test")
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=["cam_00_car_forward"])
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_ttt_batch"))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional pretrained DynHead checkpoint. When set with "
                        "--ttt-steps 0, runs in pure inference mode.")
    p.add_argument("--ttt-steps", type=int, default=500)
    p.add_argument("--ttt-lr", type=float, default=1e-3)
    p.add_argument("--ttt-batch-frames", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--photometric-threshold", type=float, default=0.10)
    p.add_argument("--photometric-consensus", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gsam-root", type=Path, default=None,
                   help="Optional dir with existing GSAM mask outputs for IoU comparison.")
    p.add_argument("--save-overlays", action="store_true",
                   help="Save RGB+mask overlay PNGs alongside the binary masks.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after processing this many (scene, camera) pairs.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    base = args.root / args.duration / args.split
    scenes = []
    for sd in sorted(base.iterdir()):
        if not sd.is_dir():
            continue
        if not any(t in sd.name for t in args.towns):
            continue
        scenes.append(sd)
    LOGGER.info("Discovered %d scenes across towns %s", len(scenes), args.towns)

    pairs: list[tuple[Path, str]] = []
    for sd in scenes:
        dyn = sd / "dynamic"
        if not dyn.is_dir():
            continue
        for cam in args.cameras:
            cd = dyn / cam
            if (cd / "rgb.mp4").exists():
                pairs.append((cd / "rgb.mp4", f"{sd.name}/{cam}"))
    if args.limit:
        pairs = pairs[: args.limit]
    LOGGER.info("Processing %d (scene, camera) pairs", len(pairs))

    # Load Pi3 once
    from sigma.pipeline.motion.pi3_dyn import Pi3DynMotionEstimator
    from sigma.data.frame_record import FrameIORecord

    estimator = Pi3DynMotionEstimator(
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        ttt_batch_frames=args.ttt_batch_frames,
        photometric_threshold=args.photometric_threshold,
        photometric_consensus=args.photometric_consensus,
    )
    estimator.setup()

    summary = []
    t_start = time.time()
    for k, (rgb_path, key) in enumerate(pairs):
        scene_name, cam_name = key.split("/")
        out_dir = args.out / scene_name / cam_name
        mask_dir = out_dir / "mask"
        if mask_dir.exists() and len(list(mask_dir.glob("mask_*.png"))) > 0:
            LOGGER.info("[%d/%d] %s — already processed, skipping", k + 1, len(pairs), key)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            frames = _load_video(rgb_path, args.max_frames)
        except Exception as e:
            LOGGER.warning("[%d/%d] %s — video load failed: %s", k + 1, len(pairs), key, e)
            continue
        if frames.shape[0] < 4:
            LOGGER.warning("[%d/%d] %s — too few frames (%d)", k + 1, len(pairs), key, frames.shape[0])
            continue

        # When TTT is on, re-init head per scene (pure per-scene TTT).
        # When TTT is off, keep the loaded checkpoint head across scenes.
        if args.ttt_steps > 0:
            from sigma.pipeline.motion.pi3_dyn_head import DynHead
            estimator.dyn_head = DynHead().to(args.device)

        recs = {i: FrameIORecord(frame_idx=i, origin_image=frames[i]) for i in range(frames.shape[0])}
        t0 = time.time()
        try:
            results = estimator.process_batch(recs)
        except Exception as e:
            LOGGER.exception("[%d/%d] %s — process_batch failed: %s", k + 1, len(pairs), key, e)
            continue
        elapsed = time.time() - t0

        masks = np.stack([results[i].data["motion"]["curr_mask"] for i in range(frames.shape[0])])
        for i in range(masks.shape[0]):
            cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), masks[i] * 255)

        if args.save_overlays:
            ov_dir = out_dir / "overlay"
            ov_dir.mkdir(exist_ok=True)
            for i in range(masks.shape[0]):
                ov = frames[i].copy()
                hot = masks[i] > 0
                if hot.any():
                    ov[hot] = (ov[hot] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
                cv2.imwrite(str(ov_dir / f"overlay_{i:04d}.png"),
                             cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        iou = _try_iou_vs_gsam(scene_name, masks, args.gsam_root)
        rec = {
            "scene": scene_name,
            "camera": cam_name,
            "n_frames": int(masks.shape[0]),
            "elapsed_s": round(elapsed, 2),
            "global_pos_frac": float(masks.mean()),
            "iou_vs_gsam": iou,
        }
        summary.append(rec)
        eta = (time.time() - t_start) / (k + 1) * (len(pairs) - k - 1)
        LOGGER.info(
            "[%d/%d] %s done in %.1fs  pos=%.3f  iou=%s  ETA=%.1fmin",
            k + 1, len(pairs), key, elapsed, rec["global_pos_frac"],
            f"{iou:.3f}" if iou is not None else "n/a",
            eta / 60,
        )

        # Persist summary as we go.
        with (args.out / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

    estimator.teardown()
    total = time.time() - t_start
    LOGGER.info("All done. Total: %.1fmin across %d pairs.", total / 60, len(summary))


if __name__ == "__main__":
    main()
