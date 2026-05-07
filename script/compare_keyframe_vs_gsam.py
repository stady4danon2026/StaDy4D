#!/usr/bin/env python3
"""Apple-to-apple comparison: current keyframe-TTT masks vs GSAM-on-all-frames.

For a sampled subset of completed scenes:
  1. Run GroundedSAM on all N frames → reference masks.
  2. Load current keyframe-TTT masks.
  3. Compute per-scene IoU, precision, recall, F1 (head vs GSAM).
  4. Print aggregated table + per-scene rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("compare")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    summary = json.load(open(args.results / "summary.json"))
    rng = random.Random(args.seed)
    rng.shuffle(summary)
    samples = summary[: args.n_samples]
    LOGGER.info("Sampling %d scenes for comparison", len(samples))

    from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator
    from sigma.data.frame_record import FrameIORecord

    gsam = GroundedSAMMotionEstimator(device=args.device)
    gsam.setup()

    rows = []
    for k, rec in enumerate(samples):
        scene = rec["scene"]; cam = rec["camera"]
        mp4 = args.root / scene / "dynamic" / cam / "rgb.mp4"
        try:
            r = imageio.get_reader(str(mp4))
            frames = np.stack([np.asarray(f) for f in r])
            r.close()
        except Exception as e:
            LOGGER.warning("skip %s/%s: %s", scene, cam, e)
            continue
        N, H, W = frames.shape[:3]

        # GSAM full coverage
        recs = {i: FrameIORecord(frame_idx=i, origin_image=frames[i]) for i in range(N)}
        out = gsam.process_batch(recs)
        gs = np.stack([(out[i].data["motion"]["curr_mask"] > 0) for i in range(N)])

        # Current keyframe-TTT masks
        mask_dir = args.results / scene / cam / "mask"
        head = []
        for i in range(N):
            f = mask_dir / f"mask_{i:04d}.png"
            if not f.exists():
                head.append(np.zeros((H, W), dtype=bool))
                continue
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            head.append(m > 127)
        head = np.stack(head)

        inter = (head & gs).sum()
        union = (head | gs).sum()
        iou = inter / max(union, 1)
        prec = inter / max(head.sum(), 1)
        rec_v = inter / max(gs.sum(), 1)
        f1 = 2 * prec * rec_v / max(prec + rec_v, 1e-6)

        rows.append({
            "scene": scene, "camera": cam,
            "gsam_pos": float(gs.mean()),
            "head_pos": float(head.mean()),
            "iou": float(iou),
            "precision": float(prec),
            "recall": float(rec_v),
            "f1": float(f1),
        })
        LOGGER.info(
            "[%d/%d] %s/%s  gsam=%.3f%%  head=%.3f%%  IoU=%.3f  P=%.3f  R=%.3f  F1=%.3f",
            k + 1, len(samples), scene, cam,
            gs.mean() * 100, head.mean() * 100, iou, prec, rec_v, f1,
        )

    gsam.teardown()

    # Aggregate
    print()
    print("=" * 80)
    print(f"{'metric':<14} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
    print("=" * 80)
    for k in ("gsam_pos", "head_pos", "iou", "precision", "recall", "f1"):
        v = np.array([r[k] for r in rows])
        print(f"{k:<14} {v.mean():>9.4f} {np.median(v):>9.4f} {v.min():>9.4f} {v.max():>9.4f}")

    out_json = args.results.parent / "compare_keyframe_vs_gsam.json"
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved per-scene rows to {out_json}")


if __name__ == "__main__":
    main()
