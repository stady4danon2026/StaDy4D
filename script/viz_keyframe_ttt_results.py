#!/usr/bin/env python3
"""Render side-by-side viz of keyframe-TTT inference results.

Layout per scene: 3 frame rows × 4 columns:
    RGB | head mask | GSAM-on-all-frames (if available) | overlay

If GSAM full-coverage masks aren't available, runs GSAM live on the same
frames and uses that as the comparison ground-truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def _label(text: str, w: int, h: int = 26, bg: int = 30) -> np.ndarray:
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return bar


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_keyframe_viz"))
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_000/cam_05_orbit_crossroad",
        "scene_T03_000/cam_00_car_forward",
        "scene_T03_001/cam_05_orbit_crossroad",
        "scene_T03_007/cam_05_orbit_crossroad",
    ])
    p.add_argument("--gsam-live", action="store_true",
                   help="Run GSAM live on the same scenes for comparison.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    gsam = None
    if args.gsam_live:
        from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator
        from sigma.data.frame_record import FrameIORecord
        gsam = GroundedSAMMotionEstimator(device="cuda")
        gsam.setup()

    for key in args.scenes:
        scene_name, cam = key.split("/")
        mask_dir = args.results / scene_name / cam / "mask"
        if not mask_dir.is_dir():
            print(f"skip {key}: no mask dir")
            continue

        mp4 = args.root / scene_name / "dynamic" / cam / "rgb.mp4"
        try:
            r = imageio.get_reader(str(mp4))
            rgb = np.stack([np.asarray(f) for f in r])
            r.close()
        except Exception as e:
            print(f"skip {key}: video read failed: {e}")
            continue

        N, H, W = rgb.shape[:3]
        masks = []
        for i in range(N):
            f = mask_dir / f"mask_{i:04d}.png"
            if not f.exists():
                masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append((m > 127).astype(np.uint8))
        masks = np.stack(masks)

        # Optional GSAM live
        gsam_masks = None
        if gsam is not None:
            from sigma.data.frame_record import FrameIORecord
            recs = {i: FrameIORecord(frame_idx=i, origin_image=rgb[i]) for i in range(N)}
            out_g = gsam.process_batch(recs)
            gsam_masks = np.stack(
                [out_g[i].data["motion"]["curr_mask"].astype(bool) for i in range(N)]
            )
            inter = (masks.astype(bool) & gsam_masks).sum()
            union = (masks.astype(bool) | gsam_masks).sum()
            iou = inter / max(union, 1)
        else:
            iou = None

        frames = [N // 6, N // 2, 5 * N // 6]
        rows = []
        for f_idx in frames:
            rgb_f = rgb[f_idx]
            mk = masks[f_idx].astype(bool)

            mk_viz = (mk[..., None].astype(np.uint8) * 255).repeat(3, axis=-1)
            ov = rgb_f.copy()
            if mk.any():
                ov[mk] = (ov[mk] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

            if gsam_masks is not None:
                gs_viz = (gsam_masks[f_idx][..., None].astype(np.uint8) * 255).repeat(3, axis=-1)
                row = np.concatenate([rgb_f, mk_viz, gs_viz, ov], axis=1)
                hdr = f"frame {f_idx} | RGB | head mask | GSAM full | overlay"
            else:
                row = np.concatenate([rgb_f, mk_viz, ov], axis=1)
                hdr = f"frame {f_idx} | RGB | head mask | overlay"
            rows.append(np.concatenate([_label(hdr, row.shape[1]), row], axis=0))

        grid = np.concatenate(rows, axis=0)
        ti = f"{scene_name}/{cam}   head_pred={masks.mean()*100:.2f}%"
        if iou is not None:
            ti += f"   gsam={gsam_masks.mean()*100:.2f}%   IoU(head,gsam)={iou:.3f}"
        title_bar = np.full((34, grid.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title_bar, ti, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        grid = np.concatenate([title_bar, grid], axis=0)

        out = args.out / f"{scene_name}__{cam}.png"
        cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out}  pred={masks.mean()*100:.2f}%  iou={iou if iou is None else round(iou, 3)}")

    if gsam is not None:
        gsam.teardown()


if __name__ == "__main__":
    main()
