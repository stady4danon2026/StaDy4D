#!/usr/bin/env python3
"""SAM3 text + head-as-point-filter ablation.

Pipeline:
  1. Sam3Model with text="car. truck. ..." → 200 candidate masks + scores.
  2. Filter candidates by ``score > score_thr`` and by IOU overlap with the
     head TTT mask (≥ ``min_head_overlap``). The head pixels act as positive
     point prompts: only candidates that contain some head pixels are kept.
  3. Union kept candidates → final mask.

This rejects parked-car candidates that the head doesn't flag while keeping
moving-car candidates with head support.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_TEXT = "car. truck. bus. motorcycle. bicycle. pedestrian."


def _bar(text: str, w: int, h: int = 26, bg: int = 30) -> np.ndarray:
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return bar


def _iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    return (inter / max(union, 1),
            inter / max(gt.sum(), 1),
            inter / max(pred.sum(), 1))


def _sam3_candidates(processor, model, image_rgb: np.ndarray, text: str,
                     score_thr: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (kept_masks (K, H, W) bool, kept_scores (K,))."""
    from PIL import Image
    pil = Image.fromarray(image_rgb)
    inputs = processor(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    if out.pred_masks is None or out.pred_logits is None:
        return np.zeros((0, *image_rgb.shape[:2]), dtype=bool), np.zeros(0)

    scores = out.pred_logits[0].sigmoid().cpu().numpy()                  # (200,)
    masks_lr = (out.pred_masks[0].sigmoid() > 0.5).cpu().numpy()         # (200, h, w)
    keep = scores > score_thr
    if keep.sum() == 0:
        return np.zeros((0, *image_rgb.shape[:2]), dtype=bool), np.zeros(0)

    masks_lr = masks_lr[keep]
    scores = scores[keep]

    # Resize each mask to original resolution
    H, W = image_rgb.shape[:2]
    out_masks = np.zeros((masks_lr.shape[0], H, W), dtype=bool)
    for i in range(masks_lr.shape[0]):
        m = cv2.resize(masks_lr[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out_masks[i] = m
    return out_masks, scores


def _head_filter(candidates: np.ndarray, head_mask: np.ndarray,
                 min_head_overlap_pixels: int = 20,
                 dilate: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Keep candidates with > N overlapping head-mask pixels."""
    if candidates.shape[0] == 0:
        return candidates, np.zeros(0, dtype=bool)
    if head_mask.sum() == 0:
        return candidates, np.ones(candidates.shape[0], dtype=bool)
    head = head_mask.astype(np.uint8)
    if dilate > 0:
        head = cv2.dilate(head, np.ones((dilate, dilate), np.uint8))
    head_bool = head.astype(bool)
    overlap = (candidates & head_bool[None]).sum(axis=(1, 2))
    keep = overlap >= min_head_overlap_pixels
    return candidates, keep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_002/cam_00_car_forward",
        "scene_T03_007/cam_05_orbit_crossroad",
        "scene_T03_005/cam_05_orbit_crossroad",
        "scene_T03_019/cam_08_pedestrian",
    ])
    p.add_argument("--frame", type=int, default=25)
    p.add_argument("--head-pred-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--gt-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle_v2"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_sam3_pointfilter"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--min-overlap", type=int, default=20)
    p.add_argument("--dilate-head", type=int, default=11)
    p.add_argument("--checkpoint", default="facebook/sam3")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import Sam3Processor, Sam3Model
    print(f"Loading {args.checkpoint} ...")
    processor = Sam3Processor.from_pretrained(args.checkpoint)
    model = Sam3Model.from_pretrained(args.checkpoint).to(args.device).eval()

    for scene_cam in args.scenes:
        scene, cam = scene_cam.split("/")
        rgb_path = args.root / scene / "dynamic" / cam / "rgb.mp4"
        head_dir = args.head_pred_dir / scene / cam / "mask"
        gt_dir = args.gt_dir / scene / cam / "mask"
        if not rgb_path.exists():
            print(f"skip {scene_cam}"); continue

        r = imageio.get_reader(str(rgb_path))
        frames = np.stack([np.asarray(f) for f in r])
        r.close()
        N, H, W = frames.shape[:3]
        fi = min(args.frame, N - 1)
        rgb = frames[fi]

        head_f = cv2.imread(str(head_dir / f"mask_{fi:04d}.png"), cv2.IMREAD_GRAYSCALE) if head_dir.is_dir() else None
        if head_f is None: head_f = np.zeros((H, W), dtype=np.uint8)
        if head_f.shape != (H, W):
            head_f = cv2.resize(head_f, (W, H), interpolation=cv2.INTER_NEAREST)
        head_bin = (head_f > 127).astype(np.uint8)

        gt_f = cv2.imread(str(gt_dir / f"mask_{fi:04d}.png"), cv2.IMREAD_GRAYSCALE) if gt_dir.is_dir() else None
        if gt_f is None: gt_f = np.zeros((H, W), dtype=np.uint8)
        if gt_f.shape != (H, W):
            gt_f = cv2.resize(gt_f, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_f > 127).astype(np.uint8)

        # SAM3 candidates
        candidates, scores = _sam3_candidates(processor, model, rgb,
                                               text=args.text,
                                               score_thr=args.score_thr,
                                               device=args.device)
        sam3_all = candidates.any(axis=0).astype(np.uint8) if len(candidates) else np.zeros((H, W), dtype=np.uint8)

        # Filter by head overlap
        cand_filtered, keep = _head_filter(candidates, head_bin,
                                            min_head_overlap_pixels=args.min_overlap,
                                            dilate=args.dilate_head)
        sam3_filtered = cand_filtered[keep].any(axis=0).astype(np.uint8) if keep.sum() else np.zeros((H, W), dtype=np.uint8)

        # Visualize candidate centers/points used as filter
        rgb_with_pts = rgb.copy()
        # Overlay head mask as cyan dots
        ys, xs = np.where(head_bin)
        for y, x in zip(ys[::20], xs[::20]):
            cv2.circle(rgb_with_pts, (int(x), int(y)), 2, (0, 220, 255), -1)

        # Compose grid
        cols = []
        cols.append(("RGB+head pts (cyan)", rgb_with_pts, None))
        cols.append(("head TTT", (head_bin[..., None] * 255).repeat(3, axis=-1), _iou(head_bin, gt_bin)))
        cols.append(("clean GT", (gt_bin[..., None] * 255).repeat(3, axis=-1), None))
        cols.append((f"SAM3 text-only (n={len(candidates)})",
                     (sam3_all[..., None] * 255).repeat(3, axis=-1), _iou(sam3_all, gt_bin)))
        cols.append((f"SAM3 + head-pt filter (n={int(keep.sum())}/{len(candidates)})",
                     (sam3_filtered[..., None] * 255).repeat(3, axis=-1), _iou(sam3_filtered, gt_bin)))

        rows = []
        for label, panel, iou in cols:
            stat = label
            if iou is not None:
                stat = f"{label}  IoU={iou[0]:.3f} R={iou[1]:.2f} P={iou[2]:.2f}"
            rows.append(np.concatenate([_bar(stat, panel.shape[1], h=26), panel], axis=0))
        grid = np.concatenate(rows, axis=0)
        title = np.full((34, grid.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title, f"{scene_cam}  frame {fi}  text='{args.text}'  score>{args.score_thr}  min_overlap={args.min_overlap}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        grid = np.concatenate([title, grid], axis=0)

        out_path = args.out / f"{scene}__{cam}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
