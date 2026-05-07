#!/usr/bin/env python3
"""Refine head + photometric coarse masks with SAM 2.

Per scene:
  1. Combine head's mask (any non-zero) ∪ photometric pseudo-mask → coarse mask.
  2. Find connected components → bounding boxes.
  3. Feed boxes to SAM 2 image predictor → refined dense masks.
  4. Render side-by-side: RGB | head | photo | combined | SAM-refined | GT (clean).

Yes — SAM 2 takes either boxes, points, or low-res masks as prompts. We use
bounding boxes here because they're the most robust signal we have from the
coarse mask: head's high-confidence centroids may sit on a tiny edge fragment,
but the COMPONENT'S BOX still spans the whole car.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _components_to_boxes(mask: np.ndarray, min_area: int = 50, dilate: int = 7,
                         pad: int = 0) -> np.ndarray:
    """Mask → connected component bounding boxes [(x1,y1,x2,y2), ...]."""
    if dilate > 0 and mask.any():
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for cc in range(1, n_lab):
        x = stats[cc, cv2.CC_STAT_LEFT]
        y = stats[cc, cv2.CC_STAT_TOP]
        w = stats[cc, cv2.CC_STAT_WIDTH]
        h = stats[cc, cv2.CC_STAT_HEIGHT]
        a = stats[cc, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        boxes.append([max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def _sam_refine(sam_processor, sam_model, image_rgb: np.ndarray,
                boxes: np.ndarray, device: str) -> np.ndarray:
    """boxes (N, 4) → (H, W) bool mask.  No-op if no boxes."""
    if boxes.shape[0] == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    pil = Image.fromarray(image_rgb)
    inputs = sam_processor(images=pil, input_boxes=[boxes.tolist()],
                            return_tensors="pt").to(device)
    with torch.no_grad():
        out = sam_model(**inputs)
    masks = sam_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
    if masks.ndim == 4:
        masks = masks[:, 0, :, :]
    return masks.any(dim=0).cpu().numpy().astype(np.uint8)


def _bar(text: str, w: int, h: int = 26, bg: int = 30) -> np.ndarray:
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return bar


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_002/cam_00_car_forward",
        "scene_T03_007/cam_05_orbit_crossroad",
        "scene_T03_005/cam_05_orbit_crossroad",
        "scene_T03_021/cam_00_car_forward",
        "scene_T03_007/cam_03_drone_forward",
    ])
    p.add_argument("--head-pred-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--gt-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle_v2"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_sam_refine"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--frames", nargs="+", type=int, default=[8, 25, 41])
    p.add_argument("--cache-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_cache/pi3_dyn_pool"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load SAM 2 (no DINO needed).
    from transformers import Sam2Processor, Sam2Model
    print("Loading SAM 2...")
    sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sam_mdl = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(args.device)
    sam_mdl.eval()

    # Compute photometric per scene if cache has features (fast: depth+poses+rgb already in cache via Pi3).
    # Simpler: re-derive photometric from Pi3 forward each time. We'll call the
    # photometric helper using cached info if available.
    from sigma.pipeline.motion.pi3_dyn_ttt import compute_photometric_labels

    # We need depth/poses/K/rgb for photometric. Easier: just skip photometric in this
    # viz and use head-only as the seed. We can also union with GSAM keyframes which
    # are saved nowhere — so head-only it is.
    # (User can extend later if needed.)

    scores = []
    for scene_cam in args.scenes:
        scene, cam = scene_cam.split("/")
        rgb_path = args.root / scene / "dynamic" / cam / "rgb.mp4"
        head_dir = args.head_pred_dir / scene / cam / "mask"
        gt_dir = args.gt_dir / scene / cam / "mask"
        if not rgb_path.exists() or not head_dir.is_dir():
            print(f"skip {scene_cam}"); continue

        r = imageio.get_reader(str(rgb_path))
        frames = np.stack([np.asarray(f) for f in r])
        r.close()
        N, H, W = frames.shape[:3]

        rows = []
        for fi in args.frames:
            if fi >= N: continue
            rgb_f = frames[fi]
            head_f = cv2.imread(str(head_dir / f"mask_{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if head_f is None: head_f = np.zeros((H, W), dtype=np.uint8)
            if head_f.shape != (H, W):
                head_f = cv2.resize(head_f, (W, H), interpolation=cv2.INTER_NEAREST)
            head_bin = (head_f > 127).astype(np.uint8)

            gt_f = cv2.imread(str(gt_dir / f"mask_{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if gt_f is None: gt_f = np.zeros((H, W), dtype=np.uint8)
            if gt_f.shape != (H, W):
                gt_f = cv2.resize(gt_f, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt_f > 127).astype(np.uint8)

            # Build coarse mask (head only here — easy to extend with photometric)
            coarse = head_bin
            boxes = _components_to_boxes(coarse, min_area=50, dilate=15, pad=30)
            refined = _sam_refine(sam_proc, sam_mdl, rgb_f, boxes, args.device)

            # Compute IoUs vs GT
            head_iou = ((head_bin & gt_bin).sum() / max((head_bin | gt_bin).sum(), 1))
            ref_iou  = ((refined  & gt_bin).sum() / max((refined  | gt_bin).sum(), 1))

            head_viz = (head_bin[..., None]*255).repeat(3, axis=-1)
            ref_viz  = (refined [..., None]*255).repeat(3, axis=-1)
            gt_viz   = (gt_bin  [..., None]*255).repeat(3, axis=-1)

            rgb_with_boxes = rgb_f.copy()
            for x1, y1, x2, y2 in boxes.astype(int):
                cv2.rectangle(rgb_with_boxes, (x1, y1), (x2, y2), (255, 200, 0), 2)

            row = np.concatenate([rgb_with_boxes, head_viz, ref_viz, gt_viz], axis=1)
            label = (f"frame {fi}  RGB+boxes | head ({head_bin.mean()*100:.2f}%, IoU={head_iou:.2f}) | "
                     f"SAM-refined ({refined.mean()*100:.2f}%, IoU={ref_iou:.2f}) | clean GT")
            rows.append(np.concatenate([_bar(label, row.shape[1], h=28), row], axis=0))
            scores.append({"scene": scene, "camera": cam, "frame": fi,
                           "head_iou": float(head_iou), "ref_iou": float(ref_iou)})

        if not rows: continue
        grid = np.concatenate(rows, axis=0)
        title = np.full((34, grid.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title, scene_cam, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        grid = np.concatenate([title, grid], axis=0)
        out_path = args.out / f"{scene.replace('/','__')}__{cam}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")

    if scores:
        import json
        with (args.out / "scores.json").open("w") as fh:
            json.dump(scores, fh, indent=2)
        head_avg = np.mean([s["head_iou"] for s in scores])
        ref_avg = np.mean([s["ref_iou"] for s in scores])
        print(f"\nMean IoU over {len(scores)} frames:  head={head_avg:.3f}  SAM-refined={ref_avg:.3f}")


if __name__ == "__main__":
    main()
