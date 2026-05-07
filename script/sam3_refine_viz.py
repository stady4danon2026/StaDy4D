#!/usr/bin/env python3
"""Per-frame Sam3 prompt ablation: text / box / text+box / text+mask.

Loads the TTT head's per-frame masks and runs Sam3 with various prompt
combinations on the same RGB. Renders side-by-side viz + computes IoU vs GT.

Goal: find the prompt combination that gives the cleanest mask for the
downstream reconstruction pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


# Default text prompt: list of dynamic categories in CARLA scenes.
DEFAULT_TEXT = "car. truck. bus. motorcycle. bicycle. pedestrian."


def _components_to_boxes(mask: np.ndarray, min_area: int = 50, dilate: int = 5,
                         pad: int = 4) -> np.ndarray:
    """Mask → connected-component bounding boxes."""
    if dilate > 0 and mask.any():
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    H, W = mask.shape
    for cc in range(1, n_lab):
        x = stats[cc, cv2.CC_STAT_LEFT]
        y = stats[cc, cv2.CC_STAT_TOP]
        w = stats[cc, cv2.CC_STAT_WIDTH]
        h = stats[cc, cv2.CC_STAT_HEIGHT]
        if stats[cc, cv2.CC_STAT_AREA] < min_area:
            continue
        boxes.append([
            max(0, x - pad),
            max(0, y - pad),
            min(W - 1, x + w + pad),
            min(H - 1, y + h + pad),
        ])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def _bar(text: str, w: int, h: int = 28, bg: int = 30) -> np.ndarray:
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    return bar


def _iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / max(union, 1)
    rec = inter / max(gt.sum(), 1)
    prec = inter / max(pred.sum(), 1)
    return float(iou), float(rec), float(prec)


def _run_sam3(processor, model, image_rgb: np.ndarray, *,
              text: str | None = None,
              boxes: np.ndarray | None = None,
              mask_init: np.ndarray | None = None,
              score_thr: float = 0.5,
              use_semantic: bool = False,
              device: str = "cuda") -> np.ndarray:
    """Run Sam3 image predictor; returns (H, W) bool mask.

    Output strategy:
      - if ``use_semantic``: take the single low-res semantic_seg map
      - else: filter the 200 query masks by ``pred_logits > score_thr``
        and union them.
    """
    from PIL import Image
    H_orig, W_orig = image_rgb.shape[:2]
    pil = Image.fromarray(image_rgb)

    kwargs: dict = {"images": pil, "return_tensors": "pt"}
    if text is not None:
        kwargs["text"] = text
    if boxes is not None and len(boxes) > 0:
        kwargs["input_boxes"] = [boxes.tolist()]
    if mask_init is not None:
        kwargs["segmentation_maps"] = Image.fromarray((mask_init * 255).astype(np.uint8))

    try:
        inputs = processor(**kwargs).to(device)
    except Exception:
        return np.zeros((H_orig, W_orig), dtype=np.uint8)

    with torch.no_grad():
        out = model(**inputs)

    if use_semantic and getattr(out, "semantic_seg", None) is not None:
        # (B, 1, h, w) → upsample
        sem = out.semantic_seg[0, 0].sigmoid()
        m = (sem > 0.5).cpu().numpy().astype(np.uint8)
        if m.shape != (H_orig, W_orig):
            m = cv2.resize(m, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        return m

    if getattr(out, "pred_masks", None) is None or getattr(out, "pred_logits", None) is None:
        return np.zeros((H_orig, W_orig), dtype=np.uint8)

    # Filter by logits/scores (sigmoid → probability)
    scores = out.pred_logits[0].sigmoid().cpu().numpy()        # (200,)
    pred_masks = out.pred_masks[0]                              # (200, h, w)
    keep = scores > score_thr
    if keep.sum() == 0:
        # Try lower threshold to keep top-K
        topk = min(5, pred_masks.shape[0])
        keep = np.zeros_like(scores, dtype=bool)
        keep[np.argsort(-scores)[:topk]] = True

    masks = (pred_masks[keep].sigmoid() > 0.5).any(dim=0)
    m = masks.cpu().numpy().astype(np.uint8)
    if m.shape != (H_orig, W_orig):
        m = cv2.resize(m, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_002/cam_00_car_forward",
        "scene_T03_007/cam_05_orbit_crossroad",
        "scene_T03_005/cam_05_orbit_crossroad",
    ])
    p.add_argument("--frame", type=int, default=25)
    p.add_argument("--head-pred-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--gt-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle_v2"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_sam3_ablation"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--checkpoint", default="facebook/sam3")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import Sam3Processor, Sam3Model
    print(f"Loading {args.checkpoint} ...")
    processor = Sam3Processor.from_pretrained(args.checkpoint)
    model = Sam3Model.from_pretrained(args.checkpoint).to(args.device).eval()

    variants = [
        ("text-only score>0.5",      {"text": args.text}),
        ("text-only score>0.3",      {"text": args.text, "score_thr": 0.3}),
        ("text-only semantic_seg",   {"text": args.text, "use_semantic": True}),
        ("text+box score>0.5",       {"text": args.text, "boxes": True}),
        ("text+mask score>0.5",      {"text": args.text, "mask_init": True}),
        ("text+box+mask score>0.5",  {"text": args.text, "boxes": True, "mask_init": True}),
    ]

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
        fi = min(args.frame, N - 1)
        rgb = frames[fi]

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

        boxes = _components_to_boxes(head_bin, min_area=30, dilate=5, pad=4)

        rgb_with_boxes = rgb.copy()
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(rgb_with_boxes, (x1, y1), (x2, y2), (255, 200, 0), 2)

        cols = []
        head_iou = _iou(head_bin, gt_bin)
        cols.append(("RGB+boxes", rgb_with_boxes, None))
        cols.append(("head TTT mask", (head_bin[..., None] * 255).repeat(3, axis=-1), head_iou))
        cols.append(("clean GT", (gt_bin[..., None] * 255).repeat(3, axis=-1), None))

        for name, kw in variants:
            kwargs = {}
            if "text" in kw: kwargs["text"] = kw["text"]
            if "boxes" in kw and kw["boxes"]: kwargs["boxes"] = boxes
            if "mask_init" in kw and kw["mask_init"]: kwargs["mask_init"] = head_bin
            if "score_thr" in kw: kwargs["score_thr"] = kw["score_thr"]
            if "use_semantic" in kw: kwargs["use_semantic"] = kw["use_semantic"]
            mask = _run_sam3(processor, model, rgb, **kwargs, device=args.device)
            cols.append((f"sam3 {name}", (mask[..., None] * 255).repeat(3, axis=-1),
                         _iou(mask, gt_bin)))

        # Compose grid
        rows = []
        for label, panel, iou in cols:
            stat = label
            if iou is not None:
                stat = f"{label}  IoU={iou[0]:.3f} R={iou[1]:.2f} P={iou[2]:.2f}"
            rows.append(np.concatenate([_bar(stat, panel.shape[1], h=28), panel], axis=0))
        # Stack vertically
        grid = np.concatenate(rows, axis=0)
        title = np.full((36, grid.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title, f"{scene_cam}  frame {fi}  text='{args.text}'",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        grid = np.concatenate([title, grid], axis=0)

        out_path = args.out / f"{scene}__{cam}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
