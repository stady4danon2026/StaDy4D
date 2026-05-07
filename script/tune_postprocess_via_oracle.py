#!/usr/bin/env python3
"""Fast post-processing tuning loop against the depth-diff oracle.

For each config in a sweep grid, applies (threshold, morph_close, morph_dilate,
hull_fill, min_blob_area) on the cached *raw probability* maps (or on already-
binarised masks) and scores aggregated IoU/Precision/Recall vs the oracle.

Two modes:
  1. **prob mode** (preferred): re-runs the head's predict path on Pi3-feature
     bundles to get raw probs, then sweeps thresholds + post-processing.
  2. **bin mode**: post-process already-saved binary masks. Threshold can't be
     varied here, only morph/dilate/hull/min_blob.

This lets us decide the *deployment* threshold + post-process before re-running
inference on the full 1458 set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_dir(d: Path) -> np.ndarray:
    files = sorted(d.glob("mask_*.png"))
    masks = []
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        masks.append((m > 127).astype(np.uint8))
    return np.stack(masks) if masks else np.zeros((0, 0, 0), dtype=np.uint8)


def _post(m: np.ndarray, close: int, dilate: int, hull_min: int, hull_max_ratio: float, min_blob: int) -> np.ndarray:
    if not m.any():
        return m
    if close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    if dilate > 0:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    if hull_min > 0 and m.any():
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        m_filled = m.copy()
        for cc in range(1, n_lab):
            area = stats[cc, cv2.CC_STAT_AREA]
            if area < hull_min: continue
            pts = np.column_stack(np.where(lab == cc))[:, [1, 0]]
            if len(pts) < 3: continue
            hull = cv2.convexHull(pts)
            if cv2.contourArea(hull) > area * hull_max_ratio: continue
            cv2.fillPoly(m_filled, [hull], 1)
        m = m_filled
    if min_blob > 0 and m.any():
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for cc in range(1, n_lab):
            if stats[cc, cv2.CC_STAT_AREA] >= min_blob:
                keep[lab == cc] = 1
        m = keep
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--oracle", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle"))
    args = p.parse_args()

    # Discover scenes present in BOTH pred and oracle.
    pairs: list[tuple[str, str]] = []
    for sd in sorted(args.oracle.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("scene_"):
            continue
        for cd in sorted(sd.iterdir()):
            if (args.pred / sd.name / cd.name / "mask").is_dir() and (cd / "mask").is_dir():
                pairs.append((sd.name, cd.name))
    print(f"Tuning on {len(pairs)} (scene, cam) pairs that exist in both pred and oracle")
    if not pairs:
        return

    # Pre-load all GT and pred masks once (cheap).
    gts: dict[tuple[str, str], np.ndarray] = {}
    preds: dict[tuple[str, str], np.ndarray] = {}
    for scene, cam in pairs:
        gt = _load_dir(args.oracle / scene / cam / "mask")
        pr = _load_dir(args.pred / scene / cam / "mask")
        if gt.size == 0 or pr.size == 0:
            continue
        n = min(gt.shape[0], pr.shape[0])
        gt = gt[:n]
        pr = pr[:n]
        if pr.shape[1:] != gt.shape[1:]:
            pr = np.stack([cv2.resize(pr[i], (gt.shape[2], gt.shape[1]), interpolation=cv2.INTER_NEAREST)
                           for i in range(n)])
        gts[(scene, cam)] = gt.astype(bool)
        preds[(scene, cam)] = pr

    # Sweep grid (close, dilate, hull_min, hull_max_ratio, min_blob)
    grid = [
        ("baseline (current)",     7, 5,   0,  0.0,   100),
        ("dilate=11",              7, 11,  0,  0.0,   100),
        ("dilate=21",              7, 21,  0,  0.0,   100),
        ("close=15+dil=11",       15, 11,  0,  0.0,   100),
        ("hull_min=300 ratio=2.5", 7, 5,   300, 2.5,  100),
        ("hull_min=500 ratio=2.5",15, 11,  500, 2.5,  100),
        ("hull_min=200 ratio=3.0",15, 11,  200, 3.0,  100),
        ("close=21+dilate=21",    21, 21,  0,  0.0,   100),
    ]

    print(f"\n{'config':<28}  {'IoU':>8}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'pred%':>7}")
    print("-" * 80)
    for name, close, dilate, hmin, hmaxr, mblob in grid:
        ious, precs, recs, f1s, ppos = [], [], [], [], []
        for key, gt in gts.items():
            pr = preds[key]
            n = pr.shape[0]
            pr_pp = np.stack([_post(pr[i], close, dilate, hmin, hmaxr, mblob) for i in range(n)]).astype(bool)
            inter = (pr_pp & gt).sum(); union = (pr_pp | gt).sum()
            iou = inter / max(union, 1)
            prec = inter / max(pr_pp.sum(), 1)
            rec = inter / max(gt.sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            ious.append(iou); precs.append(prec); recs.append(rec); f1s.append(f1); ppos.append(pr_pp.mean())
        print(f"{name:<28}  {np.mean(ious):>8.4f}  {np.mean(precs):>8.4f}  "
              f"{np.mean(recs):>8.4f}  {np.mean(f1s):>8.4f}  {np.mean(ppos)*100:>6.2f}%")


if __name__ == "__main__":
    main()
