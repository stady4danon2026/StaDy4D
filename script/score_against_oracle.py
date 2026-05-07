#!/usr/bin/env python3
"""Score any mask-output directory against the oracle GT.

Reads masks from ``--pred`` (layout: scene/camera/mask/mask_NNNN.png) and
compares against ``--oracle`` (same layout, GSAM-on-all-frames).

Outputs aggregated IoU / Precision / Recall / F1 + per-scene rows.
Runs in seconds — no GPU needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_mask_dir(d: Path) -> np.ndarray:
    files = sorted(d.glob("mask_*.png"))
    if not files:
        return np.empty((0, 0, 0), dtype=np.uint8)
    masks = []
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        masks.append((m > 127).astype(np.uint8))
    return np.stack(masks) if masks else np.empty((0, 0, 0), dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--oracle", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle"))
    p.add_argument("--label", default="pred")
    args = p.parse_args()

    rows = []
    for scene_dir in sorted(args.oracle.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if not (cam_dir / "mask").is_dir():
                continue
            pred_dir = args.pred / scene_dir.name / cam_dir.name / "mask"
            if not pred_dir.is_dir():
                continue
            gt = _load_mask_dir(cam_dir / "mask")
            pr = _load_mask_dir(pred_dir)
            if gt.size == 0 or pr.size == 0:
                continue
            n = min(gt.shape[0], pr.shape[0])
            gt = gt[:n].astype(bool)
            pr = pr[:n].astype(bool)
            # If shapes mismatch, resize pred to match GT
            if pr.shape[1:] != gt.shape[1:]:
                pr = np.stack([cv2.resize(pr[i].astype(np.uint8),
                                          (gt.shape[2], gt.shape[1]),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                               for i in range(n)])
            inter = (pr & gt).sum()
            union = (pr | gt).sum()
            iou = inter / max(union, 1)
            prec = inter / max(pr.sum(), 1)
            rec = inter / max(gt.sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            rows.append({
                "scene": scene_dir.name, "camera": cam_dir.name,
                "gt_pos": float(gt.mean()),
                "pred_pos": float(pr.mean()),
                "iou": float(iou),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            })

    if not rows:
        print("No matching scenes between pred and oracle.")
        return

    # Print aggregated
    print(f"\n=== {args.label}  ({len(rows)} scenes) ===")
    print(f"{'metric':<14} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
    for k in ("gt_pos", "pred_pos", "iou", "precision", "recall", "f1"):
        v = np.array([r[k] for r in rows])
        print(f"{k:<14} {v.mean():>9.4f} {np.median(v):>9.4f} {v.min():>9.4f} {v.max():>9.4f}")

    out = args.pred.parent / f"oracle_score_{args.label}.json"
    with out.open("w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nPer-scene rows: {out}")


if __name__ == "__main__":
    main()
