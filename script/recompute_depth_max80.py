#!/usr/bin/env python3
"""Re-aggregate NEW vs OLD depth comparison with max_depth=80 (KITTI-style)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sigma.evaluation.metrics import depth_metrics
from safetensors.numpy import load_file
from PIL import Image


def reeval_pair(scene: str, cam: str, pred_root: Path, max_depth: float = 80.0):
    gt_dir = Path(f"StaDy4D/short/test/{scene}/static/{cam}")
    pred_dir = pred_root / f"{scene}_{cam}"
    if not pred_dir.is_dir():
        return None
    gt_all = load_file(str(gt_dir / "depth.safetensors"))["depth"]
    n = min(len(gt_all), len(list((pred_dir / "depth").glob("*.npy"))))
    arpf = []
    for idx in range(n):
        pred = np.load(pred_dir / "depth" / f"depth_{idx:04d}.npy").astype(np.float32)
        gt = gt_all[idx].astype(np.float32)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR))
        m = depth_metrics(pred, gt, valid_mask=None, max_depth=max_depth, min_depth=1e-3, align="median")
        if not math.isnan(m["abs_rel"]):
            arpf.append(m["abs_rel"])
    return float(np.mean(arpf)) if arpf else None


def main():
    pred_root = Path("outputs/_eval_batch")
    failures = [
        ("scene_T10_029", "cam_01_car_forward"),
        ("scene_T10_033", "cam_00_car_forward"),
        ("scene_T10_044", "cam_00_car_forward"),
        ("scene_T03_008", "cam_00_car_forward"),
    ]
    print(f"{'pair':<48} {'abs_rel @80':>12}  {'eval-default abs_rel':>22}")
    for scene, cam in failures:
        ar80 = reeval_pair(scene, cam, pred_root, max_depth=80.0)
        # Lookup what eval reported
        json_path = Path(f"eval_results/full_short_test_NEW/short/test/{scene}/{cam}.json")
        eval_ar = json.load(open(json_path))["abs_rel"] if json_path.exists() else None
        ar80_str = f"{ar80:.3f}" if ar80 is not None else "-"
        eval_ar_str = f"{eval_ar:.3f}" if eval_ar is not None else "-"
        print(f"  {scene}/{cam:<32} {ar80_str:>12}  {eval_ar_str:>22}")


if __name__ == "__main__":
    main()
