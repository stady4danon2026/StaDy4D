#!/usr/bin/env python3
"""Compute abs_rel split by masked vs un-masked pixels.

For a small set of scenes, run NEW_sam3_ttt and NEW_gsam reconstruction,
then compute abs_rel separately on:
  - all valid pixels (current eval)
  - un-masked pixels only (where Pi3's native depth is used)
  - masked pixels only (where multi-view fill replaces depth)

Story: the abs_rel hit is concentrated in the masked region. Un-masked
abs_rel should match GSAM closely.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace

import cv2, numpy as np, safetensors.torch as st, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import build_pipeline, run_pipeline_for_camera

METHODS = {
    "NEW_gsam": {"motion": "grounded_sam", "extras": []},
    "NEW_sam3_ttt": {"motion": "sam3_ttt", "extras": []},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T07_023/cam_05_orbit_crossroad",
        "scene_T07_007/cam_04_orbit_building",
        "scene_T10_028/cam_00_car_forward",
    ])
    p.add_argument("--out", type=Path, default=ROOT / "eval_results/mask_aware")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset_root = ROOT / "StaDy4D"
    pred_runs: dict = {}

    for method, spec in METHODS.items():
        print(f"\n=== {method} ===")
        cli = SimpleNamespace(
            dataset_root=dataset_root, duration="short", split="test",
            data_format="dynamic", reconstruction="pi3", inpainting="blank",
        )
        extra = [
            "pipeline.reconstruction.use_origin_frames=true",
            "pipeline.reconstruction.fill_masked_depth=true",
            "pipeline.reconstruction.mask_dilate_px=3",
            "data.max_frames=50",
            f"pipeline/motion={spec['motion']}",
        ] + spec["extras"]
        runner, cfg = build_pipeline(cli, extra)

        pred_runs[method] = {}
        for sc in args.scenes:
            scene, cam = sc.split("/")
            out_dir = args.out / method / f"{scene}_{cam}"
            try:
                pred_dir = run_pipeline_for_camera(runner, cfg, scene, cam, out_dir)
                pred_runs[method][sc] = pred_dir
                print(f"  {sc}: ok")
            except Exception as e:
                print(f"  {sc}: FAILED — {e}")
                pred_runs[method][sc] = None

    # Now compute abs_rel split per scene per method
    print(f"\n{'='*100}")
    print(f"  Per-scene abs_rel split (all / unmasked / masked)")
    print(f"{'='*100}")
    print(f"{'scene/cam':40s} {'method':>14s} {'abs_all':>9s} {'abs_unmask':>11s} {'abs_mask':>10s} {'mask%':>7s}")
    summary = []
    for sc in args.scenes:
        scene, cam = sc.split("/")
        gt_path = dataset_root / "short/test" / scene / "static" / cam / "depth.safetensors"
        if not gt_path.exists(): continue
        gt = st.load_file(gt_path)["depth"].float().numpy()  # (N, H, W)
        for method in METHODS:
            pred_dir = pred_runs[method].get(sc)
            if pred_dir is None: continue
            pred_dir = Path(pred_dir)
            depth_files = sorted((pred_dir / "depth").glob("*.npy"))
            mask_files = sorted((pred_dir / "mask").glob("mask_*.png"))
            if not depth_files: continue

            abs_all, abs_unmask, abs_mask, mask_pix, total_pix = [], [], [], 0, 0
            n = min(len(depth_files), gt.shape[0], len(mask_files))
            for i in range(n):
                pred = np.load(depth_files[i])
                m = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
                if pred.shape != gt[i].shape:
                    pred = cv2.resize(pred, (gt[i].shape[1], gt[i].shape[0]))
                if m is None: m = np.zeros_like(pred, dtype=np.uint8)
                if m.shape != pred.shape:
                    m = cv2.resize(m, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = m > 127
                valid = (gt[i] > 0.1) & (gt[i] < 200) & (pred > 0.1)
                rel = np.abs(pred - gt[i]) / np.maximum(gt[i], 1e-3)
                if valid.any():
                    abs_all.append(rel[valid].mean())
                if (~mask & valid).any():
                    abs_unmask.append(rel[~mask & valid].mean())
                if (mask & valid).any():
                    abs_mask.append(rel[mask & valid].mean())
                mask_pix += int(mask.sum()); total_pix += int(mask.size)
            row = {
                "scene": sc, "method": method,
                "abs_all": float(np.mean(abs_all)) if abs_all else None,
                "abs_unmask": float(np.mean(abs_unmask)) if abs_unmask else None,
                "abs_mask": float(np.mean(abs_mask)) if abs_mask else None,
                "mask_pct": mask_pix * 100 / max(total_pix, 1),
            }
            summary.append(row)
            print(f"{sc:40s} {method:>14s} "
                  f"{row['abs_all'] if row['abs_all'] is not None else float('nan'):>9.3f} "
                  f"{row['abs_unmask'] if row['abs_unmask'] is not None else float('nan'):>11.3f} "
                  f"{row['abs_mask'] if row['abs_mask'] is not None else float('nan'):>10.3f} "
                  f"{row['mask_pct']:>6.1f}%")
    with (args.out / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
