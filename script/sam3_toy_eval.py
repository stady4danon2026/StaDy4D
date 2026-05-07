#!/usr/bin/env python3
"""Toy eval: run reconstruction + pose/depth/pointcloud metrics on 5 scenes
with 3 motion-stage variants:
    A. current default       (score=0.3, union, max_coverage=0.5)
    B. SAM3 thr=0.5          (score=0.5, union, max_coverage=0.5)
    C. conditional union     (score=0.3, union when head_cov<10%, else head)

Prints per-scene + aggregate comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import build_pipeline, evaluate_sample, run_pipeline_for_camera


VARIANTS = {
    "NEW_gsam":      ["pipeline/motion=grounded_sam"],
    # SAM3 only — disables TTT/head to fit in GPU memory.
    "NEW_sam3":      ["pipeline/motion=sam3_ttt",
                      "pipeline.motion.score_thr=0.3",
                      "pipeline.motion.do_ttt=false",
                      "pipeline.motion.do_head=false"],
    # Full SAM3+TTT+head (heaviest)
    "NEW_sam3_ttt":  ["pipeline/motion=sam3_ttt",
                      "pipeline.motion.score_thr=0.3",
                      "pipeline.motion.head_cov_max_for_union=1.0",
                      "pipeline.motion.do_ttt=true",
                      "pipeline.motion.do_head=true"],
}

DEFAULT_SCENES = [
    ("scene_T03_000", "cam_00_car_forward"),
    ("scene_T03_021", "cam_05_orbit_crossroad"),
    ("scene_T07_023", "cam_05_orbit_crossroad"),
    ("scene_T07_007", "cam_04_orbit_building"),
    ("scene_T10_028", "cam_00_car_forward"),
    ("scene_T03_007", "cam_03_drone_forward"),
    ("scene_T07_041", "cam_07_pedestrian"),
    ("scene_T10_029", "cam_01_car_forward"),
    ("scene_T07_034", "cam_05_orbit_crossroad"),
    ("scene_T03_002", "cam_00_car_forward"),
]

METRICS = ("AUC", "ATE", "RPEt", "RPEr", "abs_rel", "rmse", "a1", "Acc", "Comp", "NC")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=ROOT / "eval_results/sam3_toy")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS))
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    dataset_root = ROOT / "StaDy4D"
    results: dict[str, list[dict]] = {}

    for variant in args.variants:
        print(f"\n{'='*78}\nLoading {variant}\n{'='*78}", flush=True)
        cli = SimpleNamespace(
            dataset_root=dataset_root,
            duration="short",
            split="test",
            data_format="dynamic",
            reconstruction="pi3",
            inpainting="blank",
        )
        extra = VARIANTS[variant] + [
            "pipeline.reconstruction.use_origin_frames=true",
            "pipeline.reconstruction.fill_masked_depth=true",
            "pipeline.reconstruction.mask_dilate_px=3",
            f"data.max_frames={args.max_frames}",
        ]
        runner, cfg = build_pipeline(cli, extra)

        rows: list[dict] = []
        for i, (scene, cam) in enumerate(DEFAULT_SCENES, 1):
            gt_dir = dataset_root / "short" / "test" / scene / "static" / cam
            out_dir = args.out_root / variant / f"{scene}_{cam}"
            t0 = time.monotonic()
            try:
                pred_dir = run_pipeline_for_camera(runner, cfg, scene, cam, out_dir)
            except Exception as e:
                print(f"[{variant} {i}/5] {scene}/{cam} pipeline FAILED: {e}", flush=True)
                continue
            try:
                m = evaluate_sample(gt_dir, pred_dir, save_trajectory=None)
            except Exception as e:
                print(f"[{variant} {i}/5] {scene}/{cam} eval FAILED: {e}", flush=True)
                continue
            elapsed = time.monotonic() - t0
            row = {"scene": scene, "camera": cam, "elapsed_s": round(elapsed, 1), **m}
            rows.append(row)
            print(f"[{variant} {i}/5] {scene}/{cam}  done in {elapsed:.1f}s  "
                  f"abs_rel={row.get('abs_rel', 0):.3f}  ATE={row.get('ATE', 0):.3f}  "
                  f"Acc={row.get('Acc', 0):.3f}", flush=True)
        results[variant] = rows
        with (args.out_root / f"{variant}_summary.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    # Compare per-scene
    print(f"\n{'='*120}\n  Per-scene comparison\n{'='*120}")
    for scene, cam in DEFAULT_SCENES:
        print(f"\n  {scene}/{cam}")
        print(f"  {'metric':<10} " + " ".join(f"{v:>14}" for v in args.variants))
        for m in ("AUC", "ATE", "abs_rel", "Acc", "Comp", "NC"):
            line = f"  {m:<10} "
            for v in args.variants:
                row = next((r for r in results.get(v, []) if r.get("scene") == scene and r.get("camera") == cam), None)
                line += f"{row[m]:>14.4f}" if row and m in row else f"{'—':>14}"
            print(line)

    # Aggregate
    print(f"\n{'='*120}\n  Aggregate over {len(DEFAULT_SCENES)} toy scenes\n{'='*120}")
    print(f"  {'metric':<10} " + " ".join(f"{v:>14}" for v in args.variants))
    for m in METRICS:
        line = f"  {m:<10} "
        for v in args.variants:
            vs = [r[m] for r in results.get(v, []) if m in r]
            line += f"{np.mean(vs):>14.4f}" if vs else f"{'—':>14}"
        print(line)


if __name__ == "__main__":
    main()
