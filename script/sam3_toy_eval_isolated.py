#!/usr/bin/env python3
"""Toy eval with subprocess-per-scene isolation.

Each (variant, scene) pair runs in its own Python subprocess so GPU memory
is fully released between runs. Lets us include the full SAM3+TTT path
without OOM from stacked Pi3 + GroundedSAM + SAM3 models.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "NEW_gsam":      ["pipeline/motion=grounded_sam"],
    "NEW_sam3_ttt":  ["pipeline/motion=sam3_ttt",
                      "pipeline.motion.score_thr=0.3",
                      "pipeline.motion.do_ttt=true",
                      "pipeline.motion.do_head=true"],
}

SCENES = [
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

WORKER = ROOT / "script" / "_sam3_toy_eval_one_scene.py"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "eval_results/sam3_toy_iso")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        rows = []
        for i, (scene, cam) in enumerate(SCENES, 1):
            out_json = args.out / variant / f"{scene}__{cam}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            if out_json.exists():
                rows.append(json.load(open(out_json)))
                print(f"[{variant} {i}/{len(SCENES)}] {scene}/{cam}  cached", flush=True)
                continue
            t0 = time.monotonic()
            cmd = ["python", str(WORKER),
                   "--scene", scene, "--camera", cam,
                   "--out", str(out_json),
                   "--variant"] + VARIANTS[variant]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                    env=dict(__import__("os").environ,
                                             PYTHONPATH=f"<PROJECT_ROOT>/Pi3:{__import__('os').environ.get('PYTHONPATH','')}"))
                elapsed = time.monotonic() - t0
                if r.returncode != 0 or not out_json.exists():
                    print(f"[{variant} {i}/{len(SCENES)}] {scene}/{cam}  FAILED ({elapsed:.0f}s)", flush=True)
                    print(r.stderr[-500:] if r.stderr else "")
                    continue
                row = json.load(open(out_json))
                rows.append(row)
                print(f"[{variant} {i}/{len(SCENES)}] {scene}/{cam}  done in {elapsed:.0f}s  "
                      f"abs_rel={row.get('depth',{}).get('abs_rel',0):.3f}  "
                      f"Acc={row.get('pointcloud',{}).get('Acc',0):.3f}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[{variant} {i}/{len(SCENES)}] {scene}/{cam}  TIMEOUT", flush=True)

        agg_path = args.out / f"{variant}_summary.json"
        with agg_path.open("w") as fh:
            json.dump(rows, fh, indent=2)

    # Aggregate compare
    print(f"\n{'='*100}")
    print(f"  Aggregate over {len(SCENES)} toy scenes")
    print(f"{'='*100}")
    all_data = {v: json.load(open(args.out / f"{v}_summary.json")) for v in args.variants}
    print(f"  {'metric':<10} " + " ".join(f"{v:>16}" for v in args.variants))
    for sec, m in [("pose", "AUC"), ("pose", "ATE"), ("pose", "RPEt"), ("pose", "RPEr"),
                    ("depth", "abs_rel"), ("depth", "rmse"), ("depth", "a1"),
                    ("pointcloud", "Acc"), ("pointcloud", "Comp"), ("pointcloud", "NC")]:
        line = f"  {m:<10} "
        for v in args.variants:
            vs = [r.get(sec, {}).get(m) for r in all_data[v]]
            vs = [x for x in vs if x is not None]
            line += f"{np.mean(vs):>16.4f}" if vs else f"{'—':>16}"
        print(line)


if __name__ == "__main__":
    main()
