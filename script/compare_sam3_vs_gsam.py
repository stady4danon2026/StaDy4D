#!/usr/bin/env python3
"""Compare per-scene metrics: SAM3+TTT (new flat pipeline) vs NEW_gsam (existing).

Loads both outputs, runs sigma evaluator on each, prints side-by-side table.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path


SCENES = [
    ("scene_T03_000", "cam_00_car_forward"),
    ("scene_T03_021", "cam_05_orbit_crossroad"),
    ("scene_T07_023", "cam_05_orbit_crossroad"),
    ("scene_T07_007", "cam_04_orbit_building"),
    ("scene_T10_028", "cam_00_car_forward"),
]


def run_eval(pred_dir: Path, gt_dir: Path) -> dict:
    """Invoke sigma.evaluation.evaluate and parse result."""
    out_json = pred_dir.parent / f"{pred_dir.name}__eval.json"
    cmd = [
        "python", "-m", "sigma.evaluation.evaluate",
        "--gt", str(gt_dir),
        "--pred", str(pred_dir),
        "--metrics", "pose", "depth", "pointcloud",
        "--output", str(out_json),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-500:]}
    if out_json.exists():
        return json.loads(out_json.read_text())
    # fallback: parse stdout
    return {"stdout": r.stdout[-1000:]}


def fmt(v):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-root", default="outputs/sam3_compare/SAM3_TTT", type=Path)
    ap.add_argument("--gsam-root", default="eval_results/sam3_toy_iso/NEW_gsam", type=Path)
    ap.add_argument("--gt-root", default="StaDy4D/short/test", type=Path)
    args = ap.parse_args()

    rows = []
    for scene, cam in SCENES:
        gt = args.gt_root / scene / "dynamic" / cam
        sam3_pred = args.sam3_root / scene / cam
        gsam_pred = args.gsam_root / f"{scene}_{cam}_pipeline"
        if not sam3_pred.exists():
            print(f"SKIP {scene}/{cam}: SAM3 pred missing at {sam3_pred}")
            continue
        if not gsam_pred.exists():
            print(f"SKIP {scene}/{cam}: GSAM pred missing at {gsam_pred}")
            continue
        print(f"\n=== {scene}/{cam} ===")
        sam3_m = run_eval(sam3_pred, gt)
        gsam_m = run_eval(gsam_pred, gt)
        rows.append((f"{scene}/{cam}", sam3_m, gsam_m))

    # print table
    keys = [
        ("pose", "ATE"), ("pose", "AUC"), ("pose", "RPEt"), ("pose", "RPEr"),
        ("depth", "abs_rel"), ("depth", "rmse"), ("depth", "a1"),
        ("pointcloud", "Acc"), ("pointcloud", "Comp"), ("pointcloud", "NC"),
    ]
    print("\n\nMetric                | " + " | ".join(f"{r[0][:30]:30}" for r in rows))
    print("-" * (24 + 33 * len(rows)))
    for grp, k in keys:
        for who, idx in [("GSAM", 2), ("SAM3", 1)]:
            line = f"{grp}/{k:8} ({who}) | "
            for r in rows:
                v = r[idx].get(grp, {}).get(k) if isinstance(r[idx], dict) else None
                line += f"{fmt(v):30} | "
            print(line)
        print()


if __name__ == "__main__":
    main()
