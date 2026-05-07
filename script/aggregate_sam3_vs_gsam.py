#!/usr/bin/env python3
"""Aggregate _metrics/*.json from SAM3+TTT full run; compare averages with NEW_gsam.

Only averages over scene-cams where BOTH sides have metrics, so comparison is fair.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np


def load_metrics(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def eval_gsam(gsam_root: Path, gt_root: Path, scene: str, cam: str) -> dict | None:
    pred = gsam_root / f"{scene}_{cam}_pipeline"
    gt = gt_root / scene / "dynamic" / cam
    if not pred.exists() or not gt.exists():
        return None
    out = Path(f"/tmp/gsam_{scene}__{cam}.json")
    if not out.exists():
        r = subprocess.run([
            "python", "-m", "sigma.evaluation.evaluate",
            "--gt", str(gt), "--pred", str(pred),
            "--metrics", "pose", "depth", "pointcloud",
            "--output", str(out),
        ], capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not out.exists():
            return None
    return load_metrics(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-metrics", type=Path, default=Path("outputs/sam3_full_run/_metrics"))
    ap.add_argument("--gsam-root", type=Path, default=Path("eval_results/sam3_toy_iso/NEW_gsam"))
    ap.add_argument("--gt-root", type=Path, default=Path("StaDy4D/short/test"))
    args = ap.parse_args()

    sam3_files = sorted(args.sam3_metrics.glob("*.json"))
    print(f"SAM3+TTT metrics: {len(sam3_files)} files")

    paired = []   # list of (scene, cam, sam3_dict, gsam_dict)
    for f in sam3_files:
        # filename: scene__cam.json
        stem = f.stem
        if "__" not in stem: continue
        scene, cam = stem.split("__", 1)
        sm = load_metrics(f)
        if sm is None: continue
        gm = eval_gsam(args.gsam_root, args.gt_root, scene, cam)
        if gm is None: continue
        paired.append((scene, cam, sm, gm))

    print(f"paired SAM3 ↔ GSAM: {len(paired)}")
    if not paired:
        return

    keys = [
        ("pose", "ATE"), ("pose", "AUC"), ("pose", "RPEt"), ("pose", "RPEr"),
        ("depth", "abs_rel"), ("depth", "rmse"), ("depth", "a1"),
        ("pointcloud", "Acc"), ("pointcloud", "Comp"), ("pointcloud", "NC"),
    ]
    print(f"\n{'metric':18} {'GSAM mean':>14}  {'SAM3+TTT mean':>16}  {'delta %':>10}  {'sam3 wins':>10}")
    print("-" * 80)
    for grp, k in keys:
        gv, sv, win = [], [], 0
        for s, c, sm, gm in paired:
            g = gm.get(grp, {}).get(k); s_ = sm.get(grp, {}).get(k)
            if g is None or s_ is None: continue
            if not (np.isfinite(g) and np.isfinite(s_)): continue
            gv.append(g); sv.append(s_)
            higher_better = k in ("AUC", "a1", "NC")
            if (s_ > g) == higher_better:
                win += 1
        if not gv: continue
        gm_avg, sm_avg = np.mean(gv), np.mean(sv)
        delta = (sm_avg - gm_avg) / abs(gm_avg) * 100 if gm_avg != 0 else 0.0
        print(f"{grp+'/'+k:18} {gm_avg:>14.4f}  {sm_avg:>16.4f}  {delta:>+9.2f}%  {win:>5}/{len(gv)}")


if __name__ == "__main__":
    main()
