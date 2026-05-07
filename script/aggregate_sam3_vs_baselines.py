#!/usr/bin/env python3
"""Compare SIGMA (full run) against pre-computed NEW_gsam / NEW_student baselines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def index_summary(path: Path) -> dict:
    """Load per_sample list from summary JSON, key by (scene, camera)."""
    data = json.loads(path.read_text())
    return {(s["scene"], s["camera"]): s for s in data.get("per_sample", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-metrics", type=Path, default=Path("outputs/sam3_full_run/_metrics"))
    ap.add_argument("--baselines", nargs="+",
                    default=["eval_results/NEW_gsam_summary.json",
                             "eval_results/NEW_student_summary.json"])
    args = ap.parse_args()

    sam3_files = sorted(args.sam3_metrics.glob("*.json"))
    sam3 = {}
    for f in sam3_files:
        if "__" not in f.stem: continue
        scene, cam = f.stem.split("__", 1)
        m = json.loads(f.read_text())
        sam3[(scene, cam)] = {
            **m.get("pose", {}),
            **m.get("depth", {}),
            **m.get("pointcloud", {}),
        }

    print(f"SIGMA scenes done: {len(sam3)}")

    # Load each baseline
    base_idx = {Path(p).stem: index_summary(Path(p)) for p in args.baselines}

    keys = ["RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr", "abs_rel", "rmse", "a1", "Acc", "Comp", "NC"]
    higher_better = {"RRA@30", "RTA@30", "AUC", "a1", "NC"}

    for bn, bidx in base_idx.items():
        common = sorted(set(sam3.keys()) & set(bidx.keys()))
        if not common:
            print(f"\n{bn}: no overlap with SAM3"); continue
        print(f"\n=== {bn}  (paired N={len(common)}) ===")
        print(f"{'metric':10} {'baseline':>12}  {'SIGMA':>12}  {'delta %':>9}  {'SIGMA wins':>11}")
        print("-" * 65)
        for k in keys:
            bvals, svals, win = [], [], 0
            for sc in common:
                b = bidx[sc].get(k); s = sam3[sc].get(k)
                if b is None or s is None or not np.isfinite(b) or not np.isfinite(s): continue
                bvals.append(b); svals.append(s)
                if (s > b) == (k in higher_better):
                    win += 1
            if not bvals: continue
            bm, sm = np.mean(bvals), np.mean(svals)
            d = (sm - bm) / abs(bm) * 100 if bm != 0 else 0.0
            arrow = "↑" if k in higher_better else "↓"
            print(f"{k:8}{arrow}  {bm:>12.4f}  {sm:>12.4f}  {d:>+8.2f}%  {win:>5}/{len(bvals)}")


if __name__ == "__main__":
    main()
