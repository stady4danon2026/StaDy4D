#!/usr/bin/env python3
"""Compare NEW (multi-view fill) results against the cached OLD baseline.

NEW results: eval_results/compare_methods/NEW/<scene>/<camera>.json
OLD results: eval_results/short/test/<scene>/<camera>.json  (1451-sample baseline)

For each NEW pair, look up the same scene/camera in the OLD cache and emit
per-pair side-by-side metrics + an aggregate delta table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

METRIC_KEYS = [
    "RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr",
    "abs_rel", "rmse", "a1", "Acc", "Comp", "NC",
]
LOWER_BETTER = {"ATE", "RPEt", "RPEr", "abs_rel", "rmse", "Acc", "Comp"}
FMT = {
    "RRA@30": ".1f", "RTA@30": ".1f", "AUC": ".1f",
    "ATE": ".4f", "RPEt": ".4f", "RPEr": ".4f",
    "abs_rel": ".4f", "rmse": ".4f", "a1": ".4f",
    "Acc": ".4f", "Comp": ".4f", "NC": ".4f",
}


def load_pairs(root: Path) -> Dict[tuple, Dict]:
    out: Dict[tuple, Dict] = {}
    for scene_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for f in sorted(scene_dir.glob("*.json")):
            d = json.load(open(f))
            out[(scene_dir.name, f.stem)] = d
    return out


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-root", type=Path, default=Path("eval_results/compare_methods/NEW"))
    p.add_argument("--old-root", type=Path, default=Path("eval_results/short/test"))
    p.add_argument("--out", type=Path, default=Path("eval_results/compare_methods/delta_summary.json"))
    args = p.parse_args()

    new_pairs = load_pairs(args.new_root)
    old_all = load_pairs(args.old_root)

    if not new_pairs:
        print(f"No NEW results in {args.new_root}")
        return

    print(f"NEW results: {len(new_pairs)} pairs in {args.new_root}")
    print(f"OLD baseline: {len(old_all)} pairs in {args.old_root}")
    print()

    matched_new, matched_old, missing = [], [], []
    for key, new_row in new_pairs.items():
        if key in old_all:
            matched_new.append(new_row)
            matched_old.append(old_all[key])
        else:
            missing.append(key)
    if missing:
        print(f"WARNING: {len(missing)} pairs missing from OLD baseline:")
        for m in missing:
            print(f"  - {m[0]}/{m[1]}")
        print()

    n = len(matched_new)
    print(f"Matched: {n} pairs")
    print()

    # Per-pair side by side
    cols = ["abs_rel", "rmse", "a1", "Acc", "Comp", "NC", "ATE"]
    hdr = f"  {'pair':<48} | " + " ".join(f"{c:>7}" for c in cols)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for nrow, orow in zip(matched_new, matched_old):
        label = f"{nrow.get('scene','?')}/{nrow.get('camera','?')}"[:48]
        old_bits = " ".join(f"{orow.get(c, float('nan')):>7{FMT.get(c,'.3f')}}" for c in cols)
        new_bits = " ".join(f"{nrow.get(c, float('nan')):>7{FMT.get(c,'.3f')}}" for c in cols)
        print(f"  {label:<48} OLD: {old_bits}")
        print(f"  {'':<48} NEW: {new_bits}")

    new_agg = aggregate(matched_new)
    old_agg = aggregate(matched_old)

    print()
    print(f"{'='*78}")
    print(f"  Aggregate over {n} pairs (NEW = multi-view fill, OLD = baseline)")
    print(f"{'='*78}")
    print(f"  {'metric':<10} {'OLD':>11}  {'NEW':>11}  {'Δ':>11}  {'%':>8}  better")
    print(f"  {'-'*64}")
    for k in METRIC_KEYS:
        if k not in new_agg or k not in old_agg:
            continue
        o, n_ = old_agg[k], new_agg[k]
        delta = n_ - o
        pct = 100.0 * delta / o if o != 0 else float("nan")
        if k in LOWER_BETTER:
            tag = "NEW" if n_ < o else "OLD" if n_ > o else "tie"
        else:
            tag = "NEW" if n_ > o else "OLD" if n_ < o else "tie"
        print(f"  {k:<10} {o:>11.4f}  {n_:>11.4f}  {delta:>+11.4f}  {pct:>+7.1f}%  {tag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n": n,
            "new_aggregate": new_agg,
            "old_aggregate": old_agg,
            "missing_in_old": [list(m) for m in missing],
        }, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
