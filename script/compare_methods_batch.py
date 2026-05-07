#!/usr/bin/env python3
"""Paired NEW vs OLD method comparison on a chosen scene/camera set.

Loads each method's pipeline once, runs all pairs back-to-back, saves
per-pair JSONs and an aggregate. Prints a delta table.

Methods:
    NEW  = inpainting=blank   + use_origin_frames=true  + fill_masked_depth=true
    OLD  = inpainting=sdxl_lightning_uncond + use_origin_frames=false + fill_masked_depth=false
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import (  # noqa: E402
    build_pipeline,
    evaluate_sample,
    extract_flat,
    run_pipeline_for_camera,
)


METHODS = {
    "NEW": {
        "inpainting": "blank",
        "overrides": [
            "pipeline.reconstruction.use_origin_frames=true",
            "pipeline.reconstruction.fill_masked_depth=true",
            "pipeline.reconstruction.mask_dilate_px=3",
        ],
    },
    "OLD": {
        "inpainting": "sdxl_lightning_uncond",
        "overrides": [
            "pipeline.reconstruction.use_origin_frames=false",
            "pipeline.reconstruction.fill_masked_depth=false",
        ],
    },
}

METRIC_KEYS = [
    "RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr",
    "abs_rel", "rmse", "a1", "Acc", "Comp", "NC",
]
LOWER_BETTER = {"ATE", "RPEt", "RPEr", "abs_rel", "rmse", "Acc", "Comp"}


def parse_pairs(spec: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        scene, camera = tok.split("/", 1)
        out.append((scene, camera))
    return out


def run_method(
    method_name: str,
    pairs: List[Tuple[str, str]],
    dataset_root: Path,
    duration: str,
    split: str,
    data_format: str,
    max_frames: int,
    out_root: Path,
) -> List[Dict]:
    """Run a single method on all pairs. Returns list of flat dicts."""
    spec = METHODS[method_name]
    args = SimpleNamespace(
        dataset_root=dataset_root,
        duration=duration,
        split=split,
        data_format=data_format,
        reconstruction="pi3",
        inpainting=spec["inpainting"],
    )
    extra = list(spec["overrides"]) + [f"data.max_frames={max_frames}"]

    print(f"\n{'='*80}\nLoading {method_name} pipeline (inpainting={spec['inpainting']}) ...\n{'='*80}", flush=True)
    runner, cfg = build_pipeline(args, extra)

    method_dir = out_root / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for i, (scene, camera) in enumerate(pairs, start=1):
        gt_dir = dataset_root / duration / split / scene / "static" / camera
        out_dir = ROOT / "outputs" / "_compare_methods" / method_name / f"{scene}_{camera}"

        print(f"\n[{method_name} {i}/{len(pairs)}] {scene}/{camera}", flush=True)
        t0 = time.monotonic()
        try:
            pred_dir = run_pipeline_for_camera(runner, cfg, scene, camera, out_dir)
        except Exception as e:
            print(f"  pipeline FAILED: {e}", flush=True)
            continue
        t_pipe = time.monotonic() - t0

        results = evaluate_sample(gt_dir, pred_dir, save_trajectory=None)
        if results is None:
            print("  evaluation FAILED", flush=True)
            continue

        flat = extract_flat(results)
        flat["scene"] = scene
        flat["camera"] = camera
        flat["t_pipe_s"] = round(t_pipe, 2)
        rows.append(flat)

        # Save per-pair JSON
        per_dir = method_dir / scene
        per_dir.mkdir(parents=True, exist_ok=True)
        with open(per_dir / f"{camera}.json", "w") as f:
            json.dump(flat, f, indent=2)

        bits = " ".join(
            f"{k}={flat[k]:.4f}" if isinstance(flat[k], float) else ""
            for k in ("abs_rel", "rmse", "Acc", "Comp", "NC", "ATE")
        )
        print(f"  done in {t_pipe:.1f}s  {bits}", flush=True)

    # Teardown models
    for name in runner.active_modules:
        runner.stages[name].teardown()

    return rows


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def print_delta(new_agg: Dict[str, float], old_agg: Dict[str, float], n: int) -> None:
    print()
    print(f"{'='*72}")
    print(f"  NEW vs OLD aggregates over {n} (scene, camera) pairs")
    print(f"{'='*72}")
    print(f"  {'metric':<10} {'OLD':>11}  {'NEW':>11}  {'Δ':>11}  {'%':>7}  better")
    print(f"  {'-'*60}")
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
        print(f"  {k:<10} {o:>11.4f}  {n_:>11.4f}  {delta:>+11.4f}  {pct:>+6.1f}%  {tag}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True,
                   help="Comma-separated scene/camera, e.g. scene_T03_001/cam_00_car_forward,...")
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="test")
    p.add_argument("--data-format", default="dynamic")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--out-root", type=Path, default=Path("eval_results/compare_methods"))
    p.add_argument("--methods", default="NEW,OLD",
                   help="Comma-separated method names to run, in order (NEW,OLD or just NEW).")
    args = p.parse_args()

    pairs = parse_pairs(args.pairs)
    print(f"{len(pairs)} pairs:")
    for s, c in pairs:
        print(f"  {s}/{c}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    method_results: Dict[str, List[Dict]] = {}
    method_aggs: Dict[str, Dict[str, float]] = {}

    for method_name in args.methods.split(","):
        method_name = method_name.strip()
        rows = run_method(
            method_name=method_name,
            pairs=pairs,
            dataset_root=args.dataset_root,
            duration=args.duration,
            split=args.split,
            data_format=args.data_format,
            max_frames=args.max_frames,
            out_root=args.out_root,
        )
        method_results[method_name] = rows
        method_aggs[method_name] = aggregate(rows)

        with open(args.out_root / f"{method_name}_aggregate.json", "w") as f:
            json.dump({"n": len(rows), "aggregate": method_aggs[method_name],
                       "per_pair": rows}, f, indent=2)

    if "NEW" in method_aggs and "OLD" in method_aggs:
        n = min(len(method_results["NEW"]), len(method_results["OLD"]))
        print_delta(method_aggs["NEW"], method_aggs["OLD"], n)


if __name__ == "__main__":
    main()
