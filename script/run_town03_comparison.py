#!/usr/bin/env python3
"""Run NEW+student vs NEW+GSAM vs OLD+SDXL on every Town03 (scene, camera) pair.

Saves per-pair JSONs and aggregate summary per method.

Usage:
    python script/run_town03_comparison.py --max-frames 30 \
        --student-checkpoint checkpoints/student_unet.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import (  # noqa: E402
    build_pipeline,
    evaluate_sample,
    extract_flat,
    run_pipeline_for_camera,
)


CONFIGS = {
    # NEW + student (our learned head replacing GroundedSAM)
    "NEW_student": {
        "motion": "student",
        "inpainting": "blank",
        "overrides": [
            "pipeline.reconstruction.use_origin_frames=true",
            "pipeline.reconstruction.fill_masked_depth=true",
            "pipeline.reconstruction.mask_dilate_px=3",
        ],
    },
    # NEW + SAM3 ∪ TTT-head (end-to-end TTT + SAM3 + union)
    "NEW_sam3_ttt": {
        "motion": "sam3_ttt",
        "inpainting": "blank",
        "overrides": [
            "pipeline.reconstruction.use_origin_frames=true",
            "pipeline.reconstruction.fill_masked_depth=true",
            "pipeline.reconstruction.mask_dilate_px=3",
        ],
    },
    # NEW + GSAM (current baseline — multi-view fill, GSAM motion)
    "NEW_gsam": {
        "motion": "grounded_sam",
        "inpainting": "blank",
        "overrides": [
            "pipeline.reconstruction.use_origin_frames=true",
            "pipeline.reconstruction.fill_masked_depth=true",
            "pipeline.reconstruction.mask_dilate_px=3",
        ],
    },
    # OLD method — SDXL inpainting + GSAM, no fill
    "OLD_sdxl": {
        "motion": "grounded_sam",
        "inpainting": "sdxl_lightning_uncond",
        "overrides": [
            "pipeline.reconstruction.use_origin_frames=false",
            "pipeline.reconstruction.fill_masked_depth=false",
        ],
    },
}

METRIC_KEYS = ["RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr",
               "abs_rel", "rmse", "a1", "Acc", "Comp", "NC"]
LOWER_BETTER = {"ATE", "RPEt", "RPEr", "abs_rel", "rmse", "Acc", "Comp"}


def discover_pairs(dataset_root: Path, towns: list[str]) -> List[tuple[str, str]]:
    base = dataset_root / "short" / "test"
    pairs = []
    for scene_dir in sorted(base.iterdir()):
        if not any(t in scene_dir.name for t in towns):
            continue
        dyn = scene_dir / "dynamic"
        if not dyn.is_dir():
            continue
        for cam_dir in sorted(dyn.iterdir()):
            if cam_dir.name.startswith(("cam_", "camera_")) and (cam_dir / "rgb.mp4").exists():
                pairs.append((scene_dir.name, cam_dir.name))
    return pairs


def run_method(
    method_name: str,
    pairs: List[tuple[str, str]],
    dataset_root: Path,
    max_frames: int,
    student_ckpt: Path | None,
    out_root: Path,
) -> List[Dict]:
    spec = CONFIGS[method_name]
    args = SimpleNamespace(
        dataset_root=dataset_root,
        duration="short",
        split="test",
        data_format="dynamic",
        reconstruction="pi3",
        inpainting=spec["inpainting"],
    )
    extra = list(spec["overrides"]) + [f"data.max_frames={max_frames}",
                                        f"pipeline/motion={spec['motion']}"]
    if spec["motion"] == "student":
        if student_ckpt is None:
            raise SystemExit("student method requires --student-checkpoint")
        # Hydra config groups are loaded after explicit overrides; use ++ to
        # force-set the value within the chosen group's struct.
        extra.append(f"++pipeline.motion.checkpoint={student_ckpt}")

    print(f"\n{'='*78}\nLoading {method_name} (motion={spec['motion']}, "
          f"inpaint={spec['inpainting']})\n{'='*78}", flush=True)
    runner, cfg = build_pipeline(args, extra)

    method_dir = out_root / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for i, (scene, cam) in enumerate(pairs, start=1):
        gt_dir = dataset_root / "short" / "test" / scene / "static" / cam
        out_dir = ROOT / "outputs" / "_town03_compare" / method_name / f"{scene}_{cam}"

        per = method_dir / scene / f"{cam}.json"
        if per.exists():
            rows.append(json.load(open(per)))
            continue

        t0 = time.monotonic()
        try:
            pred_dir = run_pipeline_for_camera(runner, cfg, scene, cam, out_dir)
        except Exception as e:
            print(f"[{method_name} {i}/{len(pairs)}] {scene}/{cam}  pipeline FAILED: {e}", flush=True)
            continue
        t_pipe = time.monotonic() - t0

        results = evaluate_sample(gt_dir, pred_dir, save_trajectory=None)
        if results is None:
            print(f"[{method_name} {i}/{len(pairs)}] {scene}/{cam}  eval FAILED", flush=True)
            continue

        flat = extract_flat(results)
        flat["scene"], flat["camera"] = scene, cam
        flat["t_pipe_s"] = round(t_pipe, 2)
        rows.append(flat)
        per.parent.mkdir(parents=True, exist_ok=True)
        json.dump(flat, open(per, "w"), indent=2)

        if (i % 10 == 0) or i == len(pairs):
            print(f"[{method_name} {i}/{len(pairs)}] {scene}/{cam}  done in {t_pipe:.1f}s  "
                  f"abs_rel={flat.get('abs_rel', float('nan')):.4f}  "
                  f"ATE={flat.get('ATE', float('nan')):.4f}", flush=True)

        # Cleanup pred dir to save disk
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

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


def print_table(name_to_agg: Dict[str, Dict[str, float]], n: int) -> None:
    print()
    methods = list(name_to_agg)
    print(f"{'='*100}")
    print(f"  Aggregate over {n} Town03 pairs")
    print(f"{'='*100}")
    print(f"  {'metric':<10}  " + "  ".join(f"{m:>14}" for m in methods))
    print("  " + "-" * (12 + 16 * len(methods)))
    for k in METRIC_KEYS:
        line = f"  {k:<10}  "
        vals = []
        for m in methods:
            v = name_to_agg[m].get(k)
            vals.append(v)
            line += f"{v:>14.4f}  " if v is not None else f"{'-':>14}  "
        # Mark winner
        valid = [(m, v) for m, v in zip(methods, vals) if v is not None]
        if len(valid) >= 2:
            if k in LOWER_BETTER:
                winner = min(valid, key=lambda x: x[1])[0]
            else:
                winner = max(valid, key=lambda x: x[1])[0]
            line += f" ← {winner}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--student-checkpoint", type=Path, default=Path("checkpoints/student_unet.pt"))
    p.add_argument("--out-root", type=Path, default=Path("eval_results/town03_compare"))
    p.add_argument("--methods", default="NEW_student,NEW_gsam,OLD_sdxl",
                   help="Comma-separated method names from CONFIGS dict.")
    p.add_argument("--towns", nargs="+", default=["T03"],
                   help="Town filters (substring match on scene name).")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    pairs = discover_pairs(args.dataset_root, args.towns)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Pairs ({'+'.join(args.towns)}): {len(pairs)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    method_aggs: Dict[str, Dict[str, float]] = {}
    method_rows: Dict[str, List[Dict]] = {}

    for m in args.methods.split(","):
        m = m.strip()
        if m not in CONFIGS:
            print(f"Unknown method: {m}; choose from {list(CONFIGS)}")
            continue
        rows = run_method(m, pairs, args.dataset_root, args.max_frames,
                           args.student_checkpoint, args.out_root)
        method_rows[m] = rows
        method_aggs[m] = aggregate(rows)
        json.dump({"n": len(rows), "aggregate": method_aggs[m], "rows": rows},
                   open(args.out_root / f"{m}_summary.json", "w"), indent=2)
        print(f"\n>>> {m}: {len(rows)} pairs done", flush=True)

    n_min = min(len(r) for r in method_rows.values()) if method_rows else 0
    print_table(method_aggs, n_min)


if __name__ == "__main__":
    main()
