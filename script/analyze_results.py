#!/usr/bin/env python3
"""Analyze evaluation results by camera type, town, weather, and scene.

Usage:
    python script/analyze_results.py
    python script/analyze_results.py --eval-dir eval_results --duration short --split test
    python script/analyze_results.py --format latex
    python script/analyze_results.py --format csv --output results.csv

    # Evaluate + analyze raw pipeline outputs (minival / ablation runs):
    python script/analyze_results.py --minival outputs/ablation_pi3_static outputs/ablation_vggt_static
    python script/analyze_results.py --minival outputs/ablation_pi3_static --dataset-root StaDy4D
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np

METRIC_KEYS = [
    "RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr",
    "abs_rel", "rmse", "a1",
    "Acc", "Comp", "NC",
]

METRIC_FMT = {
    "RRA@30": ".1f", "RTA@30": ".1f", "AUC": ".1f",
    "ATE": ".4f", "RPEt": ".6f", "RPEr": ".4f",
    "abs_rel": ".4f", "rmse": ".4f", "a1": ".4f",
    "Acc": ".4f", "Comp": ".4f", "NC": ".4f",
}

# Higher is better for these metrics
HIGHER_IS_BETTER = {"RRA@30", "RTA@30", "AUC", "a1", "NC"}

# Minival subset: every 3rd scene, cycling through cameras 0-8.
# cam = (scene_suffix / 3) % 9, for scene_suffix in 0, 3, 6, ..., 51
# -> 18 (scene, cam) pairs per town, 54 total across T03/T07/T10
MINIVAL_TOWNS = ["T03", "T07", "T10"]
MINIVAL_PAIRS: List[tuple[str, str]] = []  # (scene_name, cam_prefix) e.g. ("scene_T03_000", "cam_00")
for _town in MINIVAL_TOWNS:
    for _i in range(18):  # 0..17
        _scene_idx = _i * 3           # 0, 3, 6, ..., 51
        _cam_idx = _i % 9             # 0, 1, 2, ..., 8, 0, 1, ..., 8
        MINIVAL_PAIRS.append((f"scene_{_town}_{_scene_idx:03d}", f"cam_{_cam_idx:02d}"))


def _is_minival_match(scene: str, camera: str) -> bool:
    """Check if (scene, camera) is in the minival subset."""
    for mv_scene, mv_cam_prefix in MINIVAL_PAIRS:
        if scene == mv_scene and camera.startswith(mv_cam_prefix + "_"):
            return True
        # exact match (e.g. cam_00 == cam_00)
        if scene == mv_scene and camera == mv_cam_prefix:
            return True
    return False


def load_all_results(eval_dir: Path, dataset_root: Path) -> List[Dict]:
    """Load all per-camera JSON results and enrich with metadata."""
    records = []
    for scene_dir in sorted(eval_dir.iterdir()):
        print(scene_dir.name)
        if not scene_dir.is_dir() or scene_dir.name == "summary.json":
            continue
        scene = scene_dir.name

        # Load metadata
        meta = {}
        meta_path = dataset_root / scene / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        town = meta.get("town", meta.get("map_name", "unknown"))
        weather = meta.get("weather", "unknown")
        camera_types = meta.get("camera_types", [])

        for cam_file in sorted(scene_dir.glob("*.json")):
            with open(cam_file) as f:
                print(cam_file)
                data = json.load(f)

            camera = data.get("camera", cam_file.stem)

            # Extract camera type from name (e.g. cam_03_drone_forward -> drone_forward)
            cam_type = re.sub(r"^cam_\d+_|^camera_\d+_", "", camera)

            # Or use metadata camera_types if available
            cam_idx_match = re.match(r"cam_(\d+)", camera)
            if cam_idx_match and camera_types:
                idx = int(cam_idx_match.group(1))
                if idx < len(camera_types):
                    cam_type = camera_types[idx]

            data["scene"] = scene
            data["camera"] = camera
            data["cam_type"] = cam_type
            data["town"] = town
            data["weather"] = weather
            records.append(data)

    return records


def _resolve_gt_dir(cfg: Dict, dataset_root: Path) -> Path | None:
    """Derive the static GT directory from a run's config.yaml data section."""
    data = cfg.get("data", {})
    root = dataset_root or Path(data.get("root_dir", "StaDy4D"))
    duration = data.get("duration", "short")
    split = data.get("split", ".")
    scene = data.get("scene", "")
    camera = data.get("camera", "")
    if not scene or not camera:
        return None
    # Build: root/duration/split/scene/static/camera  (split="." collapses)
    parts = [str(root)]
    if duration:
        parts.append(duration)
    if split and split != ".":
        parts.append(split)
    parts += [scene, "static", camera]
    return Path("/".join(parts))


def _evaluate_pred_vs_gt(pred_dir: Path, gt_dir: Path) -> Dict[str, float] | None:
    """Run evaluation on a single pred dir against GT, return flat metrics."""
    from sigma.evaluation.evaluator import Evaluator

    try:
        evaluator = Evaluator(gt_dir=gt_dir, pred_dir=pred_dir)
        evaluator.load(load_depth=True, load_rgb=False)
        results = {}
        results["pose"] = evaluator.evaluate_poses()
        results["depth"] = evaluator.evaluate_depth()
        results["pointcloud"] = evaluator.evaluate_pointcloud()
    except Exception as e:
        print(f"  [eval] ERROR on {pred_dir}: {e}")
        return None

    flat: Dict[str, float] = {}
    if "pose" in results and "error" not in results["pose"]:
        for k in ("RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr"):
            flat[k] = results["pose"].get(k, float("nan"))
    if "depth" in results and "error" not in results["depth"]:
        for k in ("abs_rel", "rmse", "a1"):
            flat[k] = results["depth"].get(k, float("nan"))
    if "pointcloud" in results and "error" not in results["pointcloud"]:
        for k in ("Acc", "Comp", "NC"):
            flat[k] = results["pointcloud"].get(k, float("nan"))
    return flat if flat else None


def load_minival_results(
    run_dirs: List[Path], dataset_root: Path,
) -> List[Dict]:
    """Scan pipeline output dirs, evaluate each against static GT, return records.

    Each run_dir (e.g. outputs/ablation_pi3_static) contains timestamped subdirs
    like 0320_0438/, each with config.yaml + depth/extrinsics/intrinsics.
    """
    records: List[Dict] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            print(f"Warning: {run_dir} not found, skipping")
            continue
        run_name = run_dir.name

        # Find timestamped subdirs (contain config.yaml)
        subdirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()],
            key=lambda p: p.name,
        )
        if not subdirs:
            # Maybe run_dir itself has config.yaml (single run)
            if (run_dir / "config.yaml").exists():
                subdirs = [run_dir]

        for pred_dir in subdirs:
            config_path = pred_dir / "config.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            data = cfg.get("data", {})
            scene = data.get("scene", "unknown")
            camera = data.get("camera", "unknown")
            data_format = data.get("data_format", "unknown")

            gt_dir = _resolve_gt_dir(cfg, dataset_root)
            if gt_dir is None or not gt_dir.exists():
                print(f"  {run_name}/{pred_dir.name}: GT not found at {gt_dir}, skipping")
                continue

            print(f"  Evaluating {run_name}/{pred_dir.name}: {scene}/{camera} ...", flush=True)
            flat = _evaluate_pred_vs_gt(pred_dir, gt_dir)
            if flat is None:
                print(f"    -> evaluation failed")
                continue

            # Enrich with metadata
            cam_type = re.sub(r"^cam_\d+_|^camera_\d+_", "", camera)

            # Try loading scene metadata
            root = dataset_root or Path(data.get("root_dir", "StaDy4D"))
            duration = data.get("duration", "short")
            split = data.get("split", ".")
            meta_parts = [str(root)]
            if duration:
                meta_parts.append(duration)
            if split and split != ".":
                meta_parts.append(split)
            meta_parts.append(scene)
            meta_path = Path("/".join(meta_parts)) / "metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            record = {
                "scene": scene,
                "camera": camera,
                "cam_type": cam_type,
                "town": meta.get("town", meta.get("map_name", "unknown")),
                "weather": meta.get("weather", "unknown"),
                "run_name": run_name,
                "run_timestamp": pred_dir.name,
                "data_format": data_format,
                **flat,
            }
            records.append(record)
            print(f"    -> OK")

    return records


def aggregate(records: List[Dict], group_key: str) -> Dict[str, Dict[str, float]]:
    """Group records by a key and compute mean metrics per group."""
    groups = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)

    result = {}
    for name, group in sorted(groups.items()):
        agg = {"count": len(group)}
        for k in METRIC_KEYS:
            vals = [r[k] for r in group if k in r and np.isfinite(r[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        result[name] = agg
    return result


def print_table(title: str, agg: Dict[str, Dict], fmt: str = "pretty"):
    """Print aggregated results as a table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    if fmt == "latex":
        _print_latex(title, agg)
        return
    if fmt == "csv":
        _print_csv(agg)
        return

    # Pretty table
    name_width = max(len(n) for n in agg) + 2
    header = f"{'Group':<{name_width}} {'N':>4} "
    header += " ".join(f"{k:>10}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))

    for name, vals in agg.items():
        row = f"{name:<{name_width}} {vals['count']:>4} "
        for k in METRIC_KEYS:
            v = vals.get(k, float("nan"))
            f = METRIC_FMT.get(k, ".4f")
            row += f"{v:>{10}{f}} "
        print(row)

    # Overall mean
    print("-" * len(header))
    all_vals = {"count": sum(v["count"] for v in agg.values())}
    for k in METRIC_KEYS:
        vals = [v[k] for v in agg.values() if np.isfinite(v[k])]
        all_vals[k] = float(np.mean(vals)) if vals else float("nan")
    row = f"{'Overall':<{name_width}} {all_vals['count']:>4} "
    for k in METRIC_KEYS:
        v = all_vals.get(k, float("nan"))
        f = METRIC_FMT.get(k, ".4f")
        row += f"{v:>{10}{f}} "
    print(row)


def _print_latex(title: str, agg: Dict[str, Dict]):
    """Print as LaTeX table."""
    print(f"% {title}")
    print(r"\begin{tabular}{l" + "c" * len(METRIC_KEYS) + "}")
    print(r"\toprule")
    header = " & ".join(METRIC_KEYS)
    print(f"Group & {header} \\\\")
    print(r"\midrule")
    for name, vals in agg.items():
        cells = []
        for k in METRIC_KEYS:
            v = vals.get(k, float("nan"))
            f = METRIC_FMT.get(k, ".4f")
            cells.append(f"{v:{f}}")
        print(f"{name} & {' & '.join(cells)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def _print_csv(agg: Dict[str, Dict]):
    """Print as CSV."""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["group", "count"] + METRIC_KEYS)
    for name, vals in agg.items():
        row = [name, vals["count"]]
        for k in METRIC_KEYS:
            row.append(vals.get(k, ""))
        writer.writerow(row)
    print(buf.getvalue(), end="")


def print_per_scene(records: List[Dict], fmt: str = "pretty"):
    """Print per-scene summary (mean across cameras for each scene)."""
    agg = aggregate(records, "scene")
    print_table("Per-Scene Performance", agg, fmt)


def print_best_worst(records: List[Dict], n: int = 5):
    """Print best and worst performing samples for each metric."""
    print(f"\n{'=' * 60}")
    print(f"  Best / Worst Samples (top-{n})")
    print(f"{'=' * 60}")

    for k in METRIC_KEYS:
        valid = [(r, r[k]) for r in records if k in r and np.isfinite(r[k])]
        if not valid:
            continue
        reverse = k in HIGHER_IS_BETTER
        sorted_vals = sorted(valid, key=lambda x: x[1], reverse=reverse)

        f = METRIC_FMT.get(k, ".4f")
        best = sorted_vals[:n]
        worst = sorted_vals[-n:]

        print(f"\n  {k} ({'higher' if reverse else 'lower'} is better):")
        print(f"    Best:  ", end="")
        print(", \t".join(f"{r['scene']}/{r['camera'][:6]}={v:{f}}" for r, v in best))
        print(f"    Worst: ", end="")
        print(", \t".join(f"{r['scene']}/{r['camera'][:6]}={v:{f}}" for r, v in worst))


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    parser.add_argument("--duration", default="short")
    parser.add_argument("--split", default="test")
    parser.add_argument("--format", choices=["pretty", "latex", "csv"], default="pretty")
    parser.add_argument("--output", type=Path, default=None, help="Save output to file")
    parser.add_argument("--top-n", type=int, default=10, help="Number of best/worst to show")
    parser.add_argument("--minival", nargs="*", type=Path, default=None,
                        help="Filter to minival subset. Optionally pass raw output dirs to evaluate.")
    args = parser.parse_args()

    # --- Load records ---
    if args.minival is not None and len(args.minival) > 0:
        # Minival with raw output dirs: evaluate then analyze
        print(f"Minival mode: evaluating {len(args.minival)} output dir(s)")
        records = load_minival_results(args.minival, args.dataset_root)
        source_label = ", ".join(str(p) for p in args.minival)
    else:
        eval_dir = args.eval_dir / args.duration / args.split
        dataset_dir = args.dataset_root / args.duration / args.split

        if not eval_dir.exists():
            print(f"Eval directory not found: {eval_dir}")
            sys.exit(1)

        records = load_all_results(eval_dir, dataset_dir)
        source_label = str(eval_dir)

        # Filter to minival subset if --minival flag given (no dirs)
        if args.minival is not None:
            before = len(records)
            records = [r for r in records if _is_minival_match(r["scene"], r["camera"])]
            print(f"Minival filter: {before} -> {len(records)} samples "
                  f"({len(MINIVAL_PAIRS)} pairs defined)")
            source_label += " [minival]"

    if not records:
        print("No results found.")
        sys.exit(1)

    # Redirect output if saving to file
    if args.output:
        sys.stdout = open(args.output, "w")

    print(f"Loaded {len(records)} samples from {source_label}")

    # By run name (useful for --minival with multiple dirs)
    if args.minival and len(args.minival) > 1:
        print_table("By Run (Method)", aggregate(records, "run_name"), args.format)

    # By camera type
    print_table("By Camera Type", aggregate(records, "cam_type"), args.format)

    # By town
    print_table("By Town", aggregate(records, "town"), args.format)

    # By weather
    print_table("By Weather", aggregate(records, "weather"), args.format)

    # Per scene
    if args.format == "pretty":
        print_per_scene(records, args.format)

    # Best / worst
    if args.format == "pretty":
        print_best_worst(records, args.top_n)

    if args.output:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
