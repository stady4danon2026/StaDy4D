#!/usr/bin/env python3
"""Aggregate per-(scene,camera) JSONs across baseline methods into one table.

Reads from eval_results_ablation_<method>/short/test/<scene>/<camera>.json,
emits a Markdown table averaging over all (scene, camera) pairs in towns
T03 + T07 + T10. Target column order matches user spec:
  RRA@30 ↑  RTA@30 ↑  AUC ↑  Abs Rel ↓  δ1.25 ↑  Acc ↓  Comp ↓  N.C. ↑
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


METRIC_ORDER = [
    ("RRA@30", "↑", ".1f"),
    ("RTA@30", "↑", ".1f"),
    ("AUC",    "↑", ".1f"),
    ("abs_rel","↓", ".4f"),
    ("a1",     "↑", ".4f"),
    ("Acc",    "↓", ".4f"),
    ("Comp",   "↓", ".4f"),
    ("NC",     "↑", ".4f"),
]
DISPLAY = {"abs_rel": "Abs Rel", "a1": "δ1.25", "NC": "N.C."}
TOWNS = ("T03", "T07", "T10")


def load_method(eval_dir: Path) -> List[Dict]:
    """Return list of per-(scene,camera) flat dicts for towns in TOWNS."""
    rows: List[Dict] = []
    split_dir = eval_dir / "short" / "test"
    if not split_dir.is_dir():
        return rows
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        if not any(t in scene_dir.name for t in TOWNS):
            continue
        for cam_json in sorted(scene_dir.glob("cam_*.json")):
            try:
                with open(cam_json) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
    return rows


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, _arrow, _fmt in METRIC_ORDER:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["_n"] = len(rows)
    return out


def per_scene_breakdown(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Average across cameras within each scene, then group."""
    by_scene: Dict[str, List[Dict]] = {}
    for r in rows:
        by_scene.setdefault(r.get("scene", "?"), []).append(r)
    out = {}
    for scene, recs in by_scene.items():
        out[scene] = aggregate(recs)
    return out


def _town(r: Dict) -> str:
    s = r.get("scene", "")
    for t in TOWNS:
        if t in s:
            return t
    return "?"


def _camera_type(r: Dict) -> str:
    """Strip the leading cam_NN_ prefix to get the camera type label."""
    c = r.get("camera", "")
    parts = c.split("_", 2)
    return parts[2] if len(parts) >= 3 else c


def per_town_breakdown(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(_town(r), []).append(r)
    return {k: aggregate(v) for k, v in by.items()}


def per_camera_breakdown(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(_camera_type(r), []).append(r)
    return {k: aggregate(v) for k, v in by.items()}


def fmt(v: float, spec: str) -> str:
    if not np.isfinite(v):
        return "  n/a"
    return f"{v:{spec}}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("<REPO_ROOT>"),
                   help="Project root containing eval_results_ablation_<method>/")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Method names (eval_results_ablation_<NAME> dirs); auto-discover if omitted")
    p.add_argument("--out", type=Path, default=None,
                   help="Output Markdown path (default stdout)")
    p.add_argument("--per-scene", action="store_true",
                   help="Also dump per-scene breakdown")
    p.add_argument("--per-town", action="store_true",
                   help="Also dump per-town breakdown (T03/T07/T10)")
    p.add_argument("--per-camera", action="store_true",
                   help="Also dump per-camera-type breakdown (car_forward / drone / cctv / ...)")
    args = p.parse_args()

    if args.methods is None:
        args.methods = sorted(
            d.name.removeprefix("eval_results_ablation_")
            for d in args.root.iterdir()
            if d.is_dir() and d.name.startswith("eval_results_ablation_")
        )

    lines = []
    head = "| Method | N | " + " | ".join(
        f"{DISPLAY.get(k, k)} {a}" for k, a, _ in METRIC_ORDER
    ) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(METRIC_ORDER))) + "|"
    lines.append(f"### Benchmark on StaDy4D short/test (towns {', '.join(TOWNS)})")
    lines.append("")
    lines.append(head)
    lines.append(sep)

    summaries = {}
    for method in args.methods:
        eval_dir = args.root / f"eval_results_ablation_{method}"
        rows = load_method(eval_dir)
        agg = aggregate(rows)
        summaries[method] = (rows, agg)
        cells = [method, str(agg["_n"])]
        cells += [fmt(agg[k], spec) for k, _a, spec in METRIC_ORDER]
        lines.append("| " + " | ".join(cells) + " |")

    def _emit_breakdown(title: str, fn, label: str):
        lines.append("")
        lines.append(f"### {title}")
        for method, (rows, _agg) in summaries.items():
            grouped = fn(rows)
            lines.append("")
            lines.append(f"#### {method}")
            lines.append(f"| {label} | N | " + " | ".join(
                f"{DISPLAY.get(k, k)} {a}" for k, a, _ in METRIC_ORDER
            ) + " |")
            lines.append(sep)
            for k_ in sorted(grouped):
                a = grouped[k_]
                cells = [k_, str(a["_n"])]
                cells += [fmt(a[k], spec) for k, _a, spec in METRIC_ORDER]
                lines.append("| " + " | ".join(cells) + " |")

    if args.per_town:
        _emit_breakdown("Per-town breakdown", per_town_breakdown, "Town")
    if args.per_camera:
        _emit_breakdown("Per-camera-type breakdown", per_camera_breakdown, "Camera type")
    if args.per_scene:
        _emit_breakdown("Per-scene breakdown", per_scene_breakdown, "Scene")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
