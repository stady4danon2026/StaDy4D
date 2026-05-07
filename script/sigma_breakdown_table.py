#!/usr/bin/env python3
"""SIGMA performance breakdown: overall + per-camera-type + per-weather + per-town."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np


METRICS = [
    ("pose", "RRA@30", True),
    ("pose", "RTA@30", True),
    ("pose", "AUC", True),
    ("depth", "abs_rel", False),
    ("depth", "a1", True),
    ("pointcloud", "Acc", False),
    ("pointcloud", "Comp", False),
    ("pointcloud", "NC", True),
]


def load_sigma():
    """Load SIGMA metrics with patched scenes overriding original."""
    out = {}
    for f in Path("outputs/sam3_full_run/_metrics").glob("*.json"):
        out[f.stem] = json.loads(f.read_text())
    for d in ["outputs/sigma_top20_worst_nc/_metrics",
              "outputs/sigma_top20_worst_nc_v2/_metrics",
              "outputs/sigma_alignment_failures/_metrics"]:
        for f in Path(d).glob("*.json"):
            out[f.stem] = json.loads(f.read_text())
    return out


def collect_value(m, grp, k):
    if grp not in m or k not in m[grp]: return None
    v = m[grp][k]
    if v is None or not np.isfinite(v): return None
    return v


def aggregate(records, group_key):
    """records: list of (group_value, metric_dict). Returns DataFrame."""
    by_group = {}
    for gv, m in records:
        by_group.setdefault(gv, []).append(m)
    rows = []
    for gv, ms in sorted(by_group.items()):
        row = {"group": gv, "n": len(ms)}
        for grp, k, _ in METRICS:
            vals = [v for v in (collect_value(m, grp, k) for m in ms) if v is not None]
            row[k] = np.mean(vals) if vals else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sigma = load_sigma()
    print(f"Loaded {len(sigma)} SIGMA scenes")

    # Build (scene, cam) -> metadata (town, weather). Filter to short/test only.
    idx_full = pd.read_parquet("StaDy4D/index.parquet")
    idx_full = idx_full[idx_full["path"].str.startswith("short/test/")]
    idx = idx_full.drop_duplicates("scene_name").set_index("scene_name")

    records = []
    for stem, m in sigma.items():
        scene, cam = stem.split("__", 1)
        if scene not in idx.index: continue
        meta = idx.loc[scene]
        # camera type: extract everything after cam_NN_ → "car_forward", etc.
        parts = cam.split("_", 2)
        cam_type = parts[2] if len(parts) > 2 else cam
        records.append((scene, cam, cam_type, meta["town"], meta["weather"], m))

    print(f"\nTotal: {len(records)}")

    # Overall
    overall_metrics = {}
    for grp, k, _ in METRICS:
        vals = [v for v in (collect_value(r[5], grp, k) for r in records) if v is not None]
        overall_metrics[k] = np.mean(vals) if vals else float("nan")
    print("\n=== OVERALL ===")
    print("  N =", len(records))
    for grp, k, hi in METRICS:
        arrow = "↑" if hi else "↓"
        print(f"  {k:8}{arrow}  {overall_metrics[k]:.4f}")

    def fmt_table(df, title):
        print(f"\n=== {title} ===")
        cols = ["group", "n"] + [k for _,k,_ in METRICS]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Per camera type
    df_cam = aggregate([(r[2], r[5]) for r in records], "cam_type")
    fmt_table(df_cam, "PER CAMERA TYPE")

    # Per weather
    df_w = aggregate([(r[4], r[5]) for r in records], "weather")
    fmt_table(df_w, "PER WEATHER")

    # Per town/CityMap
    df_t = aggregate([(r[3], r[5]) for r in records], "town")
    fmt_table(df_t, "PER CITYMAP (TOWN)")


if __name__ == "__main__":
    main()
