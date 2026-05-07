#!/usr/bin/env python3
"""Add depth-diff GT labels to the existing pool cache bundles.

Loads each cached bundle, computes |dynamic_depth - static_depth| > thr from
the StaDy4D safetensors files, resizes to the bundle's Pi3 input resolution,
and saves back as ``bundle["depthdiff"]`` (uint8 binary mask).

Cheap — file I/O + resize. ~1s per bundle.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("augment_depthdiff")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path,
                   default=Path("<DATA_ROOT>/sigma_cache/pi3_dyn_pool"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--thr", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=900.0)
    args = p.parse_args()

    bundles = sorted(args.cache_dir.glob("*.pt"))
    LOGGER.info("Augmenting %d bundles with depth-diff labels", len(bundles))

    t0 = time.time()
    for k, bp in enumerate(bundles):
        b = torch.load(bp, map_location="cpu")
        if "depthdiff" in b:
            continue

        # Bundle name → scene/cam paths
        name = bp.stem
        parts = name.split("__")
        scene = parts[0]
        cam = "__".join(parts[1:])
        sd_path = args.root / scene / "static" / cam / "depth.safetensors"
        dd_path = args.root / scene / "dynamic" / cam / "depth.safetensors"
        if not sd_path.exists() or not dd_path.exists():
            LOGGER.warning("[%d/%d] %s — missing depth pair, skipped", k + 1, len(bundles), name)
            continue

        d_sta = st.load_file(sd_path)["depth"].float()                  # (N, H, W)
        d_dyn = st.load_file(dd_path)["depth"].float()
        diff = (d_dyn - d_sta).abs()
        valid = (d_sta < args.max_depth) & (d_dyn < args.max_depth)
        gt_orig = ((diff > args.thr) & valid).float().unsqueeze(1)      # (N, 1, H, W)

        # Resize to Pi3 input shape
        TH, TW = b["labels"].shape[-2:]
        gt_resized = F.interpolate(gt_orig, size=(TH, TW), mode="nearest").squeeze(1)
        b["depthdiff"] = (gt_resized > 0.5).to(torch.uint8)

        torch.save(b, bp)
        if (k + 1) % 20 == 0 or (k + 1) == len(bundles):
            LOGGER.info("[%d/%d] %s  pos=%.3f%%  cum=%.1fs",
                        k + 1, len(bundles), name,
                        b["depthdiff"].float().mean() * 100, time.time() - t0)

    LOGGER.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
