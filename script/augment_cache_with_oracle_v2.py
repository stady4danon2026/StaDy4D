#!/usr/bin/env python3
"""Add clean GT (depth+RGB diff) to the cache as bundle["gt_clean"]."""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger("aug2")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=Path("<DATA_ROOT>/sigma_cache/pi3_dyn_pool"))
    p.add_argument("--oracle-v2", type=Path, default=Path("<DATA_ROOT>/sigma_outputs/dyn_oracle_v2"))
    args = p.parse_args()

    bundles = sorted(args.cache_dir.glob("*.pt"))
    LOGGER.info("Augmenting %d bundles with oracle_v2 (clean) labels", len(bundles))
    t0 = time.time()
    for k, bp in enumerate(bundles):
        b = torch.load(bp, map_location="cpu")
        if "gt_clean" in b:
            continue
        name = bp.stem
        parts = name.split("__")
        scene = parts[0]
        cam = "__".join(parts[1:])
        mask_dir = args.oracle_v2 / scene / cam / "mask"
        if not mask_dir.is_dir():
            continue
        masks = []
        files = sorted(mask_dir.glob("mask_*.png"))
        for f in files:
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                masks.append(None); continue
            masks.append((m > 127).astype(np.uint8))
        if not masks or any(m is None for m in masks):
            continue
        gt = torch.from_numpy(np.stack(masks)).float().unsqueeze(1)   # (N, 1, H_orig, W_orig)
        TH, TW = b["labels"].shape[-2:]
        gt_resized = F.interpolate(gt, size=(TH, TW), mode="nearest").squeeze(1)
        b["gt_clean"] = (gt_resized > 0.5).to(torch.uint8)
        torch.save(b, bp)
        if (k + 1) % 30 == 0 or (k + 1) == len(bundles):
            LOGGER.info("[%d/%d] %s pos=%.2f%%  cum=%.1fs",
                        k + 1, len(bundles), name, b["gt_clean"].float().mean() * 100, time.time() - t0)
    LOGGER.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
