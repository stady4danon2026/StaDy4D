#!/usr/bin/env python3
"""Pre-decode mp4s + save resized RGB into the existing mask npz files.

After this, training reads (rgb, mask) from a single .npz, no mp4 decode at
sample time. Resize RGB to a fixed working size to keep total disk small.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/dynamic_train"))
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--img-size", type=int, default=320,
                   help="Resize longer edge to this and pad to square.")
    args = p.parse_args()

    files = sorted(args.data_dir.glob("*.npz"))
    print(f"Pre-decoding {len(files)} sequences to {args.img_size}x{args.img_size}...")
    t0 = time.monotonic()

    for i, f in enumerate(files):
        data = np.load(f)
        if "rgb" in data.files:
            continue
        masks = data["masks"]                                   # (N, H, W) uint8
        N, H, W = masks.shape
        # Decode mp4
        scene, cam = f.stem.split("__", 1)
        video = args.dataset_root / "short" / "test" / scene / "dynamic" / cam / "rgb.mp4"
        if not video.exists():
            print(f"  {f.stem}: video missing, skipping")
            continue
        reader = imageio.get_reader(str(video))
        try:
            rgb = np.stack([np.asarray(reader.get_data(j)) for j in range(N)])
        finally:
            reader.close()

        # Resize keeping aspect, pad to square at img_size
        S = args.img_size
        scale = S / max(H, W)
        nh, nw = int(H * scale), int(W * scale)
        rgb_small = np.zeros((N, S, S, 3), dtype=np.uint8)
        mask_small = np.zeros((N, S, S), dtype=np.uint8)
        for j in range(N):
            rgb_small[j, :nh, :nw] = cv2.resize(rgb[j], (nw, nh), interpolation=cv2.INTER_AREA)
            mask_small[j, :nh, :nw] = cv2.resize(masks[j], (nw, nh),
                                                  interpolation=cv2.INTER_NEAREST)

        np.savez_compressed(
            f, masks=masks, rgb=rgb_small, mask_resized=mask_small,
            orig_size=np.array([H, W], dtype=np.int32),
            pad_size=np.array([nh, nw], dtype=np.int32),
        )
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            elapsed = time.monotonic() - t0
            remain = (len(files) - i - 1) * elapsed / (i + 1)
            print(f"[{i+1}/{len(files)}] {elapsed:.0f}s elapsed, ETA {remain:.0f}s",
                  flush=True)

    print(f"\nDone: {time.monotonic()-t0:.0f}s.  "
          f"Disk usage: {sum(f.stat().st_size for f in files)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
