#!/usr/bin/env python3
"""Build TUM-dynamics training data for cross-domain LoRA fine-tuning.

Loads RGB+depth pairs from a TUM sequence, resizes to a Pi3-compatible
grid, and saves as a single npz with the same schema as the StaDy4D LoRA
data (rgb, depth, K, c2w, size, orig_size).

Output is a single npz per sequence, like data/lora_train/*, so the
existing LoRAStaDy4DDataset reads them with no changes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def pi3_size(H: int, W: int, pl: int = 255000) -> tuple[int, int]:
    s = math.sqrt(pl / max(H * W, 1))
    Wf, Hf = W * s, H * s
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > pl:
        if k / m > Wf / Hf: k -= 1
        else: m -= 1
    return max(1, k) * 14, max(1, m) * 14


def parse_list(p: Path) -> list[tuple[float, str]]:
    """Returns [(timestamp, rest_of_line)]."""
    rows = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), parts[1]))
        except ValueError:
            continue
    return rows


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/realworld/tum_dynamics", type=Path)
    p.add_argument("--out", default="data/lora_train", type=Path,
                   help="Append to the same training-data directory.")
    p.add_argument("--max-frames-per-seq", type=int, default=80)
    p.add_argument("--stride", type=int, default=2)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    seqs = sorted([p for p in args.root.iterdir() if p.is_dir() and "rgbd_dataset" in p.name])
    print(f"Found {len(seqs)} TUM sequences")

    # TUM fr3 intrinsics (Kinect color)
    K_orig = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]], dtype=np.float32)

    for seq_dir in seqs:
        rgb_rows = parse_list(seq_dir / "rgb.txt")
        depth_rows = parse_list(seq_dir / "depth.txt")
        gt_rows = parse_list(seq_dir / "groundtruth.txt") if (seq_dir / "groundtruth.txt").exists() else []
        depth_t = np.array([r[0] for r in depth_rows])
        gt_t = np.array([r[0] for r in gt_rows]) if gt_rows else None

        rgb_rows = rgb_rows[::args.stride][: args.max_frames_per_seq]
        if not rgb_rows:
            continue

        rgb_buf, depth_buf, c2w_buf = [], [], []
        # Get target Pi3 size from first frame
        first = imageio.imread(seq_dir / rgb_rows[0][1])
        H, W = first.shape[:2]
        TW, TH = pi3_size(H, W)

        K_resized = K_orig.copy()
        K_resized[0, 0] *= TW / W; K_resized[0, 2] *= TW / W
        K_resized[1, 1] *= TH / H; K_resized[1, 2] *= TH / H
        K_arr = np.tile(K_resized, (len(rgb_rows), 1, 1)).astype(np.float32)

        for t_rgb, rgb_path in rgb_rows:
            rgb = imageio.imread(seq_dir / rgb_path)
            d_idx = int(np.argmin(np.abs(depth_t - t_rgb)))
            d_raw = imageio.imread(seq_dir / depth_rows[d_idx][1])
            d = d_raw.astype(np.float32) / 5000.0
            d[d_raw == 0] = 0
            rgb_r = cv2.resize(rgb, (TW, TH), interpolation=cv2.INTER_AREA)
            d_r = cv2.resize(d, (TW, TH), interpolation=cv2.INTER_NEAREST)
            rgb_buf.append(rgb_r)
            depth_buf.append(d_r)
            if gt_t is not None:
                g_idx = int(np.argmin(np.abs(gt_t - t_rgb)))
                tx, ty, tz, qx, qy, qz, qw = map(float, gt_rows[g_idx][1].split()[:7])
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = quat_to_R(qx, qy, qz, qw)
                c2w[:3, 3] = [tx, ty, tz]
                c2w_buf.append(c2w)
            else:
                c2w_buf.append(np.eye(4, dtype=np.float32))

        rgb_arr = np.stack(rgb_buf, axis=0).astype(np.uint8)
        depth_arr = np.stack(depth_buf, axis=0).astype(np.float32)
        c2w_arr = np.stack(c2w_buf, axis=0).astype(np.float32)

        out_path = args.out / f"TUM__{seq_dir.name}.npz"
        np.savez_compressed(
            out_path, rgb=rgb_arr, depth=depth_arr, c2w=c2w_arr, K=K_arr,
            size=np.array([TH, TW], dtype=np.int32),
            orig_size=np.array([H, W], dtype=np.int32),
        )
        print(f"  saved {out_path}: {rgb_arr.shape[0]} frames @ {TH}x{TW}")

    print("\nDone.")


if __name__ == "__main__":
    main()
