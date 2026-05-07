#!/usr/bin/env python3
"""Collect (origin RGB, static depth GT, K, c2w) for Pi3X LoRA fine-tuning.

Source: StaDy4D train split (Towns 01/02/04/05/06).
Output: per-(scene, camera) npz with frame stack ready for training.

The static depth in StaDy4D is depth-of-the-clean-scene (no actors). When
Pi3X is supervised on (origin RGB → static depth), it learns to predict
non-dynamic depth even with cars in the input — the desired behaviour.

Usage:
    python script/collect_lora_train_data.py --out data/lora_train --max-frames 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from safetensors.numpy import load_file


def discover(root: Path, towns: list[str]) -> list[tuple[str, str, Path, Path]]:
    pairs = []
    for scene_dir in sorted(root.iterdir()):
        if not any(t in scene_dir.name for t in towns):
            continue
        dyn = scene_dir / "dynamic"
        sta = scene_dir / "static"
        if not dyn.is_dir() or not sta.is_dir():
            continue
        for cam_dyn in sorted(dyn.iterdir()):
            if not cam_dyn.is_dir() or not cam_dyn.name.startswith(("cam_", "camera_")):
                continue
            cam_sta = sta / cam_dyn.name
            if not cam_sta.is_dir():
                continue
            mp4 = cam_dyn / "rgb.mp4"
            depth_st = cam_sta / "depth.safetensors"
            extr_st = cam_sta / "extrinsics.safetensors"
            intr_st = cam_sta / "intrinsics.safetensors"
            if mp4.exists() and depth_st.exists() and extr_st.exists() and intr_st.exists():
                pairs.append((scene_dir.name, cam_dyn.name, cam_dyn, cam_sta))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/train"))
    p.add_argument("--towns", nargs="+", default=["T01", "T02", "T04", "T05", "T06"])
    p.add_argument("--out", type=Path, default=Path("data/lora_train"))
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--img-size", type=int, default=434, help="resize longer edge to this")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pairs = discover(args.root, args.towns)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Found {len(pairs)} (scene, camera) pairs across towns={args.towns}")

    # Pi3-compatible target dimensions.
    import math
    def _pi3_size(H: int, W: int, pixel_limit: int = 255000) -> tuple[int, int]:
        scale = math.sqrt(pixel_limit / max(H * W, 1))
        Wf, Hf = W * scale, H * scale
        k, m = round(Wf / 14), round(Hf / 14)
        while (k * 14) * (m * 14) > pixel_limit:
            if k / m > Wf / Hf:
                k -= 1
            else:
                m -= 1
        return max(1, k) * 14, max(1, m) * 14

    t0 = time.monotonic()
    saved, skipped, failed = 0, 0, 0
    for i, (scene, cam, cam_dyn_dir, cam_sta_dir) in enumerate(pairs):
        out = args.out / f"{scene}__{cam}.npz"
        if out.exists():
            skipped += 1
            continue
        try:
            depth_all = load_file(str(cam_sta_dir / "depth.safetensors"))["depth"].astype(np.float32)
            c2w_all = load_file(str(cam_sta_dir / "extrinsics.safetensors"))["c2w"].astype(np.float32)
            K_all = load_file(str(cam_sta_dir / "intrinsics.safetensors"))["K"].astype(np.float32)
            reader = imageio.get_reader(str(cam_dyn_dir / "rgb.mp4"))
            try:
                rgb_full = np.stack([np.asarray(reader.get_data(j))
                                      for j in range(min(args.max_frames, len(depth_all)))])
            finally:
                reader.close()
            n = min(rgb_full.shape[0], depth_all.shape[0], c2w_all.shape[0], K_all.shape[0],
                    args.max_frames)
            rgb_full = rgb_full[:n]
            depth_all = depth_all[:n]
            c2w_all = c2w_all[:n]
            K_all = K_all[:n]

            # Resize RGB + depth to the Pi3-compatible size used at inference.
            H, W = rgb_full.shape[1:3]
            TW, TH = _pi3_size(H, W)
            import cv2
            rgb_resized = np.zeros((n, TH, TW, 3), dtype=np.uint8)
            depth_resized = np.zeros((n, TH, TW), dtype=np.float32)
            for j in range(n):
                rgb_resized[j] = cv2.resize(rgb_full[j], (TW, TH), interpolation=cv2.INTER_AREA)
                depth_resized[j] = cv2.resize(depth_all[j], (TW, TH), interpolation=cv2.INTER_NEAREST)
            # Scale intrinsics
            K_resized = K_all.copy()
            K_resized[:, 0, 0] *= TW / W
            K_resized[:, 0, 2] *= TW / W
            K_resized[:, 1, 1] *= TH / H
            K_resized[:, 1, 2] *= TH / H

            np.savez_compressed(
                out,
                rgb=rgb_resized,
                depth=depth_resized,
                c2w=c2w_all,
                K=K_resized,
                size=np.array([TH, TW], dtype=np.int32),
                orig_size=np.array([H, W], dtype=np.int32),
            )
            saved += 1
            if (i + 1) % 25 == 0 or i == len(pairs) - 1:
                elapsed = time.monotonic() - t0
                rate = elapsed / max(saved, 1)
                eta = (len(pairs) - i - 1) * rate
                print(f"[{i+1}/{len(pairs)}] {scene}/{cam}  saved {saved} skip {skipped} "
                      f"fail {failed}  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(pairs)}] {scene}/{cam}  FAILED: {e}", flush=True)

    total_size = sum(f.stat().st_size for f in args.out.glob("*.npz")) / 1e9
    print(f"\nDone: {time.monotonic()-t0:.0f}s, {saved} saved / {skipped} skip / {failed} fail. "
          f"Disk: {total_size:.1f} GB")


if __name__ == "__main__":
    main()
