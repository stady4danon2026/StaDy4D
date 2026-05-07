#!/usr/bin/env python3
"""Time a single Pi3X forward pass on N frames (model already loaded).

Usage:
    python script/time_pi3x_forward.py --frames 30 --warmup 2 --runs 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from safetensors.numpy import load_file


def _decode_video(p: Path) -> np.ndarray:
    r = imageio.get_reader(str(p))
    out = [np.asarray(f) for f in r]
    r.close()
    return np.stack(out, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="StaDy4D/short/test/scene_T03_008/dynamic/cam_00_car_forward")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--pixel_limit", type=int, default=255000)
    args = p.parse_args()

    print(f"Loading Pi3X from {args.checkpoint} ...", flush=True)
    t = time.monotonic()
    from pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained(args.checkpoint).to("cuda").eval()
    print(f"  model load: {time.monotonic()-t:.1f}s", flush=True)

    print(f"Loading {args.frames} frames from {args.scene} ...", flush=True)
    rgb = _decode_video(Path(args.scene) / "rgb.mp4")[: args.frames]
    H, W = rgb.shape[1:3]

    # Pi3-compatible target size (multiples of 14 within pixel_limit)
    import math
    scale = math.sqrt(args.pixel_limit / max(H * W, 1))
    Wf, Hf = W * scale, H * scale
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > args.pixel_limit:
        if k / m > Wf / Hf:
            k -= 1
        else:
            m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    print(f"  H,W = {H}x{W}  -> Pi3 input {TH}x{TW}", flush=True)

    from PIL import Image
    from torchvision import transforms
    to_t = transforms.ToTensor()
    tlist = []
    for i in range(args.frames):
        pil = Image.fromarray(rgb[i]).resize((TW, TH), Image.Resampling.LANCZOS)
        tlist.append(to_t(pil))
    imgs = torch.stack(tlist, dim=0).unsqueeze(0).to("cuda")  # (1, N, 3, H, W)
    print(f"  tensor: {tuple(imgs.shape)}  dtype={imgs.dtype}", flush=True)

    cap = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    print(f"  amp dtype: {dtype}", flush=True)

    def fwd():
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                _ = model(imgs)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(args.warmup):
        fwd()

    times = []
    for _ in range(args.runs):
        t = time.monotonic()
        fwd()
        times.append(time.monotonic() - t)

    print()
    print(f"=== Pi3X forward pass on {args.frames} frames, {args.runs} runs ===")
    print(f"  mean   : {np.mean(times):.3f} s")
    print(f"  median : {np.median(times):.3f} s")
    print(f"  min/max: {min(times):.3f} / {max(times):.3f} s")
    print(f"  per-frame: {np.mean(times)/args.frames*1000:.1f} ms/frame")


if __name__ == "__main__":
    main()
