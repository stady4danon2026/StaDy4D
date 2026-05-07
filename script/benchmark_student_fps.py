#!/usr/bin/env python3
"""Benchmark the student head: raw forward, batched throughput, end-to-end inference.

Usage:
    python script/benchmark_student_fps.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from sigma.pipeline.motion.student import StudentMotionEstimator
from sigma.pipeline.motion.student_unet import StudentUNet


def bench(name, fn, runs=10, warmup=3):
    """Run fn() multiple times after warmup, report mean / median / min / max."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t = time.monotonic()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.monotonic() - t)
    return name, times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/student_unet.pt")
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--orig-size", type=int, nargs=2, default=[480, 640],
                   help="Origin frame size (H, W) used in the StaDy4D pipeline.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    H, W = args.orig_size

    # ----- 1. Raw model forward -----
    model = StudentUNet().to(device).eval()
    if Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Note: checkpoint {args.checkpoint} not found; using random init for timing only")

    print(f"Device: {device}")
    print(f"Origin frame size: {H}x{W}")
    print(f"Student input size: {args.img_size}x{args.img_size}")
    print()

    @torch.no_grad()
    def raw_fwd(bs):
        x = torch.randn(bs, 3, args.img_size, args.img_size, device=device)
        return lambda: model(x)

    @torch.no_grad()
    def amp_fwd(bs):
        x = torch.randn(bs, 3, args.img_size, args.img_size, device=device)
        def fn():
            with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                     dtype=torch.float16):
                return model(x)
        return fn

    print(f"=== Raw forward (fp32) ===")
    for bs in [1, 4, 8, 16, 32]:
        try:
            _, ts = bench(f"bs={bs}", raw_fwd(bs), runs=args.runs)
            t_mean = np.mean(ts)
            print(f"  bs={bs:>3}: {t_mean*1000:>7.2f} ms total  =  {t_mean*1000/bs:>6.2f} ms/img  =  {bs/t_mean:>7.1f} FPS")
        except RuntimeError as e:
            print(f"  bs={bs}: OOM ({e})")

    print()
    print(f"=== AMP fp16 forward ===")
    for bs in [1, 4, 8, 16, 32]:
        try:
            _, ts = bench(f"bs={bs}", amp_fwd(bs), runs=args.runs)
            t_mean = np.mean(ts)
            print(f"  bs={bs:>3}: {t_mean*1000:>7.2f} ms total  =  {t_mean*1000/bs:>6.2f} ms/img  =  {bs/t_mean:>7.1f} FPS")
        except RuntimeError as e:
            print(f"  bs={bs}: OOM ({e})")

    # ----- 2. End-to-end via StudentMotionEstimator -----
    print()
    print(f"=== End-to-end (StudentMotionEstimator.process_batch, includes resize+filter) ===")
    if Path(args.checkpoint).exists():
        from sigma.data.frame_record import FrameIORecord
        # Fake N frames at origin resolution
        for n_frames in [1, 5, 10, 30, 60]:
            est = StudentMotionEstimator(checkpoint=args.checkpoint, device=device,
                                          img_size=args.img_size)
            est.setup()
            frame_records = {
                i: FrameIORecord(frame_idx=i,
                                  origin_image=np.random.randint(0, 256, (H, W, 3), dtype=np.uint8))
                for i in range(n_frames)
            }
            # Warmup
            for _ in range(2):
                est.process_batch(frame_records)
            torch.cuda.synchronize() if device.startswith("cuda") else None
            # Bench
            ts = []
            for _ in range(args.runs):
                t = time.monotonic()
                est.process_batch(frame_records)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                ts.append(time.monotonic() - t)
            t_mean = np.mean(ts)
            print(f"  N={n_frames:>3} frames: {t_mean*1000:>7.2f} ms total  "
                  f"=  {t_mean*1000/n_frames:>6.2f} ms/frame  =  {n_frames/t_mean:>7.1f} FPS")

    # ----- 3. Compare to GroundedSAM and SDXL costs -----
    print()
    print(f"=== Reference: stages from production pipeline (measured earlier) ===")
    print(f"  GroundedSAM (DINO+SAM batch):  ~3,900 ms / 30-frame scene  =  ~130 ms/frame  =  ~7.7 FPS")
    print(f"  SDXL inpainting:               ~100 sec / 30-frame scene  =  ~3,300 ms/frame  =  ~0.3 FPS")
    print(f"  Pi3X reconstruction (offline): ~1,800 ms / 30-frame scene  =  ~60 ms/frame  =  ~17 FPS")

    # Param count
    params = sum(p.numel() for p in model.parameters())
    print()
    print(f"Student head: {params/1e6:.2f}M params, {params * 4 / 1e6:.1f} MB fp32")


if __name__ == "__main__":
    main()
