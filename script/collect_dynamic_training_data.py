#!/usr/bin/env python3
"""Collect (RGB, GSAM mask) training pairs from Town07+Town10 (held-out from Town03 eval).

Runs only the motion stage. Saves masks (RGB is re-decoded from mp4 at train time).

Usage:
    python script/collect_dynamic_training_data.py \
        --towns T07 T10 \
        --out-dir data/dynamic_train \
        --max-frames 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import build_pipeline  # noqa: E402


def discover_pairs(dataset_root: Path, towns: list[str]) -> list[tuple[str, str]]:
    base = dataset_root / "short" / "test"
    pairs = []
    for scene_dir in sorted(base.iterdir()):
        if not scene_dir.is_dir():
            continue
        if not any(t in scene_dir.name for t in towns):
            continue
        dyn = scene_dir / "dynamic"
        if not dyn.is_dir():
            continue
        for cam_dir in sorted(dyn.iterdir()):
            if cam_dir.name.startswith(("cam_", "camera_")) and (cam_dir / "rgb.mp4").exists():
                pairs.append((scene_dir.name, cam_dir.name))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--towns", nargs="+", default=["T07", "T10"])
    p.add_argument("--out-dir", type=Path, default=Path("data/dynamic_train"))
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of (scene, camera) pairs (for quick iteration).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.dataset_root, args.towns)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Found {len(pairs)} pairs in towns={args.towns}")

    pipeline_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        duration="short",
        split="test",
        data_format="dynamic",
        reconstruction="pi3",  # required by config; we won't run it
        inpainting="blank",
    )
    extras = [f"data.max_frames={args.max_frames}"]
    print("Loading motion stage (one-time)...")
    runner, cfg = build_pipeline(pipeline_args, extras)
    print("Loaded.\n")

    from omegaconf import OmegaConf
    from sigma.data import FrameDataModule, FrameSequenceConfig
    from sigma.data.frame_record import FrameIORecord

    t0 = time.monotonic()
    skipped = 0
    failed = 0
    for i, (scene, cam) in enumerate(pairs):
        out = args.out_dir / f"{scene}__{cam}.npz"
        if out.exists():
            skipped += 1
            continue
        try:
            OmegaConf.update(cfg, "data.scene", scene)
            OmegaConf.update(cfg, "data.camera", cam)
            frames_dir = f"{cfg.data.root_dir}/{cfg.data.duration}/{cfg.data.split}/{scene}"
            OmegaConf.update(cfg, "data.frames_dir", frames_dir)

            dm_cfg = FrameSequenceConfig(
                frames_dir=Path(frames_dir),
                frame_stride=cfg.data.frame_stride,
                max_frames=cfg.data.max_frames,
                data_format=cfg.data.get("data_format", "dynamic"),
                camera=cam,
                load_metadata=cfg.data.get("load_metadata", True),
            )
            dm = FrameDataModule(dm_cfg)
            dm.setup()
            frames = dm.load_all_frames()

            frame_records: dict = {}
            for j, f in enumerate(frames):
                frame_records[j] = FrameIORecord(frame_idx=j, origin_image=f)

            motion_results = runner.motion_stage.process_batch(frame_records)
            masks = []
            for j in sorted(motion_results.keys()):
                m = motion_results[j].data["motion"].get("curr_mask")
                if m is None:
                    m = np.zeros(frames[0].shape[:2], dtype=np.uint8)
                masks.append(m.astype(np.uint8))

            np.savez_compressed(out, masks=np.stack(masks))
            dm.teardown()

            if (i + 1) % 5 == 0 or i == len(pairs) - 1:
                elapsed = time.monotonic() - t0
                done = i + 1 - skipped
                rate = elapsed / max(done, 1)
                remain = (len(pairs) - i - 1) * rate
                print(f"[{i+1}/{len(pairs)}]  {scene}/{cam}   "
                      f"({elapsed:>5.0f}s elapsed, ETA {remain:>5.0f}s, "
                      f"{done} done / {skipped} skip / {failed} fail)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(pairs)}]  {scene}/{cam}   FAILED: {e}", flush=True)
            failed += 1

    runner.motion_stage.teardown()
    print(f"\nDone: {time.monotonic()-t0:.0f}s total, "
          f"{len(list(args.out_dir.glob('*.npz')))} files in {args.out_dir}")


if __name__ == "__main__":
    main()
