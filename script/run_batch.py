#!/usr/bin/env python3
"""Batch mode: run the SIGMA pipeline on every (scene, camera) pair found under
a StaDy4D root directory.

Directory layout expected (new format)
--------------------------------------
<dataset_root>/<duration>/<split>/
  scene_T03_008/
    dynamic/
      cam_00_car_forward/rgb.mp4, depth.safetensors, ...
      cam_01_car_forward/rgb.mp4, ...
    metadata.json
  scene_T03_009/
    ...

One pipeline run is spawned per (scene, camera) pair discovered.

Usage examples
--------------
# Run all scenes in short/test:
python script/run_batch.py --duration short --split test

# Run a specific scene:
python script/run_batch.py --duration short --split test --scene scene_T03_008

# Limit to a specific camera name pattern:
python script/run_batch.py --duration short --split test --camera cam_00

# 4 parallel workers (safe only with CPU or multi-GPU):
python script/run_batch.py --duration short --split test --workers 4

# Pass extra Hydra overrides after '--':
python script/run_batch.py --duration short --split test -- data.max_frames=30 device=cpu

# Legacy: pass a full path directly (still supported):
python script/run_batch.py StaDy4D/short/test/scene_T03_008
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from threading import Semaphore, Thread
from typing import List, Tuple

SIGMA_MAIN = Path(__file__).resolve().parent.parent / "sigma" / "main.py"


# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------

def discover_jobs(
    split_dir: Path,
    scene_filter: str | None,
    data_format: str,
    camera_filter: str | None,
) -> List[Tuple[str, str]]:
    """Return a sorted list of (scene, camera) pairs to process.

    Scans ``<split_dir>/<scene_pattern>/<data_format>/(cam_*|camera_*)/``
    to find all available camera sub-directories.
    """
    jobs: List[Tuple[str, str]] = []
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_filter and scene_filter not in scene_dir.name:
            continue
        fmt_dir = scene_dir / data_format
        if not fmt_dir.is_dir():
            continue
        for cam_dir in sorted(fmt_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            if not (cam_dir.name.startswith("cam_") or cam_dir.name.startswith("camera_")):
                continue
            # New format: check for rgb.mp4; legacy: check for rgb/ subdir
            if not (cam_dir / "rgb.mp4").exists() and not (cam_dir / "rgb").is_dir():
                continue
            if camera_filter and camera_filter not in cam_dir.name:
                continue
            jobs.append((scene_dir.name, cam_dir.name))
    return jobs


# ---------------------------------------------------------------------------
# Single-job runner
# ---------------------------------------------------------------------------

def _run_one(
    scene: str,
    camera: str,
    dataset_root: str,
    duration: str,
    split: str,
    reconstruction: str,
    inpainting: str,
    data_format: str,
    extra_overrides: List[str],
    dry_run: bool = False,
) -> Tuple[str, str, int]:
    """Invoke sigma/main.py for a single (scene, camera) pair.

    Returns:
        (scene, camera, returncode)  -- returncode 0 means success.
    """
    run_name = f"batch_{scene}_{camera}"
    cmd = [
        sys.executable,
        str(SIGMA_MAIN),
        f"data.root_dir={dataset_root}",
        f"data.duration={duration}",
        f"data.split={split}",
        f"data.scene={scene}",
        f"data.camera={camera}",
        f"data.data_format={data_format}",
        f"pipeline/reconstruction={reconstruction}",
        f"pipeline/inpainting={inpainting}",
        f"run.name={run_name}",
    ] + extra_overrides

    label = f"{scene}/{camera}"
    if dry_run:
        print(f"DRY-RUN: {' '.join(cmd)}")
        return scene, camera, 0

    print(f"\n[batch] Starting {label}", flush=True)
    result = subprocess.run(cmd, cwd=SIGMA_MAIN.parent.parent)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[batch] {label}: {status}", flush=True)
    return scene, camera, result.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SIGMA pipeline on every (scene, camera) pair under a StaDy4D directory.",
        epilog="Arguments after '--' are forwarded verbatim to sigma/main.py as Hydra overrides.",
    )
    parser.add_argument(
        "scene_path",
        nargs="?",
        type=Path,
        default=None,
        help="(Legacy) Full path to a scene directory, e.g. StaDy4D/short/test/scene_T03_008.",
    )
    parser.add_argument(
        "--duration",
        default="short",
        help="Dataset duration: 'short' or 'mid' (default: short).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split: 'test', 'train', etc. (default: test).",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Filter to scenes whose name contains this string (default: all scenes).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("StaDy4D"),
        help="Root of the StaDy4D dataset (default: StaDy4D).",
    )
    parser.add_argument(
        "--data-format",
        default="dynamic",
        help="Data format sub-directory inside each scene folder (default: 'dynamic').",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Filter to cameras whose name contains this string (default: all cameras).",
    )
    parser.add_argument(
        "--reconstruction",
        default="pi3",
        choices=["pi3", "pi3_online", "vggt_offline", "vggt_online", "megasam"],
        help="Reconstruction method config name (default: pi3).",
    )
    parser.add_argument(
        "--inpainting",
        default="sdxl_lightning_uncond",
        choices=["sdxl_lightning_uncond", "blank"],
        help="Inpainting method config name (default: sdxl_lightning_uncond).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel pipeline workers (default: 1 = sequential). "
            "WARNING: GPU memory is shared -- use --workers > 1 only with CPU or multi-GPU."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )

    # Split off Hydra overrides that appear after '--'.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        my_args, extra_overrides = argv[:sep], argv[sep + 1:]
    else:
        my_args, extra_overrides = argv, []

    args = parser.parse_args(my_args)

    # Handle legacy positional argument (direct scene path).
    if args.scene_path is not None:
        scene_dir = args.scene_path.resolve()
        if not scene_dir.is_dir():
            parser.error(f"Scene path does not exist: {scene_dir}")
        # Discover cameras in this single scene
        fmt_dir = scene_dir / args.data_format
        if not fmt_dir.is_dir():
            parser.error(f"No {args.data_format}/ directory in {scene_dir}")
        jobs = []
        for cam_dir in sorted(fmt_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            if not (cam_dir.name.startswith("cam_") or cam_dir.name.startswith("camera_")):
                continue
            if args.camera and args.camera not in cam_dir.name:
                continue
            jobs.append((scene_dir.name, cam_dir.name))
        if not jobs:
            parser.error(f"No camera directories found in {fmt_dir}")
        # Infer dataset_root, duration, split from path
        # scene_path = dataset_root/duration/split/scene
        dataset_root = str(scene_dir.parent.parent.parent)
        duration = scene_dir.parent.parent.name
        split = scene_dir.parent.name
        all_jobs = [(dataset_root, duration, split, scene, cam) for scene, cam in jobs]
    else:
        # New-style: --duration + --split
        split_dir = args.dataset_root / args.duration / args.split
        if not split_dir.is_dir():
            parser.error(f"Split directory does not exist: {split_dir}")

        jobs = discover_jobs(split_dir, args.scene, args.data_format, args.camera)
        if not jobs:
            print(
                f"No (scene, camera) pairs found under {split_dir} "
                f"(format='{args.data_format}'"
                + (f", scene filter='{args.scene}'" if args.scene else "")
                + (f", camera filter='{args.camera}'" if args.camera else "")
                + ")"
            )
            sys.exit(1)

        all_jobs = [
            (str(args.dataset_root), args.duration, args.split, scene, cam)
            for scene, cam in jobs
        ]

    print(f"Found {len(all_jobs)} job(s):")
    for _, dur, spl, scene, cam in all_jobs:
        print(f"  {dur}/{spl}/{scene}/{cam}")

    if args.dry_run:
        for ds_root, dur, spl, scene, cam in all_jobs:
            _run_one(
                scene, cam, ds_root, dur, spl,
                args.reconstruction, args.inpainting, args.data_format,
                extra_overrides, dry_run=True,
            )
        return

    t0 = time.monotonic()

    if args.workers <= 1:
        results = [
            _run_one(
                scene, cam, ds_root, dur, spl,
                args.reconstruction, args.inpainting, args.data_format,
                extra_overrides,
            )
            for ds_root, dur, spl, scene, cam in all_jobs
        ]
    else:
        print(f"\nRunning {len(all_jobs)} job(s) with up to {args.workers} parallel workers ...")
        sem = Semaphore(args.workers)
        results: List[Tuple[str, str, int]] = []

        def worker(ds_root: str, dur: str, spl: str, scene: str, cam: str) -> None:
            with sem:
                outcome = _run_one(
                    scene, cam, ds_root, dur, spl,
                    args.reconstruction, args.inpainting, args.data_format,
                    extra_overrides,
                )
            results.append(outcome)

        threads = [
            Thread(target=worker, args=(ds_root, dur, spl, scene, cam))
            for ds_root, dur, spl, scene, cam in all_jobs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    elapsed = time.monotonic() - t0
    successes = [(s, c) for s, c, rc in results if rc == 0]
    failures  = [(s, c, rc) for s, c, rc in results if rc != 0]

    print("\n" + "=" * 60)
    print(f"Batch complete -- {len(results)} job(s) in {elapsed:.1f}s")
    print(f"  Succeeded: {len(successes)}")
    if failures:
        print(f"  Failed:    {len(failures)}")
        for scene, cam, rc in failures:
            print(f"    {scene}/{cam}  (exit {rc})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
