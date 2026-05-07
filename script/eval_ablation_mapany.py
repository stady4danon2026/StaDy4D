#!/usr/bin/env python3
"""Ablation: MapAnything on dynamic RGB, evaluated against static GT.

MapAnything's load_images expects file paths on disk, so we dump the
dynamic mp4 frames to a temp PNG dir before inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from eval_ablation_vggt import (
    discover_jobs, load_dynamic_frames, evaluate_sample, extract_flat,
    print_header, print_row, print_summary, METRIC_KEYS,
)


def setup_mapany(checkpoint: str = "facebook/map-anything", device: str = "cuda"):
    import torch
    from mapanything.models import MapAnything
    print(f"Loading MapAnything from {checkpoint}...", flush=True)
    model = MapAnything.from_pretrained(checkpoint).to(torch.device(device))
    model.eval()
    print("MapAnything loaded.", flush=True)
    return model


def run_mapany_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
    apply_mask: bool = True,
    mask_edges: bool = True,
) -> Path:
    import imageio.v2 as imageio
    from mapanything.utils.image import load_images

    timesteps = sorted(frames.keys())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, fidx in enumerate(timesteps):
            p = tmp / f"frame_{i:04d}.png"
            imageio.imwrite(str(p), frames[fidx])
            paths.append(str(p))

        views = load_images(paths)
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=apply_mask,
            mask_edges=mask_edges,
        )

    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, (fidx, pred) in enumerate(zip(timesteps, outputs)):
        depth = pred["depth_z"][0].squeeze(-1).cpu().numpy().astype(np.float32)
        intr  = pred["intrinsics"][0].cpu().numpy().astype(np.float64)
        pose  = pred["camera_poses"][0].cpu().numpy().astype(np.float64)  # 4x4 c2w

        np.save(depth_dir / f"depth_{fidx:04d}.npy", depth)
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", pose)
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", intr)
        imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), frames[fidx])

    return output_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="test")
    p.add_argument("--data-format", default="dynamic")
    p.add_argument("--scene", default=None)
    p.add_argument("--camera", default=None)
    p.add_argument("--checkpoint", default="facebook/map-anything")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--no-save-trajectory", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument("--scratch", type=Path, default=None)
    p.add_argument("--no-mask", action="store_true",
                   help="Disable MapAnything's confidence/edge masking (ablation)")
    args = p.parse_args()

    if args.eval_dir is None:
        args.eval_dir = Path("eval_results_ablation_mapany_nomask" if args.no_mask
                             else "eval_results_ablation_mapany")
    if args.scratch is None:
        args.scratch = Path("<BASELINES_ROOT>/outputs/mapany_nomask" if args.no_mask
                            else "<BASELINES_ROOT>/outputs/mapany")

    split_dir = args.dataset_root / args.duration / args.split
    if not split_dir.is_dir():
        print(f"Split directory not found: {split_dir}"); sys.exit(1)

    jobs = discover_jobs(split_dir, args.scene, args.data_format, args.camera)
    if not jobs:
        print("No (scene, camera) pairs found."); sys.exit(1)

    eval_dir = args.eval_dir / args.duration / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation: MapAnything on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    model = None
    if not args.eval_only:
        model = setup_mapany(checkpoint=args.checkpoint, device=args.device)

    all_flat: List[Dict[str, float]] = []
    all_labels: List[Tuple[str, str]] = []
    print_header()

    t0 = time.monotonic()
    for i, (scene, camera) in enumerate(jobs):
        label = f"{scene}/{camera}"
        gt_dir = split_dir / scene / "static" / camera
        output_dir = args.scratch / f"{scene}_{camera}"

        out_json = eval_dir / scene / f"{camera}.json"
        if out_json.exists() and not args.eval_only:
            try:
                with open(out_json) as f: cached = json.load(f)
                flat = {k: cached[k] for k in METRIC_KEYS if k in cached}
                all_flat.append(flat); all_labels.append((scene, camera))
                print_row(scene, camera, flat); continue
            except Exception: pass

        if args.eval_only:
            pred_dir = output_dir
            if not pred_dir.exists():
                print(f"  [{i+1}/{len(jobs)}] {label}: no output, skipping"); continue
        else:
            try:
                frames = load_dynamic_frames(split_dir, scene, camera, args.data_format,
                                             max_frames=args.max_frames)
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: load FAILED — {e}"); continue
            try:
                pred_dir = run_mapany_on_frames(
                    model, frames, output_dir,
                    apply_mask=not args.no_mask,
                    mask_edges=not args.no_mask,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: MapAny FAILED — {e}")
                import shutil; shutil.rmtree(output_dir, ignore_errors=True); continue

        traj_dir = None if args.no_save_trajectory else \
            Path("eval_vis_ablation_mapany") / f"{scene}_{camera}"
        results = evaluate_sample(gt_dir, pred_dir, save_trajectory=traj_dir)

        if results is not None:
            flat = extract_flat(results)
            all_flat.append(flat); all_labels.append((scene, camera))
            print_row(scene, camera, flat)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump({"scene": scene, "camera": camera, **flat}, f, indent=2)
        else:
            print(f"  {label}: evaluation failed")

        if not args.eval_only and not args.keep_outputs:
            import shutil; shutil.rmtree(pred_dir, ignore_errors=True)

    elapsed = time.monotonic() - t0
    if model is not None: del model

    print()
    if all_flat:
        print_summary(all_flat)
        summary = {"num_samples": len(all_flat), "ablation": "mapany_dynamic_rgb"}
        for k in METRIC_KEYS:
            vals = [f[k] for f in all_flat if k in f and np.isfinite(f[k])]
            summary[k] = float(np.mean(vals)) if vals else None
        summary["per_sample"] = [{"scene": s, "camera": c, **flat}
                                 for (s, c), flat in zip(all_labels, all_flat)]
        with open(eval_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {eval_dir}/")

    print(f"\nTotal time: {elapsed:.1f}s ({len(all_flat)}/{len(jobs)} succeeded)")


if __name__ == "__main__":
    main()
