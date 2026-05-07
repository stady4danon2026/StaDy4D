#!/usr/bin/env python3
"""Ablation: Depth-Anything-3 on dynamic RGB, evaluated against static GT.

Mirrors eval_ablation_vggt.py — swaps VGGT for DA3.
DA3 emits per-frame depth + w2c extrinsics + intrinsics; we convert w2c→c2w
and dump to SIGMA per-frame format so sigma.evaluation.evaluator can score it.

Usage:
    python script/eval_ablation_da3.py --duration short --split test
    python script/eval_ablation_da3.py --duration short --split test --scene scene_T03
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

# Reuse infrastructure from VGGT ablation script
sys.path.insert(0, str(Path(__file__).parent))
from eval_ablation_vggt import (
    discover_jobs,
    load_dynamic_frames,
    evaluate_sample,
    extract_flat,
    print_header,
    print_row,
    print_summary,
    METRIC_KEYS,
)


# ---------------------------------------------------------------------------
# DA3 runner
# ---------------------------------------------------------------------------

def setup_da3(checkpoint: str = "depth-anything/DA3NESTED-GIANT-LARGE", device: str = "cuda"):
    import torch
    from depth_anything_3.api import DepthAnything3
    print(f"Loading DA3 from {checkpoint}...", flush=True)
    model = DepthAnything3.from_pretrained(checkpoint).to(device=torch.device(device))
    model.eval()
    print("DA3 loaded.", flush=True)
    return model


def _w2c_to_c2w(ext_w2c: np.ndarray) -> np.ndarray:
    """[..., 3, 4] w2c → [..., 4, 4] c2w."""
    R = ext_w2c[..., :3, :3]
    t = ext_w2c[..., :3, 3]
    R_c2w = np.swapaxes(R, -1, -2)
    t_c2w = -np.einsum("...ij,...j->...i", R_c2w, t)
    out = np.broadcast_to(np.eye(4, dtype=np.float64), ext_w2c.shape[:-2] + (4, 4)).copy()
    out[..., :3, :3] = R_c2w
    out[..., :3, 3] = t_c2w
    return out


def run_da3_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
    process_res: int = 504,
) -> Path:
    import imageio.v2 as imageio

    timesteps = sorted(frames.keys())
    images = [frames[t] for t in timesteps]  # list of HWC uint8

    pred = model.inference(images, process_res=process_res, align_to_input_ext_scale=False)

    # pred.depth: (N, H, W); pred.extrinsics: (N, 3, 4) w2c; pred.intrinsics: (N, 3, 3)
    depths = np.asarray(pred.depth, dtype=np.float32)
    ext_w2c = np.asarray(pred.extrinsics, dtype=np.float64)  # (N, 3, 4)
    intrs = np.asarray(pred.intrinsics, dtype=np.float64)    # (N, 3, 3)

    # Convert w2c → c2w (4x4)
    ext_c2w = _w2c_to_c2w(ext_w2c)  # (N, 4, 4)

    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, fidx in enumerate(timesteps):
        np.save(depth_dir / f"depth_{fidx:04d}.npy", depths[i].astype(np.float32))
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", ext_c2w[i])
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", intrs[i])
        imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), frames[fidx])

    return output_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="test")
    p.add_argument("--data-format", default="dynamic")
    p.add_argument("--scene", default=None)
    p.add_argument("--camera", default=None)
    p.add_argument("--checkpoint", default="depth-anything/DA3NESTED-GIANT-LARGE")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--no-save-trajectory", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--eval-dir", type=Path, default=Path("eval_results_ablation_da3"))
    p.add_argument("--scratch", type=Path, default=Path("<BASELINES_ROOT>/outputs/da3"))
    args = p.parse_args()

    split_dir = args.dataset_root / args.duration / args.split
    if not split_dir.is_dir():
        print(f"Split directory not found: {split_dir}")
        sys.exit(1)

    jobs = discover_jobs(split_dir, args.scene, args.data_format, args.camera)
    if not jobs:
        print("No (scene, camera) pairs found.")
        sys.exit(1)

    eval_dir = args.eval_dir / args.duration / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation: DA3 on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    model = None
    if not args.eval_only:
        model = setup_da3(checkpoint=args.checkpoint, device=args.device)

    all_flat: List[Dict[str, float]] = []
    all_labels: List[Tuple[str, str]] = []
    print_header()

    t0 = time.monotonic()
    for i, (scene, camera) in enumerate(jobs):
        label = f"{scene}/{camera}"
        gt_dir = split_dir / scene / "static" / camera
        output_dir = args.scratch / f"{scene}_{camera}"

        # Skip if already evaluated
        out_json = eval_dir / scene / f"{camera}.json"
        if out_json.exists() and not args.eval_only:
            try:
                with open(out_json) as f:
                    cached = json.load(f)
                flat = {k: cached[k] for k in METRIC_KEYS if k in cached}
                all_flat.append(flat); all_labels.append((scene, camera))
                print_row(scene, camera, flat)
                continue
            except Exception:
                pass

        if args.eval_only:
            pred_dir = output_dir
            if not pred_dir.exists():
                print(f"  [{i+1}/{len(jobs)}] {label}: no output, skipping")
                continue
        else:
            try:
                frames = load_dynamic_frames(
                    split_dir, scene, camera, args.data_format,
                    max_frames=args.max_frames,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: load FAILED — {e}")
                continue

            try:
                pred_dir = run_da3_on_frames(
                    model, frames, output_dir, process_res=args.process_res,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: DA3 FAILED — {e}")
                import shutil; shutil.rmtree(output_dir, ignore_errors=True)
                continue

        traj_dir = None
        if not args.no_save_trajectory:
            traj_dir = Path("eval_vis_ablation_da3") / f"{scene}_{camera}"

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
    if model is not None:
        del model

    print()
    if all_flat:
        print_summary(all_flat)
        summary = {"num_samples": len(all_flat), "ablation": "da3_dynamic_rgb"}
        for k in METRIC_KEYS:
            vals = [f[k] for f in all_flat if k in f and np.isfinite(f[k])]
            summary[k] = float(np.mean(vals)) if vals else None
        summary["per_sample"] = [
            {"scene": s, "camera": c, **flat}
            for (s, c), flat in zip(all_labels, all_flat)
        ]
        with open(eval_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {eval_dir}/")

    print(f"\nTotal time: {elapsed:.1f}s ({len(all_flat)}/{len(jobs)} succeeded)")


if __name__ == "__main__":
    main()
