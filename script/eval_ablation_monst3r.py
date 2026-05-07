#!/usr/bin/env python3
"""Ablation: MonST3R on dynamic RGB, evaluated against static GT.

Same dust3r/cloud_opt pipeline as Easi3R, without the attention-adapted
mask refinement. Lighter than the official demo (skips RAFT flow loss
and the mask-driven re-inference) and uses niter=100 for ~30 s/scene.
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from eval_ablation_vggt import (
    discover_jobs, load_dynamic_frames, evaluate_sample, extract_flat,
    print_header, print_row, print_summary, METRIC_KEYS,
)

MONST3R_REPO = "<BASELINES_ROOT>/repos/monst3r"
MONST3R_WEIGHTS = "<BASELINES_ROOT>/weights/monst3r"  # HF snapshot dir


def setup_monst3r(weights: str = MONST3R_WEIGHTS, device: str = "cuda"):
    import torch

    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    sys.path.insert(0, MONST3R_REPO)
    from dust3r.model import AsymmetricCroCo3DStereo

    print(f"Loading MonST3R from {weights}", flush=True)
    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)
    model.eval()
    print("MonST3R loaded.", flush=True)
    return model


def run_monst3r_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
    image_size: int = 512,
    niter: int = 100,
    winsize: int = 5,
    schedule: str = "cosine",
) -> Path:
    import torch
    import imageio.v2 as imageio
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = next(model.parameters()).device
    timesteps = sorted(frames.keys())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, fidx in enumerate(timesteps):
            p = tmp / f"frame_{i:04d}.png"
            imageio.imwrite(str(p), frames[fidx])
            paths.append(str(p))

        loaded = load_images(paths, size=image_size, verbose=False)
        imgs = loaded[0] if isinstance(loaded, tuple) else loaded

        pairs = make_pairs(imgs, scene_graph=f"swin-{winsize}-noncyclic",
                           prefilter=None, symmetrize=True)

        with torch.no_grad():
            output = inference(pairs, model, device, batch_size=4, verbose=False)

        scene = global_aligner(
            output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=False, shared_focal=True,
            temporal_smoothing_weight=0.01, translation_weight=1.0,
            flow_loss_weight=0.0, use_self_mask=True,
            num_total_iter=niter, empty_cache=True, batchify=True,
        )
        scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=0.01)

        focals = scene.get_focals().detach().cpu().numpy().squeeze(-1)
        c2ws = scene.get_im_poses().detach().cpu().numpy()
        depths = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]

    H_pred, W_pred = depths[0].shape
    H_orig, W_orig = next(iter(frames.values())).shape[:2]

    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    import cv2
    for i, fidx in enumerate(timesteps):
        d_orig = cv2.resize(depths[i], (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        f = float(focals[i])
        K = np.array([
            [f * W_orig / W_pred, 0, W_orig / 2.0],
            [0, f * H_orig / H_pred, H_orig / 2.0],
            [0, 0, 1.0],
        ], dtype=np.float64)

        np.save(depth_dir / f"depth_{fidx:04d}.npy", d_orig.astype(np.float32))
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", c2ws[i].astype(np.float64))
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", K)
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
    p.add_argument("--weights", default=MONST3R_WEIGHTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--niter", type=int, default=100)
    p.add_argument("--winsize", type=int, default=5)
    p.add_argument("--no-save-trajectory", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--eval-dir", type=Path, default=Path("eval_results_ablation_monst3r"))
    p.add_argument("--scratch", type=Path, default=Path("<BASELINES_ROOT>/outputs/monst3r"))
    args = p.parse_args()

    split_dir = args.dataset_root / args.duration / args.split
    if not split_dir.is_dir():
        print(f"Split directory not found: {split_dir}"); sys.exit(1)

    jobs = discover_jobs(split_dir, args.scene, args.data_format, args.camera)
    if not jobs:
        print("No (scene, camera) pairs found."); sys.exit(1)

    eval_dir = args.eval_dir / args.duration / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation: MonST3R on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    model = None
    if not args.eval_only:
        model = setup_monst3r(weights=args.weights, device=args.device)

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
                pred_dir = run_monst3r_on_frames(
                    model, frames, output_dir,
                    image_size=args.image_size, niter=args.niter, winsize=args.winsize,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: MonST3R FAILED — {e}")
                import shutil; shutil.rmtree(output_dir, ignore_errors=True); continue

        traj_dir = None if args.no_save_trajectory else \
            Path("eval_vis_ablation_monst3r") / f"{scene}_{camera}"
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
        summary = {"num_samples": len(all_flat), "ablation": "monst3r_dynamic_rgb"}
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
