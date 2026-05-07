#!/usr/bin/env python3
"""Ablation: VGGT4D on dynamic RGB, evaluated against static GT.

Two-stage inference: (1) depth + dynamic mask, (2) extrinsics refined
using dynamic mask. Outputs per-frame depth, intrinsic, c2w.
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

VGGT4D_REPO = "<BASELINES_ROOT>/repos/vggt4d"
VGGT4D_WEIGHTS = "<BASELINES_ROOT>/weights/vggt4d/model_tracker_fixed_e20.pt"


def setup_vggt4d(weights: str = VGGT4D_WEIGHTS, device: str = "cuda"):
    import torch

    sys.path.insert(0, VGGT4D_REPO)
    from vggt4d.models.vggt4d import VGGTFor4D

    print(f"Loading VGGT4D from {weights}...", flush=True)
    model = VGGTFor4D()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    print("VGGT4D loaded.", flush=True)
    return model


def run_vggt4d_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
) -> Path:
    import torch
    import torch.nn.functional as F
    import imageio.v2 as imageio
    from einops import rearrange

    from vggt4d.masks.dynamic_mask import (
        adaptive_multiotsu_variance, cluster_attention_maps, extract_dyn_map,
    )
    from vggt4d.utils.model_utils import inference, organize_qk_dict
    from vggt.utils.load_fn import load_and_preprocess_images

    device = next(model.parameters()).device
    timesteps = sorted(frames.keys())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, fidx in enumerate(timesteps):
            p = tmp / f"frame_{i:04d}.png"
            imageio.imwrite(str(p), frames[fidx])
            paths.append(str(p))
        images = load_and_preprocess_images([str(p) for p in paths]).to(device)

    n_img, _, h_img, w_img = images.shape
    h_tok, w_tok = h_img // 14, w_img // 14

    with torch.no_grad():
        # Stage 1: depth + dyn mask
        predictions1, qk_dict, enc_feat, agg_tokens_list = inference(model, images)
        del agg_tokens_list
        qk_dict = organize_qk_dict(qk_dict, n_img)
        dyn_maps = extract_dyn_map(qk_dict, images)

        feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
        norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
        upsampled = F.interpolate(
            rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
            size=(h_img, w_img), mode="bilinear", align_corners=False,
        )
        upsampled = rearrange(upsampled, "n_img 1 h w -> n_img h w")
        thres = adaptive_multiotsu_variance(upsampled.cpu().numpy())
        dyn_masks = upsampled > thres

        del enc_feat, feat_map
        torch.cuda.empty_cache()

        # Stage 2: pose refinement with dyn mask
        predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

    # depth: (N, H, W, 1) per VGGT convention -> squeeze to (N,H,W)
    depths = np.asarray(predictions1["depth"]).astype(np.float32)
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    intr = np.asarray(predictions1["intrinsic"]).astype(np.float64)        # (N, 3, 3)
    c2w = np.asarray(predictions2["cam2world"]).astype(np.float64)         # (N, 4, 4)

    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, fidx in enumerate(timesteps):
        np.save(depth_dir / f"depth_{fidx:04d}.npy", depths[i])
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", c2w[i])
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", intr[i])
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
    p.add_argument("--weights", default=VGGT4D_WEIGHTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--no-save-trajectory", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--eval-dir", type=Path, default=Path("eval_results_ablation_vggt4d"))
    p.add_argument("--scratch", type=Path, default=Path("<BASELINES_ROOT>/outputs/vggt4d"))
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

    print(f"[Ablation: VGGT4D on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    model = None
    if not args.eval_only:
        model = setup_vggt4d(weights=args.weights, device=args.device)

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
                pred_dir = run_vggt4d_on_frames(model, frames, output_dir)
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: VGGT4D FAILED — {e}")
                import shutil; shutil.rmtree(output_dir, ignore_errors=True); continue

        traj_dir = None if args.no_save_trajectory else \
            Path("eval_vis_ablation_vggt4d") / f"{scene}_{camera}"
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
        summary = {"num_samples": len(all_flat), "ablation": "vggt4d_dynamic_rgb"}
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
