#!/usr/bin/env python3
"""Ablation: CUT3R on dynamic RGB, evaluated against static GT.

CUT3R is feed-forward / online — emits per-frame depth + c2w + intrinsics
directly via prepare_output. We dump frames to a temp PNG dir, run
CUT3R inference, then convert outputs to SIGMA per-frame format.
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

CUT3R_REPO = "<BASELINES_ROOT>/repos/cut3r"
CUT3R_WEIGHTS = "<BASELINES_ROOT>/weights/cut3r/cut3r_512_dpt_4_64.pth"


def setup_cut3r(weights: str = CUT3R_WEIGHTS, device: str = "cuda"):
    import torch

    # CUT3R checkpoint pickles omegaconf objects -> torch>=2.6 weights_only blocks it.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    sys.path.insert(0, CUT3R_REPO)
    sys.path.insert(0, os.path.dirname(weights))  # mirrors add_path_to_dust3r
    from src.dust3r.model import ARCroco3DStereo

    print(f"Loading CUT3R from {weights}...", flush=True)
    model = ARCroco3DStereo.from_pretrained(weights).to(device)
    model.eval()
    print("CUT3R loaded.", flush=True)
    return model


def run_cut3r_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
    size: int = 512,
) -> Path:
    import torch
    import imageio.v2 as imageio
    from src.dust3r.inference import inference
    from src.dust3r.utils.image import load_images
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth

    timesteps = sorted(frames.keys())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, fidx in enumerate(timesteps):
            p = tmp / f"frame_{i:04d}.png"
            imageio.imwrite(str(p), frames[fidx])
            paths.append(str(p))

        images = load_images(paths, size=size)

        views = []
        for i, im in enumerate(images):
            views.append({
                "img": im["img"],
                "ray_map": torch.full(
                    (im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(im["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            })

        with torch.no_grad():
            outputs, _state = inference(views, model, "cuda")

    pts3ds_self = torch.cat([o["pts3d_in_self_view"].cpu() for o in outputs["pred"]], 0)
    pr_poses = torch.cat([
        pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in outputs["pred"]
    ], 0)

    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2]).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    depths = pts3ds_self[..., 2].numpy().astype(np.float32)  # (N, H, W)
    poses = pr_poses.numpy().astype(np.float64)              # (N, 4, 4) c2w
    intrs_np = np.zeros((B, 3, 3), dtype=np.float64)
    intrs_np[:, 0, 0] = focal.cpu().numpy()
    intrs_np[:, 1, 1] = focal.cpu().numpy()
    intrs_np[:, 0, 2] = pp[:, 0].numpy()
    intrs_np[:, 1, 2] = pp[:, 1].numpy()
    intrs_np[:, 2, 2] = 1.0

    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, fidx in enumerate(timesteps):
        np.save(depth_dir / f"depth_{fidx:04d}.npy", depths[i])
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", poses[i])
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", intrs_np[i])
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
    p.add_argument("--weights", default=CUT3R_WEIGHTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--no-save-trajectory", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--eval-dir", type=Path, default=Path("eval_results_ablation_cut3r"))
    p.add_argument("--scratch", type=Path, default=Path("<BASELINES_ROOT>/outputs/cut3r"))
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

    print(f"[Ablation: CUT3R on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    model = None
    if not args.eval_only:
        model = setup_cut3r(weights=args.weights, device=args.device)

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
                pred_dir = run_cut3r_on_frames(model, frames, output_dir, size=args.size)
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: CUT3R FAILED — {e}")
                import shutil; shutil.rmtree(output_dir, ignore_errors=True); continue

        traj_dir = None if args.no_save_trajectory else \
            Path("eval_vis_ablation_cut3r") / f"{scene}_{camera}"
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
        summary = {"num_samples": len(all_flat), "ablation": "cut3r_dynamic_rgb"}
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
