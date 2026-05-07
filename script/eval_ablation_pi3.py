#!/usr/bin/env python3
"""Ablation: Pi3-only reconstruction on dynamic RGB, evaluated against static GT.

No motion detection, no inpainting — just raw dynamic frames → Pi3 → eval.
This isolates Pi3's reconstruction quality from the rest of the pipeline.

Usage:
    # All scenes/cameras:
    python script/eval_ablation_pi3.py --duration short --split test

    # Single scene:
    python script/eval_ablation_pi3.py --duration short --split test --scene scene_T03_001

    # Filter camera:
    python script/eval_ablation_pi3.py --duration short --split test --camera cam_00

    # Skip reconstruction (evaluate existing outputs only):
    python script/eval_ablation_pi3.py --duration short --split test --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Suppress third-party noise
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")


# ---------------------------------------------------------------------------
# Job discovery (reused from eval_batch)
# ---------------------------------------------------------------------------

def discover_jobs(
    split_dir: Path,
    scene_filter: str | None,
    data_format: str,
    camera_filter: str | None,
) -> List[Tuple[str, str]]:
    """Return sorted list of (scene, camera) pairs."""
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
            if not (cam_dir / "rgb.mp4").exists() and not (cam_dir / "rgb").is_dir():
                continue
            if camera_filter and camera_filter not in cam_dir.name:
                continue
            jobs.append((scene_dir.name, cam_dir.name))
    return jobs


# ---------------------------------------------------------------------------
# Frame loading (dynamic RGB only)
# ---------------------------------------------------------------------------

def load_dynamic_frames(
    split_dir: Path, scene: str, camera: str, data_format: str,
    max_frames: int = 50,
) -> Dict[int, np.ndarray]:
    """Load dynamic RGB frames as {frame_idx: uint8 array}."""
    cam_dir = split_dir / scene / data_format / camera

    # Try safetensors + mp4 first
    mp4_path = cam_dir / "rgb.mp4"
    if mp4_path.exists():
        import imageio.v3 as iio
        frames_raw = iio.imread(str(mp4_path), plugin="pyav")
        frames = {}
        for i, frame in enumerate(frames_raw[:max_frames]):
            frames[i] = np.asarray(frame)
        return frames

    # Fallback: per-frame PNGs
    rgb_dir = cam_dir / "rgb"
    if rgb_dir.is_dir():
        import imageio.v2 as imageio
        pngs = sorted(rgb_dir.glob("*.png"))[:max_frames]
        frames = {}
        for i, p in enumerate(pngs):
            frames[i] = imageio.imread(str(p))
        return frames

    raise FileNotFoundError(f"No RGB data found in {cam_dir}")


# ---------------------------------------------------------------------------
# Pi3 runner (standalone, no pipeline)
# ---------------------------------------------------------------------------

def setup_pi3(checkpoint: str = "yyfz233/Pi3X", device: str = "cuda"):
    """Load Pi3X model once."""
    import torch
    from pi3.models.pi3x import Pi3X

    print(f"Loading Pi3X from {checkpoint}...", flush=True)
    model = Pi3X.from_pretrained(checkpoint).to(device).eval()
    print("Pi3X loaded.", flush=True)
    return model


def run_pi3_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
    device: str = "cuda",
    pixel_limit: int = 255000,
    min_confidence: float = 0.1,
    calibrate_poses: bool = False,
    pose_window_size: int = 5,
) -> Path:
    """Run Pi3X on raw frames and save depth/extrinsics/intrinsics for evaluation."""
    import math
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    import imageio.v2 as imageio

    timesteps = sorted(frames.keys())
    H_orig, W_orig = frames[timesteps[0]].shape[:2]

    # Compute Pi3-compatible size
    scale = math.sqrt(pixel_limit / max(W_orig * H_orig, 1))
    W_f, H_f = W_orig * scale, H_orig * scale
    k, m = round(W_f / 14), round(H_f / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_f / H_f:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    # Preprocess
    to_tensor = transforms.ToTensor()
    tensor_list = []
    for t in timesteps:
        frame = frames[t]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        pil = Image.fromarray(frame).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        tensor_list.append(to_tensor(pil))

    imgs = torch.stack(tensor_list, dim=0).unsqueeze(0).to(device)

    # Forward pass
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs)

    # Optional sliding-window pose calibration
    calibrated_poses: Dict[int, np.ndarray] = {}
    if calibrate_poses and len(timesteps) > pose_window_size:
        from sigma.pipeline.reconstruction.pi3_adapter import Pi3Reconstructor
        recon = Pi3Reconstructor.__new__(Pi3Reconstructor)
        recon.model = model
        recon.pose_window_size = pose_window_size
        recon.pixel_limit = pixel_limit
        recon.device = device
        calibrated_poses = recon._calibrate_poses(frames, timesteps, dtype)

    # Create output dirs
    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Extract per-frame results
    for i, fidx in enumerate(timesteps):
        local_pts = res["local_points"][0, i]  # (H_pi3, W_pi3, 3)
        camera_pose = res["camera_poses"][0, i]  # (4, 4)

        # Depth
        depth_pi3 = local_pts[..., 2].cpu().float().numpy()
        t_d = torch.from_numpy(depth_pi3).float().unsqueeze(0).unsqueeze(0)
        depth_orig = F.interpolate(
            t_d, size=(H_orig, W_orig), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        np.save(depth_dir / f"depth_{fidx:04d}.npy", depth_orig.astype(np.float32))

        # Extrinsic (use calibrated if available)
        if fidx in calibrated_poses:
            ext = calibrated_poses[fidx][:3, :].astype(np.float64)
        else:
            ext = camera_pose[:3, :].cpu().float().numpy()

        # Pad to 4x4
        ext_4x4 = np.eye(4, dtype=np.float64)
        ext_4x4[:3, :] = ext
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", ext_4x4)

        # Intrinsics (derived from local_points)
        intrinsic = _derive_intrinsics(local_pts.cpu(), TARGET_H, TARGET_W)
        # Scale to original resolution
        intrinsic[0, 0] *= W_orig / TARGET_W
        intrinsic[0, 2] *= W_orig / TARGET_W
        intrinsic[1, 1] *= H_orig / TARGET_H
        intrinsic[1, 2] *= H_orig / TARGET_H
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", intrinsic)

        # Save dynamic RGB as pred RGB (for visualization in evaluator)
        imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), frames[fidx])

    return output_dir


def _derive_intrinsics(local_pts, H, W):
    """Estimate intrinsics from Pi3X local_points."""
    import math
    import torch

    cx, cy = W / 2.0, H / 2.0
    z = local_pts[:, :, 2]
    x = local_pts[:, :, 0]
    y = local_pts[:, :, 1]

    u = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    v = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)

    valid_z = z > 0.01
    mask_fx = valid_z & ((u - cx).abs() > W * 0.1)
    mask_fy = valid_z & ((v - cy).abs() > H * 0.1)

    default_fx = W / (2 * math.tan(math.radians(60)))
    default_fy = H / (2 * math.tan(math.radians(60)))
    try:
        fx = (
            float(torch.median((u[mask_fx] - cx) / (x[mask_fx] / z[mask_fx])).item())
            if mask_fx.sum() > 10 else default_fx
        )
        fy = (
            float(torch.median((v[mask_fy] - cy) / (y[mask_fy] / z[mask_fy])).item())
            if mask_fy.sum() > 10 else default_fy
        )
    except Exception:
        fx, fy = default_fx, default_fy

    fx = float(np.clip(fx, W * 0.2, W * 5.0))
    fy = float(np.clip(fy, H * 0.2, H * 5.0))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation (same as eval_batch)
# ---------------------------------------------------------------------------

def evaluate_sample(
    gt_dir: Path, pred_dir: Path,
    save_trajectory: Path | None = None,
) -> Dict | None:
    from sigma.evaluation.evaluator import Evaluator
    try:
        evaluator = Evaluator(gt_dir=gt_dir, pred_dir=pred_dir)
        load_rgb = save_trajectory is not None
        evaluator.load(load_depth=True, load_rgb=load_rgb)

        results = {}
        results["pose"] = evaluator.evaluate_poses()
        results["depth"] = evaluator.evaluate_depth()
        results["pointcloud"] = evaluator.evaluate_pointcloud()

        if save_trajectory is not None:
            save_trajectory.mkdir(parents=True, exist_ok=True)
            evaluator.visualize(save_trajectory)

        return results
    except Exception as e:
        print(f"  [eval] ERROR: {e}", flush=True)
        return None


def extract_flat(results: Dict) -> Dict[str, float]:
    flat = {}
    if "pose" in results and "error" not in results["pose"]:
        p = results["pose"]
        for k in ("RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr"):
            flat[k] = p.get(k, float("nan"))
    if "depth" in results and "error" not in results["depth"]:
        d = results["depth"]
        for k in ("abs_rel", "rmse", "a1"):
            flat[k] = d.get(k, float("nan"))
    if "pointcloud" in results and "error" not in results["pointcloud"]:
        pc = results["pointcloud"]
        for k in ("Acc", "Comp", "NC"):
            flat[k] = pc.get(k, float("nan"))
    return flat


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "RRA@30", "RTA@30", "AUC", "ATE", "RPEt", "RPEr",
    "abs_rel", "rmse", "a1",
    "Acc", "Comp", "NC",
]

METRIC_FMT = {
    "RRA@30": ".1f", "RTA@30": ".1f", "AUC": ".1f",
    "ATE": ".4f", "RPEt": ".6f", "RPEr": ".4f",
    "abs_rel": ".4f", "rmse": ".4f", "a1": ".4f",
    "Acc": ".4f", "Comp": ".4f", "NC": ".4f",
}


def print_header():
    header = f"{'scene':<20} {'camera':<25} "
    header += " ".join(f"{k:>10}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))


def print_row(scene: str, camera: str, flat: Dict[str, float]):
    row = f"{scene:<20} {camera:<25} "
    for k in METRIC_KEYS:
        v = flat.get(k, float("nan"))
        fmt = METRIC_FMT.get(k, ".4f")
        row += f"{v:>{10}{fmt}} "
    print(row)


def print_summary(all_flat: List[Dict[str, float]]):
    print("=" * 60)
    print(f"  Aggregate over {len(all_flat)} samples")
    print("=" * 60)
    for k in METRIC_KEYS:
        vals = [f[k] for f in all_flat if k in f and np.isfinite(f[k])]
        if vals:
            mean = np.mean(vals)
            fmt = METRIC_FMT.get(k, ".4f")
            print(f"  {k:<25} {mean:{fmt}}")
        else:
            print(f"  {k:<25} N/A")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: Pi3-only on dynamic RGB, eval against static GT.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    parser.add_argument("--duration", default="short")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-format", default="dynamic")
    parser.add_argument("--scene", default=None, help="Filter scenes")
    parser.add_argument("--camera", default=None, help="Filter cameras")
    parser.add_argument("--checkpoint", default="yyfz233/Pi3X")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--pixel-limit", type=int, default=255000)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--calibrate-poses", action="store_true",
                        help="Enable sliding-window pose calibration")
    parser.add_argument("--pose-window-size", type=int, default=5)
    parser.add_argument("--no-save-trajectory", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Pi3, evaluate existing outputs")
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_results_ablation_pi3"),
                        help="Directory to save results")
    args = parser.parse_args()

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

    print(f"[Ablation: Pi3-only on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    # Load Pi3 model once
    model = None
    if not args.eval_only:
        model = setup_pi3(checkpoint=args.checkpoint, device=args.device)

    all_flat: List[Dict[str, float]] = []
    all_labels: List[Tuple[str, str]] = []
    print_header()

    t0 = time.monotonic()
    for i, (scene, camera) in enumerate(jobs):
        label = f"{scene}/{camera}"
        gt_dir = split_dir / scene / "static" / camera

        output_dir = Path("outputs") / "_ablation_pi3" / f"{scene}_{camera}"

        if args.eval_only:
            pred_dir = output_dir
            if not pred_dir.exists():
                print(f"  [{i+1}/{len(jobs)}] {label}: no output found, skipping")
                continue
        else:
            # Load dynamic frames
            try:
                frames = load_dynamic_frames(
                    split_dir, scene, camera, args.data_format,
                    max_frames=args.max_frames,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: load FAILED — {e}")
                continue

            # Run Pi3
            try:
                pred_dir = run_pi3_on_frames(
                    model, frames, output_dir,
                    device=args.device,
                    pixel_limit=args.pixel_limit,
                    min_confidence=args.min_confidence,
                    calibrate_poses=args.calibrate_poses,
                    pose_window_size=args.pose_window_size,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: Pi3 FAILED — {e}")
                continue

        # Evaluate against static GT
        traj_dir = None
        if not args.no_save_trajectory:
            traj_dir = Path("eval_vis_ablation_pi3") / f"{scene}_{camera}"

        results = evaluate_sample(gt_dir, pred_dir, save_trajectory=traj_dir)

        if results is not None:
            flat = extract_flat(results)
            all_flat.append(flat)
            all_labels.append((scene, camera))
            print_row(scene, camera, flat)

            # Save per-camera JSON
            out_json = eval_dir / scene / f"{camera}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump({"scene": scene, "camera": camera, **flat}, f, indent=2)
        else:
            print(f"  {label}: evaluation failed")

        # Clean up outputs
        if not args.eval_only and not args.keep_outputs and pred_dir is not None:
            import shutil
            shutil.rmtree(pred_dir, ignore_errors=True)

    elapsed = time.monotonic() - t0

    # Teardown
    if model is not None:
        del model

    print()
    if all_flat:
        print_summary(all_flat)

        summary = {"num_samples": len(all_flat), "ablation": "pi3_only_dynamic_rgb"}
        for k in METRIC_KEYS:
            vals = [f[k] for f in all_flat if k in f and np.isfinite(f[k])]
            summary[k] = float(np.mean(vals)) if vals else None
        summary["per_sample"] = [
            {"scene": s, "camera": c, **flat}
            for (s, c), flat in zip(all_labels, all_flat)
        ]
        summary_path = eval_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {eval_dir}/")
        print(f"  Per-camera: {eval_dir}/<scene>/<camera>.json")
        print(f"  Summary:    {summary_path}")

    print(f"\nTotal time: {elapsed:.1f}s ({len(all_flat)}/{len(jobs)} succeeded)")


if __name__ == "__main__":
    main()
