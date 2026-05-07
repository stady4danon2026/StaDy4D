#!/usr/bin/env python3
"""Ablation: LoGeR on dynamic RGB, evaluated against static GT.

LoGeR is built on Pi3 with a hybrid memory (sliding-window attention +
test-time training) for long-context geometric reconstruction. Outputs
local_points (depth = z) + camera_poses (c2w) per frame.

Two checkpoints supported: LoGeR and LoGeR_star.
"""

from __future__ import annotations

import argparse
import json
import math
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

LOGER_REPO = "<BASELINES_ROOT>/repos/loger"
LOGER_WEIGHTS_DIR = Path("<BASELINES_ROOT>/weights/loger")


def setup_loger(
    variant: str = "LoGeR",
    config_path: str | None = None,
    weights: str | None = None,
    device: str = "cuda",
):
    """variant ∈ {LoGeR, LoGeR_star}"""
    import torch
    import yaml
    import inspect

    sys.path.insert(0, LOGER_REPO)
    from loger.models.pi3 import Pi3

    if config_path is None:
        config_path = f"{LOGER_REPO}/ckpts/{variant}/original_config.yaml"
    if weights is None:
        weights = str(LOGER_WEIGHTS_DIR / variant / "latest.pt")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    pi3_sig = inspect.signature(Pi3.__init__)
    valid = {n for n, p in pi3_sig.parameters.items()
             if n not in {"self", "args", "kwargs"}}

    def _maybe_seq(v):
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return list(yaml.safe_load(v.strip()))
            except Exception:
                return v
        return v

    kwargs = {}
    for k in sorted(valid):
        if k in model_cfg:
            v = model_cfg[k]
            if k in {"ttt_insert_after", "attn_insert_after"}:
                v = _maybe_seq(v)
            kwargs[k] = v

    print(f"Loading {variant} from {weights} (config: {config_path})", flush=True)
    model = Pi3(**kwargs)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"{variant} loaded.", flush=True)
    return model


def _derive_intrinsics(local_pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """Same as Pi3 adapter — recover fx/fy from local_points."""
    cx, cy = W / 2.0, H / 2.0
    z = local_pts[..., 2]
    x = local_pts[..., 0]
    y = local_pts[..., 1]
    u = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    v = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))

    valid = z > 0.01
    mask_fx = valid & (np.abs(u - cx) > W * 0.1)
    mask_fy = valid & (np.abs(v - cy) > H * 0.1)

    default_fx = W / (2 * math.tan(math.radians(60)))
    default_fy = H / (2 * math.tan(math.radians(60)))
    try:
        fx = float(np.median((u[mask_fx] - cx) / (x[mask_fx] / z[mask_fx]))) \
             if mask_fx.sum() > 10 else default_fx
        fy = float(np.median((v[mask_fy] - cy) / (y[mask_fy] / z[mask_fy]))) \
             if mask_fy.sum() > 10 else default_fy
    except Exception:
        fx, fy = default_fx, default_fy

    fx = float(np.clip(fx, W * 0.2, W * 5.0))
    fy = float(np.clip(fy, H * 0.2, H * 5.0))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def run_loger_on_frames(
    model,
    frames: Dict[int, np.ndarray],
    output_dir: Path,
    window_size: int = 32,
    overlap_size: int = 3,
    target_w: int = 518,   # 37 × 14
    target_h: int = 294,   # 21 × 14 (closest 16:9 fit divisible by 14)
) -> Path:
    import torch
    import imageio.v2 as imageio
    from loger.utils.basic import load_images_as_tensor

    timesteps = sorted(frames.keys())
    device = next(model.parameters()).device

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, fidx in enumerate(timesteps):
            imageio.imwrite(str(tmp / f"frame_{i:06d}.png"), frames[fidx])

        images = load_images_as_tensor(str(tmp), Target_W=target_w, Target_H=target_h).to(device)

        forward_kwargs = {
            "window_size": window_size,
            "overlap_size": overlap_size,
        }

        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            preds = model(images[None], **forward_kwargs)

    # local_points: [B=1, S, H, W, 3]; depth = z
    local_pts = preds["local_points"][0].float().cpu().numpy()      # (S, H, W, 3)
    depths = local_pts[..., 2].astype(np.float32)                   # (S, H, W)

    # camera_poses: [B=1, S, 4, 4] c2w
    cam_poses = preds["camera_poses"][0].float().cpu().numpy().astype(np.float64)

    # Derive intrinsics per frame
    S, H, W, _ = local_pts.shape
    intrs = np.stack([_derive_intrinsics(local_pts[i], H, W) for i in range(S)])

    # Rescale intrinsics to original frame size for fair eval
    H_orig, W_orig = next(iter(frames.values())).shape[:2]
    intrs_orig = intrs.copy()
    intrs_orig[:, 0, 0] *= W_orig / W
    intrs_orig[:, 0, 2] *= W_orig / W
    intrs_orig[:, 1, 1] *= H_orig / H
    intrs_orig[:, 1, 2] *= H_orig / H

    # Resize depth to original H_orig × W_orig
    import cv2
    depths_orig = np.stack([
        cv2.resize(depths[i], (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        for i in range(S)
    ])

    depth_dir = output_dir / "depth"
    ext_dir = output_dir / "extrinsics"
    intr_dir = output_dir / "intrinsics"
    rgb_dir = output_dir / "rgb"
    for d in (depth_dir, ext_dir, intr_dir, rgb_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, fidx in enumerate(timesteps):
        np.save(depth_dir / f"depth_{fidx:04d}.npy", depths_orig[i])
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", cam_poses[i])
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", intrs_orig[i])
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
    p.add_argument("--variant", default="LoGeR", choices=["LoGeR", "LoGeR_star"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--overlap-size", type=int, default=3)
    p.add_argument("--no-save-trajectory", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument("--scratch", type=Path, default=None)
    args = p.parse_args()

    if args.eval_dir is None:
        args.eval_dir = Path(f"eval_results_ablation_loger_{args.variant.lower()}")
    if args.scratch is None:
        args.scratch = Path(f"<BASELINES_ROOT>/outputs/loger_{args.variant.lower()}")

    split_dir = args.dataset_root / args.duration / args.split
    if not split_dir.is_dir():
        print(f"Split directory not found: {split_dir}"); sys.exit(1)

    jobs = discover_jobs(split_dir, args.scene, args.data_format, args.camera)
    if not jobs:
        print("No (scene, camera) pairs found."); sys.exit(1)

    eval_dir = args.eval_dir / args.duration / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation: {args.variant} on dynamic RGB]")
    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}\n")

    model = None
    if not args.eval_only:
        model = setup_loger(variant=args.variant, device=args.device)

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
                pred_dir = run_loger_on_frames(
                    model, frames, output_dir,
                    window_size=args.window_size, overlap_size=args.overlap_size,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(jobs)}] {label}: LoGeR FAILED — {e}")
                import shutil; shutil.rmtree(output_dir, ignore_errors=True); continue

        traj_dir = None if args.no_save_trajectory else \
            Path(f"eval_vis_ablation_loger_{args.variant.lower()}") / f"{scene}_{camera}"
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
        summary = {"num_samples": len(all_flat), "ablation": f"{args.variant.lower()}_dynamic_rgb"}
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
