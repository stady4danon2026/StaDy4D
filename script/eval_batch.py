#!/usr/bin/env python3
"""Batch evaluation: run pipeline + evaluate on every (scene, camera) pair.

Runs in-process — models are loaded once and reused across all cameras.

Usage:
    # Evaluate all scenes/cameras in short/test:
    python script/eval_batch.py --duration short --split test

    # Single scene:
    python script/eval_batch.py --duration short --split test --scene scene_T03_001

    # Filter camera:
    python script/eval_batch.py --duration short --split test --camera cam_00

    # Disable trajectory PNGs (saved by default to eval_vis/):
    python script/eval_batch.py --duration short --split test --no-save-trajectory

    # Skip pipeline (evaluate existing outputs only):
    python script/eval_batch.py --duration short --split test --eval-only

    # Pass extra Hydra overrides after '--':
    python script/eval_batch.py --duration short --split test -- data.max_frames=30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress third-party noise before any HF/diffusers imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")
warnings.filterwarnings("ignore", message=".*model of type sam2_video.*")
warnings.filterwarnings("ignore", message=".*No LoRA keys.*")
warnings.filterwarnings("ignore", message=".*not expected and will be ignored.*")

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job discovery
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
# In-process pipeline runner (models loaded once)
# ---------------------------------------------------------------------------

def build_pipeline(args, extra_overrides: List[str]):
    """Build PipelineRunner once using Hydra compose. Returns (runner, cfg)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()

    config_dir = str(Path(__file__).resolve().parent.parent / "sigma" / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Use first job's scene/camera as placeholder for initial config
        overrides = [
            f"data.root_dir={args.dataset_root}",
            f"data.duration={args.duration}",
            f"data.split={args.split}",
            f"data.scene=_placeholder",
            f"data.camera=_placeholder",
            f"data.data_format={args.data_format}",
            f"pipeline/reconstruction={args.reconstruction}",
            f"pipeline/inpainting={args.inpainting}",
            "run.name=_eval_batch",
            "logging.level=WARNING",
        ] + extra_overrides
        cfg = compose(config_name="config", overrides=overrides)

    # Setup logging once (WARNING level to suppress pipeline chatter)
    from sigma.utils import setup_logging
    setup_logging(level="WARNING", save_dir=None)

    # Build stages (loads models)
    from sigma.runners.pipeline_runner import PipelineRunner

    # Temporarily set valid paths so PipelineRunner.__init__ doesn't crash
    OmegaConf.update(cfg, "run.output_dir", "/tmp/_eval_batch_init")
    OmegaConf.update(cfg, "run.visualization_dir", "/tmp/_eval_batch_init")
    OmegaConf.update(cfg, "logging.save_dir", "/tmp/_eval_batch_init/logs")
    OmegaConf.update(cfg, "data.scene", "_placeholder")
    OmegaConf.update(cfg, "data.camera", "_placeholder")

    runner = PipelineRunner(cfg)

    # Setup all stages once (loads models into GPU)
    for name in runner.active_modules:
        runner.stages[name].setup()

    return runner, cfg


def run_pipeline_for_camera(
    runner, cfg, scene: str, camera: str, output_dir: Path,
):
    """Re-run the pipeline for a new camera, reusing loaded models."""
    from omegaconf import OmegaConf
    from sigma.data import FrameDataModule, FrameSequenceConfig
    from sigma.data.frame_record import FrameIORecord
    from sigma.pipeline.reconstruction.aggregator import SceneAggregator
    from sigma.utils import ensure_dir

    # Update config for this camera
    OmegaConf.update(cfg, "data.scene", scene)
    OmegaConf.update(cfg, "data.camera", camera)
    frames_dir = f"{cfg.data.root_dir}/{cfg.data.duration}/{cfg.data.split}/{scene}"
    OmegaConf.update(cfg, "data.frames_dir", frames_dir)

    # Build fresh data module
    dm_cfg = FrameSequenceConfig(
        frames_dir=Path(frames_dir),
        frame_stride=cfg.data.frame_stride,
        max_frames=cfg.data.max_frames,
        data_format=cfg.data.get("data_format", "dynamic"),
        camera=camera,
        load_metadata=cfg.data.get("load_metadata", True),
    )
    data_module = FrameDataModule(dm_cfg)
    data_module.setup()

    # Set output dirs
    runner.output_dir = output_dir
    runner.vis_dir = output_dir
    ensure_dir(output_dir)

    # Reset frame records
    runner.frame_records = {}

    # Reset reconstructor aggregator (accumulated point clouds)
    if hasattr(runner.reconstructor, 'aggregator'):
        runner.reconstructor.aggregator = SceneAggregator()

    # Load frames
    all_frames = data_module.load_all_frames()
    for i, frame in enumerate(all_frames):
        runner.frame_records[i] = FrameIORecord(frame_idx=i, origin_image=frame)

    frame_records = runner.frame_records

    # Phase 2: Preprocess first frame
    if runner.preprocessor and "preprocess" in runner.active_modules:
        first_frame = all_frames[0]
        background_mask, _ = runner.preprocessor.process(first_frame)
        frame_records[0].mask = background_mask
        inpaint_result = runner.inpainter.process(frame_records, 0)
        runner._update_frame_record_from_inpainting(
            frame_records[0], inpaint_result.data.get("inpainting")
        )

    # Phase 3: Batch motion (skip frame 0 if preprocessor already set its mask)
    if "motion" in runner.active_modules:
        motion_results = runner.motion_stage.process_batch(frame_records)
        for fidx in sorted(motion_results):
            if fidx == 0 and frame_records[0].mask is not None:
                continue
            result = motion_results[fidx]
            runner._update_frame_record_from_motion(
                frame_records[fidx], result.data.get("motion")
            )

    # Phase 4: Batch inpainting
    if "inpainting" in runner.active_modules:
        inpaint_results = runner.inpainter.process_batch(frame_records)
        for fidx in sorted(inpaint_results):
            result = inpaint_results[fidx]
            runner._update_frame_record_from_inpainting(
                frame_records[fidx], result.data.get("inpainting")
            )

    # Phase 5: Reconstruction — save only depth/extrinsics/intrinsics for eval
    if "reconstruction" in runner.active_modules:
        depth_dir = output_dir / "depth"
        ext_dir = output_dir / "extrinsics"
        intr_dir = output_dir / "intrinsics"
        rgb_dir = output_dir / "rgb"
        mask_dir = output_dir / "mask"
        inpainted_dir = output_dir / "inpainted"
        for d in (depth_dir, ext_dir, intr_dir, rgb_dir, mask_dir, inpainted_dir):
            d.mkdir(parents=True, exist_ok=True)

        def _save_recon(fidx, recon_result):
            import imageio.v2 as imageio
            assets = recon_result.visualization_assets
            if assets.get("depth_map") is not None:
                np.save(depth_dir / f"depth_{fidx:04d}.npy", assets["depth_map"])
            np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", assets["extrinsic"])
            np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", assets["intrinsic"])
            # Save inpainted frame (no car) as pred RGB, not composited
            rec = frame_records.get(fidx)
            if rec is not None and rec.inpainted_image is not None:
                imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), rec.inpainted_image)
            elif assets.get("current_frame") is not None:
                imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), assets["current_frame"])
            # Save mask and dynamic input from frame records
            rec = frame_records.get(fidx)
            if rec is not None:
                if rec.mask is not None:
                    mask_img = (rec.mask * 255).astype(np.uint8) if rec.mask.max() <= 1 else rec.mask.astype(np.uint8)
                    imageio.imwrite(str(mask_dir / f"mask_{fidx:04d}.png"), mask_img)
                if rec.origin_image is not None:
                    imageio.imwrite(str(inpainted_dir / f"inpainted_{fidx:04d}.png"), rec.origin_image)

        if runner.reconstructor.run_deferred:
            for fidx, recon_result in runner.reconstructor.finalize(frame_records):
                _save_recon(fidx, recon_result)
        else:
            online_results = []
            for fidx in sorted(frame_records.keys()):
                recon_result = runner.reconstructor.process(frame_records, fidx)
                online_results.append((fidx, recon_result))

            # If the online reconstructor exposes a post-process hook (e.g.
            # multi-view depth fill), run it once after the per-frame loop
            # so the global aggregated cloud is available.
            if hasattr(runner.reconstructor, "post_process_fill"):
                masks_dict = {
                    fidx: rec.mask
                    for fidx, rec in frame_records.items()
                    if rec.mask is not None
                }
                runner.reconstructor.post_process_fill(
                    frame_records, online_results, masks_dict
                )

            for fidx, recon_result in online_results:
                _save_recon(fidx, recon_result)

    data_module.teardown()
    return output_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    gt_dir: Path, pred_dir: Path, flip_gt_y: bool = True,
    save_trajectory: Path | None = None,
) -> Dict | None:
    """Run evaluation on one sample."""
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
    """Extract metrics into a flat dict."""
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
# Entry point
# ---------------------------------------------------------------------------

def find_latest_output(run_name: str) -> Path | None:
    """Find the most recently created output dir for a run name."""
    run_dir = Path("outputs") / run_name
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        my_args, extra_overrides = argv[:sep], argv[sep + 1:]
    else:
        my_args, extra_overrides = argv, []

    parser = argparse.ArgumentParser(
        description="Batch pipeline + evaluation on StaDy4D (in-process, models loaded once).",
        epilog="Arguments after '--' are forwarded as Hydra overrides.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    parser.add_argument("--duration", default="short")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-format", default="dynamic")
    parser.add_argument("--scene", default=None, help="Filter scenes")
    parser.add_argument("--camera", default=None, help="Filter cameras")
    parser.add_argument("--reconstruction", default="pi3")
    parser.add_argument("--inpainting", default="sdxl_lightning_uncond")
    parser.add_argument("--no-flip-gt-y", action="store_true")
    parser.add_argument("--no-save-trajectory", action="store_true",
                        help="Disable saving trajectory PNG per sample")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip pipeline, evaluate existing outputs")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="Don't delete pipeline outputs after evaluation")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_results"),
                        help="Directory to save per-camera JSON results (default: eval_results/)")
    args = parser.parse_args(my_args)

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

    print(f"Found {len(jobs)} (scene, camera) pairs in {split_dir}")

    # Build pipeline once (loads all models)
    runner, cfg = None, None
    if not args.eval_only:
        print("Loading models...", flush=True)
        runner, cfg = build_pipeline(args, extra_overrides)
        print("Models loaded.\n", flush=True)

    all_flat: List[Dict[str, float]] = []
    all_labels: List[Tuple[str, str]] = []
    print_header()

    t0 = time.monotonic()
    for i, (scene, camera) in enumerate(jobs):
        label = f"{scene}/{camera}"
        gt_dir = split_dir / scene / "static" / camera

        if args.eval_only:
            pred_dir = find_latest_output(f"_eval_tmp_{scene}_{camera}")
            if pred_dir is None:
                pred_dir = find_latest_output(f"batch_{scene}_{camera}")
            if pred_dir is None:
                print(f"  [{i+1}/{len(jobs)}] {label}: no output found, skipping")
                continue
        else:
            output_dir = Path("outputs") / "_eval_batch" / f"{scene}_{camera}"
            try:
                pred_dir = run_pipeline_for_camera(runner, cfg, scene, camera, output_dir)
            except Exception as e:
                print(f"    pipeline FAILED — {e}")
                continue

        # Evaluate
        traj_dir = None
        if not args.no_save_trajectory:
            traj_dir = Path("eval_vis") / f"{scene}_{camera}"

        results = evaluate_sample(
            gt_dir, pred_dir,
            flip_gt_y=not args.no_flip_gt_y,
            save_trajectory=traj_dir,
        )

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

        # Clean up pipeline outputs
        if not args.eval_only and not args.keep_outputs and pred_dir is not None:
            shutil.rmtree(pred_dir, ignore_errors=True)

    elapsed = time.monotonic() - t0

    # Teardown models
    if runner is not None:
        for name in runner.active_modules:
            runner.stages[name].teardown()

    print()
    if all_flat:
        print_summary(all_flat)

        # Save aggregate summary
        summary = {"num_samples": len(all_flat)}
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
