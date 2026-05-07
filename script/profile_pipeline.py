#!/usr/bin/env python3
"""Profile each phase of the NEW pipeline on one (scene, camera) pair.

Reports per-phase wall time so we can see where the 27s budget goes vs
the 1.8s raw Pi3X forward.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import build_pipeline  # noqa: E402


SCENE = "scene_T03_008"
CAMERA = "cam_00_car_forward"


def main() -> None:
    args = SimpleNamespace(
        dataset_root=Path("StaDy4D"),
        duration="short",
        split="test",
        data_format="dynamic",
        reconstruction="pi3",
        inpainting="blank",
    )
    extra = [
        "pipeline.reconstruction.use_origin_frames=true",
        "pipeline.reconstruction.fill_masked_depth=true",
        "pipeline.reconstruction.mask_dilate_px=3",
        "data.max_frames=30",
    ]

    print("Loading pipeline (one-time) ...", flush=True)
    t = time.monotonic()
    runner, cfg = build_pipeline(args, extra)
    print(f"  pipeline init: {time.monotonic()-t:.2f}s", flush=True)

    from omegaconf import OmegaConf
    from sigma.data import FrameDataModule, FrameSequenceConfig
    from sigma.data.frame_record import FrameIORecord
    from sigma.pipeline.reconstruction.aggregator import SceneAggregator

    OmegaConf.update(cfg, "data.scene", SCENE)
    OmegaConf.update(cfg, "data.camera", CAMERA)
    frames_dir = f"{cfg.data.root_dir}/{cfg.data.duration}/{cfg.data.split}/{SCENE}"
    OmegaConf.update(cfg, "data.frames_dir", frames_dir)

    dm_cfg = FrameSequenceConfig(
        frames_dir=Path(frames_dir),
        frame_stride=cfg.data.frame_stride,
        max_frames=cfg.data.max_frames,
        data_format=cfg.data.get("data_format", "dynamic"),
        camera=CAMERA,
        load_metadata=cfg.data.get("load_metadata", True),
    )

    out_dir = Path("outputs/_profile_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.output_dir = out_dir
    runner.vis_dir = out_dir

    if hasattr(runner.reconstructor, "aggregator"):
        runner.reconstructor.aggregator = SceneAggregator()

    timings = {}

    def tick(label):
        t = time.monotonic()
        return label, t

    # --- 1. Data load
    t = time.monotonic()
    data_module = FrameDataModule(dm_cfg)
    data_module.setup()
    all_frames = data_module.load_all_frames()
    runner.frame_records = {}
    for i, frame in enumerate(all_frames):
        runner.frame_records[i] = FrameIORecord(frame_idx=i, origin_image=frame)
    timings["1_data_load"] = time.monotonic() - t
    n = len(all_frames)
    print(f"  loaded {n} frames", flush=True)

    frame_records = runner.frame_records

    # --- 2. Preprocess first frame
    t = time.monotonic()
    if runner.preprocessor and "preprocess" in runner.active_modules:
        first = all_frames[0]
        bg, _ = runner.preprocessor.process(first)
        frame_records[0].mask = bg
        ip = runner.inpainter.process(frame_records, 0)
        runner._update_frame_record_from_inpainting(frame_records[0], ip.data.get("inpainting"))
    timings["2_preprocess_first"] = time.monotonic() - t

    # --- 3. Motion (GroundedSAM batch)
    t = time.monotonic()
    if "motion" in runner.active_modules:
        motion_results = runner.motion_stage.process_batch(frame_records)
        for fidx in sorted(motion_results):
            if fidx == 0 and frame_records[0].mask is not None:
                continue
            runner._update_frame_record_from_motion(
                frame_records[fidx], motion_results[fidx].data.get("motion")
            )
    timings["3_motion_groundedsam"] = time.monotonic() - t

    # --- 4. Inpainting (BlankInpainter)
    t = time.monotonic()
    if "inpainting" in runner.active_modules:
        ip_results = runner.inpainter.process_batch(frame_records)
        for fidx in sorted(ip_results):
            runner._update_frame_record_from_inpainting(
                frame_records[fidx], ip_results[fidx].data.get("inpainting")
            )
    timings["4_inpainting_blank"] = time.monotonic() - t

    # --- 5. Reconstruction (Pi3X + multi-view fill)
    t = time.monotonic()
    if "reconstruction" in runner.active_modules:
        # Time pi3x forward inside finalize separately by monkey-patching
        import torch
        recon = runner.reconstructor
        orig_full_batch = recon._finalize_full_batch
        sub_t: dict[str, float] = {}

        def timed_full_batch(frame_records, frames_dict, pi3_input_dict, masks_dict, origin_dict, timesteps):
            t0 = time.monotonic()
            imgs_tensor, resize_info = recon._preprocess_frames(pi3_input_dict, timesteps)
            sub_t["5a_pi3_preprocess"] = time.monotonic() - t0
            H_pi3, W_pi3 = resize_info["pi3_size"]
            H_orig, W_orig = resize_info["orig_size"]

            t0 = time.monotonic()
            dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = recon.model(imgs_tensor)
            torch.cuda.synchronize()
            sub_t["5b_pi3_forward"] = time.monotonic() - t0

            t0 = time.monotonic()
            calibrated_poses: dict = {}
            if recon.calibrate_poses and len(timesteps) > recon.pose_window_size:
                calibrated_poses = recon._calibrate_poses(pi3_input_dict, timesteps, dtype)
            sub_t["5c_pose_calib"] = time.monotonic() - t0

            t0 = time.monotonic()
            results = recon._build_all_frame_results(
                frame_records, res, frames_dict, masks_dict, origin_dict,
                timesteps, H_pi3, W_pi3, H_orig, W_orig, calibrated_poses,
            )
            sub_t["5d_build_results"] = time.monotonic() - t0

            t0 = time.monotonic()
            recon._fill_masked_depth_from_global(frame_records, results, masks_dict)
            sub_t["5e_multiview_fill"] = time.monotonic() - t0
            return results

        recon._finalize_full_batch = timed_full_batch
        recon_results = list(recon.finalize(frame_records))
        recon._finalize_full_batch = orig_full_batch

        timings.update(sub_t)
    timings["5_reconstruction_total"] = time.monotonic() - t

    # --- 6. Save
    t = time.monotonic()
    import imageio.v2 as imageio
    depth_dir = out_dir / "depth"; depth_dir.mkdir(exist_ok=True)
    ext_dir = out_dir / "extrinsics"; ext_dir.mkdir(exist_ok=True)
    intr_dir = out_dir / "intrinsics"; intr_dir.mkdir(exist_ok=True)
    rgb_dir = out_dir / "rgb"; rgb_dir.mkdir(exist_ok=True)
    mask_dir = out_dir / "mask"; mask_dir.mkdir(exist_ok=True)
    for fidx, recon_result in recon_results:
        assets = recon_result.visualization_assets
        if assets.get("depth_map") is not None:
            np.save(depth_dir / f"depth_{fidx:04d}.npy", assets["depth_map"])
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", assets["extrinsic"])
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", assets["intrinsic"])
        rec = frame_records.get(fidx)
        if rec is not None and rec.inpainted_image is not None:
            imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), rec.inpainted_image)
        if rec is not None and rec.mask is not None:
            mimg = (rec.mask * 255).astype(np.uint8) if rec.mask.max() <= 1 else rec.mask.astype(np.uint8)
            imageio.imwrite(str(mask_dir / f"mask_{fidx:04d}.png"), mimg)
    timings["6_save"] = time.monotonic() - t

    # --- 7. Eval
    t = time.monotonic()
    from sigma.evaluation.evaluator import Evaluator
    gt_dir = Path(f"StaDy4D/short/test/{SCENE}/static/{CAMERA}")
    ev = Evaluator(gt_dir=gt_dir, pred_dir=out_dir)
    ev.load(load_depth=True, load_rgb=False)
    pose_r = ev.evaluate_poses()
    depth_r = ev.evaluate_depth()
    pc_r = ev.evaluate_pointcloud()
    timings["7_eval"] = time.monotonic() - t

    # --- Report
    total = sum(v for k, v in timings.items() if not k.startswith("5_"))  # exclude double count
    total += timings["5_reconstruction_total"]
    print()
    print(f"=== Profile on {SCENE}/{CAMERA}, {n} frames ===")
    print(f"  {'phase':<32} {'time (s)':>10} {'%':>6}")
    print("  " + "-" * 50)
    for k in sorted(timings):
        v = timings[k]
        pct = 100 * v / total
        marker = "  └ " if k.startswith("5") and not k.endswith("_total") else "    "
        print(f"  {marker}{k:<28} {v:>10.2f} {pct:>5.1f}%")
    print(f"  {'TOTAL':<32} {total:>10.2f}")


if __name__ == "__main__":
    main()
