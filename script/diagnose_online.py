#!/usr/bin/env python3
"""Diagnose online Pi3X mode — capture per-frame depth statistics + scale evolution.

Calls online's process() one frame at a time and inspects:
- depth distribution per call (median, p10, p90, max)
- pose translation magnitude
- self._global_poses growth and scale per Sim(3) chain step
- whether any obvious garbage shows up
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import build_pipeline


def main() -> None:
    pipeline_args = SimpleNamespace(
        dataset_root=Path("StaDy4D"),
        duration="short", split="test", data_format="dynamic",
        reconstruction="pi3_online", inpainting="blank",
    )
    extras = [
        "pipeline.reconstruction.use_origin_frames=true",
        "pipeline.reconstruction.fill_masked_depth=false",  # no fill — see raw per-frame
        "pipeline.reconstruction.max_frames=5",
        "data.max_frames=10",
    ]

    print("Loading pipeline ...", flush=True)
    runner, cfg = build_pipeline(pipeline_args, extras)

    from omegaconf import OmegaConf
    from sigma.data import FrameDataModule, FrameSequenceConfig
    from sigma.data.frame_record import FrameIORecord
    from sigma.pipeline.reconstruction.aggregator import SceneAggregator

    scene = "scene_T03_001"
    cam = "cam_00_car_forward"
    OmegaConf.update(cfg, "data.scene", scene)
    OmegaConf.update(cfg, "data.camera", cam)
    frames_dir = f"{cfg.data.root_dir}/short/test/{scene}"
    OmegaConf.update(cfg, "data.frames_dir", frames_dir)

    dm = FrameDataModule(FrameSequenceConfig(
        frames_dir=Path(frames_dir),
        frame_stride=1, max_frames=10, data_format="dynamic", camera=cam,
    ))
    dm.setup()
    runner.reconstructor.aggregator = SceneAggregator()
    runner.frame_records = {}
    for i, f in enumerate(dm.load_all_frames()):
        runner.frame_records[i] = FrameIORecord(frame_idx=i, origin_image=f)

    # Run motion + inpainting (so masks/inpainted are set)
    motion = runner.motion_stage.process_batch(runner.frame_records)
    for fidx in sorted(motion):
        if fidx == 0 and runner.frame_records[0].mask is not None:
            continue
        runner._update_frame_record_from_motion(runner.frame_records[fidx], motion[fidx].data.get("motion"))
    inpaint = runner.inpainter.process_batch(runner.frame_records)
    for fidx in sorted(inpaint):
        runner._update_frame_record_from_inpainting(runner.frame_records[fidx], inpaint[fidx].data.get("inpainting"))

    print(f"\n=== Per-frame online process() diagnostics on {scene}/{cam} ===")
    print(f"{'idx':>4} {'win':>4} {'depth_med':>10} {'depth_p10':>10} {'depth_p90':>10} "
          f"{'depth_max':>10} {'pose_t_norm':>12} {'global_n':>9}")
    print("-" * 78)

    for frame_idx in sorted(runner.frame_records.keys()):
        recon_result = runner.reconstructor.process(runner.frame_records, frame_idx)
        recon_outputs = recon_result.data["reconstruction"]
        depth = recon_outputs.depth_map.astype(np.float32)
        valid = depth[depth > 0]
        ext = recon_outputs.extrinsic
        t_norm = float(np.linalg.norm(ext[:, 3]))
        win_size = min(frame_idx + 1, 5)
        print(f"{frame_idx:>4} {win_size:>4} "
              f"{np.median(valid):>10.3f} "
              f"{np.percentile(valid,10):>10.3f} "
              f"{np.percentile(valid,90):>10.3f} "
              f"{valid.max():>10.3f} "
              f"{t_norm:>12.4f} "
              f"{len(runner.reconstructor._global_poses):>9}")

    # Print accumulated _global_poses translation evolution
    print(f"\n=== _global_poses translation evolution ===")
    for k, v in sorted(runner.reconstructor._global_poses.items()):
        t = v[:3, 3]
        print(f"  frame {k:>3}: t = ({t[0]:>+8.3f}, {t[1]:>+8.3f}, {t[2]:>+8.3f})  |t|={np.linalg.norm(t):>7.4f}")


if __name__ == "__main__":
    main()
