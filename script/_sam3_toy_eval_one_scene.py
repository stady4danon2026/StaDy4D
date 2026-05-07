#!/usr/bin/env python3
"""Worker: run reconstruction + eval on ONE (scene, camera) and write JSON.

Used by sam3_toy_eval_isolated.py via subprocess.run() so each scene gets a
fresh Python process (no GPU memory stacking from previous runs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script.eval_batch import build_pipeline, evaluate_sample, run_pipeline_for_camera


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", nargs="+", default=[],
                   help="Hydra overrides defining the variant.")
    p.add_argument("--max-frames", type=int, default=50)
    args = p.parse_args()

    dataset_root = ROOT / "StaDy4D"
    cli = SimpleNamespace(
        dataset_root=dataset_root, duration="short", split="test",
        data_format="dynamic", reconstruction="pi3", inpainting="blank",
    )
    extra = list(args.variant) + [
        "pipeline.reconstruction.use_origin_frames=true",
        "pipeline.reconstruction.fill_masked_depth=true",
        "pipeline.reconstruction.mask_dilate_px=3",
        f"data.max_frames={args.max_frames}",
    ]
    runner, cfg = build_pipeline(cli, extra)

    out_dir = args.out.parent / f"{args.scene}_{args.camera}_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    pred_dir = run_pipeline_for_camera(runner, cfg, args.scene, args.camera, out_dir)
    gt_dir = dataset_root / "short/test" / args.scene / "static" / args.camera
    metrics = evaluate_sample(gt_dir, pred_dir, save_trajectory=None)

    elapsed = time.monotonic() - t0
    row = {
        "scene": args.scene, "camera": args.camera,
        "elapsed_s": round(elapsed, 1),
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(row, fh, indent=2)
    print(f"OK {args.scene}/{args.camera}  abs_rel={metrics.get('depth',{}).get('abs_rel',0):.3f}  "
          f"Acc={metrics.get('pointcloud',{}).get('Acc',0):.3f}  in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
