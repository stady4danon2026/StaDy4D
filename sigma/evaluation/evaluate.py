#!/usr/bin/env python3
"""CLI entry point for 4D reconstruction evaluation.

Usage:
    python -m sigma.evaluation.evaluate \
        --gt data/carla/output/video_00 \
        --pred outputs/cc49 \
        --output eval_results.json \
        --depth-align median \
        --metrics pose depth intrinsics pointcloud

    # Quick pose-only eval:
    python -m sigma.evaluation.evaluate \
        --gt data/carla/output/video_00 \
        --pred outputs/cc49 \
        --metrics pose

    # All metrics with custom thresholds:
    python -m sigma.evaluation.evaluate \
        --gt data/carla/output/video_00 \
        --pred outputs/cc49 \
        --depth-align lsq \
        --max-depth 100 \
        --f-score-threshold 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys

from sigma.evaluation.evaluator import Evaluator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIGMA 4D Reconstruction Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gt", required=True, help="Path to ground truth directory")
    parser.add_argument("--pred", required=True, help="Path to prediction directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path for results")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pose", "depth", "pointcloud"],
        choices=["pose", "depth", "pointcloud", "all"],
        help="Which metrics to evaluate (default: all)",
    )
    parser.add_argument(
        "--depth-align",
        default="median",
        choices=["median", "lsq", "none"],
        help="Depth scale alignment method (default: median)",
    )
    parser.add_argument("--max-depth", type=float, default=500.0,
                        help="Max depth clamp (m). Pixels at or above this value (e.g. sky at 1000 m) are excluded. Default 500 suits CARLA outdoor scenes.")
    parser.add_argument("--min-depth", type=float, default=1e-3, help="Min depth clamp (m)")
    parser.add_argument("--pc-max-points", type=int, default=50000,
                        help="Max points for point cloud metrics (subsample for speed)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate depth comparison and trajectory visualizations",
    )
    parser.add_argument(
        "--vis-dir", default=None,
        help="Directory for visualization outputs (default: <output_dir>/eval_vis or ./eval_vis)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    evaluator = Evaluator(
        gt_dir=args.gt,
        pred_dir=args.pred,
        depth_align=args.depth_align,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        pc_max_points=args.pc_max_points,
    )

    metrics = args.metrics
    if "all" in metrics:
        metrics = ["pose", "depth", "pointcloud"]

    load_depth = "depth" in metrics or "pointcloud" in metrics
    evaluator.load(load_depth=load_depth)

    results = {}

    if "pose" in metrics:
        logging.info("Evaluating camera poses...")
        results["pose"] = evaluator.evaluate_poses()

    if "depth" in metrics:
        logging.info("Evaluating depth...")
        results["depth"] = evaluator.evaluate_depth()

    if "pointcloud" in metrics:
        logging.info("Evaluating point clouds...")
        results["pointcloud"] = evaluator.evaluate_pointcloud()

    # Print report
    report = evaluator.print_results(results)
    print(report)

    # Save JSON
    if args.output:
        evaluator.save_results(results, args.output)
        print(f"\nResults saved to {args.output}")

    # Visualization
    if args.visualize:
        from pathlib import Path
        if args.vis_dir:
            vis_dir = Path(args.vis_dir)
        elif args.output:
            vis_dir = Path(args.output).parent / "eval_vis"
        else:
            vis_dir = Path("eval_vis")
        logging.info("Generating visualizations in %s ...", vis_dir)
        evaluator.visualize(vis_dir)
        print(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
