"""Main evaluator that compares ground truth vs predictions and produces a report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from sigma.evaluation.data_loader import SequenceData, load_sequence
from sigma.evaluation.metrics import (
    accuracy_completeness_nc,
    align_pose,
    ate,
    compute_pose_alignment,
    depth_metrics,
    pose_auc,
    relative_rotation_accuracy,
    relative_translation_accuracy,
    rpe,
)


def _depth_to_pointcloud(
    depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray,
) -> np.ndarray:
    """Back-project depth map to 3D world-space points. Returns (N, 3)."""
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten().astype(np.float64)
    x = (u.flatten() - cx) * z / fx
    y = (v.flatten() - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    R_c2w = extrinsic[:3, :3]
    t_c2w = extrinsic[:3, 3]
    return (pts_cam @ R_c2w.T) + t_c2w

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate predicted 4D reconstruction against ground truth.

    Supports:
    - Camera pose evaluation (ATE, RPE, per-frame error, accuracy thresholds)
    - Depth evaluation (AbsRel, SqRel, RMSE, delta thresholds)
    - Point cloud evaluation (Chamfer distance, F-score)
    - Intrinsics evaluation (focal length and principal point errors)
    """

    def __init__(
        self,
        gt_dir: str | Path,
        pred_dir: str | Path,
        depth_align: str = "median",
        max_depth: float = 500.0,
        min_depth: float = 1e-3,
        pc_max_points: int = 50000,
    ):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.depth_align = depth_align
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.pc_max_points = pc_max_points

        self.gt: SequenceData | None = None
        self.pred: SequenceData | None = None

    def load(self, load_depth: bool = True, load_rgb: bool = False) -> None:
        """Load ground truth and prediction data."""
        logger.info("Loading ground truth from %s", self.gt_dir)
        self.gt = load_sequence(self.gt_dir, load_depth=load_depth, load_rgb=load_rgb)
        logger.info("  -> %d GT frames", self.gt.num_frames)

        logger.info("Loading predictions from %s", self.pred_dir)
        self.pred = load_sequence(self.pred_dir, load_depth=load_depth, load_rgb=load_rgb)
        logger.info("  -> %d predicted frames", self.pred.num_frames)

        # Find common frame indices
        common = sorted(set(self.gt.frame_indices) & set(self.pred.frame_indices))
        logger.info("  -> %d common frames", len(common))
        if len(common) == 0:
            logger.warning("No overlapping frame indices between GT and prediction!")

    def _common_indices(self) -> List[int]:
        assert self.gt is not None and self.pred is not None
        return sorted(set(self.gt.frame_indices) & set(self.pred.frame_indices))

    # ------------------------------------------------------------------
    # Camera Pose
    # ------------------------------------------------------------------
    def evaluate_poses(self) -> Dict[str, float | Dict]:
        """Camera pose evaluation: RRA@30, RTA@30, AUC, ATE, RPEt, RPEr."""
        common = self._common_indices()
        if not common:
            return {"error": "no common frames"}

        gt_has_ext = any(self.gt.frames[i].extrinsic is not None for i in common)
        pred_has_ext = any(self.pred.frames[i].extrinsic is not None for i in common)
        if not gt_has_ext or not pred_has_ext:
            return {"error": "missing extrinsics"}

        gt_poses, pred_poses = [], []
        for idx in common:
            gt_ext = self.gt.frames[idx].extrinsic
            pred_ext = self.pred.frames[idx].extrinsic
            if gt_ext is None or pred_ext is None:
                continue
            gt_poses.append(gt_ext)
            pred_poses.append(pred_ext)

        ate_results = ate(pred_poses, gt_poses)
        rpe_results = rpe(pred_poses, gt_poses, delta=1)

        return {
            "num_frames": len(gt_poses),
            "RRA@30": relative_rotation_accuracy(pred_poses, gt_poses, threshold=30.0),
            "RTA@30": relative_translation_accuracy(pred_poses, gt_poses, threshold=30.0),
            "AUC": pose_auc(pred_poses, gt_poses, max_threshold=30.0),
            "ATE": ate_results["ate_rmse"],
            "RPEt": rpe_results["rpe_trans_rmse"],
            "RPEr": rpe_results["rpe_rot_rmse"],
            "alignment_scale": ate_results["alignment_scale"],
        }

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------
    def evaluate_depth(self) -> Dict[str, float | Dict]:
        """Depth evaluation across all common frames."""
        common = self._common_indices()
        if not common:
            return {"error": "no common frames"}

        all_metrics: List[Dict[str, float]] = []

        for idx in common:
            gt_frame = self.gt.frames[idx]
            pred_frame = self.pred.frames[idx]
            if gt_frame.depth is None or pred_frame.depth is None:
                continue

            gt_d = gt_frame.depth
            pred_d = pred_frame.depth

            # Resize prediction to match GT if needed
            if gt_d.shape != pred_d.shape:
                from PIL import Image
                pred_d = np.array(Image.fromarray(pred_d).resize(
                    (gt_d.shape[1], gt_d.shape[0]), Image.BILINEAR
                ))

            # Confidence as valid mask
            valid_mask = None
            if pred_frame.confidence is not None:
                conf = pred_frame.confidence
                if conf.shape == pred_d.shape:
                    valid_mask = conf > 0

            m = depth_metrics(
                pred_d, gt_d,
                valid_mask=valid_mask,
                max_depth=self.max_depth,
                min_depth=self.min_depth,
                align=self.depth_align,
            )
            all_metrics.append(m)

        if not all_metrics:
            return {"error": "no valid depth frames"}

        # Average only the metrics we care about: abs_rel, rmse, a1
        def _avg(key):
            vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "num_frames": len(all_metrics),
            "abs_rel": _avg("abs_rel"),
            "rmse": _avg("rmse"),
            "a1": _avg("a1"),
        }

    # ------------------------------------------------------------------
    # Point Cloud
    # ------------------------------------------------------------------
    def evaluate_pointcloud(self) -> Dict[str, float]:
        """Build point clouds from depth + pose and compare via Chamfer distance.

        Pred point clouds are transformed into GT world space using the same
        Procrustes alignment (scale + rotation + translation) computed from poses.
        """
        common = self._common_indices()
        if not common:
            return {"error": "no common frames"}

        # Gather valid poses for alignment
        gt_poses_for_align, pred_poses_for_align = [], []
        valid_indices = []
        for idx in common:
            gt_f = self.gt.frames[idx]
            pred_f = self.pred.frames[idx]
            if (gt_f.depth is None or gt_f.extrinsic is None or gt_f.intrinsic is None
                    or pred_f.depth is None or pred_f.extrinsic is None or pred_f.intrinsic is None):
                continue
            gt_poses_for_align.append(gt_f.extrinsic)
            pred_poses_for_align.append(pred_f.extrinsic)
            valid_indices.append(idx)

        if not valid_indices:
            return {"error": "no valid frames for point cloud"}

        # Compute Procrustes alignment from poses.
        # For degenerate cases (orbit/static cameras with near-zero translation),
        # compute_pose_alignment returns scale=1 with rotation-based R_align.
        # We then refine the scale using median depth ratio.
        scale, R_align, t_align = compute_pose_alignment(pred_poses_for_align, gt_poses_for_align)

        # Always cross-check with depth median ratio. Procrustes scale fits
        # trajectories — for near-static cameras it can disagree wildly with
        # the depth scale and squash the point cloud.
        # Compute ratio on CO-VALID pixels per frame (where both pred and GT
        # have valid depth), then take median across frames. Independent
        # filtering biases the ratio when sky/ground coverage differs.
        per_frame_ratios = []
        for idx in valid_indices:
            gt_d, pred_d = self.gt.frames[idx].depth, self.pred.frames[idx].depth
            if gt_d is None or pred_d is None: continue
            valid_mask = (
                (gt_d > self.min_depth) & (gt_d < self.max_depth)
                & (pred_d > self.min_depth) & (pred_d < self.max_depth)
                & np.isfinite(gt_d) & np.isfinite(pred_d)
            )
            if valid_mask.sum() < 100: continue
            per_frame_ratios.append(float(np.median(gt_d[valid_mask]) / np.median(pred_d[valid_mask])))
        if per_frame_ratios:
            depth_scale = float(np.median(per_frame_ratios))
            ratio = scale / depth_scale if depth_scale > 0 else 1.0
            # When Procrustes (trajectory-based) and depth scales disagree, the
            # trajectory was too small to be reliable — fall back to rotation-
            # based alignment (degenerate path) and depth scale.
            if scale == 1.0 or ratio < 0.5 or ratio > 2.0:
                scale = depth_scale
                # Re-derive R_align from camera rotations (degenerate-style)
                M = np.zeros((3, 3))
                for pred_p, gt_p in zip(pred_poses_for_align, gt_poses_for_align):
                    M += gt_p[:3, :3] @ pred_p[:3, :3].T
                U, _, Vt = np.linalg.svd(M)
                d = np.linalg.det(U @ Vt)
                D = np.diag([1.0, 1.0, d])
                R_align = U @ D @ Vt
                pred_pos = np.array([p[:3, 3] for p in pred_poses_for_align])
                gt_pos = np.array([p[:3, 3] for p in gt_poses_for_align])
                t_align = gt_pos.mean(0) - scale * R_align @ pred_pos.mean(0)

        gt_points_all, pred_points_all = [], []
        for idx, gt_ext_raw, pred_ext_raw in zip(
            valid_indices, gt_poses_for_align, pred_poses_for_align
        ):
            gt_f = self.gt.frames[idx]
            pred_f = self.pred.frames[idx]

            # Mask out sky / far-plane pixels before back-projection
            # Symmetric clamp on both sides — without filtering pred>max_depth,
            # methods that predict realistic far depth at sky pixels get phantom
            # points in mid-air (no nearby GT) that inflate Acc relative to
            # methods that confidence-filter sky to zero.
            gt_depth = gt_f.depth.copy().astype(np.float32)
            pred_depth = pred_f.depth.copy().astype(np.float32)
            gt_depth[(gt_depth < self.min_depth) | (gt_depth > self.max_depth)] = np.nan
            pred_depth[(pred_depth < self.min_depth) | (pred_depth > self.max_depth)] = np.nan

            gt_pts = _depth_to_pointcloud(gt_depth, gt_f.intrinsic, gt_ext_raw)

            # Apply alignment to pred extrinsic so points land in GT world space
            pred_ext_aligned = align_pose(pred_ext_raw, scale, R_align, t_align)
            pred_pts = _depth_to_pointcloud(pred_depth, pred_f.intrinsic, pred_ext_aligned)

            gt_pts = gt_pts[np.isfinite(gt_pts).all(axis=1)]
            pred_pts = pred_pts[np.isfinite(pred_pts).all(axis=1)]

            gt_points_all.append(gt_pts)
            pred_points_all.append(pred_pts)

        gt_pc = np.concatenate(gt_points_all, axis=0)
        pred_pc = np.concatenate(pred_points_all, axis=0)

        results = accuracy_completeness_nc(pred_pc, gt_pc, max_points=self.pc_max_points)
        results["alignment_scale"] = float(scale)

        return results

    # ------------------------------------------------------------------
    # Run All
    # ------------------------------------------------------------------
    def evaluate_all(self) -> Dict[str, Dict]:
        """Run all evaluations and return combined results."""
        if self.gt is None:
            self.load()

        results = {}

        logger.info("Evaluating camera poses...")
        results["pose"] = self.evaluate_poses()

        logger.info("Evaluating depth...")
        results["depth"] = self.evaluate_depth()

        logger.info("Evaluating point clouds...")
        results["pointcloud"] = self.evaluate_pointcloud()

        return results

    def print_results(self, results: Dict[str, Dict]) -> str:
        """Format results into a readable report string."""
        lines = []
        lines.append("=" * 70)
        lines.append("  SIGMA 4D Reconstruction Evaluation Report")
        lines.append("=" * 70)
        lines.append(f"  GT:   {self.gt_dir}")
        lines.append(f"  Pred: {self.pred_dir}")
        lines.append("")

        D = "↓"
        U = "↑"

        if "pose" in results and "error" not in results["pose"]:
            p = results["pose"]
            lines.append("--- Camera Pose ---")
            lines.append(f"  Frames:   {p['num_frames']}")
            lines.append(f"  RRA@30:   {p['RRA@30']:.1f}%  {U}")
            lines.append(f"  RTA@30:   {p['RTA@30']:.1f}%  {U}")
            lines.append(f"  AUC:      {p['AUC']:.1f}%  {U}")
            lines.append(f"  ATE:      {p['ATE']:.4f} m  {D}")
            lines.append(f"  RPEt:     {p['RPEt']:.6f} m  {D}")
            lines.append(f"  RPEr:     {p['RPEr']:.4f} deg  {D}")
            lines.append("")

        if "depth" in results and "error" not in results["depth"]:
            d = results["depth"]
            lines.append("--- Depth ---")
            lines.append(f"  Frames:   {d.get('num_frames', '?')}")
            lines.append(f"  Abs Rel:  {d['abs_rel']:.4f}  {D}")
            lines.append(f"  RMSE:     {d['rmse']:.4f}  {D}")
            lines.append(f"  d1:       {d['a1']:.4f}  {U}")
            lines.append("")

        if "pointcloud" in results and "error" not in results["pointcloud"]:
            pc = results["pointcloud"]
            lines.append("--- Point Cloud ---")
            lines.append(f"  Acc.:     {pc['Acc']:.4f}  {D}")
            lines.append(f"  Comp.:    {pc['Comp']:.4f}  {D}")
            lines.append(f"  N.C.:     {pc['NC']:.4f}  {U}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize(self, vis_dir: str | Path) -> None:
        """Generate 2x2 comparison images per frame.

        Layout::

            GT depth   | Pred depth
            GT RGB     | Pred RGB

        Saved as ``vis_dir/frame_NNNN.png``.
        Also saves ``trajectory.png`` (bird's-eye + side view).

        Args:
            vis_dir: Directory to write visualization images.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping visualization")
            return

        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        common = self._common_indices()
        if not common:
            logger.warning("No common frames — nothing to visualize")
            return

        # ---- 3x2 comparison for sampled frames (first, last, ~5 evenly spaced) ----
        max_vis = 5
        if len(common) <= max_vis:
            vis_indices = common
        else:
            step = (len(common) - 1) / (max_vis - 1)
            vis_indices = [common[int(round(i * step))] for i in range(max_vis)]

        for idx in vis_indices:
            gt_f = self.gt.frames[idx]
            pred_f = self.pred.frames[idx]

            has_depth = gt_f.depth is not None and pred_f.depth is not None
            has_rgb = gt_f.rgb is not None and pred_f.rgb is not None
            if not has_depth and not has_rgb:
                continue

            fig, axes = plt.subplots(3, 2, figsize=(12, 12))

            # Row 0: Depth
            if has_depth:
                gt_d = gt_f.depth.astype(np.float32)
                gt_d[(gt_d < self.min_depth) | (gt_d > self.max_depth)] = np.nan
                pred_d = pred_f.depth.astype(np.float32)
                pred_d[pred_d < self.min_depth] = np.nan

                if gt_d.shape != pred_d.shape:
                    from PIL import Image as _PILImage
                    pred_d = np.array(_PILImage.fromarray(pred_d).resize(
                        (gt_d.shape[1], gt_d.shape[0]), _PILImage.BILINEAR))

                gt_valid = gt_d[(gt_d > 1e-3) & np.isfinite(gt_d)]
                pred_valid = pred_d[(pred_d > 1e-3) & np.isfinite(pred_d)]
                vis_scale = float(np.median(gt_valid) / np.median(pred_valid)) if len(gt_valid) > 0 and len(pred_valid) > 0 else 1.0
                pred_d_vis = pred_d * vis_scale

                vmax = float(np.percentile(gt_valid, 95)) if len(gt_valid) > 0 else float(np.nanmax(gt_d))
                vmin = float(gt_d[gt_d > 0].min()) if (gt_d > 0).any() else 0.0

                axes[0, 0].imshow(gt_d, cmap="turbo", vmin=vmin, vmax=vmax)
                axes[0, 0].set_title("GT Depth")
                axes[0, 1].imshow(pred_d_vis, cmap="turbo", vmin=vmin, vmax=vmax)
                axes[0, 1].set_title(f"Pred Depth (x{vis_scale:.1f})")
            else:
                axes[0, 0].text(0.5, 0.5, "No GT depth", ha="center", va="center")
                axes[0, 1].text(0.5, 0.5, "No pred depth", ha="center", va="center")

            # Row 1: RGB
            if has_rgb:
                gt_rgb = gt_f.rgb
                pred_rgb = pred_f.rgb
                if gt_rgb.shape != pred_rgb.shape:
                    from PIL import Image as _PILImage
                    pred_rgb = np.array(_PILImage.fromarray(pred_rgb).resize(
                        (gt_rgb.shape[1], gt_rgb.shape[0]), _PILImage.BILINEAR))
                axes[1, 0].imshow(gt_rgb)
                axes[1, 0].set_title("GT RGB")
                axes[1, 1].imshow(pred_rgb)
                axes[1, 1].set_title("Pred RGB")
            else:
                axes[1, 0].text(0.5, 0.5, "No GT RGB", ha="center", va="center")
                axes[1, 1].text(0.5, 0.5, "No pred RGB", ha="center", va="center")

            # Row 2: Input RGB (dynamic scene) + Mask
            if pred_f.inpainted is not None:
                # inpainted field stores the dynamic input RGB in this context
                axes[2, 0].imshow(pred_f.inpainted)
                axes[2, 0].set_title("Input RGB (dynamic)")
            else:
                axes[2, 0].text(0.5, 0.5, "No input RGB", ha="center", va="center")

            if pred_f.mask is not None and pred_f.inpainted is not None:
                # Show RGB masked: keep only non-mask region
                mask_bin = (pred_f.mask > 127) if pred_f.mask.max() > 1 else (pred_f.mask > 0)
                masked_rgb = pred_f.inpainted.copy()
                masked_rgb[mask_bin] = 0
                axes[2, 1].imshow(masked_rgb)
                axes[2, 1].set_title("Masked RGB")
            elif pred_f.mask is not None:
                axes[2, 1].imshow(pred_f.mask, cmap="gray", vmin=0, vmax=255)
                axes[2, 1].set_title("Mask")
            else:
                axes[2, 1].text(0.5, 0.5, "No mask", ha="center", va="center")

            for ax in axes.flat:
                ax.axis("off")

            fig.suptitle(f"Frame {idx}", fontsize=14)
            fig.tight_layout()
            fig.savefig(vis_dir / f"frame_{idx:04d}.png", dpi=100)
            plt.close(fig)

        # ---- Trajectory plot ----
        gt_poses, pred_poses = [], []
        for idx in common:
            gt_ext = self.gt.frames[idx].extrinsic
            pred_ext = self.pred.frames[idx].extrinsic
            if gt_ext is None or pred_ext is None:
                continue
            gt_poses.append(gt_ext)
            pred_poses.append(pred_ext)

        if len(gt_poses) < 2:
            return

        gt_pos = np.array([p[:3, 3] for p in gt_poses])
        pred_pos = np.array([p[:3, 3] for p in pred_poses])
        # Camera forward direction: -Z axis of c2w (OpenGL convention)
        gt_fwd = np.array([-p[:3, 2] for p in gt_poses])

        scale, R_align, t_align = compute_pose_alignment(pred_poses, gt_poses)
        aligned_pos = scale * (pred_pos @ R_align.T) + t_align
        # Align pred forward directions too
        pred_fwd_raw = np.array([-p[:3, 2] for p in pred_poses])
        aligned_fwd = pred_fwd_raw @ R_align.T

        def _plot_traj(ax, idx_h, idx_v, xlabel, ylabel, title):
            n = len(gt_pos)
            all_h = np.concatenate([gt_pos[:, idx_h], aligned_pos[:, idx_h]])
            all_v = np.concatenate([gt_pos[:, idx_v], aligned_pos[:, idx_v]])

            # Use percentile-based range to exclude outliers
            lo_h, hi_h = np.percentile(all_h, 2), np.percentile(all_h, 98)
            lo_v, hi_v = np.percentile(all_v, 2), np.percentile(all_v, 98)
            span = max(hi_h - lo_h, hi_v - lo_v, 1.0)
            arrow_len = span * 0.08

            # Trajectory lines (no markers)
            ax.plot(gt_pos[:, idx_h], gt_pos[:, idx_v], "b-", linewidth=1.2, label="GT", alpha=0.4)
            ax.plot(aligned_pos[:, idx_h], aligned_pos[:, idx_v], "g-", linewidth=1.2,
                    label=f"Pred (aligned x{scale:.2f})", alpha=0.4)

            # Camera direction arrow at every timestep
            for i in range(n):
                for pos, fwd, color in [
                    (gt_pos, gt_fwd, "blue"),
                    (aligned_pos, aligned_fwd, "green"),
                ]:
                    dh = fwd[i, idx_h]
                    dv = fwd[i, idx_v]
                    norm = np.sqrt(dh**2 + dv**2) + 1e-8
                    ax.annotate(
                        "", xy=(pos[i, idx_h] + dh/norm*arrow_len,
                                pos[i, idx_v] + dv/norm*arrow_len),
                        xytext=(pos[i, idx_h], pos[i, idx_v]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                        mutation_scale=10),
                    )

            # Start marker
            ax.plot(gt_pos[0, idx_h], gt_pos[0, idx_v], "bs", markersize=8)
            ax.plot(aligned_pos[0, idx_h], aligned_pos[0, idx_v], "gs", markersize=8)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ch, cv = all_h.mean(), all_v.mean()
            pad = span * 0.15
            half = span / 2 + pad
            ax.set_xlim(ch - half, ch + half)
            ax.set_ylim(cv - half, cv + half)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        _plot_traj(axes[0], 0, 2, "X (m)", "Z (m)", "Bird's-eye view (X-Z)")
        _plot_traj(axes[1], 0, 1, "X (m)", "Y (m)", "Side view (X-Y)")
        fig.tight_layout()
        fig.savefig(vis_dir / "trajectory.png", dpi=120)
        plt.close(fig)

    def save_results(self, results: Dict, output_path: str | Path) -> None:
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove per-frame lists for JSON (they can be huge)
        save_results = {}
        for category, vals in results.items():
            if isinstance(vals, dict):
                save_results[category] = {
                    k: v for k, v in vals.items()
                    if not isinstance(v, list)
                }
            else:
                save_results[category] = vals

        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2, default=_json_default)
        logger.info("Results saved to %s", output_path)


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
