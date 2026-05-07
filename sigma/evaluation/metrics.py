"""Core metric functions for 4D reconstruction evaluation.

Covers camera pose, depth, point cloud, and intrinsics metrics.
All functions are pure numpy — no torch dependency.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Camera Pose Metrics
# ---------------------------------------------------------------------------

def rotation_geodesic_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angular distance between two rotation matrices (degrees).

    err = arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    """
    R_rel = R_pred.T @ R_gt
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Euclidean distance between two translation vectors."""
    return float(np.linalg.norm(t_pred - t_gt))


def pose_error(ext_pred: np.ndarray, ext_gt: np.ndarray) -> Tuple[float, float]:
    """Compute translation error (m) and rotation error (deg) from 3x4 c2w matrices."""
    R_pred, t_pred = ext_pred[:3, :3], ext_pred[:3, 3]
    R_gt, t_gt = ext_gt[:3, :3], ext_gt[:3, 3]
    t_err = translation_error(t_pred, t_gt)
    r_err = rotation_geodesic_error(R_pred, R_gt)
    return t_err, r_err


def compute_pose_alignment(
    pred_poses: list[np.ndarray], gt_poses: list[np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the similarity transform that best aligns pred trajectory to GT.

    Finds s, R, t such that s * R @ p_pred + t ≈ p_gt for all positions.

    Returns:
        (scale, R_align (3x3), t_align (3,))
    """
    pred_positions = np.array([p[:3, 3] for p in pred_poses])
    gt_positions = np.array([p[:3, 3] for p in gt_poses])

    pred_centered = pred_positions - pred_positions.mean(axis=0)
    gt_centered = gt_positions - gt_positions.mean(axis=0)
    pred_var = float(np.sum(pred_centered ** 2))
    gt_var = float(np.sum(gt_centered ** 2))
    degenerate = pred_var < 1e-6 or gt_var < 1e-6

    # Build the cross-covariance matrix whose SVD gives R_align.
    # Normal: Procrustes on positions.  Degenerate (orbit/static): align rotations.
    if degenerate:
        H = sum(gt_p[:3, :3] @ pred_p[:3, :3].T for pred_p, gt_p in zip(pred_poses, gt_poses))
    else:
        H = pred_centered.T @ gt_centered

    U, S, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])
    R_align = (U @ D @ Vt) if degenerate else (Vt.T @ D @ U.T)

    scale = 1.0 if degenerate else float(np.sum(S) / pred_var)
    t_align = gt_positions.mean(axis=0) - scale * R_align @ pred_positions.mean(axis=0)
    return scale, R_align, t_align


def align_pose(
    ext: np.ndarray, scale: float, R_align: np.ndarray, t_align: np.ndarray
) -> np.ndarray:
    """Apply a similarity transform to a c2w pose matrix.

    The transform maps pred world space to GT world space:
        p_gt = scale * R_align @ p_pred + t_align

    The returned 4x4 matrix has the scale absorbed into both R and the
    camera-space depth, so that ``_depth_to_pointcloud`` can use it directly.

    Args:
        ext: 3x4 or 4x4 camera-to-world extrinsic.
        scale, R_align, t_align: output of :func:`compute_pose_alignment`.

    Returns:
        4x4 aligned c2w extrinsic.
    """
    R_c2w = ext[:3, :3]
    t_c2w = ext[:3, 3]
    # scale * R_align brings camera-space points into GT world scale+orientation
    R_new = scale * (R_align @ R_c2w)
    t_new = scale * (R_align @ t_c2w) + t_align
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = R_new
    result[:3, 3] = t_new
    return result


def ate(pred_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> Dict[str, float]:
    """Absolute Trajectory Error (ATE).

    After Procrustes alignment (similarity transform), compute RMSE of positions.
    Returns aligned ATE and per-frame stats.
    """
    scale, R_align, t_align = compute_pose_alignment(pred_poses, gt_poses)

    pred_positions = np.array([p[:3, 3] for p in pred_poses])
    gt_positions = np.array([p[:3, 3] for p in gt_poses])

    aligned = scale * (pred_positions @ R_align.T) + t_align
    errors = np.linalg.norm(aligned - gt_positions, axis=1)

    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_std": float(np.std(errors)),
        "ate_max": float(np.max(errors)),
        "alignment_scale": float(scale),
    }


def rpe(pred_poses: list[np.ndarray], gt_poses: list[np.ndarray],
        delta: int = 1) -> Dict[str, float]:
    """Relative Pose Error (RPE).

    Measures local consistency: error of relative transforms between frame pairs.
    """
    t_errors, r_errors = [], []
    for i in range(len(pred_poses) - delta):
        j = i + delta
        # Relative transform
        gt_rel = _relative_transform(gt_poses[i], gt_poses[j])
        pred_rel = _relative_transform(pred_poses[i], pred_poses[j])
        # Error of relative transform
        err = _relative_transform(gt_rel, pred_rel)
        t_errors.append(float(np.linalg.norm(err[:3, 3])))
        r_errors.append(rotation_geodesic_error(err[:3, :3], np.eye(3)))

    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(np.array(t_errors) ** 2))),
        "rpe_trans_mean": float(np.mean(t_errors)),
        "rpe_trans_median": float(np.median(t_errors)),
        "rpe_rot_rmse": float(np.sqrt(np.mean(np.array(r_errors) ** 2))),
        "rpe_rot_mean": float(np.mean(r_errors)),
        "rpe_rot_median": float(np.median(r_errors)),
    }


def _to_4x4(ext: np.ndarray) -> np.ndarray:
    """Convert 3x4 to 4x4 by adding [0,0,0,1] row."""
    if ext.shape == (4, 4):
        return ext
    out = np.eye(4)
    out[:3, :] = ext
    return out


def _relative_transform(ext_a: np.ndarray, ext_b: np.ndarray) -> np.ndarray:
    """Compute relative transform from A to B: T_B @ T_A^{-1}."""
    A = _to_4x4(ext_a)
    B = _to_4x4(ext_b)
    return np.linalg.inv(A) @ B


def relative_rotation_accuracy(
    pred_poses: list[np.ndarray], gt_poses: list[np.ndarray],
    threshold: float = 30.0, delta: int = 1,
) -> float:
    """RRA@θ: fraction of consecutive pairs where relative rotation error < θ degrees."""
    n_correct, n_total = 0, 0
    for i in range(len(pred_poses) - delta):
        j = i + delta
        gt_rel = _relative_transform(gt_poses[i], gt_poses[j])
        pred_rel = _relative_transform(pred_poses[i], pred_poses[j])
        r_err = rotation_geodesic_error(pred_rel[:3, :3], gt_rel[:3, :3])
        if r_err < threshold:
            n_correct += 1
        n_total += 1
    return 100.0 * n_correct / max(n_total, 1)


def _translation_angular_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Angular error (degrees) between two translation direction vectors."""
    n_pred = np.linalg.norm(t_pred)
    n_gt = np.linalg.norm(t_gt)
    if n_pred < 1e-10 or n_gt < 1e-10:
        return 0.0
    cos_angle = np.dot(t_pred, t_gt) / (n_pred * n_gt)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def relative_translation_accuracy(
    pred_poses: list[np.ndarray], gt_poses: list[np.ndarray],
    threshold: float = 30.0, delta: int = 1,
) -> float:
    """RTA@θ: fraction of consecutive pairs where relative translation direction error < θ degrees."""
    n_correct, n_total = 0, 0
    for i in range(len(pred_poses) - delta):
        j = i + delta
        gt_rel = _relative_transform(gt_poses[i], gt_poses[j])
        pred_rel = _relative_transform(pred_poses[i], pred_poses[j])
        t_err = _translation_angular_error(pred_rel[:3, 3], gt_rel[:3, 3])
        if t_err < threshold:
            n_correct += 1
        n_total += 1
    return 100.0 * n_correct / max(n_total, 1)


def pose_auc(
    pred_poses: list[np.ndarray], gt_poses: list[np.ndarray],
    max_threshold: float = 30.0, delta: int = 1, num_bins: int = 300,
) -> float:
    """AUC of cumulative pose accuracy curve.

    For each threshold in [0, max_threshold], accuracy = fraction of pairs
    where max(rot_err, trans_angular_err) < threshold.
    Returns AUC normalized to [0, 100].
    """
    rot_errors, trans_angular_errors = [], []
    for i in range(len(pred_poses) - delta):
        j = i + delta
        gt_rel = _relative_transform(gt_poses[i], gt_poses[j])
        pred_rel = _relative_transform(pred_poses[i], pred_poses[j])
        rot_errors.append(rotation_geodesic_error(pred_rel[:3, :3], gt_rel[:3, :3]))
        trans_angular_errors.append(_translation_angular_error(pred_rel[:3, 3], gt_rel[:3, 3]))

    max_errors = np.maximum(rot_errors, trans_angular_errors)
    thresholds = np.linspace(0, max_threshold, num_bins + 1)
    accuracies = np.array([np.mean(max_errors < t) for t in thresholds])
    auc = float(np.trapz(accuracies, thresholds) / max_threshold * 100.0)
    return auc


# ---------------------------------------------------------------------------
# Depth Metrics
# ---------------------------------------------------------------------------

def depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray | None = None,
    max_depth: float = 80.0,
    min_depth: float = 1e-3,
    align: str = "median",
) -> Dict[str, float]:
    """Standard monocular depth evaluation metrics.

    Args:
        pred: Predicted depth (H, W).
        gt: Ground truth depth (H, W).
        valid_mask: Optional boolean mask for valid pixels.
        max_depth: Upper depth clamp.
        min_depth: Lower depth clamp.
        align: Scale alignment method — "median", "lsq" (least squares), or "none".

    Returns dict with: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, si_log.
    """
    # Filter pred>min_depth too: some pipelines write 0 for "no prediction"
    # (e.g. exclude_moving_objects). Including those pixels corrupts the
    # median scale alignment and post-alignment range filter.
    if valid_mask is None:
        valid_mask = (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & np.isfinite(gt) & np.isfinite(pred)
    else:
        valid_mask = valid_mask & (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & np.isfinite(gt) & np.isfinite(pred)

    pred_v = pred[valid_mask].astype(np.float64)
    gt_v = gt[valid_mask].astype(np.float64)

    if len(pred_v) == 0:
        return {k: float("nan") for k in [
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "si_log",
            "scale", "shift"
        ]}

    # Scale/shift alignment (predictions may be up-to-scale)
    scale, shift = 1.0, 0.0
    if align == "median":
        scale = float(np.median(gt_v) / np.median(pred_v))
        pred_v = pred_v * scale
    elif align == "lsq":
        # Affine: pred_aligned = scale * pred + shift
        A = np.stack([pred_v, np.ones_like(pred_v)], axis=-1)
        result = np.linalg.lstsq(A, gt_v, rcond=None)
        scale, shift = result[0]
        pred_v = pred_v * scale + shift
    # else "none": no alignment

    # Re-filter after alignment: exclude pixels where aligned pred falls outside valid range
    post_valid = (pred_v > min_depth) & (pred_v < max_depth)
    pred_v = pred_v[post_valid]
    gt_v = gt_v[post_valid]

    if len(pred_v) == 0:
        return {k: float("nan") for k in [
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "si_log",
            "scale", "shift"
        ]}

    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    a1 = float(np.mean(thresh < 1.25))
    a2 = float(np.mean(thresh < 1.25 ** 2))
    a3 = float(np.mean(thresh < 1.25 ** 3))

    diff = pred_v - gt_v
    abs_rel = float(np.mean(np.abs(diff) / gt_v))
    sq_rel = float(np.mean(diff ** 2 / gt_v))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(pred_v) - np.log(gt_v)) ** 2)))

    # Scale-invariant log error (Eigen et al.)
    log_diff = np.log(pred_v) - np.log(gt_v)
    si_log = float(np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "si_log": si_log,
        "scale": scale,
        "shift": shift,
    }


# ---------------------------------------------------------------------------
# Point Cloud Metrics
# ---------------------------------------------------------------------------

def _estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate surface normals via local PCA on k-nearest neighbors."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)
    normals = np.zeros_like(points)
    for i, nbr_idx in enumerate(indices):
        nbrs = points[nbr_idx]
        centered = nbrs - nbrs.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue = normal direction
    return normals


def accuracy_completeness_nc(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    max_points: int = 50000,
) -> Dict[str, float]:
    """Accuracy (Acc.), Completeness (Comp.), and Normal Consistency (N.C.).

    - Accuracy: mean L1 distance from each pred point to nearest GT point.
    - Completeness: mean L1 distance from each GT point to nearest pred point.
    - Normal Consistency: mean |dot(n_pred, n_gt)| at corresponding nearest points.
    """
    if len(pc_pred) > max_points:
        idx = np.random.choice(len(pc_pred), max_points, replace=False)
        pc_pred = pc_pred[idx]
    if len(pc_gt) > max_points:
        idx = np.random.choice(len(pc_gt), max_points, replace=False)
        pc_gt = pc_gt[idx]

    from scipy.spatial import cKDTree
    tree_gt = cKDTree(pc_gt)
    tree_pred = cKDTree(pc_pred)
    d_pred2gt, idx_pred2gt = tree_gt.query(pc_pred)
    d_gt2pred, idx_gt2pred = tree_pred.query(pc_gt)

    accuracy = float(np.mean(d_pred2gt))
    completeness = float(np.mean(d_gt2pred))

    # Normal consistency
    normals_pred = _estimate_normals(pc_pred)
    normals_gt = _estimate_normals(pc_gt)
    # pred->gt direction: dot normals at corresponding nearest points
    nc_pred2gt = np.abs(np.sum(normals_pred * normals_gt[idx_pred2gt], axis=1))
    nc_gt2pred = np.abs(np.sum(normals_gt * normals_pred[idx_gt2pred], axis=1))
    normal_consistency = float((np.mean(nc_pred2gt) + np.mean(nc_gt2pred)) / 2)

    return {
        "Acc": accuracy,
        "Comp": completeness,
        "NC": normal_consistency,
    }


# ---------------------------------------------------------------------------
# Intrinsics Metrics
# ---------------------------------------------------------------------------

def intrinsics_error(K_pred: np.ndarray, K_gt: np.ndarray) -> Dict[str, float]:
    """Relative error on focal length and principal point."""
    fx_err = abs(K_pred[0, 0] - K_gt[0, 0]) / K_gt[0, 0]
    fy_err = abs(K_pred[1, 1] - K_gt[1, 1]) / K_gt[1, 1]
    cx_err = abs(K_pred[0, 2] - K_gt[0, 2]) / K_gt[0, 2]
    cy_err = abs(K_pred[1, 2] - K_gt[1, 2]) / K_gt[1, 2]
    return {
        "fx_rel_err": float(fx_err),
        "fy_rel_err": float(fy_err),
        "cx_rel_err": float(cx_err),
        "cy_rel_err": float(cy_err),
        "focal_rel_err": float((fx_err + fy_err) / 2),
    }
