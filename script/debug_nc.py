#!/usr/bin/env python3
"""Recalculate NC step-by-step on a worst-NC scene to diagnose the issue."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import safetensors.torch as st
sys.path.insert(0, ".")
from sigma.evaluation.metrics import compute_pose_alignment, align_pose, _estimate_normals


def back_project(depth, K, ext, max_d=80.0):
    """depth: (H,W); K: (3,3); ext: c2w (4,4) or (3,4) — return (N,3) world points (skip sky)."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(W); v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    valid = np.isfinite(depth) & (depth > 0.5) & (depth < max_d)
    d = depth[valid]
    x_cam = (uu[valid] - cx) * d / fx
    y_cam = (vv[valid] - cy) * d / fy
    z_cam = d
    cam_pts = np.stack([x_cam, y_cam, z_cam], axis=-1)
    R = ext[:3, :3]; t = ext[:3, 3]
    world = (R @ cam_pts.T).T + t
    return world


def main():
    scene = "scene_T03_020"
    cam = "cam_01_car_forward"
    pred_root = Path(f"outputs/sigma_worst_nc/{scene}/{cam}")
    gt_root = Path(f"StaDy4D/short/test/{scene}/dynamic/{cam}")

    N = 50
    pred_d = np.stack([np.load(pred_root / "depth" / f"depth_{i:04d}.npy") for i in range(N)]).astype(np.float32)
    pred_K = np.stack([np.load(pred_root / "intrinsics" / f"intrinsic_{i:04d}.npy") for i in range(N)])
    pred_e = np.stack([np.load(pred_root / "extrinsics" / f"extrinsic_{i:04d}.npy") for i in range(N)])
    pred_e_4x4 = []
    for e in pred_e:
        m = np.eye(4); m[:3, :4] = e; pred_e_4x4.append(m)
    pred_e_4x4 = np.stack(pred_e_4x4)

    gt_d = st.load_file(str(gt_root / "depth.safetensors"))["depth"].numpy()[:N].astype(np.float32)
    gt_K_all = st.load_file(str(gt_root / "intrinsics.safetensors"))["K"].numpy()[:N]
    gt_e = st.load_file(str(gt_root / "extrinsics.safetensors"))["c2w"].numpy()[:N]

    print(f"=== {scene}/{cam} ===")
    print(f"pred K[0]:\n{pred_K[0]}\npred E[0,:3,:]:\n{pred_e_4x4[0,:3]}")
    print(f"GT K[0]:\n{gt_K_all[0]}\nGT E[0,:3,:]:\n{gt_e[0,:3]}")

    # Compute alignment from poses
    scale, R_align, t_align = compute_pose_alignment(list(pred_e_4x4), list(gt_e))
    print(f"\nPose-Procrustes: scale={scale:.4f}  degenerate={scale==1.0}")
    if scale == 1.0:
        # Use depth-based rescale
        ratios = []
        for i in range(N):
            v = (pred_d[i] > 0.5) & (pred_d[i] < 80) & (gt_d[i] > 0.5) & (gt_d[i] < 80)
            if v.sum() > 100:
                ratios.append(np.median(gt_d[i][v]) / np.median(pred_d[i][v]))
        if ratios:
            scale = float(np.median(ratios))
            pred_pos = np.array([p[:3, 3] for p in pred_e_4x4])
            gt_pos = np.array([p[:3, 3] for p in gt_e])
            t_align = gt_pos.mean(0) - scale * R_align @ pred_pos.mean(0)
        print(f"Depth-refined scale={scale:.4f}")

    # Build clouds
    pred_pts_all, gt_pts_all = [], []
    for i in range(N):
        ext_aligned = align_pose(pred_e_4x4[i], scale, R_align, t_align)
        p = back_project(pred_d[i], pred_K[i], ext_aligned, max_d=80)
        g = back_project(gt_d[i], gt_K_all[i], gt_e[i], max_d=80)
        # subsample stride 8
        if len(p) > 5000: p = p[::len(p)//5000][:5000]
        if len(g) > 5000: g = g[::len(g)//5000][:5000]
        pred_pts_all.append(p); gt_pts_all.append(g)
    pred_pc = np.concatenate(pred_pts_all)
    gt_pc = np.concatenate(gt_pts_all)
    print(f"\npred cloud: {len(pred_pc)} pts, range Z: {pred_pc[:,2].min():.1f} to {pred_pc[:,2].max():.1f}")
    print(f"GT cloud  : {len(gt_pc)} pts, range Z: {gt_pc[:,2].min():.1f} to {gt_pc[:,2].max():.1f}")
    print(f"pred mean: {pred_pc.mean(0)}")
    print(f"GT  mean: {gt_pc.mean(0)}")

    # NC computation
    from scipy.spatial import cKDTree
    np.random.seed(0)
    if len(pred_pc) > 50000:
        pred_pc = pred_pc[np.random.choice(len(pred_pc), 50000, replace=False)]
    if len(gt_pc) > 50000:
        gt_pc = gt_pc[np.random.choice(len(gt_pc), 50000, replace=False)]

    print("\nEstimating normals...")
    n_pred = _estimate_normals(pred_pc)
    n_gt = _estimate_normals(gt_pc)
    tree_gt = cKDTree(gt_pc); tree_pred = cKDTree(pred_pc)
    d_p2g, idx_p2g = tree_gt.query(pred_pc)
    d_g2p, idx_g2p = tree_pred.query(gt_pc)

    nc_p2g = np.abs(np.sum(n_pred * n_gt[idx_p2g], axis=1))
    nc_g2p = np.abs(np.sum(n_gt * n_pred[idx_g2p], axis=1))
    print(f"\nAcc (pred->gt): {d_p2g.mean():.3f}  (lower better)")
    print(f"Comp (gt->pred): {d_g2p.mean():.3f}")
    print(f"NC: {(nc_p2g.mean() + nc_g2p.mean()) / 2:.4f}")
    print(f"  nc_p2g mean: {nc_p2g.mean():.4f}, median: {np.median(nc_p2g):.4f}")
    print(f"  nc_g2p mean: {nc_g2p.mean():.4f}, median: {np.median(nc_g2p):.4f}")
    print(f"  nc_p2g distribution: <0.3 = {(nc_p2g<0.3).mean()*100:.1f}%, >0.9 = {(nc_p2g>0.9).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
