#!/usr/bin/env python3
"""Run SIGMA on real-world datasets (TUM-dynamics, DTU, ETH3D) and
report metrics. Optionally swap in a LoRA-tuned Pi3X via --lora-checkpoint.

Usage:
    # TUM-dynamics, pose
    python script/eval_realworld.py --dataset tum --root data/realworld/tum_dynamics

    # DTU, point cloud
    python script/eval_realworld.py --dataset dtu --root /DATA/dtu_test/spann3r/dtu_test_mvsnet_release \
        --max-scenes 5

    # ETH3D, point cloud (needs read perms on /DATA/eth3d)
    python script/eval_realworld.py --dataset eth3d --root /DATA/eth3d --max-scenes 5

    # With LoRA
    python script/eval_realworld.py --dataset tum --root data/realworld/tum_dynamics \
        --lora-checkpoint checkpoints/pi3_lora_static_depth.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.data.realworld_loaders import (  # noqa: E402
    load_tum_dynamics, load_dtu, load_eth3d, RealWorldSequence, FrameData,
)


def discover_sequences(dataset: str, root: Path) -> list[Path]:
    if dataset == "tum":
        return sorted(p for p in root.iterdir() if p.is_dir() and "rgbd_dataset" in p.name)
    if dataset == "dtu":
        return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("scan"))
    if dataset == "eth3d":
        return sorted(p for p in root.iterdir() if p.is_dir())
    raise ValueError(dataset)


def load_sequence(dataset: str, seq_path: Path, max_frames: int, stride: int) -> RealWorldSequence:
    if dataset == "tum":
        return load_tum_dynamics(seq_path, max_frames=max_frames, stride=stride)
    if dataset == "dtu":
        return load_dtu(seq_path, max_frames=max_frames, stride=stride)
    if dataset == "eth3d":
        return load_eth3d(seq_path, max_frames=max_frames, stride=stride)
    raise ValueError(dataset)


def build_pipeline(args, lora_checkpoint: str | None):
    from script.eval_batch import build_pipeline as _bp

    pipeline_args = SimpleNamespace(
        dataset_root=Path("StaDy4D"),  # config requires it; not used at inference
        duration="short", split="test", data_format="dynamic",
        reconstruction="pi3", inpainting="blank",
    )
    extras = [
        "pipeline.reconstruction.use_origin_frames=true",
        "pipeline.reconstruction.fill_masked_depth=true",
        "pipeline.reconstruction.mask_dilate_px=3",
        f"data.max_frames={args.max_frames}",
    ]
    if lora_checkpoint:
        extras.append(f"++pipeline.reconstruction.lora_checkpoint={lora_checkpoint}")
    return _bp(pipeline_args, extras)


def run_on_sequence(runner, cfg, seq: RealWorldSequence, out_dir: Path):
    """Feed a RealWorldSequence into the pipeline and save per-frame outputs."""
    from sigma.data.frame_record import FrameIORecord
    from sigma.pipeline.reconstruction.aggregator import SceneAggregator
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    runner.frame_records = {}
    if hasattr(runner.reconstructor, "aggregator"):
        runner.reconstructor.aggregator = SceneAggregator()
    if hasattr(runner.reconstructor, "_global_poses"):
        runner.reconstructor._global_poses = {}
    if hasattr(runner.reconstructor, "_pending_warmup"):
        runner.reconstructor._pending_warmup = {}

    for i, fd in enumerate(seq.frames):
        runner.frame_records[i] = FrameIORecord(frame_idx=i, origin_image=fd.rgb)

    # Skip motion + inpainting on real-world benchmarks: those stages target
    # CARLA-style moving cars, but on TUM/DTU/ETH3D they spuriously fire
    # (e.g. detect "car" on Buddha statues), zeroing huge swaths of the frame
    # and corrupting Pi3 reconstruction. Pi3 paper's protocol feeds raw images.

    # Phase 5: reconstruction
    pred_per_frame = []
    if runner.reconstructor.run_deferred:
        pred_per_frame = list(runner.reconstructor.finalize(runner.frame_records))
    else:
        for fidx in sorted(runner.frame_records.keys()):
            pred_per_frame.append((fidx, runner.reconstructor.process(runner.frame_records, fidx)))
        if hasattr(runner.reconstructor, "post_process_fill"):
            masks = {f: r.mask for f, r in runner.frame_records.items() if r.mask is not None}
            runner.reconstructor.post_process_fill(runner.frame_records, pred_per_frame, masks)

    # Save per-frame outputs
    depth_dir = out_dir / "depth"; depth_dir.mkdir(exist_ok=True)
    ext_dir = out_dir / "extrinsics"; ext_dir.mkdir(exist_ok=True)
    intr_dir = out_dir / "intrinsics"; intr_dir.mkdir(exist_ok=True)
    rgb_dir = out_dir / "rgb"; rgb_dir.mkdir(exist_ok=True)
    for fidx, recon_result in pred_per_frame:
        assets = recon_result.visualization_assets
        if assets.get("depth_map") is not None:
            np.save(depth_dir / f"depth_{fidx:04d}.npy", assets["depth_map"])
        np.save(ext_dir / f"extrinsic_{fidx:04d}.npy", assets["extrinsic"])
        np.save(intr_dir / f"intrinsic_{fidx:04d}.npy", assets["intrinsic"])
        rec = runner.frame_records.get(fidx)
        if rec is not None and rec.origin_image is not None:
            imageio.imwrite(str(rgb_dir / f"rgb_{fidx:04d}.png"), rec.origin_image)

    # Also save the global cloud if available
    if pred_per_frame:
        last_assets = pred_per_frame[-1][1].visualization_assets
        gpc = last_assets.get("global_point_cloud")
        if gpc is not None:
            np.save(out_dir / "global_point_cloud.npy", gpc)
        gpcc = last_assets.get("global_point_color")
        if gpcc is not None:
            np.save(out_dir / "global_point_color.npy", gpcc)

    return pred_per_frame


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_tum_pose(seq: RealWorldSequence, pred_per_frame: list) -> dict:
    """ATE + RPE between pred c2w trajectory and TUM GT trajectory + per-pixel
    depth abs_rel against the Kinect GT depth (LoRA-sensitive metric)."""
    gt_pose_full = []
    for f in seq.frames:
        c = np.eye(4); c[:3, :3] = f.c2w[:3, :3]; c[:3, 3] = f.c2w[:3, 3]
        gt_pose_full.append(c)
    gt_t = np.stack([p[:3, 3] for p in gt_pose_full])

    pred_t_list, pred_pose_full = [], []
    for _, r in pred_per_frame:
        ext = r.data["reconstruction"].extrinsic
        if ext is None: continue
        c = np.eye(4)
        if ext.shape == (3, 4):
            c[:3, :] = ext
        else:
            c = np.asarray(ext, dtype=np.float64)
        pred_t_list.append(c[:3, 3])
        pred_pose_full.append(c)
    pred_t = np.stack(pred_t_list[: len(gt_t)])
    pred_pose_full = pred_pose_full[: len(gt_t)]
    gt_t = gt_t[: len(pred_t)]
    gt_pose_full = gt_pose_full[: len(pred_t)]

    # Procrustes (Sim3) align pred → gt
    pc = pred_t - pred_t.mean(0); gc = gt_t - gt_t.mean(0)
    H = pc.T @ gc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    pred_var = float(np.sum(pc * pc))
    scale = float(np.sum(S) / pred_var) if pred_var > 1e-9 else 1.0
    t = gt_t.mean(0) - scale * (R @ pred_t.mean(0))
    pred_aligned = scale * (R @ pred_t.T).T + t
    err = np.linalg.norm(pred_aligned - gt_t, axis=1)

    # ---- RPE (relative pose error) — TUM RGB-D benchmark ----
    # Apply Sim(3) (s,R,t) to entire pred pose: c2w' = [s R | t] @ c2w
    sim3 = np.eye(4); sim3[:3, :3] = scale * R; sim3[:3, 3] = t
    pred_aligned_full = [sim3 @ P for P in pred_pose_full]
    rpe_t, rpe_r = [], []
    for i in range(len(pred_aligned_full) - 1):
        # GT relative motion
        rel_gt = np.linalg.inv(gt_pose_full[i]) @ gt_pose_full[i + 1]
        rel_pr = np.linalg.inv(pred_aligned_full[i]) @ pred_aligned_full[i + 1]
        E = np.linalg.inv(rel_gt) @ rel_pr
        rpe_t.append(float(np.linalg.norm(E[:3, 3])))
        cos_th = (np.trace(E[:3, :3]) - 1.0) / 2.0
        cos_th = float(np.clip(cos_th, -1.0, 1.0))
        rpe_r.append(float(np.degrees(np.arccos(cos_th))))

    # ---- Depth abs_rel against TUM Kinect GT ----
    import cv2
    rels = []
    for (idx, r), gt_frame in zip(pred_per_frame, seq.frames):
        recon = r.data["reconstruction"]
        if recon.depth_map is None or gt_frame.depth is None:
            continue
        pred = np.asarray(recon.depth_map, dtype=np.float32)
        gt = np.asarray(gt_frame.depth, dtype=np.float32)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        valid = (gt > 0.1) & (gt < 5.0) & np.isfinite(pred) & (pred > 0.01)
        if int(valid.sum()) < 100:
            continue
        gt_med = np.median(gt[valid])
        pr_med = np.median(pred[valid])
        if gt_med < 1e-3 or pr_med < 1e-3:
            continue
        pred_aligned_d = pred * (gt_med / pr_med)
        rel = np.abs(pred_aligned_d[valid] - gt[valid]) / gt[valid]
        rels.append(float(rel.mean()))

    out = {
        "ATE_mean": float(err.mean()),
        "ATE_median": float(np.median(err)),
        "ATE_rmse": float(np.sqrt((err * err).mean())),
        "alignment_scale": scale,
        "n_frames": int(len(pred_t)),
    }
    if rpe_t:
        rpe_t_arr = np.array(rpe_t); rpe_r_arr = np.array(rpe_r)
        out["RPE_t_mean"] = float(rpe_t_arr.mean())
        out["RPE_t_rmse"] = float(np.sqrt((rpe_t_arr ** 2).mean()))
        out["RPE_r_mean_deg"] = float(rpe_r_arr.mean())
        out["RPE_r_rmse_deg"] = float(np.sqrt((rpe_r_arr ** 2).mean()))
    if rels:
        out["depth_abs_rel_mean"] = float(np.mean(rels))
        out["depth_abs_rel_median"] = float(np.median(rels))
    return out


def evaluate_pointcloud_pi3_protocol(seq: RealWorldSequence, pred_per_frame: list,
                                      icp_threshold: float = 0.1) -> dict:
    """Pi3's MV-recon protocol: dense point correspondences via Umeyama Sim(3).

    For each frame: build per-pixel (3D) gt point from gt depth + K + c2w, and
    pred point from pred depth + same K + same c2w.  Stack across frames; Sim(3)
    align using only valid pixels (gt > 0 and finite).  Then Chamfer.

    Mirrors third_party/pi3_eval/mv_recon/eval.py + utils.umeyama.
    """
    pred_pts_per_frame = []
    gt_pts_per_frame = []
    valid_per_frame = []

    for (idx, r), gt_frame in zip(pred_per_frame, seq.frames):
        recon = r.data["reconstruction"]
        if recon.depth_map is None or recon.intrinsic is None or recon.extrinsic is None:
            continue
        if gt_frame.depth is None:
            continue
        pred_depth = np.asarray(recon.depth_map, dtype=np.float32)
        gt_depth = np.asarray(gt_frame.depth, dtype=np.float32)

        # GT camera params (for GT unprojection)
        K_gt = np.asarray(gt_frame.K, dtype=np.float64)
        ext_gt = np.asarray(gt_frame.c2w, dtype=np.float64)
        if ext_gt.shape == (3, 4):
            c2w_gt = np.eye(4); c2w_gt[:3, :] = ext_gt
        else:
            c2w_gt = ext_gt

        # PRED camera params (for pred unprojection — Pi3 official protocol)
        K_pred = np.asarray(recon.intrinsic, dtype=np.float64)
        ext_pred = np.asarray(recon.extrinsic, dtype=np.float64)
        if ext_pred.shape == (3, 4):
            c2w_pred = np.eye(4); c2w_pred[:3, :] = ext_pred
        else:
            c2w_pred = ext_pred

        # Resize pred_depth to GT shape so the per-pixel valid masks line up.
        if pred_depth.shape != gt_depth.shape:
            import cv2
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            # Rescale K_pred to match the resized pred_depth resolution
            sx = gt_depth.shape[1] / recon.depth_map.shape[1]
            sy = gt_depth.shape[0] / recon.depth_map.shape[0]
            K_pred = K_pred.copy()
            K_pred[0, 0] *= sx; K_pred[0, 2] *= sx
            K_pred[1, 1] *= sy; K_pred[1, 2] *= sy

        H, W = gt_depth.shape
        v, u = np.indices((H, W))
        u = u.astype(np.float64); v = v.astype(np.float64)

        def unproject(d, K, c2w):
            x = (u - K[0, 2]) * d / K[0, 0]
            y = (v - K[1, 2]) * d / K[1, 1]
            cam = np.stack([x, y, d], axis=-1)
            cam_h = np.concatenate([cam, np.ones_like(d)[..., None]], axis=-1)
            return cam_h @ c2w.T  # (H, W, 4)

        gt_world = unproject(gt_depth.astype(np.float64), K_gt, c2w_gt)[..., :3]
        pred_world = unproject(pred_depth.astype(np.float64), K_pred, c2w_pred)[..., :3]
        valid = (gt_depth > 1e-4) & np.isfinite(gt_depth) & np.isfinite(pred_depth) \
                  & (pred_depth > 1e-4)
        gt_pts_per_frame.append(gt_world)
        pred_pts_per_frame.append(pred_world)
        valid_per_frame.append(valid)

    if not gt_pts_per_frame:
        return {"error": "no GT depth available"}

    gt_pts = np.stack(gt_pts_per_frame, axis=0)        # (N, H, W, 3)
    pred_pts = np.stack(pred_pts_per_frame, axis=0)    # (N, H, W, 3)
    valid_mask = np.stack(valid_per_frame, axis=0)     # (N, H, W)

    # Umeyama Sim(3) on point pairs (Pi3 protocol).
    P = pred_pts[valid_mask].T  # (3, M)
    G = gt_pts[valid_mask].T    # (3, M)
    if P.shape[1] < 100:
        return {"error": "too few valid point pairs"}
    mu_p = P.mean(axis=1, keepdims=True)
    mu_g = G.mean(axis=1, keepdims=True)
    var_p = float(np.sum((P - mu_p) ** 2)) / P.shape[1]
    cov = ((G - mu_g) @ (P - mu_p).T) / P.shape[1]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    c = float(np.trace(np.diag(D) @ S) / max(var_p, 1e-12))
    R = U @ S @ Vt
    t = mu_g - c * R @ mu_p

    pred_aligned = c * np.einsum("nhwj,ij->nhwi", pred_pts, R) + t.T  # (N, H, W, 3)
    pred_valid = pred_aligned[valid_mask]
    gt_valid = gt_pts[valid_mask]

    # Subsample for tractability
    rng = np.random.default_rng(0)
    if len(gt_valid) > 200000:
        sel = rng.choice(len(gt_valid), 200000, replace=False)
        gt_valid_sub = gt_valid[sel]; pred_valid_sub = pred_valid[sel]
    else:
        gt_valid_sub, pred_valid_sub = gt_valid, pred_valid

    # ---- Pi3 protocol: ICP refinement + normals → NC ----
    import open3d as o3d
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_valid_sub)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_valid_sub)

    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd_pred = pcd_pred.transform(reg.transformation)
    pcd_pred.estimate_normals()
    pcd_gt.estimate_normals()
    pred_pts_icp = np.asarray(pcd_pred.points)
    gt_pts_icp = np.asarray(pcd_gt.points)
    pred_normals = np.asarray(pcd_pred.normals)
    gt_normals = np.asarray(pcd_gt.normals)

    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_pts_icp)
    tree_pred = cKDTree(pred_pts_icp)
    d_pred, idx_p2g = tree_gt.query(pred_pts_icp, workers=-1)   # pred → nearest GT
    d_gt, idx_g2p = tree_pred.query(gt_pts_icp, workers=-1)     # GT → nearest pred

    # NC1: pred normal vs GT-nearest normal; NC2: GT normal vs pred-nearest normal.
    nc1 = float(np.mean(np.abs(np.sum(gt_normals[idx_p2g] * pred_normals, axis=-1))))
    nc1_med = float(np.median(np.abs(np.sum(gt_normals[idx_p2g] * pred_normals, axis=-1))))
    nc2 = float(np.mean(np.abs(np.sum(gt_normals * pred_normals[idx_g2p], axis=-1))))
    nc2_med = float(np.median(np.abs(np.sum(gt_normals * pred_normals[idx_g2p], axis=-1))))

    return {
        "Acc_mean": float(d_pred.mean()),
        "Comp_mean": float(d_gt.mean()),
        "Acc_median": float(np.median(d_pred)),
        "Comp_median": float(np.median(d_gt)),
        "NC_mean": float((nc1 + nc2) / 2),
        "NC_median": float((nc1_med + nc2_med) / 2),
        "NC1_mean": nc1, "NC2_mean": nc2,
        "Chamfer": float(0.5 * (d_pred.mean() + d_gt.mean())),
        "alignment_scale": float(c),
        "alignment_method": "umeyama+icp",
        "icp_threshold": float(icp_threshold),
        "n_valid_pts": int(P.shape[1]),
    }


def evaluate_pointcloud(seq: RealWorldSequence, pred_per_frame: list, gt_ply: Path) -> dict:
    """Chamfer distance between pred cloud (built from per-frame depth) and GT ply.

    Uses camera-trajectory Sim(3) alignment when GT poses are available
    (much better than centroid-only).  Falls back to cloud-centroid alignment.
    """
    try:
        import open3d as o3d
        gt_pc = np.asarray(o3d.io.read_point_cloud(str(gt_ply)).points)
    except Exception:
        from plyfile import PlyData
        gt_data = PlyData.read(str(gt_ply))
        gt_pc = np.stack([gt_data["vertex"][a] for a in ("x", "y", "z")], axis=-1)

    # Build pred cloud (in pred world frame)
    pred_pts = []
    pred_c2w_list = []
    for (idx, r), gt_frame in zip(pred_per_frame, seq.frames):
        recon = r.data["reconstruction"]
        if recon.depth_map is None or recon.intrinsic is None or recon.extrinsic is None:
            continue
        depth = np.asarray(recon.depth_map, dtype=np.float32)
        K = np.asarray(recon.intrinsic, dtype=np.float64)
        ext = np.asarray(recon.extrinsic, dtype=np.float64)
        if ext.shape == (3, 4):
            c2w = np.eye(4); c2w[:3, :] = ext
        else:
            c2w = ext
        H, W = depth.shape
        valid = (depth > 0.1) & np.isfinite(depth)
        v, u = np.indices((H, W))
        v = v[valid].astype(np.float64); u = u[valid].astype(np.float64)
        z = depth[valid].astype(np.float64)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        cam = np.stack([x, y, z], axis=1)
        pts_h = np.hstack([cam, np.ones((cam.shape[0], 1))])
        world = (c2w @ pts_h.T).T[:, :3]
        pred_pts.append(world)
        pred_c2w_list.append(c2w)

    if not pred_pts:
        return {"error": "no pred points"}
    pred_pc = np.concatenate(pred_pts, axis=0).astype(np.float32)

    # ---- Try camera-trajectory Sim(3) alignment (proper) ----
    s_align = R_align = t_align = None
    gt_translations = []
    pred_translations = []
    for fd, c2w in zip(seq.frames[:len(pred_c2w_list)], pred_c2w_list):
        if fd.c2w is not None:
            gt_translations.append(np.asarray(fd.c2w[:3, 3], dtype=np.float64))
            pred_translations.append(np.asarray(c2w[:3, 3], dtype=np.float64))
    if len(gt_translations) >= 3:
        gt_t = np.stack(gt_translations); pr_t = np.stack(pred_translations)
        pc, gc = pr_t - pr_t.mean(0), gt_t - gt_t.mean(0)
        H_ = pc.T @ gc
        U, S, Vt = np.linalg.svd(H_)
        R_a = Vt.T @ U.T
        if np.linalg.det(R_a) < 0:
            Vt[-1] *= -1; R_a = Vt.T @ U.T
        var = float(np.sum(pc * pc))
        if var > 1e-9:
            s_align = float(np.sum(S) / var)
            t_align = gt_t.mean(0) - s_align * (R_a @ pr_t.mean(0))
            R_align = R_a

    if R_align is not None:
        pred_aligned = (s_align * (R_align @ pred_pc.T)).T + t_align
        align_method = "camera-traj-sim3"
    else:
        pred_c, gt_c = pred_pc.mean(0), gt_pc.mean(0)
        pred_centred = pred_pc - pred_c; gt_centred = gt_pc - gt_c
        pred_scale = float(np.linalg.norm(pred_centred, axis=1).mean()) + 1e-9
        gt_scale = float(np.linalg.norm(gt_centred, axis=1).mean()) + 1e-9
        s_align = gt_scale / pred_scale
        pred_aligned = pred_centred * s_align + gt_c
        align_method = "centroid-scale"

    # Subsample for tractability
    rng = np.random.default_rng(0)
    if len(gt_pc) > 200000:
        gt_pc = gt_pc[rng.choice(len(gt_pc), 200000, replace=False)]
    if len(pred_aligned) > 200000:
        pred_aligned = pred_aligned[rng.choice(len(pred_aligned), 200000, replace=False)]

    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_pc)
    tree_pred = cKDTree(pred_aligned)
    d_pred, _ = tree_gt.query(pred_aligned)
    d_gt, _ = tree_pred.query(gt_pc)
    return {
        "Acc_mean": float(d_pred.mean()),
        "Comp_mean": float(d_gt.mean()),
        "Acc_median": float(np.median(d_pred)),
        "Comp_median": float(np.median(d_gt)),
        "Chamfer": float(0.5 * (d_pred.mean() + d_gt.mean())),
        "alignment_scale": float(s_align),
        "alignment_method": align_method,
        "n_pred_pts": int(len(pred_aligned)),
        "n_gt_pts": int(len(gt_pc)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["tum", "dtu", "eth3d"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out-root", type=Path, default=Path("eval_results/realworld"))
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-scenes", type=int, default=5)
    p.add_argument("--lora-checkpoint", type=str, default=None)
    args = p.parse_args()

    seq_paths = discover_sequences(args.dataset, args.root)[: args.max_scenes]
    print(f"Found {len(seq_paths)} sequences for {args.dataset}")
    out_root = args.out_root / args.dataset / ("lora" if args.lora_checkpoint else "zeroshot")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    runner, cfg = build_pipeline(args, args.lora_checkpoint)

    rows = []
    for seq_path in seq_paths:
        try:
            t0 = time.monotonic()
            print(f"\n=== {seq_path.name} ===", flush=True)
            seq = load_sequence(args.dataset, seq_path, args.max_frames, args.stride)
            print(f"  loaded {len(seq.frames)} frames")
            out_dir = out_root / seq_path.name
            pred = run_on_sequence(runner, cfg, seq, out_dir)
            t_pipe = time.monotonic() - t0

            row = {"sequence": seq_path.name, "n_frames": len(seq.frames),
                   "t_pipe_s": round(t_pipe, 2)}
            if args.dataset == "tum":
                row.update(evaluate_tum_pose(seq, pred))
            elif args.dataset == "dtu":
                # Use Pi3's protocol: per-image GT depth + Umeyama on dense
                # point pairs.  Falls back to fused PLY only if no per-image
                # depth was loaded (legacy path).
                if any(f.depth is not None for f in seq.frames):
                    row.update(evaluate_pointcloud_pi3_protocol(seq, pred, icp_threshold=100.0))
                else:
                    ply = sorted(seq_path.glob("*_pcd.ply"))
                    if ply:
                        row.update(evaluate_pointcloud(seq, pred, ply[0]))
                    else:
                        row["note"] = f"no GT depth or ply in {seq_path}"
            elif args.dataset == "eth3d":
                # ETH3D GT = per-image float32 depth; use Pi3's mv_recon
                # protocol with Umeyama Sim(3) on dense point correspondences.
                row.update(evaluate_pointcloud_pi3_protocol(seq, pred, icp_threshold=0.1))
            rows.append(row)
            print(f"  {row}", flush=True)
            with open(out_root / f"{seq_path.name}.json", "w") as f:
                json.dump(row, f, indent=2)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    summary = {"dataset": args.dataset, "n": len(rows), "lora": args.lora_checkpoint, "rows": rows}
    summary_path = out_root / "summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
