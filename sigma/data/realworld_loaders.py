"""Adapters that convert real-world datasets (TUM-dynamics, DTU, ETH3D)
into the (origin RGB, optional GT depth, optional GT pose) tuples the
SIGMA pipeline expects.

For TUM-dynamics: video-style sequence with per-frame depth+pose GT.
For DTU / ETH3D:  multi-view stills with calibration + sparse/dense GT.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class FrameData:
    rgb: np.ndarray                  # (H, W, 3) uint8
    depth: Optional[np.ndarray]      # (H, W) float32 metric metres, or None
    K: Optional[np.ndarray]          # (3, 3) intrinsics
    c2w: Optional[np.ndarray]        # (4, 4) camera-to-world (OpenCV convention)
    timestamp: Optional[float]       # seconds, for TUM


@dataclass
class RealWorldSequence:
    name: str
    frames: List[FrameData]
    pose_convention: str = "opencv"  # "opencv" or "opengl"


# ---------------------------------------------------------------------------
# TUM-dynamics
# ---------------------------------------------------------------------------

def load_tum_dynamics(seq_dir: Path, max_frames: int = 30,
                      stride: int = 5) -> RealWorldSequence:
    """Load a TUM RGB-D Dataset sequence (e.g. fr3 walking_xyz).

    TUM provides RGB at ~30Hz, depth at 30Hz (Kinect raw), and groundtruth
    pose at 100Hz from a Vicon mocap.  We pick frames at the given stride
    and find the closest depth/pose by timestamp.

    Args:
        seq_dir: e.g. data/realworld/tum_dynamics/rgbd_dataset_freiburg3_walking_xyz
        max_frames: cap on returned frame count.
        stride: take every Nth RGB frame.
    """
    import imageio.v2 as imageio

    rgb_list = (seq_dir / "rgb.txt").read_text().splitlines()
    depth_list = (seq_dir / "depth.txt").read_text().splitlines()
    gt_list = (seq_dir / "groundtruth.txt").read_text().splitlines()

    def _parse(lines: list[str]):
        rows: list[tuple[float, str]] = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            try:
                rows.append((float(parts[0]), " ".join(parts[1:])))
            except ValueError:
                continue
        return rows

    rgb_rows = _parse(rgb_list)
    depth_rows = _parse(depth_list)
    gt_rows = _parse(gt_list)
    gt_t = np.array([r[0] for r in gt_rows])

    # TUM fr3 intrinsics (Kinect color, factory calibration)
    fx, fy = 535.4, 539.2
    cx, cy = 320.1, 247.6
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def closest(rows_t: np.ndarray, t: float) -> int:
        return int(np.argmin(np.abs(rows_t - t)))

    def quat_to_rotmat(qx, qy, qz, qw):
        # tx ty tz qx qy qz qw → R, t (OpenCV convention)
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ], dtype=np.float32)
        return R

    frames: List[FrameData] = []
    chosen_rgb = rgb_rows[::stride][:max_frames]
    depth_t_arr = np.array([r[0] for r in depth_rows])
    for t_rgb, rgb_path in chosen_rgb:
        rgb = imageio.imread(seq_dir / rgb_path)
        # Depth
        d_idx = closest(depth_t_arr, t_rgb)
        depth_path = seq_dir / depth_rows[d_idx][1]
        depth_raw = imageio.imread(depth_path)
        depth = depth_raw.astype(np.float32) / 5000.0   # TUM scale factor
        depth[depth == 0] = np.nan
        # Pose
        g_idx = closest(gt_t, t_rgb)
        parts = gt_rows[g_idx][1].split()
        tx, ty, tz, qx, qy, qz, qw = map(float, parts[:7])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
        c2w[:3, 3] = [tx, ty, tz]

        frames.append(FrameData(rgb=np.asarray(rgb), depth=depth, K=K, c2w=c2w,
                                 timestamp=t_rgb))

    return RealWorldSequence(name=seq_dir.name, frames=frames, pose_convention="opencv")


# ---------------------------------------------------------------------------
# ETH3D (multi-view DSLR)
# ---------------------------------------------------------------------------

def load_eth3d(scene_dir: Path, max_frames: int = 30,
               stride: int = 1) -> RealWorldSequence:
    """Load an ETH3D high-res multi-view scene.

    Supports three layouts in this preference order:
      0. Per-image npz cache (custom_undistorted) — cleanest:
         <scene>/images/custom_undistorted/<name>.JPG
         <scene>/custom_undistorted_cam/<name>.npz   (keys: intrinsics, extrinsics)
      1. Undistorted COLMAP (typical 7z release):
         <scene>/images/dslr_images_undistorted/*.JPG
         <scene>/dslr_calibration_undistorted/cameras.txt + images.txt
      2. JPG distorted COLMAP (Pi3X assumes pinhole; will be approximate):
         <scene>/images/dslr_images/*.JPG
         <scene>/dslr_calibration_jpg/cameras.txt + images.txt
    """
    import imageio.v2 as imageio

    # ---------- Layout 0: per-image npz ----------
    cam_npz_dir = scene_dir / "custom_undistorted_cam"
    img_undist_dir = scene_dir / "images" / "custom_undistorted"
    gt_depth_dir = scene_dir / "ground_truth_depth" / "custom_undistorted"
    if cam_npz_dir.is_dir() and img_undist_dir.is_dir():
        img_paths = sorted(img_undist_dir.glob("*.JPG")) + sorted(img_undist_dir.glob("*.jpg"))

        # ---- Pi3 protocol: pick keyframes from seq-id-map if available ----
        seq_id_map_path = (Path(__file__).resolve().parent.parent.parent
                            / "third_party" / "pi3_eval" / "datasets"
                            / "seq-id-maps" / "ETH3D_mv-recon_seq-id-map-kf5.json")
        chosen_ids = None
        if seq_id_map_path.exists():
            import json
            seq_map = json.load(open(seq_id_map_path))
            if scene_dir.name in seq_map:
                chosen_ids = seq_map[scene_dir.name]
                if chosen_ids:
                    img_paths = [img_paths[i] for i in chosen_ids if i < len(img_paths)]

        if chosen_ids is None:
            img_paths = img_paths[::stride][:max_frames]

        frames: List[FrameData] = []
        # ETH3D images are 4032×6048 — way too big for GroundedSAM/Pi3X
        # without pre-resize.  Resize the longer edge to MAX_LONG; rescale K.
        MAX_LONG = 1024
        import cv2
        for p in img_paths:
            cam_npz = cam_npz_dir / f"{p.stem}.npz"
            if not cam_npz.exists():
                continue
            cam = np.load(cam_npz)
            K = np.asarray(cam["intrinsics"], dtype=np.float32)
            ext = np.asarray(cam["extrinsics"], dtype=np.float32)
            # ETH3D npz extrinsics: world-to-camera (Pi3 unprojects with this)
            if ext.shape == (4, 4):
                c2w = np.linalg.inv(ext)
            else:
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :] = ext

            rgb = np.asarray(imageio.imread(p))
            H, W = rgb.shape[:2]

            # Load GT depth if available (binary float32, .JPG-named).
            depth = None
            depth_path = gt_depth_dir / p.name
            if depth_path.exists():
                try:
                    raw = np.fromfile(depth_path, dtype=np.float32)
                    if raw.size == H * W:
                        depth = raw.reshape(H, W)
                        depth[~np.isfinite(depth)] = -1.0
                except Exception:
                    depth = None

            # Resize for memory.
            long_edge = max(H, W)
            if long_edge > MAX_LONG:
                scale = MAX_LONG / long_edge
                new_W, new_H = int(round(W * scale)), int(round(H * scale))
                rgb = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
                K = K.copy()
                K[0, 0] *= new_W / W; K[0, 2] *= new_W / W
                K[1, 1] *= new_H / H; K[1, 2] *= new_H / H
                if depth is not None:
                    depth = cv2.resize(depth, (new_W, new_H),
                                        interpolation=cv2.INTER_NEAREST)
            frames.append(FrameData(rgb=rgb, depth=depth, K=K, c2w=c2w,
                                     timestamp=None))
        if frames:
            return RealWorldSequence(name=scene_dir.name, frames=frames,
                                       pose_convention="opencv")

    # ---------- Layout 1/2: COLMAP cameras.txt + images.txt ----------
    candidates = [
        ("dslr_calibration_undistorted", "dslr_images_undistorted"),
        ("dslr_calibration_jpg", "dslr_images"),
    ]
    cams_file = imgs_file = img_dir = None
    for cal_dir_name, img_subdir in candidates:
        c = scene_dir / cal_dir_name / "cameras.txt"
        i = scene_dir / cal_dir_name / "images.txt"
        d = scene_dir / "images" / img_subdir
        if c.exists() and i.exists() and d.exists():
            cams_file, imgs_file, img_dir = c, i, d
            break

    if cams_file is None:
        raise FileNotFoundError(f"Missing ETH3D calibration files in {scene_dir}")

    # Parse cameras.txt -> intrinsics by camera_id
    K_by_cam: dict[int, np.ndarray] = {}
    for ln in cams_file.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        cam_id = int(parts[0])
        # PINHOLE model: w h fx fy cx cy
        # SIMPLE_PINHOLE: f cx cy
        w, h = int(parts[2]), int(parts[3])
        if parts[1] == "PINHOLE":
            fx, fy, cx, cy = map(float, parts[4:8])
        elif parts[1] == "SIMPLE_PINHOLE":
            f = float(parts[4]); cx, cy = float(parts[5]), float(parts[6])
            fx = fy = f
        else:
            continue
        K_by_cam[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                     dtype=np.float32)

    # Parse images.txt -> per-image (qw qx qy qz tx ty tz cam_id name)
    # Two lines per image (second line = points2D) — skip the second.
    img_entries: list[tuple[str, np.ndarray, int]] = []
    lines = [ln for ln in imgs_file.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        # Convert quaternion+translation (world-to-camera) to camera-to-world
        # ETH3D/COLMAP convention: P_cam = R P_world + t
        R = _quat_wxyz_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float32)
        # c2w = inv([R|t]) -> R^T, -R^T t
        Rt = R.T
        tt = -Rt @ t
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = Rt; c2w[:3, 3] = tt
        img_entries.append((name, c2w, cam_id))

    # Sample
    img_entries = img_entries[::stride][:max_frames]

    frames: List[FrameData] = []
    for name, c2w, cam_id in img_entries:
        rgb_path = img_dir / name
        if not rgb_path.exists():
            # ETH3D nests by camera_id sometimes
            cands = list(img_dir.rglob(Path(name).name))
            if cands:
                rgb_path = cands[0]
            else:
                continue
        rgb = imageio.imread(rgb_path)
        K = K_by_cam.get(cam_id)
        frames.append(FrameData(rgb=np.asarray(rgb), depth=None, K=K, c2w=c2w,
                                 timestamp=None))

    return RealWorldSequence(name=scene_dir.name, frames=frames, pose_convention="opencv")


def _quat_wxyz_to_R(qw, qx, qy, qz) -> np.ndarray:
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# DTU MVS (rectified)
# ---------------------------------------------------------------------------

def load_dtu(scene_dir: Path, max_frames: int = 30,
             stride: int = 1) -> RealWorldSequence:
    """Load a DTU MVS scene from the standard MVSNet-style rectified layout.

    Loads per-image depth from <scene>/depths/<id>.npy too, so the Pi3-style
    point-pair Sim(3) protocol can be used for evaluation.

    Honours Pi3's seq-id-map (kf5) when available so we evaluate the same
    keyframes the paper does.
    """
    import imageio.v2 as imageio

    img_dir = scene_dir / "images"
    cam_dir = scene_dir / "cams"
    depth_dir = scene_dir / "depths"
    mask_dir = scene_dir / "binary_masks"
    if not (img_dir.is_dir() and cam_dir.is_dir()):
        raise FileNotFoundError(f"Unrecognised DTU layout in {scene_dir}")

    # Pi3 kf5 seq-id-map
    seq_id_map_path = (Path(__file__).resolve().parent.parent.parent
                        / "third_party" / "pi3_eval" / "datasets"
                        / "seq-id-maps" / "DTU_mv-recon_seq-id-map-kf5.json")
    chosen_ids = None
    if seq_id_map_path.exists():
        import json
        seq_map = json.load(open(seq_id_map_path))
        if scene_dir.name in seq_map:
            chosen_ids = seq_map[scene_dir.name]

    rgb_all = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if chosen_ids is not None:
        rgb_files = [rgb_all[i] for i in chosen_ids if i < len(rgb_all)]
    else:
        rgb_files = rgb_all[::stride][:max_frames]

    frames: List[FrameData] = []
    for rgb_path in rgb_files:
        stem = rgb_path.stem
        cam_file = cam_dir / f"{stem}_cam.txt"
        if not cam_file.exists():
            continue
        txt = cam_file.read_text().splitlines()
        ext_lines = [ln.split() for ln in txt[1:5]]
        ext = np.array([[float(x) for x in row] for row in ext_lines],
                       dtype=np.float32)               # world-to-camera (MVSNet)
        intr_lines = [ln.split() for ln in txt[7:10]]
        K = np.array([[float(x) for x in row] for row in intr_lines],
                     dtype=np.float32)
        c2w = np.linalg.inv(ext)
        rgb = np.asarray(imageio.imread(rgb_path))

        # Per-image GT depth — DTU has it as numpy float32 (1200, 1600).
        depth = None
        depth_path = depth_dir / f"{stem}.npy"
        if depth_path.exists():
            try:
                depth = np.load(depth_path).astype(np.float32)
                depth = np.nan_to_num(depth, 0.0)
            except Exception:
                depth = None

        # Apply Pi3's binary-mask + 10x10 erosion to remove background.
        if depth is not None:
            mask_path = mask_dir / f"{stem}.png"
            if mask_path.exists():
                import cv2 as _cv2
                m = _cv2.imread(str(mask_path), _cv2.IMREAD_UNCHANGED)
                if m is not None:
                    m = (m / 255.0).astype(np.float32)
                    m = (m > 0.5).astype(np.float32)
                    if m.shape != depth.shape:
                        m = _cv2.resize(m, (depth.shape[1], depth.shape[0]),
                                          interpolation=_cv2.INTER_NEAREST)
                    kernel = np.ones((10, 10), np.uint8)
                    m = _cv2.erode(m, kernel, iterations=1)
                    depth = depth * m

        frames.append(FrameData(rgb=rgb, depth=depth, K=K, c2w=c2w, timestamp=None))

    return RealWorldSequence(name=scene_dir.name, frames=frames,
                               pose_convention="opencv")
