"""StaDy4D dataset analysis: camera trajectories + dynamic-actor distribution.

Outputs (under --out):
  trajectories/<scene>/stacked.png           all cameras overlaid (top-down)
  trajectories/<scene>/<cam>.png             one figure per camera
  actors/<scene>/world_xy.png                 all-actor world (CARLA xy) scatter
  actors/<scene>/visible_xy.png               actors that fall inside any cam frustum
  actors/<scene>/occupancy_<cam>.png          per-camera 2D occupancy heatmap
  summary.json                                aggregated counts

Convention notes (see CLAUDE.md):
  c2w stored in OpenGL world frame: World X = CARLA y, World Y = CARLA z (up),
  World Z = CARLA x (forward). Camera follows OpenGL: x=right, y=up, z=back.
  Top-down plots use CARLA xy directly (z is height).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patheffects import withStroke
from safetensors.numpy import load_file


# ---------------------------------------------------------------------------
# Research-style global aesthetics
# ---------------------------------------------------------------------------

# Okabe–Ito palette (colorblind-safe, widely used in publications)
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#5A5A5A",
}
PALETTE_LIST = list(PALETTE.values())


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi":            140,
        "savefig.dpi":           220,
        "savefig.bbox":          "tight",
        "font.family":           "DejaVu Sans",   # widely available, clean
        "font.size":             10.5,
        "axes.titlesize":        12,
        "axes.titleweight":      "semibold",
        "axes.labelsize":        10.5,
        "axes.labelweight":      "medium",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.edgecolor":        "#222222",
        "axes.linewidth":        0.9,
        "axes.titlepad":         9,
        "axes.labelpad":         5,
        "axes.grid":             True,
        "axes.axisbelow":        True,
        "grid.color":            "#BFBFBF",
        "grid.linestyle":        "--",
        "grid.linewidth":        0.55,
        "grid.alpha":            0.6,
        "xtick.color":           "#222222",
        "ytick.color":           "#222222",
        "xtick.labelsize":       9.5,
        "ytick.labelsize":       9.5,
        "xtick.major.size":      3,
        "ytick.major.size":      3,
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        "legend.frameon":        True,
        "legend.framealpha":     0.92,
        "legend.edgecolor":      "#CCCCCC",
        "legend.fontsize":       9.0,
        "legend.title_fontsize": 9.5,
        "figure.titlesize":      13,
        "figure.titleweight":    "bold",
    })


_apply_style()


def _annotate_bars(ax, bars, fmt: str = "{:.1f}", offset: float = 0.5,
                   color: str = "#222222") -> None:
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va="bottom", fontsize=8, color=color,
                path_effects=[withStroke(linewidth=2.0, foreground="white")])


def _hide_top_right(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cam_pose_carla_xy(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions_xy, forwards_xy) in CARLA ground-plane frame.

    c2w: (N, 4, 4) OpenGL convention with axes mapped CARLA->OpenGL as
    (x_carla, y_carla, z_carla) -> (Z_world, X_world, Y_world).
    """
    pos_x = c2w[:, 2, 3]   # CARLA x  <- world Z
    pos_y = c2w[:, 0, 3]   # CARLA y  <- world X
    # camera forward in OpenGL world = -Z_cam = -c2w[:,:3,2]
    fwd_x = -c2w[:, 2, 2]
    fwd_y = -c2w[:, 0, 2]
    return np.stack([pos_x, pos_y], axis=1), np.stack([fwd_x, fwd_y], axis=1)


def carla_world_to_opengl(p_carla: np.ndarray) -> np.ndarray:
    """(..., 3) CARLA xyz -> the c2w-world frame (OpenCV, Y-down).

    Empirically (and matching ``visualize.py``'s unprojection): the c2w used
    in StaDy4D is the OpenCV camera convention (X right, Y down, Z forward)
    in a world frame whose axes map to CARLA as:
        world X =  CARLA y
        world Y = -CARLA z   (Y is down, CARLA z is up → opposite sign)
        world Z =  CARLA x
    """
    out = np.empty_like(p_carla)
    out[..., 0] =  p_carla[..., 1]
    out[..., 1] = -p_carla[..., 2]
    out[..., 2] =  p_carla[..., 0]
    return out


def project_points(points_world: np.ndarray, c2w: np.ndarray, K: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV pinhole projection (X right, Y down, Z forward).

    Returns (uv (M,2), valid_mask (M,)) where valid means in front of the
    camera. Out-of-image filtering is left to the caller.
    """
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pc = points_world @ R.T + t
    z_eye = pc[:, 2]
    in_front = z_eye > 1e-3
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * pc[:, 0] / np.where(in_front, z_eye, 1.0) + cx
    v = fy * pc[:, 1] / np.where(in_front, z_eye, 1.0) + cy
    uv = np.stack([u, v], axis=1)
    return uv, in_front


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scene(scene_dir: Path) -> dict:
    meta = json.loads((scene_dir / "metadata.json").read_text())
    actors = json.loads((scene_dir / "actors.json").read_text())

    # ego/attached flag per camera id (e.g. "camera_00_car_forward")
    attached = {}
    for ext in meta.get("initial_extrinsics", []):
        attached[ext.get("camera", "")] = bool(ext.get("is_attached", False))

    cams_root = scene_dir / "dynamic"
    if not cams_root.exists():
        cams_root = scene_dir / "static"
    cam_dirs = sorted(d for d in cams_root.iterdir() if d.is_dir() and d.name.startswith("cam_"))
    cameras = {}
    for d in cam_dirs:
        c2w = load_file(str(d / "extrinsics.safetensors"))["c2w"].astype(np.float64)
        K = load_file(str(d / "intrinsics.safetensors"))["K"].astype(np.float64)
        # match cam_NN_* against camera_NN_* in metadata
        meta_key = "camera_" + d.name.removeprefix("cam_")
        cameras[d.name] = {"c2w": c2w, "K": K, "dir": d,
                            "is_ego": attached.get(meta_key, False)}
    return {"meta": meta, "actors": actors, "cameras": cameras}


# ---------------------------------------------------------------------------
# Plot pieces
# ---------------------------------------------------------------------------

def _add_camera_triangles(ax, pos_xy: np.ndarray, fwd_xy: np.ndarray,
                          color, scale: float, alpha: float = 0.7) -> None:
    """Draw a small triangle at each camera position pointing along forward."""
    fwd = fwd_xy / (np.linalg.norm(fwd_xy, axis=1, keepdims=True) + 1e-9)
    side = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    tip = pos_xy + fwd * scale
    base_l = pos_xy - fwd * (scale * 0.4) + side * (scale * 0.5)
    base_r = pos_xy - fwd * (scale * 0.4) - side * (scale * 0.5)
    polys = np.stack([tip, base_l, base_r], axis=1)
    ax.add_collection(PolyCollection(polys, facecolors=color, edgecolors=color,
                                     alpha=alpha, linewidths=0.4))


def plot_trajectory_stacked(scene: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    all_pos = []
    for i, (name, cam) in enumerate(scene["cameras"].items()):
        pos, fwd = cam_pose_carla_xy(cam["c2w"])
        all_pos.append(pos)
        color = cmap(i % 10)
        ax.plot(pos[:, 0], pos[:, 1], "-", color=color, lw=1.0, alpha=0.6)
        _add_camera_triangles(ax, pos[::max(1, len(pos) // 12)],
                              fwd[::max(1, len(fwd) // 12)],
                              color, scale=_auto_scale(np.concatenate(all_pos)))
        ax.scatter(pos[0, 0], pos[0, 1], color=color, marker="o", s=30,
                   edgecolors="k", linewidths=0.5, label=name, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title(f"Camera trajectories — {scene['meta'].get('map_name','')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_per_camera(scene: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cam in scene["cameras"].items():
        pos, fwd = cam_pose_carla_xy(cam["c2w"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pos[:, 0], pos[:, 1], "-", color="C0", lw=1.2, alpha=0.7)
        scale = _auto_scale(pos)
        _add_camera_triangles(ax, pos, fwd, "C3", scale=scale, alpha=0.8)
        ax.scatter(pos[0, 0], pos[0, 1], color="green", s=40, label="start",
                   edgecolors="k", zorder=3)
        ax.scatter(pos[-1, 0], pos[-1, 1], color="red", s=40, label="end",
                   edgecolors="k", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x  (m)")
        ax.set_ylabel("y  (m)")
        ax.set_title(f"{name}  ({len(pos)} frames)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def _auto_scale(pos: np.ndarray) -> float:
    span = np.ptp(pos, axis=0).max() if len(pos) > 1 else 1.0
    return float(max(span * 0.02, 0.5))


# ---------------------------------------------------------------------------
# Actor / occupancy plots
# ---------------------------------------------------------------------------

def collect_actor_locations(actors: dict) -> dict:
    """Return per-frame world-xyz arrays grouped by track type (vehicle/walker)."""
    n_frames = actors["num_frames"]
    by_type: dict[str, list[np.ndarray]] = {"vehicle": [], "walker": []}
    per_frame: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
    for tr in actors["tracks"]:
        if tr.get("is_ego"):
            continue
        kind = "vehicle" if tr["type_id"].startswith("vehicle") else "walker"
        for f in tr["frames"]:
            loc = f["location"]
            p = np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
            by_type[kind].append(p)
            fi = f["frame"]
            if 0 <= fi < n_frames:
                per_frame[fi].append(p)
    by_type = {k: (np.stack(v) if v else np.zeros((0, 3))) for k, v in by_type.items()}
    return {"by_type": by_type, "per_frame": per_frame}


def plot_world_actors(scene: dict, actor_data: dict, out_path: Path) -> None:
    by_type = actor_data["by_type"]
    fig, ax = plt.subplots(figsize=(8, 8))
    # also show camera trajectories faintly for context
    for name, cam in scene["cameras"].items():
        pos, _ = cam_pose_carla_xy(cam["c2w"])
        ax.plot(pos[:, 0], pos[:, 1], "-", color="0.6", lw=0.6, alpha=0.5)
    if len(by_type["vehicle"]):
        ax.scatter(by_type["vehicle"][:, 0], by_type["vehicle"][:, 1],
                   s=4, c="C0", alpha=0.4, label=f"vehicle ({len(by_type['vehicle'])})")
    if len(by_type["walker"]):
        ax.scatter(by_type["walker"][:, 0], by_type["walker"][:, 1],
                   s=4, c="C3", alpha=0.4, label=f"walker ({len(by_type['walker'])})")
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title(f"Actor world distribution — {scene['meta'].get('map_name','')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_camera_occupancy(scene: dict, actor_data: dict, out_dir: Path
                          ) -> dict:
    """For each camera, project actors per frame and accumulate a 2D occupancy
    heatmap in image coordinates. Returns per-camera visible counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    W = scene["meta"]["resolution"]["width"]
    H = scene["meta"]["resolution"]["height"]
    bins = (64, 48)  # ~ W/10, H/10
    counts = {}
    visible_world = []  # (x_carla, y_carla) of any actor projected into any cam

    for name, cam in scene["cameras"].items():
        c2w = cam["c2w"]
        K = cam["K"]
        n_frames = min(len(c2w), len(actor_data["per_frame"]))
        heat = np.zeros(bins[::-1], dtype=np.float64)  # (H_bins, W_bins)
        n_visible = 0
        for fi in range(n_frames):
            pts = actor_data["per_frame"][fi]
            if not pts:
                continue
            pts = np.stack(pts)
            uv, in_front = project_points(carla_world_to_opengl(pts), c2w[fi],
                                          K[fi] if K.ndim == 3 else K)
            inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                              & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.any():
                u = uv[inside, 0]
                v = uv[inside, 1]
                hh, _, _ = np.histogram2d(v, u, bins=bins[::-1],
                                          range=[[0, H], [0, W]])
                heat += hh
                n_visible += int(inside.sum())
                visible_world.append(pts[inside])
        counts[name] = n_visible
        # plot heatmap
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(heat, origin="upper", extent=[0, W, H, 0],
                       cmap="magma", aspect="auto")
        ax.set_title(f"{name}\nactor occupancy (sum over frames) — {n_visible} hits")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        fig.colorbar(im, ax=ax, label="actor-frame count")
        fig.tight_layout()
        fig.savefig(out_dir / f"occupancy_{name}.png", dpi=150)
        plt.close(fig)

    # plot world-xy of actors that were visible in any camera
    if visible_world:
        vw = np.concatenate(visible_world)
        fig, ax = plt.subplots(figsize=(8, 8))
        for name, cam in scene["cameras"].items():
            pos, _ = cam_pose_carla_xy(cam["c2w"])
            ax.plot(pos[:, 0], pos[:, 1], "-", color="0.6", lw=0.6, alpha=0.5)
        ax.scatter(vw[:, 0], vw[:, 1], s=3, c="C2", alpha=0.4,
                   label=f"visible-in-any-cam ({len(vw)})")
        ax.set_aspect("equal")
        ax.set_xlabel("x  (m)")
        ax.set_ylabel("y  (m)")
        ax.set_title("Visible actor world distribution")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "visible_xy.png", dpi=150)
        plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# RGB overlays with oriented 3D bounding boxes
# ---------------------------------------------------------------------------

_BBOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),    # bottom face
               (4, 5), (5, 7), (7, 6), (6, 4),    # top face
               (0, 4), (1, 5), (2, 6), (3, 7)]    # verticals


def _actor_corners_world_carla(center: np.ndarray, ext: np.ndarray,
                               yaw_deg: float,
                               bbox_center: np.ndarray | None = None
                               ) -> np.ndarray:
    """Return 8 corner positions in CARLA world coords (oriented by yaw).

    `center` is the actor's world location (wheel base for vehicles).
    `bbox_center` is the bbox-center offset in the actor's local frame
    (e.g. vehicles have z ≈ half height because the location is at ground).
    """
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    signs = np.array([[sx, syg, sz] for sx in (-1, 1) for syg in (-1, 1)
                      for sz in (-1, 1)], dtype=np.float64)
    local = signs * ext
    if bbox_center is not None:
        local = local + bbox_center
    return center + (R @ local.T).T


def _decode_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Decode requested frame indices from an mp4 file. Returns {idx: RGB}."""
    import cv2
    out = {}
    cap = cv2.VideoCapture(str(path))
    target = sorted(set(indices))
    cur = 0
    nxt = 0
    while nxt < len(target):
        ok, frame = cap.read()
        if not ok:
            break
        if cur == target[nxt]:
            out[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            nxt += 1
        cur += 1
    cap.release()
    return out


def aggregate_dataset_dynamic_occupancy(scene_dirs: list[Path], out_path: Path,
                                        thresh_m: float = 0.3,
                                        max_depth: float = 80.0,
                                        device: str | None = None,
                                        target_hw: tuple[int, int] = (480, 640)
                                        ) -> dict[str, dict[str, float]]:
    """Single dataset-wide overlay: per camera id, accumulate the frequency
    of pixels where depth(dynamic) != depth(static) > thresh_m, across every
    scene/frame, then overlay the heatmap on a representative RGB.

    Output is one PNG (3×3 grid for the 9-camera StaDy4D layout).
    """
    import torch
    import cv2
    from tqdm import tqdm
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = target_hw

    # cam_id -> {"dyn": (H,W) float, "tot": (H,W) float, "rgb": uint8 H,W,3}
    accum: dict[str, dict] = {}
    per_scene: dict[str, dict[str, float]] = {}   # scene_name -> cam -> ratio

    use_fp16 = (device == "cuda")
    pin = (device == "cuda")
    dtype_t = torch.float16 if use_fp16 else torch.float32

    from concurrent.futures import ThreadPoolExecutor

    def _load_pair(d_dir: Path, s_dir: Path):
        d_path = d_dir / "depth.safetensors"
        s_path = s_dir / "depth.safetensors"
        if not (d_path.exists() and s_path.exists()):
            return None
        try:
            d_np = load_file(str(d_path))["depth"]
            s_np = load_file(str(s_path))["depth"]
        except Exception:
            return None
        if d_np.shape != s_np.shape or d_np.size == 0:
            return None
        td = torch.from_numpy(d_np)
        ts = torch.from_numpy(s_np)
        if pin:
            td = td.pin_memory(); ts = ts.pin_memory()
        return d_dir, s_dir, td, ts

    pairs = []
    for sd in scene_dirs:
        dyn_root = sd / "dynamic"; sta_root = sd / "static"
        if not (dyn_root.exists() and sta_root.exists()):
            continue
        for d_dir in sorted(dyn_root.iterdir()):
            if not (d_dir.is_dir() and d_dir.name.startswith("cam_")):
                continue
            s_dir = sta_root / d_dir.name
            if s_dir.exists():
                pairs.append((sd, d_dir, s_dir))

    pool = ThreadPoolExecutor(max_workers=4)
    in_flight = []
    PREFETCH = 8
    pair_iter = iter(pairs)
    def _submit_next():
        try:
            sd, d_dir, s_dir = next(pair_iter)
        except StopIteration:
            return False
        in_flight.append((sd, pool.submit(_load_pair, d_dir, s_dir)))
        return True
    for _ in range(PREFETCH):
        if not _submit_next():
            break

    pbar = tqdm(total=len(pairs), desc="agg dynamic-vs-static")
    while in_flight:
        sd, fut = in_flight.pop(0)
        _submit_next()
        loaded = fut.result()
        pbar.update(1)
        if loaded is None:
            continue
        d_dir, s_dir, td, ts = loaded
        d_dyn = td.to(device, non_blocking=pin).to(dtype_t)
        d_sta = ts.to(device, non_blocking=pin).to(dtype_t)
        if d_dyn.shape[-2:] != (H, W):
            d_dyn = torch.nn.functional.interpolate(
                d_dyn[:, None].float(), size=(H, W), mode="nearest")[:, 0].to(dtype_t)
            d_sta = torch.nn.functional.interpolate(
                d_sta[:, None].float(), size=(H, W), mode="nearest")[:, 0].to(dtype_t)
        valid = (d_dyn > 0.05) & (d_dyn < max_depth) \
              & (d_sta > 0.05) & (d_sta < max_depth)
        mask = ((d_dyn - d_sta).abs() > thresh_m) & valid
        dyn_sum = mask.to(torch.float32).sum(0)
        tot_sum = valid.to(torch.float32).sum(0)
        denom = valid.to(torch.float32).reshape(len(valid), -1).sum(1).clamp(min=1)
        ratio = (mask.to(torch.float32).reshape(len(mask), -1).sum(1) / denom).mean()
        # one combined H2D-back (smaller payloads)
        dyn_sum_np = dyn_sum.cpu().numpy()
        tot_sum_np = tot_sum.cpu().numpy()
        ratio_f    = float(ratio.cpu().item())
        per_scene.setdefault(sd.name, {})[d_dir.name] = ratio_f
        slot = accum.setdefault(d_dir.name,
                                {"dyn": np.zeros((H, W), np.float64),
                                 "tot": np.zeros((H, W), np.float64),
                                 "rgb": None})
        slot["dyn"] += dyn_sum_np
        slot["tot"] += tot_sum_np
        if slot["rgb"] is None:
            cap = cv2.VideoCapture(str(d_dir / "rgb.mp4"))
            ok, frame = cap.read(); cap.release()
            if ok:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))
                slot["rgb"] = img
    pbar.close()
    pool.shutdown(wait=True)

    if not accum:
        print("[overlay] no dynamic/static pairs found")
        return per_scene

    cams = sorted(accum.keys())
    import math
    cols = min(3, len(cams))
    rows = math.ceil(len(cams) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.9 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    # use a global vmax (95th percentile across all panels) for comparability
    global_vals = np.concatenate([
        (slot["dyn"] / np.maximum(slot["tot"], 1)).ravel()
        for slot in accum.values()])
    vmax = max(0.02, float(np.percentile(global_vals, 99)))

    last_im = None
    for i, cam in enumerate(cams):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        slot = accum[cam]
        freq = slot["dyn"] / np.maximum(slot["tot"], 1)
        if slot["rgb"] is not None:
            ax.imshow(slot["rgb"])
        im = ax.imshow(freq, cmap="inferno", alpha=0.78,
                       vmin=0, vmax=vmax)
        last_im = im
        # camera label inside the panel
        ax.text(0.02, 0.97, cam.replace("cam_", ""),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="white", fontweight="bold",
                path_effects=[withStroke(linewidth=2.5, foreground="black")])
        # mean dyn ratio bottom-right
        ax.text(0.98, 0.04, f"mean = {freq.mean()*100:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="white",
                path_effects=[withStroke(linewidth=2.5, foreground="black")])

    # one shared horizontal colorbar at the bottom
    cax = fig.add_axes([0.25, -0.015, 0.5, 0.018])
    cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cb.set_label("Per-pixel dynamic-occupancy frequency",
                 fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)

    fig.suptitle("Dataset-wide dynamic-pixel occupancy", y=1.005)
    fig.text(0.5, 0.985,
             rf"$|\,\mathrm{{depth}}_{{\mathrm{{dyn}}}} - "
             rf"\mathrm{{depth}}_{{\mathrm{{static}}}}\,| > {thresh_m:g}\,\mathrm{{m}}$"
             f"   ·   {len(scene_dirs)} scenes",
             ha="center", va="bottom", fontsize=10.5,
             color=PALETTE["grey"])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return per_scene


# ---------------------------------------------------------------------------
# Dynamics analysis
# ---------------------------------------------------------------------------

def actor_speeds(actors: dict, fps: float) -> dict:
    """Per-actor mean speed (m/s) from CARLA velocity field, grouped by type.

    Falls back to finite-difference of `location` if velocity is missing.
    """
    out = {"vehicle": [], "walker": []}
    for tr in actors["tracks"]:
        if tr.get("is_ego"):
            continue
        kind = "vehicle" if tr["type_id"].startswith("vehicle") else "walker"
        speeds = []
        prev_loc = None
        for f in tr["frames"]:
            v = f.get("velocity")
            if v is not None and any(abs(v[k]) > 1e-9 for k in ("x", "y", "z")):
                speeds.append(np.linalg.norm([v["x"], v["y"], v["z"]]))
            elif prev_loc is not None:
                loc = f["location"]
                speeds.append(np.linalg.norm([
                    loc["x"] - prev_loc["x"],
                    loc["y"] - prev_loc["y"],
                    loc["z"] - prev_loc["z"],
                ]) * fps)
            prev_loc = f["location"]
        if speeds:
            out[kind].append(float(np.mean(speeds)))
    return {k: np.array(v) for k, v in out.items()}


def per_camera_static_dyn_ratio(scene_dir: Path, thresh_m: float = 0.3,
                                max_depth: float = 80.0,
                                device: str | None = None) -> dict:
    """For each camera return the fraction of pixels (averaged over frames)
    where |depth_dynamic - depth_static| > thresh — the GT dynamic mask."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    dyn_root = scene_dir / "dynamic"
    sta_root = scene_dir / "static"
    if not (dyn_root.exists() and sta_root.exists()):
        return out
    for d_dir in sorted(dyn_root.iterdir()):
        if not (d_dir.is_dir() and d_dir.name.startswith("cam_")):
            continue
        s_dir = sta_root / d_dir.name
        if not s_dir.exists():
            continue
        try:
            d = torch.from_numpy(load_file(str(d_dir / "depth.safetensors"))
                                  ["depth"].astype(np.float32)).to(device)
            s = torch.from_numpy(load_file(str(s_dir / "depth.safetensors"))
                                  ["depth"].astype(np.float32)).to(device)
        except Exception:
            continue
        if d.shape != s.shape or d.numel() == 0:
            continue
        valid = (d > 0.05) & (d < max_depth) & (s > 0.05) & (s < max_depth)
        moving = ((d - s).abs() > thresh_m) & valid
        denom = valid.reshape(len(valid), -1).sum(1).clamp(min=1)
        ratio = moving.reshape(len(moving), -1).sum(1) / denom
        out[d_dir.name] = float(ratio.mean().cpu())
    return out


def per_camera_depth_diff(scene: dict, max_depth: float = 80.0,
                          dynamic_thresh: float = 0.5) -> dict:
    """For each camera return:
        mean_abs_diff: mean |Δdepth| between consecutive frames (m, valid pixels)
        dynamic_ratio: fraction of valid pixels with |Δdepth| > dynamic_thresh

    The ratio is a depth-based proxy for the dynamic-pixel mask: pixels whose
    depth changes substantially between frames are likely on a moving object
    or revealed/occluded by one (camera ego-motion still contributes some
    background change, so this overestimates real dynamic content but works
    as a relative measure).
    """
    out = {"mean_abs_diff": {}, "dynamic_ratio": {}}
    for name, cam in scene["cameras"].items():
        cam_dir = cam.get("dir")
        if cam_dir is None:
            continue
        try:
            d = load_file(str(cam_dir / "depth.safetensors"))["depth"].astype(np.float32)
        except Exception:
            continue
        if len(d) < 2:
            out["mean_abs_diff"][name] = 0.0
            out["dynamic_ratio"][name] = 0.0
            continue
        valid = (d > 0.05) & (d < max_depth)
        v = valid[:-1] & valid[1:]
        delta = np.abs(d[1:] - d[:-1])
        out["mean_abs_diff"][name] = float(delta[v].mean()) if v.any() else 0.0
        # per-pair dynamic ratio, then mean over pairs
        moving = (delta > dynamic_thresh) & v
        denom = v.reshape(len(v), -1).sum(axis=1).clip(min=1)
        ratio = moving.reshape(len(v), -1).sum(axis=1) / denom
        out["dynamic_ratio"][name] = float(ratio.mean())
    return out


def ego_speeds_per_camera(scene: dict, fps: float) -> dict:
    """Per ego-camera mean speed (m/s) and yaw-rate (deg/s) from extrinsics."""
    out_speed, out_yaw = {}, {}
    for name, cam in scene["cameras"].items():
        if not cam.get("is_ego"):
            continue
        c2w = cam["c2w"]
        if len(c2w) < 2:
            continue
        pos = np.stack([c2w[:, 2, 3], c2w[:, 0, 3], c2w[:, 1, 3]], axis=1)
        step = np.linalg.norm(pos[1:] - pos[:-1], axis=1) * fps     # m/s
        out_speed[name] = float(step.mean())
        # forward vector in CARLA xy (yaw in ground plane)
        fwd_xy = np.stack([-c2w[:, 2, 2], -c2w[:, 0, 2]], axis=1)
        n = np.linalg.norm(fwd_xy, axis=1, keepdims=True).clip(min=1e-9)
        fwd_xy = fwd_xy / n
        cosang = np.clip((fwd_xy[1:] * fwd_xy[:-1]).sum(axis=1), -1, 1)
        yaw_step = np.degrees(np.arccos(cosang)) * fps              # deg/s
        out_yaw[name] = float(yaw_step.mean())
    return {"speed_mps": out_speed, "yaw_dps": out_yaw}


def per_camera_actor_coverage(scene: dict, actor_data: dict) -> dict:
    """Approximate fraction of image pixels covered by actor bboxes per camera.

    Uses each actor's `bbox_extent` projected at the actor's world location
    (8 corners in CARLA frame), takes the convex hull (axis-aligned rect)
    of the projected corners. Returns mean coverage over frames in [0,1].
    """
    W = scene["meta"]["resolution"]["width"]
    H = scene["meta"]["resolution"]["height"]
    img_area = float(W * H)
    # build per-frame list of (location, extent, bbox_center, yaw)
    n_frames = scene["actors"]["num_frames"]
    per_frame: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = [
        [] for _ in range(n_frames)]
    for tr in scene["actors"]["tracks"]:
        if tr.get("is_ego"):
            continue
        ext = tr.get("bbox_extent", {"x": 0.5, "y": 0.5, "z": 0.5})
        e = np.array([ext["x"], ext["y"], ext["z"]], dtype=np.float64)
        if e.max() < 1e-3:
            e = np.array([0.4, 0.4, 0.9])
        bc = tr.get("bbox_center", {"x": 0.0, "y": 0.0, "z": 0.0})
        bcv = np.array([bc["x"], bc["y"], bc["z"]], dtype=np.float64)
        for f in tr["frames"]:
            fi = f["frame"]
            if 0 <= fi < n_frames:
                loc = f["location"]
                per_frame[fi].append((
                    np.array([loc["x"], loc["y"], loc["z"]]),
                    e, bcv, float(f["rotation"]["yaw"])))

    out = {}
    for name, cam in scene["cameras"].items():
        c2w = cam["c2w"]
        K = cam["K"]
        n = min(len(c2w), n_frames)
        cov_per_frame = []
        for fi in range(n):
            pf = per_frame[fi]
            if not pf:
                cov_per_frame.append(0.0)
                continue
            area = 0.0
            Ki = K[fi] if K.ndim == 3 else K
            for center, ext, bcv, yaw in pf:
                corners = _actor_corners_world_carla(center, ext, yaw, bcv)
                uv, in_front = project_points(carla_world_to_opengl(corners),
                                              c2w[fi], Ki)
                if in_front.sum() < 2:
                    continue
                uv = uv[in_front]
                u0, v0 = uv[:, 0].min(), uv[:, 1].min()
                u1, v1 = uv[:, 0].max(), uv[:, 1].max()
                u0, u1 = max(0.0, u0), min(float(W), u1)
                v0, v1 = max(0.0, v0), min(float(H), v1)
                if u1 > u0 and v1 > v0:
                    area += (u1 - u0) * (v1 - v0)
            cov_per_frame.append(min(1.0, area / img_area))
        out[name] = float(np.mean(cov_per_frame)) if cov_per_frame else 0.0
    return out


def plot_dynamics_bars(scene_name: str, speeds: dict, depth_diff: dict,
                       coverage: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) actor speed distribution
    ax = axes[0]
    data = [speeds.get("vehicle", np.array([])), speeds.get("walker", np.array([]))]
    parts = ax.boxplot([d for d in data if len(d)], showfliers=False,
                       labels=[lbl for d, lbl in zip(data, ["vehicle", "walker"]) if len(d)],
                       patch_artist=True)
    for p, c in zip(parts["boxes"], ["C0", "C3"]):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_ylabel("mean speed per actor (m/s)")
    ax.set_title("Actor speed distribution")
    ax.grid(axis="y", alpha=0.3)

    # (b) per-camera mean |Δdepth|
    ax = axes[1]
    cams = list(depth_diff.keys())
    vals = [depth_diff[c] for c in cams]
    ax.bar(range(len(cams)), vals, color="C2", alpha=0.7)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([c.replace("cam_", "") for c in cams], rotation=40, ha="right",
                       fontsize=8)
    ax.set_ylabel("mean |Δdepth| (m)")
    ax.set_title("Per-camera frame-to-frame depth change")
    ax.grid(axis="y", alpha=0.3)

    # (c) per-camera actor pixel coverage
    ax = axes[2]
    cams = list(coverage.keys())
    vals = [coverage[c] * 100 for c in cams]
    ax.bar(range(len(cams)), vals, color="C1", alpha=0.7)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([c.replace("cam_", "") for c in cams], rotation=40, ha="right",
                       fontsize=8)
    ax.set_ylabel("mean actor bbox coverage (%)")
    ax.set_title("Per-camera dynamic-actor pixel coverage")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Dynamics — {scene_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_scene_stacked(summaries: list[dict], out_path: Path) -> None:
    """Single stacked bar chart: each scene = one bar, layers = the four
    dynamics metrics rescaled to [0,1] across scenes (so they're comparable)."""
    rows = [s for s in summaries if "dynamics" in s]
    if not rows:
        return
    names = [r["scene"].replace("scene_", "") for r in rows]

    def _safe(arr):
        a = np.asarray(arr, dtype=np.float64)
        m = a.max() if a.size and a.max() > 0 else 1.0
        return a / m

    veh = _safe([np.mean(r["dynamics"]["actor_speed"]["vehicle"] or [0]) for r in rows])
    wlk = _safe([np.mean(r["dynamics"]["actor_speed"]["walker"] or [0]) for r in rows])
    dpt = _safe([np.mean(list(r["dynamics"]["depth_diff"].values()) or [0]) for r in rows])
    cov = _safe([np.mean(list(r["dynamics"]["actor_coverage"].values()) or [0]) for r in rows])

    fig, ax = plt.subplots(figsize=(max(12, 0.18 * len(names) + 4), 5))
    x = np.arange(len(names))
    bottom = np.zeros_like(veh)
    for arr, lbl, color in [(veh, "vehicle speed", "C0"),
                            (wlk, "walker speed", "C3"),
                            (dpt, "Δdepth", "C2"),
                            (cov, "actor coverage", "C1")]:
        ax.bar(x, arr, bottom=bottom, color=color, alpha=0.8, label=lbl, width=0.9)
        bottom += arr
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("normalized contribution (each layer / max-of-layer)")
    ax.set_title(f"Stacked dynamics signature per scene ({len(rows)} scenes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_master(out_root: Path, save_path: Path) -> Path | None:
    """Stitch the four overview grids + cross-scene chart into a single PNG."""
    panels = [
        ("Camera trajectories",       out_root / "overview_trajectories_stacked.png"),
        ("Actor world distribution",  out_root / "overview_actors_world_xy.png"),
        ("Visible-actor distribution", out_root / "overview_actors_visible_xy.png"),
        ("Per-scene dynamics bars",   out_root / "overview_dynamics_dynamics_bars.png"),
        ("Cross-scene summary",       out_root / "cross_scene_dynamics.png"),
    ]
    panels = [(t, p) for t, p in panels if p.exists()]
    if not panels:
        return None
    import math
    cols = 2
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, (title, path) in enumerate(panels):
        r, c = divmod(i, cols)
        axes[r, c].imshow(plt.imread(path))
        axes[r, c].set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_cross_scene(summaries: list[dict], out_path: Path) -> None:
    rows = [s for s in summaries if "dynamics" in s]
    if not rows:
        return
    names = [r["scene"] for r in rows]
    veh_speed = [np.mean(r["dynamics"]["actor_speed"]["vehicle"] or [0]) for r in rows]
    wlk_speed = [np.mean(r["dynamics"]["actor_speed"]["walker"] or [0]) for r in rows]
    depth_d = [np.mean(list(r["dynamics"]["depth_diff"].values()) or [0]) for r in rows]
    cov = [np.mean(list(r["dynamics"]["actor_coverage"].values()) or [0]) * 100 for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(max(10, 0.3 * len(names) + 6), 8))
    x = np.arange(len(names))
    short_names = [n.replace("scene_", "") for n in names]

    axes[0, 0].bar(x - 0.2, veh_speed, 0.4, label="vehicle", color="C0", alpha=0.7)
    axes[0, 0].bar(x + 0.2, wlk_speed, 0.4, label="walker", color="C3", alpha=0.7)
    axes[0, 0].set_title("Mean actor speed per scene (m/s)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x, depth_d, color="C2", alpha=0.7)
    axes[0, 1].set_title("Mean |Δdepth| across cameras (m)")

    axes[1, 0].bar(x, cov, color="C1", alpha=0.7)
    axes[1, 0].set_title("Mean actor bbox pixel coverage (%)")

    axes[1, 1].bar(x, [r["num_actors"] for r in rows], color="C4", alpha=0.7)
    axes[1, 1].set_title("Actor count per scene")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=70, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-town trajectory overlays + recentered xyz views
# ---------------------------------------------------------------------------

def _load_all_camera_poses(scene_dirs: list[Path]) -> list[dict]:
    """Light per-scene loader: returns [{scene, town, cameras: {name: c2w}}]."""
    out = []
    for sd in scene_dirs:
        meta_path = sd / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        city = meta.get("map_name", "unknown")
        cams_root = sd / "dynamic"
        if not cams_root.exists():
            cams_root = sd / "static"
        cams = {}
        for d in sorted(cams_root.iterdir()):
            if not (d.is_dir() and d.name.startswith("cam_")):
                continue
            ext_path = d / "extrinsics.safetensors"
            if not ext_path.exists():
                continue
            cams[d.name] = load_file(str(ext_path))["c2w"].astype(np.float64)
        out.append({"scene": sd.name, "city": city, "cameras": cams})
    return out


def build_town_bev(scenes_in_town: list[Path], extent_xy: tuple[float, float,
                                                                 float, float],
                   resolution_m: float = 1.0,
                   z_band: tuple[float, float] = (-2.0, 8.0),
                   sample_stride: int = 6,
                   max_frames_per_cam: int = 4,
                   max_depth: float = 80.0,
                   device: str | None = None) -> np.ndarray:
    """Render a top-down density BEV image of `town` by back-projecting
    static-camera depth into world XY and accumulating a 2D histogram.

    Returns log-normalized density in [0,1], shape (H_px, W_px).
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    xmin, xmax, ymin, ymax = extent_xy
    W = max(2, int(round((xmax - xmin) / resolution_m)))
    H = max(2, int(round((ymax - ymin) / resolution_m)))
    counts = torch.zeros((H, W), dtype=torch.float32, device=device)

    pref_cams = ("cam_06_cctv", "cam_04_orbit_building", "cam_05_orbit_crossroad",
                 "cam_03_drone_forward", "cam_07_pedestrian", "cam_08_pedestrian")

    for sd in scenes_in_town:
        sta_root = sd / "static"
        if not sta_root.exists():
            continue
        cam_dirs = [sta_root / n for n in pref_cams if (sta_root / n).exists()]
        if not cam_dirs:
            continue
        for cdir in cam_dirs:
            try:
                d = load_file(str(cdir / "depth.safetensors"))["depth"].astype(np.float32)
                K = load_file(str(cdir / "intrinsics.safetensors"))["K"].astype(np.float64)
                c2w = load_file(str(cdir / "extrinsics.safetensors"))["c2w"].astype(np.float64)
            except Exception:
                continue
            n = len(d)
            if n == 0: continue
            idx = np.linspace(0, n - 1, min(max_frames_per_cam, n)).round().astype(int)
            d_t = torch.from_numpy(d[idx]).to(device)                # (F,H,W)
            K_t = torch.from_numpy(K[idx] if K.ndim == 3 else
                                    np.broadcast_to(K, (len(idx), 3, 3))).to(device)
            c_t = torch.from_numpy(c2w[idx]).to(device)              # (F,4,4)
            F, Hd, Wd = d_t.shape
            us = torch.arange(0, Wd, sample_stride, device=device)
            vs = torch.arange(0, Hd, sample_stride, device=device)
            vv, uu = torch.meshgrid(vs, us, indexing="ij")
            uu = uu.reshape(-1).float()
            vv = vv.reshape(-1).float()
            depth_s = d_t[:, vs][:, :, us].reshape(F, -1)            # (F,P)
            valid = (depth_s > 0.05) & (depth_s < max_depth)
            fx = K_t[:, 0, 0:1]; fy = K_t[:, 1, 1:2]
            cx = K_t[:, 0, 2:3]; cy = K_t[:, 1, 2:3]
            x_cam = (uu[None] - cx) * depth_s / fx
            y_cam = -(vv[None] - cy) * depth_s / fy
            z_cam = -depth_s
            ones = torch.ones_like(x_cam)
            P_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # (F,P,4)
            P_world = torch.einsum("fij,fpj->fpi", c_t, P_cam)[..., :3]
            # OpenGL world -> CARLA: (carla_x, carla_y, carla_z) = (Z, X, Y)
            cx_w = P_world[..., 2]
            cy_w = P_world[..., 0]
            cz_w = P_world[..., 1]
            keep = valid & (cz_w >= z_band[0]) & (cz_w <= z_band[1]) \
                       & (cx_w >= xmin) & (cx_w < xmax) \
                       & (cy_w >= ymin) & (cy_w < ymax)
            cx_w = cx_w[keep]; cy_w = cy_w[keep]
            if cx_w.numel() == 0:
                continue
            ix = ((cx_w - xmin) / resolution_m).long().clamp(0, W - 1)
            iy = ((cy_w - ymin) / resolution_m).long().clamp(0, H - 1)
            flat = iy * W + ix
            counts.view(-1).index_add_(0,
                flat, torch.ones_like(flat, dtype=torch.float32))

    arr = counts.cpu().numpy()
    if arr.max() <= 0:
        return arr
    # log-normalize for visibility
    arr = np.log1p(arr)
    arr = arr / arr.max()
    return arr


def plot_city_overlay(city: str, scenes: list[dict], out_path: Path,
                      bev: np.ndarray | None = None,
                      bev_extent: tuple[float, float, float, float] | None = None
                      ) -> None:
    """All scenes / all cameras for one city, overlaid in world xy."""
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.set_facecolor("white")
    if bev is not None and bev_extent is not None:
        xmin, xmax, ymin, ymax = bev_extent
        if bev.ndim == 3:
            # CARLA BEVs need a horizontal flip to align with our world-xy frame.
            # Brighten by blending halfway toward white so the map reads as a
            # light backdrop rather than a dark foreground.
            light = np.fliplr(bev.astype(np.float32))
            if light.max() > 1.5:        # uint8 path
                light = light / 255.0
            light = 1.0 - (1.0 - light) * 0.45    # pulls blacks toward white
            light = np.clip(light, 0.0, 1.0)
            ax.imshow(light, origin="lower",
                      extent=(xmin, xmax, ymin, ymax),
                      alpha=0.85, interpolation="bilinear")
        else:
            ax.imshow(bev, origin="lower", extent=(xmin, xmax, ymin, ymax),
                      cmap="bone_r", alpha=0.55, interpolation="nearest")
    cam_names = sorted({n for s in scenes for n in s["cameras"].keys()})
    color_for = {n: PALETTE_LIST[i % len(PALETTE_LIST)] for i, n in enumerate(cam_names)}
    seen_legend = set()
    for s in scenes:
        for cname, c2w in s["cameras"].items():
            pos, fwd = cam_pose_carla_xy(c2w)
            color = color_for[cname]
            label = cname if cname not in seen_legend else None
            seen_legend.add(cname)
            ax.plot(pos[:, 0], pos[:, 1], "-", color=color, lw=0.9,
                    alpha=0.55, label=label,
                    solid_capstyle="round", solid_joinstyle="round")
            mid = len(pos) // 2
            _add_camera_triangles(ax, pos[mid:mid + 1], fwd[mid:mid + 1],
                                  color, scale=2.4, alpha=0.95)
            ax.scatter(pos[0, 0], pos[0, 1], s=14, facecolor="white",
                       edgecolor=color, linewidths=1.0, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title(f"{city}", loc="left")
    ax.text(0.99, 0.985,
            f"{len(scenes)} scenes · {len(cam_names)} cameras",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=PALETTE["grey"])
    ax.grid(True, alpha=0.45)
    leg = ax.legend(loc="best", ncol=2, fontsize=8, title="Camera",
                    title_fontsize=9)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _carla_xyz_from_c2w(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (N,3) CARLA position and forward direction.

    Mapping: CARLA (x,y,z) <- world (Z, X, Y). Forward = -Z_cam in OpenGL.
    """
    pos = np.stack([c2w[:, 2, 3], c2w[:, 0, 3], c2w[:, 1, 3]], axis=1)
    fwd = np.stack([-c2w[:, 2, 2], -c2w[:, 0, 2], -c2w[:, 1, 2]], axis=1)
    return pos, fwd


def plot_recentered_axis_views(scenes: list[dict], out_dir: Path) -> None:
    """3 figures (xy, xz, yz). Every camera trajectory translated so its
    first frame sits at origin; small triangle = direction at start."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_names = sorted({n for s in scenes for n in s["cameras"].keys()})
    color_for = {n: PALETTE_LIST[i % len(PALETTE_LIST)] for i, n in enumerate(cam_names)}
    plane_specs = [
        ("xy", (0, 1), ("x", "y")),
        ("xz", (0, 2), ("x", "z  (height)")),
        ("yz", (1, 2), ("y", "z  (height)")),
    ]
    for tag, (i, j), (xlbl, ylbl) in plane_specs:
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.set_facecolor("#F7F7F8")
        seen_legend = set()
        for s in scenes:
            for cname, c2w in s["cameras"].items():
                pos, fwd = _carla_xyz_from_c2w(c2w)
                pos = pos - pos[0]
                pts = pos[:, [i, j]]
                fpts = fwd[:, [i, j]]
                color = color_for[cname]
                label = cname if cname not in seen_legend else None
                seen_legend.add(cname)
                ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=0.55,
                        alpha=0.45, label=label,
                        solid_capstyle="round", solid_joinstyle="round")
                _add_camera_triangles(ax, pts[:1], fpts[:1], color,
                                      scale=1.4, alpha=0.95)
        ax.scatter([0], [0], color="black", marker="+", s=80,
                   linewidths=1.4, zorder=6, label="origin (per-cam start)")
        ax.set_aspect("equal")
        ax.set_xlabel(f"{xlbl}  (m, recentered)")
        ax.set_ylabel(f"{ylbl}  (m, recentered)")
        ax.set_title(f"Recentered camera trajectories — {tag.upper()}",
                     loc="left")
        ax.text(0.99, 0.985, f"{len(scenes)} scenes",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=PALETTE["grey"])
        ax.grid(True, alpha=0.45)
        leg = ax.legend(loc="best", ncol=2, fontsize=8, title="Camera",
                        title_fontsize=9)
        leg.get_frame().set_linewidth(0.6)
        fig.tight_layout()
        fig.savefig(out_dir / f"recentered_{tag}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-town bar charts (mean ± std)
# ---------------------------------------------------------------------------

def plot_ego_speed_per_camera(summaries: list[dict], out_path: Path,
                              pose_data: list[dict] | None = None,
                              fps: float = 10.0) -> None:
    """Twin-axis bar chart per ego camera: speed (km/h, left) vs
    yaw-rate (deg/s, right). Aggregated across every scene."""
    speed_by_cam: dict[str, list[float]] = {}
    yaw_by_cam:   dict[str, list[float]] = {}
    for r in summaries:
        dyn = r.get("dynamics", {})
        for cam, mps in dyn.get("ego_speed_mps", {}).items():
            speed_by_cam.setdefault(cam, []).append(mps * 3.6)
        for cam, dps in dyn.get("ego_yawrate_dps", {}).items():
            yaw_by_cam.setdefault(cam, []).append(dps)
    # fallback: derive yaw-rate from extrinsics if cached scenes are missing it
    if pose_data is not None and not yaw_by_cam:
        for s in pose_data:
            for cname, c2w in s["cameras"].items():
                if not cname.endswith("_car_forward") or len(c2w) < 2:
                    continue
                fwd_xy = np.stack([-c2w[:, 2, 2], -c2w[:, 0, 2]], axis=1)
                n = np.linalg.norm(fwd_xy, axis=1, keepdims=True).clip(min=1e-9)
                fwd_xy /= n
                cosang = np.clip((fwd_xy[1:] * fwd_xy[:-1]).sum(1), -1, 1)
                yaw = float((np.degrees(np.arccos(cosang)) * fps).mean())
                yaw_by_cam.setdefault(cname, []).append(yaw)

    cams = sorted(speed_by_cam.keys())
    if not cams:
        return
    s_mean = np.array([np.mean(speed_by_cam[c]) for c in cams])
    s_std  = np.array([np.std(speed_by_cam[c])  for c in cams])
    y_mean = np.array([np.mean(yaw_by_cam.get(c, [0])) for c in cams])
    y_std  = np.array([np.std(yaw_by_cam.get(c, [0])) for c in cams])
    n_samples = sum(len(v) for v in speed_by_cam.values())

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(cams) + 3), 4.8))
    ax.set_facecolor("#FBFBFC")
    ax2 = ax.twinx()
    ax2.grid(False)

    x = np.arange(len(cams))
    w = 0.36
    speed_color = PALETTE["blue"]
    yaw_color = PALETTE["red"]

    b1 = ax.bar(x - w / 2, s_mean, w, color=speed_color,
                edgecolor=speed_color, linewidth=0,
                label="Speed", zorder=3)
    ax.errorbar(x - w / 2, s_mean, yerr=s_std, fmt="none",
                ecolor=PALETTE["grey"], elinewidth=1.0, capsize=0, alpha=0.9,
                zorder=4)
    b2 = ax2.bar(x + w / 2, y_mean, w, color=yaw_color,
                 edgecolor=yaw_color, linewidth=0,
                 label="Yaw-rate", zorder=3)
    ax2.errorbar(x + w / 2, y_mean, yerr=y_std, fmt="none",
                 ecolor=PALETTE["grey"], elinewidth=1.0, capsize=0, alpha=0.9,
                 zorder=4)
    _annotate_bars(ax, b1, fmt="{:.1f}", offset=0.3, color=speed_color)
    _annotate_bars(ax2, b2, fmt="{:.1f}", offset=0.15, color=yaw_color)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("cam_", "") for c in cams],
                       rotation=22, ha="right")
    ax.set_ylabel("Speed   (km/h)", color=speed_color, fontweight="medium")
    ax2.set_ylabel("Yaw-rate   (deg/s)", color=yaw_color, fontweight="medium")
    ax.tick_params(axis="y", colors=speed_color)
    ax2.tick_params(axis="y", colors=yaw_color)
    for spine in (ax.spines["left"],): spine.set_color(speed_color)
    for spine in (ax2.spines["right"],): spine.set_color(yaw_color)
    ax2.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Ego camera ID")
    ax.set_title("Ego-camera motion profile", loc="left")
    ax.text(0.99, 0.96,
            f"{len(cams)} ego cams · {n_samples} scenes",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=PALETTE["grey"])
    ax.grid(axis="x", visible=False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="upper left",
                    frameon=True, ncol=2, framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_scene_dynamic_bars(summaries: list[dict], out_path: Path) -> None:
    """Per city (=CARLA map): twin-axis bar chart.
        left  y → mean visible-actor count per frame (visible in any camera)
        right y → mean per-pixel coverage by projected actor bboxes (%)
    Both aggregated over every (scene × camera) in the city.
    """
    by_city: dict[str, dict] = {}
    use_static_dyn = any(r.get("dynamics", {}).get("depth_static_dyn_ratio")
                         for r in summaries)
    cov_label = (r"Dynamic coverage  $\,(|d_{\mathrm{dyn}} - d_{\mathrm{static}}| > 0.3\,\mathrm{m})$"
                 if use_static_dyn else "Actor 3D bbox coverage")
    for r in summaries:
        if not r.get("map"):
            continue
        c = r["map"]
        slot = by_city.setdefault(c, {"obj_per_frame": [], "coverage_pct": []})
        n_frames = max(1, int(r.get("num_frames", 1)))
        for cam, n_vis in r.get("visible_per_camera", {}).items():
            slot["obj_per_frame"].append(n_vis / n_frames)
        cov_src = (r["dynamics"].get("depth_static_dyn_ratio")
                   if use_static_dyn
                   else r.get("dynamics", {}).get("actor_coverage", {}))
        for cam, cov in (cov_src or {}).items():
            slot["coverage_pct"].append(cov * 100.0)

    cities = sorted(by_city.keys())
    if not cities:
        return
    obj_m = np.array([np.mean(by_city[c]["obj_per_frame"]) if by_city[c]["obj_per_frame"] else 0 for c in cities])
    obj_s = np.array([np.std(by_city[c]["obj_per_frame"])  if by_city[c]["obj_per_frame"] else 0 for c in cities])
    cov_m = np.array([np.mean(by_city[c]["coverage_pct"])  if by_city[c]["coverage_pct"] else 0 for c in cities])
    cov_s = np.array([np.std(by_city[c]["coverage_pct"])   if by_city[c]["coverage_pct"] else 0 for c in cities])

    obj_color = PALETTE["blue"]
    cov_color = PALETTE["green"]

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(cities) + 3), 5.0))
    ax.set_facecolor("#FBFBFC")
    ax2 = ax.twinx()
    ax2.grid(False)

    x = np.arange(len(cities))
    w = 0.36
    b1 = ax.bar(x - w / 2, obj_m, w, color=obj_color,
                label="Visible objects per frame", zorder=3)
    ax.errorbar(x - w / 2, obj_m, yerr=obj_s, fmt="none",
                ecolor=PALETTE["grey"], elinewidth=1.0, capsize=0, alpha=0.85)
    b2 = ax2.bar(x + w / 2, cov_m, w, color=cov_color,
                 label=cov_label, zorder=3)
    ax2.errorbar(x + w / 2, cov_m, yerr=cov_s, fmt="none",
                 ecolor=PALETTE["grey"], elinewidth=1.0, capsize=0, alpha=0.85)
    _annotate_bars(ax, b1, fmt="{:.1f}", offset=obj_m.max() * 0.015, color=obj_color)
    _annotate_bars(ax2, b2, fmt="{:.1f}", offset=max(cov_m.max() * 0.015, 0.1), color=cov_color)

    ax.set_xticks(x); ax.set_xticklabels(cities, rotation=12, ha="right")
    ax.set_ylabel("Visible objects per frame", color=obj_color, fontweight="medium")
    ax2.set_ylabel("Dynamic-pixel coverage   (%)", color=cov_color, fontweight="medium")
    ax.tick_params(axis="y", colors=obj_color)
    ax2.tick_params(axis="y", colors=cov_color)
    ax.spines["left"].set_color(obj_color)
    ax2.spines["right"].set_color(cov_color)
    ax2.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-city dynamic content", loc="left")
    ax.grid(axis="x", visible=False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="upper left",
                    frameon=True, framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-scene overview grid
# ---------------------------------------------------------------------------

def make_overview_grid(out_root: Path, kind: str = "trajectories",
                       fname: str = "stacked.png") -> Path | None:
    """Tile every scene's `<kind>/<scene>/<fname>` into one PNG."""
    src_dir = out_root / kind
    if not src_dir.exists():
        return None
    imgs = sorted(src_dir.glob(f"*/{fname}"))
    if not imgs:
        return None
    import math
    n = len(imgs)
    cols = min(6, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, img_path in enumerate(imgs):
        r, c = divmod(i, cols)
        img = plt.imread(img_path)
        axes[r, c].imshow(img)
        axes[r, c].set_title(img_path.parent.name, fontsize=7)
    fig.suptitle(f"{kind} — {fname} ({n} scenes)", fontsize=10)
    fig.tight_layout()
    tag = Path(fname).stem
    out = out_root / f"overview_{kind}_{tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze_scene(scene_dir: Path, out_root: Path,
                  do_dynamics: bool = True) -> dict:
    scene = load_scene(scene_dir)
    actor_data = collect_actor_locations(scene["actors"])
    name = scene_dir.name

    traj_dir = out_root / "trajectories" / name
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_stacked(scene, traj_dir / "stacked.png")
    plot_trajectory_per_camera(scene, traj_dir)

    actor_dir = out_root / "actors" / name
    actor_dir.mkdir(parents=True, exist_ok=True)
    plot_world_actors(scene, actor_data, actor_dir / "world_xy.png")
    visible_counts = plot_camera_occupancy(scene, actor_data, actor_dir)

    summary = {
        "scene": name,
        "map": scene["meta"].get("map_name"),
        "weather": scene["meta"].get("weather"),
        "num_frames": scene["meta"].get("num_frames"),
        "num_cameras": len(scene["cameras"]),
        "num_actors": scene["actors"]["num_actors"],
        "num_vehicles_pts": int(len(actor_data["by_type"]["vehicle"])),
        "num_walkers_pts": int(len(actor_data["by_type"]["walker"])),
        "visible_per_camera": visible_counts,
    }

    if do_dynamics:
        fps = float(scene["meta"].get("fps", 10))
        speeds = actor_speeds(scene["actors"], fps)
        depth = per_camera_depth_diff(scene)
        coverage = per_camera_actor_coverage(scene, actor_data)
        ego_data = ego_speeds_per_camera(scene, fps)
        ego = ego_data["speed_mps"]
        ego_yaw = ego_data["yaw_dps"]
        dyn_dir = out_root / "dynamics" / name
        dyn_dir.mkdir(parents=True, exist_ok=True)
        plot_dynamics_bars(name, speeds, depth["mean_abs_diff"], coverage,
                           dyn_dir / "dynamics_bars.png")
        summary["dynamics"] = {
            "actor_speed": {k: v.tolist() for k, v in speeds.items()},
            "depth_diff": depth["mean_abs_diff"],
            "depth_dynamic_ratio": depth["dynamic_ratio"],
            "actor_coverage": coverage,
            "ego_speed_mps": ego,
            "ego_yawrate_dps": ego_yaw,
        }
    return summary


def discover_scenes(root: Path, duration: str, split: str,
                    scene_filter: str | None) -> list[Path]:
    base = root / duration / split
    scenes = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("scene_"))
    if scene_filter:
        scenes = [s for s in scenes if scene_filter in s.name]
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("StaDy4D"))
    ap.add_argument("--duration", default="short")
    ap.add_argument("--split", default="train,test",
                    help="comma-separated list of splits to scan")
    ap.add_argument("--scene", default=None,
                    help="substring filter; default = all scenes in split")
    ap.add_argument("--out", type=Path, default=Path("outputs/dataset_analysis"))
    ap.add_argument("--no-dynamics", action="store_true",
                    help="skip depth-diff / actor-coverage / speed analysis")
    ap.add_argument("--no-overview", action="store_true",
                    help="skip multi-scene overview grids")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="per-scene JSON cache (default: <out>/.cache)")
    ap.add_argument("--bev-dir", type=Path, default=Path("StaDy4D/town_bev"),
                    help="dir with <Town>.png + <Town>.json (extent meters); "
                         "falls back to data-driven BEV if missing")
    ap.add_argument("--force", action="store_true",
                    help="ignore cache and recompute every scene")
    ap.add_argument("--no-occupancy", action="store_true",
                    help="skip the dataset-wide dynamic-occupancy aggregator "
                         "(saves ~10 min); per-scene bars then fall back to "
                         "the cached frame-to-frame depth-diff metric")
    args = ap.parse_args()
    cache_dir = args.cache_dir or (args.out / ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    scenes = []
    for sp in splits:
        scenes.extend(discover_scenes(args.root, args.duration, sp, args.scene))
    if not scenes:
        raise SystemExit(f"no scenes found under {args.root}/{args.duration}/{splits}")

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    REQUIRED_KEYS = {"depth_dynamic_ratio", "ego_speed_mps"}  # bump on schema change
    for s in scenes:
        cache_file = cache_dir / f"{s.name}.json"
        if (not args.force and cache_file.exists()):
            try:
                cached = json.loads(cache_file.read_text())
                if (args.no_dynamics
                        or REQUIRED_KEYS.issubset(cached.get("dynamics", {}).keys())):
                    summaries.append(cached)
                    print(f"[cache ] {s.name}")
                    continue
            except Exception:
                pass
        print(f"[analyze] {s.name}")
        try:
            r = analyze_scene(s, args.out, do_dynamics=not args.no_dynamics)
            cache_file.write_text(json.dumps(r))
            summaries.append(r)
        except Exception as e:  # keep going on bad scenes
            print(f"  !! failed: {e}")
            summaries.append({"scene": s.name, "error": str(e)})
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))

    # Per-town trajectory overlays + recentered xyz views
    print("[city] loading camera poses for city aggregation")
    pose_data = _load_all_camera_poses(scenes)
    by_city: dict[str, list[dict]] = {}
    for s in pose_data:
        by_city.setdefault(s["city"], []).append(s)
    city_dir = args.out / "cities"
    city_dir.mkdir(parents=True, exist_ok=True)
    # collect scene_dir -> city for BEV building
    name2dir = {sd.name: sd for sd in scenes}
    for city, lst in by_city.items():
        # bounding box across every camera trajectory in the town
        all_xy = np.concatenate([
            cam_pose_carla_xy(c2w)[0]
            for s in lst for c2w in s["cameras"].values()
        ], axis=0)
        pad = 30.0
        xmin, ymin = all_xy.min(0) - pad
        xmax, ymax = all_xy.max(0) + pad
        # cap BEV side-length so we don't blow memory on sparse maps
        side = max(xmax - xmin, ymax - ymin, 80.0)
        if side > 600:
            scale = 600 / side
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            xmin = cx - 600 / 2; xmax = cx + 600 / 2
            ymin = cy - 600 / 2; ymax = cy + 600 / 2
        ext = (float(xmin), float(xmax), float(ymin), float(ymax))
        city_scene_dirs = [name2dir[s["scene"]] for s in lst if s["scene"] in name2dir]
        bev = None
        bev_png = args.bev_dir / f"{city}.png"
        bev_meta = args.bev_dir / f"{city}.json"
        if bev_png.exists() and bev_meta.exists():
            try:
                bev = plt.imread(bev_png)
                m = json.loads(bev_meta.read_text())
                ext = (float(m["xmin"]), float(m["xmax"]),
                       float(m["ymin"]), float(m["ymax"]))
                print(f"[city] {city}: using pre-rendered BEV {bev_png.name}")
            except Exception as e:
                print(f"  !! load BEV {bev_png}: {e}")
                bev = None
        if bev is None:
            try:
                bev = build_town_bev(city_scene_dirs, ext, resolution_m=1.5,
                                      max_frames_per_cam=3, sample_stride=8)
            except Exception as e:
                print(f"  !! BEV failed for {city}: {e}")
                bev = None
        plot_city_overlay(city, lst, city_dir / f"{city}_overlay.png",
                          bev=bev, bev_extent=ext)
        plot_recentered_axis_views(lst, city_dir / city)
        print(f"[city] {city}: overlay (BEV {'ok' if bev is not None else 'skipped'}) + recentered")
    plot_recentered_axis_views(pose_data, city_dir / "ALL")
    print("[city] ALL: recentered xy/xz/yz across every city")

    # Aggregator: builds the dataset-wide overlay AND returns per-scene
    # static-vs-dynamic ratios (so we can patch the cache once instead of
    # re-loading every depth pair separately for the per-scene bars).
    per_scene_ratios = {}
    if not args.no_occupancy:
        per_scene_ratios = aggregate_dataset_dynamic_occupancy(
            scenes, args.out / "dataset_dynamic_occupancy.png") or {}
        print(f"[overlay] {args.out / 'dataset_dynamic_occupancy.png'}")
    else:
        print("[overlay] skipped (--no-occupancy)")
    if per_scene_ratios:
        for s, summ in zip(scenes, summaries):
            r = per_scene_ratios.get(s.name)
            if r is None or "dynamics" not in summ:
                continue
            summ["dynamics"]["depth_static_dyn_ratio"] = r
            try:
                (cache_dir / f"{s.name}.json").write_text(json.dumps(summ))
            except Exception:
                pass

    plot_ego_speed_per_camera(summaries, args.out / "ego_speed_per_camera.png",
                              pose_data=pose_data)
    plot_per_scene_dynamic_bars(summaries, args.out / "per_scene_dynamic_bars.png")
    print(f"[bar ] {args.out / 'ego_speed_per_camera.png'}")
    print(f"[bar ] {args.out / 'per_scene_dynamic_bars.png'}")

    if not args.no_overview:
        for kind, fname in [("trajectories", "stacked.png"),
                            ("actors", "world_xy.png"),
                            ("actors", "visible_xy.png"),
                            ("dynamics", "dynamics_bars.png")]:
            out = make_overview_grid(args.out, kind, fname)
            if out:
                print(f"[overview] {out}")
        plot_cross_scene(summaries, args.out / "cross_scene_dynamics.png")
        master = plot_master(args.out, args.out / "master.png")
        print(f"[cross-scene] {args.out / 'cross_scene_dynamics.png'}")
        if master:
            print(f"[master]      {master}")

    print(f"[analyze] wrote {len(summaries)} scene summaries to {args.out}")


if __name__ == "__main__":
    main()
