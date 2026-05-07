"""Multi-camera world visualizer for StaDy4D dataset.

Loads all cameras from a single scene directory, unprojects RGB+depth into
world-space point clouds using GT extrinsics, and visualizes the combined
scene in viser.

Supports both new format (safetensors + mp4) and legacy format (per-frame files).

Usage:
    python -m sigma.visualization.build_world \
        --scene_path StaDy4D/short/test/scene_T03_008 \
        --scene_type dynamic \
        --downsample_factor 2 \
        --point_stride 4 \
        --max_frames 30
"""

import argparse
import colorsys
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as tf
from tqdm.auto import tqdm

from sigma.visualization.visualize import create_loader


CAMERA_HUES = [0.0, 0.11, 0.33, 0.55, 0.75, 0.90, 0.05, 0.22, 0.44, 0.66, 0.83, 0.16]


def _tint(rgb: np.ndarray, hue: float, strength: float = 0.25) -> np.ndarray:
    tint = np.array(colorsys.hsv_to_rgb(hue, 0.8, 1.0), dtype=np.float32)
    return np.clip((1.0 - strength) * rgb.astype(np.float32) + strength * tint, 0.0, 1.0)


def _discover_cameras(scene_dir: Path) -> list[Path]:
    """Find camera directories, supporting both cam_* and camera_* prefixes."""
    cam_dirs = sorted([
        d for d in scene_dir.iterdir()
        if d.is_dir() and (d.name.startswith("cam_") or d.name.startswith("camera_"))
    ])
    return cam_dirs


def main(
    scene_path: Path,
    scene_type: str = "dynamic",
    downsample_factor: int = 1,
    point_stride: int = 1,
    max_frames: int = 100,
    max_depth: float = 200.0,
    tint_strength: float = 0.25,
    share: bool = False,
) -> None:
    scene_dir = scene_path / scene_type
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    cam_dirs = _discover_cameras(scene_dir)
    if not cam_dirs:
        raise FileNotFoundError(f"No cam_* or camera_* directories in {scene_dir}")

    cam_names = [d.name for d in cam_dirs]
    n_cams = len(cam_names)
    print(f"Found {n_cams} cameras: {cam_names}")

    loaders = [create_loader(d) for d in tqdm(cam_dirs, desc="Init")]
    n_frames = min(max_frames, min(l.num_frames for l in loaders))
    print(f"Loading {n_frames} frames x {n_cams} cameras")

    # For display rotation conversion (handled by loader.display_rotation)

    # ------------------------------------------------------------------
    # Pre-load all frames.
    # ------------------------------------------------------------------
    camera_data: list[list[tuple]] = []  # [cam][frame] = (pts, colors, c2w)
    centroid_pts: list[np.ndarray] = []

    for cam_idx, loader in enumerate(tqdm(loaders, desc="Loading")):
        frames = []
        for fi in range(n_frames):
            pts, colors, _, c2w, _ = loader.get_frame_data(fi, downsample_factor)

            # Filter sky / invalid points.
            depth = loader.load_depth(fi)
            if downsample_factor > 1:
                depth = depth[::downsample_factor, ::downsample_factor]
            depth_flat = depth.flatten()
            valid = np.isfinite(pts).all(axis=1) & (depth_flat < max_depth)

            pts    = pts[valid]
            colors = colors[valid]

            if point_stride > 1:
                pts    = pts[::point_stride]
                colors = colors[::point_stride]

            colors = _tint(colors, CAMERA_HUES[cam_idx % len(CAMERA_HUES)], tint_strength)
            frames.append((pts.astype(np.float32), colors, c2w))

            if len(pts) > 0:
                centroid_pts.append(pts.mean(axis=0))

        camera_data.append(frames)

    world_centroid = np.mean(centroid_pts, axis=0)
    print(f"World centroid: {world_centroid.round(2)}")

    # Recenter everything.
    for ci in range(n_cams):
        for fi in range(n_frames):
            pts, col, c2w = camera_data[ci][fi]
            c2w = c2w.copy()
            c2w[:3, 3] -= world_centroid
            camera_data[ci][fi] = (pts - world_centroid, col, c2w)

    # ------------------------------------------------------------------
    # Viser
    # ------------------------------------------------------------------
    server = viser.ViserServer()
    server.scene.set_up_direction("-y")
    if share:
        server.request_share_url()

    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.0001 * point_stride,
            max=0.01 * point_stride,
            step=0.0001 * point_stride,
            initial_value=0.001 * point_stride,
        )
        gui_timestep   = server.gui.add_slider("Timestep", min=0, max=n_frames - 1, step=1, initial_value=0, disabled=True)
        gui_next       = server.gui.add_button("Next Frame", disabled=True)
        gui_prev       = server.gui.add_button("Prev Frame", disabled=True)
        gui_play       = server.gui.add_checkbox("Playing", True)
        gui_fps        = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=10)
        gui_show_all   = server.gui.add_checkbox("Show All Frames", False)

    cam_toggles: list[viser.GuiInputHandle] = []
    with server.gui.add_folder("Cameras"):
        for name in cam_names:
            cam_toggles.append(server.gui.add_checkbox(name, initial_value=True))

    @gui_next.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % n_frames

    @gui_prev.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % n_frames

    @gui_play.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_play.value
        gui_next.disabled = gui_play.value
        gui_prev.disabled = gui_play.value

    # ------------------------------------------------------------------
    # Scene nodes
    # ------------------------------------------------------------------
    frame_nodes:   list[list] = []
    point_nodes:   list[list] = []
    frustum_nodes: list[list] = []

    for ci, (loader, name) in enumerate(zip(loaders, cam_names)):
        server.scene.add_frame(f"/cameras/{name}", show_axes=False)
        server.scene.add_frame(f"/frustums/{name}", show_axes=False)

        fov    = 2 * np.arctan2(loader.height / 2, loader.K[0, 0])
        aspect = loader.width / loader.height

        cf, cp, cfr = [], [], []
        for fi in tqdm(range(n_frames), desc=f"Build {name}", leave=False):
            pts, col, c2w = camera_data[ci][fi]

            cf.append(server.scene.add_frame(f"/cameras/{name}/t{fi}", show_axes=False))
            cp.append(server.scene.add_point_cloud(
                name=f"/cameras/{name}/t{fi}/pts",
                points=pts, colors=col,
                point_size=gui_point_size.value,
                point_shape="rounded",
            ))
            cfr.append(server.scene.add_camera_frustum(
                f"/frustums/{name}/t{fi}",
                fov=fov, aspect=aspect, scale=0.5,
                wxyz=tf.SO3.from_matrix(loader.display_rotation(c2w[:3, :3])).wxyz,
                position=c2w[:3, 3],
            ))

        frame_nodes.append(cf)
        point_nodes.append(cp)
        frustum_nodes.append(cfr)

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------
    def _apply_visibility(t: int) -> None:
        show_all = gui_show_all.value
        for ci in range(n_cams):
            on = cam_toggles[ci].value
            for fi in range(n_frames):
                frame_nodes[ci][fi].visible   = on and (show_all or fi == t)
                frustum_nodes[ci][fi].visible = on and fi == t

    _apply_visibility(0)

    @gui_timestep.on_update
    def _(_) -> None:
        with server.atomic():
            _apply_visibility(gui_timestep.value)
        server.flush()

    @gui_show_all.on_update
    def _(_) -> None:
        with server.atomic():
            if gui_show_all.value:
                gui_play.value = False
                gui_play.disabled = True
            else:
                gui_play.disabled = False
            _apply_visibility(gui_timestep.value)

    for ci in range(n_cams):
        def _make_cb(idx: int):
            def _cb(_) -> None:
                with server.atomic():
                    _apply_visibility(gui_timestep.value)
            return _cb
        cam_toggles[ci].on_update(_make_cb(ci))

    # Set viewer to look at scene center from first camera.
    _, _, first_c2w = camera_data[0][0]
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = tuple(first_c2w[:3, 3])
        client.camera.look_at = tuple(np.zeros(3))

    print("Viser server running -> http://localhost:8080")

    while True:
        if gui_play.value:
            gui_timestep.value = (gui_timestep.value + 1) % n_frames

        t      = gui_timestep.value
        t_next = (t + 1) % n_frames
        for ci in range(n_cams):
            point_nodes[ci][t].point_size      = gui_point_size.value
            point_nodes[ci][t_next].point_size = gui_point_size.value

        time.sleep(1.0 / gui_fps.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path",         type=Path, required=True,
                        help="Path to scene dir (e.g. StaDy4D/short/test/scene_T03_008)")
    parser.add_argument("--scene_type",         type=str,  default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--downsample_factor",  type=int,  default=2)
    parser.add_argument("--point_stride",       type=int,  default=4)
    parser.add_argument("--max_frames",         type=int,  default=100)
    parser.add_argument("--max_depth",          type=float, default=200.0)
    parser.add_argument("--tint_strength",      type=float, default=0.25)
    parser.add_argument("--share",              action="store_true")
    args = parser.parse_args()

    main(
        scene_path=args.scene_path,
        scene_type=args.scene_type,
        downsample_factor=args.downsample_factor,
        point_stride=args.point_stride,
        max_frames=args.max_frames,
        max_depth=args.max_depth,
        tint_strength=args.tint_strength,
        share=args.share,
    )
