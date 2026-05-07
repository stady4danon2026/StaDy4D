"""Export the multi-camera world (point cloud + camera frustums) as a GLB.

Mirrors `sigma/visualization/build_world.py` but writes a single .glb that you
can drop into a `<model-viewer>` on a static webpage.

Usage:
    python script/export_world_glb.py \
        --scene_path StaDy4D/mid/test/scene_T03_002 \
        --scene_type static \
        --out world.glb \
        --frame 0 \
        --downsample_factor 2 \
        --point_stride 4 \
        --max_depth 200
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import trimesh
from tqdm.auto import tqdm

from sigma.visualization.visualize import create_loader

CAMERA_HUES = [0.0, 0.11, 0.33, 0.55, 0.75, 0.90, 0.05, 0.22, 0.44]
SKIP_CAMERA_KEYWORDS = ("orbit_building",)


def _discover_cameras(scene_dir: Path) -> list[Path]:
    return sorted(
        d for d in scene_dir.iterdir()
        if d.is_dir()
        and (d.name.startswith("cam_") or d.name.startswith("camera_"))
        and not any(kw in d.name for kw in SKIP_CAMERA_KEYWORDS)
    )


def _frustum_lines(c2w: np.ndarray, hue: float, scale: float = 3.0):
    """Build a hollow wireframe frustum (line segments) for one camera."""
    apex = np.array([0.0, 0.0, 0.0])
    f = scale
    base = np.array([
        [-f * 0.6, -f * 0.4, -f],
        [ f * 0.6, -f * 0.4, -f],
        [ f * 0.6,  f * 0.4, -f],
        [-f * 0.6,  f * 0.4, -f],
    ])
    verts_cam = np.vstack([apex[None, :], base])
    verts_world = (c2w[:3, :3] @ verts_cam.T).T + c2w[:3, 3]
    # 8 edges: 4 base + 4 apex-to-corner
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    segments = np.array([[verts_world[a], verts_world[b]] for a, b in pairs])
    path = trimesh.load_path(segments)
    rgb = (np.array(colorsys.hsv_to_rgb(hue, 0.95, 0.55)) * 255).astype(np.uint8)
    color = np.array([[*rgb, 255]] * len(path.entities), dtype=np.uint8)
    path.colors = color
    return path


def _resolve_scene_paths(args) -> list[Path]:
    if args.scene_paths:
        return [Path(p) for p in args.scene_paths]
    if args.town and args.weather:
        import pandas as pd
        idx_path = args.dataset_root / "index.parquet"
        df = pd.read_parquet(idx_path)
        sub = df[(df.town == args.town) & (df.weather == args.weather)]
        if args.duration:
            sub = sub[sub.path.str.startswith(f"{args.duration}/")]
        if sub.empty:
            raise SystemExit(f"No scenes match town={args.town} weather={args.weather}")
        paths = [args.dataset_root / p for p in sub.path.tolist()]
        print(f"Index match: {len(paths)} scenes for town={args.town} weather={args.weather}")
        return paths
    return [args.scene_path]


def main(args):
    scene_paths = _resolve_scene_paths(args)
    print(f"Fusing {len(scene_paths)} scene(s)")

    all_pts = []
    all_cols = []
    frustums = []

    for si, scene_path in enumerate(scene_paths):
        scene_dir = scene_path / args.scene_type
        cam_dirs = _discover_cameras(scene_dir)
        if not cam_dirs:
            print(f"  [skip] No cam_*/camera_* dirs in {scene_dir}")
            continue
        print(f"[{si+1}/{len(scene_paths)}] {scene_path.name}: {len(cam_dirs)} cameras")
        loaders = [create_loader(d) for d in cam_dirs]
        for ci, (name_dir, loader) in enumerate(zip(cam_dirs, tqdm(loaders, desc=scene_path.name, leave=False))):
            n = min(args.max_frames, loader.num_frames)
            hue = CAMERA_HUES[ci % len(CAMERA_HUES)]
            for fi in range(n):
                pts, colors, _, c2w, _ = loader.get_frame_data(fi, args.downsample_factor)
                if pts is None or len(pts) == 0:
                    continue
                depth = np.linalg.norm(pts - c2w[:3, 3], axis=1)
                mask = depth < args.max_depth
                pts = pts[mask][::args.point_stride]
                colors = colors[mask][::args.point_stride]
                all_pts.append(pts.astype(np.float32))
                all_cols.append((colors * 255).astype(np.uint8))
                if args.all_frustums or fi == args.frame:
                    frustums.append(_frustum_lines(c2w, hue, scale=args.frustum_scale))

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    print(f"Raw points: {len(pts):,}")

    if args.voxel_size > 0:
        keys = np.round(pts / args.voxel_size).astype(np.int64)
        # build unique key index
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        pts = pts[unique_idx]
        cols = cols[unique_idx]
        print(f"After voxel dedupe @ {args.voxel_size}m: {len(pts):,}")

    centroid = pts.mean(axis=0)
    pts -= centroid
    for fr in frustums:
        fr.vertices = fr.vertices - centroid

    # viser uses -Y as up; flip Y so the GLB matches glTF's +Y-up convention
    if args.flip_y:
        pts[:, 1] *= -1
        for fr in frustums:
            v = fr.vertices.copy()
            v[:, 1] *= -1
            fr.vertices = v

    cloud = trimesh.PointCloud(vertices=pts, colors=np.hstack([cols, np.full((len(cols), 1), 255, dtype=np.uint8)]))

    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name="points")
    for i, fr in enumerate(frustums):
        scene.add_geometry(fr, geom_name=f"frustum_{i}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene_path", type=Path, required=False)
    p.add_argument("--scene_paths", nargs="+", default=None,
                   help="Multiple scene paths to fuse into one GLB")
    p.add_argument("--dataset_root", type=Path, default=Path("StaDy4D"),
                   help="Root containing index.parquet (for --town/--weather lookup)")
    p.add_argument("--town", default=None, help="Filter scenes by town, e.g. T03")
    p.add_argument("--weather", default=None, help="Filter scenes by weather, e.g. ClearNoon")
    p.add_argument("--duration", default=None, choices=[None, "short", "mid"],
                   help="Optional duration prefix filter when using --town/--weather")
    p.add_argument("--scene_type", default="static", choices=["static", "dynamic"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0, help="Which frame to draw frustums for")
    p.add_argument("--max_frames", type=int, default=30, help="How many frames to fuse into the cloud")
    p.add_argument("--downsample_factor", type=int, default=2)
    p.add_argument("--point_stride", type=int, default=4)
    p.add_argument("--max_depth", type=float, default=200.0)
    p.add_argument("--frustum_scale", type=float, default=1.5)
    p.add_argument("--voxel_size", type=float, default=0.0,
                   help="Voxel grid edge length in metres for dedupe; 0 disables")
    p.add_argument("--all_frustums", action="store_true",
                   help="Draw a camera frustum at every frame (camera trajectory)")
    p.add_argument("--flip_y", action="store_true", default=True,
                   help="Flip Y to convert viser's -Y-up to glTF's +Y-up (default on)")
    p.add_argument("--no_flip_y", dest="flip_y", action="store_false")
    main(p.parse_args())
