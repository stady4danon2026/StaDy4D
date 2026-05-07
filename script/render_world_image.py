"""Render a single hero image of the fused multi-scene world.

Reuses the same scene/camera discovery as `export_world_glb.py` but rasterizes
to a PNG via pyrender (headless EGL). Camera frustums are colored by *type*
(Dashcam / Pedestrian / Drone / CCTV / Orbit) and a legend overlay is added.

Workflow:
  1) Open the GLB you already exported in the VSCode glTF Viewer (or any GLB
     viewer) to find a good camera pose. Read off camera position + target.
  2) Run this script with `--cam_pos x,y,z --cam_target x,y,z` (post-flip,
     i.e. the values you see in the viewer).

Example:
  PYOPENGL_PLATFORM=egl python script/render_world_image.py \
      --town T03 --weather ClearNoon --duration mid \
      --scene_type dynamic \
      --out outputs/teaser/T03_hero.png \
      --cam_pos 0,120,260 --cam_target 0,0,0 \
      --width 2400 --height 1600 \
      --point_size 2.0 \
      --point_stride 2 --voxel_size 0.05
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402
import trimesh  # noqa: E402

from sigma.visualization.visualize import create_loader  # noqa: E402


CAMERA_TYPES = [
    ("Dashcam",    ("car_forward",),                (0.95, 0.30, 0.25)),
    ("Pedestrian", ("pedestrian",),                 (0.30, 0.75, 0.95)),
    ("Drone",      ("drone",),                      (0.95, 0.75, 0.20)),
    ("CCTV",       ("cctv",),                       (0.45, 0.85, 0.45)),
    ("Orbit",      ("orbit_crossroad",),            (0.85, 0.45, 0.85)),
]
SKIP_KEYWORDS = ("orbit_building",)


def _classify(cam_name: str):
    for label, kws, color in CAMERA_TYPES:
        if any(kw in cam_name for kw in kws):
            return label, color
    return None, (0.7, 0.7, 0.7)


def _discover_cameras(scene_dir: Path):
    return sorted(
        d for d in scene_dir.iterdir()
        if d.is_dir()
        and (d.name.startswith("cam_") or d.name.startswith("camera_"))
        and not any(kw in d.name for kw in SKIP_KEYWORDS)
    )


def _resolve_scenes(args):
    if args.scene_paths:
        return [Path(p) for p in args.scene_paths]
    if args.town and args.weather:
        import pandas as pd
        df = pd.read_parquet(args.dataset_root / "index.parquet")
        sub = df[(df.town == args.town) & (df.weather == args.weather)]
        if args.duration:
            sub = sub[sub.path.str.startswith(f"{args.duration}/")]
        return [args.dataset_root / p for p in sub.path.tolist()]
    return [args.scene_path]


def _frustum_mesh(c2w, color_rgb, scale=1.5):
    apex = np.zeros(3)
    f = scale
    base = np.array([
        [-f * 0.6, -f * 0.4, -f],
        [ f * 0.6, -f * 0.4, -f],
        [ f * 0.6,  f * 0.4, -f],
        [-f * 0.6,  f * 0.4, -f],
    ])
    verts_cam = np.vstack([apex[None], base])
    verts = (c2w[:3, :3] @ verts_cam.T).T + c2w[:3, 3]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    segs = np.array([[verts[a], verts[b]] for a, b in pairs])
    # cylinder per edge for visibility
    meshes = []
    radius = 0.03 * scale
    for a, b in segs:
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=8)
        # align +Z to v
        z = np.array([0, 0, 1.0])
        axis = np.cross(z, v / L)
        s = np.linalg.norm(axis)
        if s < 1e-6:
            R = np.eye(3) if v[2] > 0 else np.diag([1, -1, -1])
        else:
            axis /= s
            ang = np.arctan2(s, np.dot(z, v / L))
            R = trimesh.transformations.rotation_matrix(ang, axis)[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (a + b) / 2
        cyl.apply_transform(T)
        meshes.append(cyl)
    fr = trimesh.util.concatenate(meshes)
    color = (np.array([*color_rgb, 1.0]) * 255).astype(np.uint8)
    fr.visual.face_colors = np.tile(color, (len(fr.faces), 1))
    return fr


def _look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = eye
    return M


def _draw_frustum_icon(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, rgb_a):
    """Wireframe frustum icon. Apex at left, rectangular base at right.
    (x,y) is the top-left of the icon's bounding box; size is its width."""
    h = size
    apex = (x + 4, y + h // 2)
    bx = x + size - 2  # base x
    by_top = y + 4
    by_bot = y + h - 4
    bw = int(size * 0.55)
    base = [
        (bx, by_top),
        (bx, by_bot),
        (bx - bw, by_bot),
        (bx - bw, by_top),
    ]
    w = 3
    # apex-to-corner edges
    for c in base:
        draw.line([apex, c], fill=rgb_a, width=w)
    # base rectangle
    for a, b in zip(base, base[1:] + base[:1]):
        draw.line([a, b], fill=rgb_a, width=w)


def _add_legend(img: Image.Image, items, font_size=28):
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    pad = 18
    icon = 52
    line_h = max(icon, font_size) + 12
    width = 320
    height = pad * 2 + line_h * len(items)
    x0, y0 = 30, 30
    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(20, 20, 24, 210))
    for i, (label, color) in enumerate(items):
        cy = y0 + pad + i * line_h
        rgba = tuple(int(c * 255) for c in color) + (255,)
        _draw_frustum_icon(draw, x0 + pad, cy, icon, rgba)
        draw.text((x0 + pad + icon + 14, cy + 6), label, fill=(240, 240, 240, 255), font=font)
    return img


def _load_extrinsics(cam_dir: Path) -> np.ndarray | None:
    """Return [N,4,4] c2w from extrinsics.safetensors or per-frame JSONs."""
    st = cam_dir / "extrinsics.safetensors"
    if st.exists():
        from safetensors.numpy import load_file
        d = load_file(str(st))
        return d["c2w"] if "c2w" in d else next(iter(d.values()))
    ext_dir = cam_dir / "extrinsics"
    if ext_dir.exists():
        import json
        files = sorted(ext_dir.glob("*.json"))
        mats = []
        for f in files:
            mats.append(np.array(json.load(open(f))["c2w"], dtype=np.float32))
        return np.stack(mats) if mats else None
    return None


def _build_from_glb(args):
    """Fast path: read points/colors from GLB, frustum poses from extrinsics."""
    glb = trimesh.load(args.glb)
    pts_list, col_list = [], []
    for geom in (glb.geometry.values() if isinstance(glb, trimesh.Scene) else [glb]):
        if isinstance(geom, trimesh.PointCloud):
            pts_list.append(np.asarray(geom.vertices, dtype=np.float32))
            c = geom.colors[:, :3] if geom.colors is not None else np.full((len(geom.vertices), 3), 200, dtype=np.uint8)
            col_list.append(c.astype(np.uint8))
    if not pts_list:
        raise SystemExit(f"No PointCloud geometry in {args.glb}")
    pts = np.concatenate(pts_list, 0)
    cols = np.concatenate(col_list, 0)
    print(f"GLB points: {len(pts):,}")

    # The GLB has already had centroid subtracted and Y flipped (per export script).
    # Re-derive the same centroid/flip transform so frustums land in the same frame.
    scenes = _resolve_scenes(args)
    used_types = set()
    frustums_world = []  # (label, color, c2w in original world frame)
    raw_world_pts = []   # to recompute centroid

    for scene_path in scenes:
        scene_dir = scene_path / args.scene_type
        cam_dirs = _discover_cameras(scene_dir)
        for cam_dir in cam_dirs:
            label, color = _classify(cam_dir.name)
            if label is None:
                continue
            ext = _load_extrinsics(cam_dir)
            if ext is None:
                continue
            used_types.add(label)
            n = min(args.max_frames, len(ext))
            frames = range(n) if args.all_frustums else [min(args.frame, n - 1)]
            for fi in frames:
                frustums_world.append((label, color, ext[fi]))
            raw_world_pts.append(ext[:n, :3, 3])

    # Approximate centroid: GLB cloud already centered, so its mean ~ 0.
    # We need to map frustum world->GLB-frame: subtract original centroid + flip Y.
    # Original centroid was the *raw point cloud* centroid. We don't have raw points
    # cheaply, so use the camera-position centroid as a proxy. Then add a residual
    # offset so the GLB cloud and frustums coincide: shift frustums so the median
    # camera position aligns with the cloud centroid (which is ~0 by construction).
    cam_centers = np.concatenate(raw_world_pts, 0) if raw_world_pts else np.zeros((1, 3))
    centroid = cam_centers.mean(0)

    return pts, cols, frustums_world, used_types, centroid


def main(args):
    if args.glb:
        pts, cols, frustums_world, used_types, centroid = _build_from_glb(args)
    else:
        scenes = _resolve_scenes(args)
        print(f"Fusing {len(scenes)} scene(s)")

        all_pts, all_cols = [], []
        frustums_world = []
        used_types = set()

        for si, scene_path in enumerate(scenes):
            scene_dir = scene_path / args.scene_type
            cam_dirs = _discover_cameras(scene_dir)
            if not cam_dirs:
                continue
            loaders = [create_loader(d) for d in cam_dirs]
            for cam_dir, loader in tqdm(list(zip(cam_dirs, loaders)), desc=f"[{si+1}/{len(scenes)}] {scene_path.name}", leave=False):
                label, color = _classify(cam_dir.name)
                if label is None:
                    continue
                used_types.add(label)
                n = min(args.max_frames, loader.num_frames)
                for fi in range(n):
                    pts, colors, _, c2w, _ = loader.get_frame_data(fi, args.downsample_factor)
                    if pts is None or len(pts) == 0:
                        continue
                    d = np.linalg.norm(pts - c2w[:3, 3], axis=1)
                    m = d < args.max_depth
                    pts = pts[m][::args.point_stride]
                    colors = colors[m][::args.point_stride]
                    all_pts.append(pts.astype(np.float32))
                    all_cols.append((colors * 255).astype(np.uint8))
                    if args.all_frustums or fi == args.frame:
                        frustums_world.append((label, color, c2w))

        pts = np.concatenate(all_pts, 0)
        cols = np.concatenate(all_cols, 0)
        print(f"Raw points: {len(pts):,}")
        if args.voxel_size > 0:
            keys = np.round(pts / args.voxel_size).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts, cols = pts[idx], cols[idx]
            print(f"Voxel @ {args.voxel_size}m: {len(pts):,}")
        centroid = pts.mean(0)
        pts -= centroid
        if args.flip_y:
            pts[:, 1] *= -1

    # frustums into the (already-centered, already-Y-flipped) cloud frame
    fmeshes = []
    for label, color, c2w in frustums_world:
        c2w_local = c2w.copy().astype(np.float64)
        c2w_local[:3, 3] -= centroid
        fr = _frustum_mesh(c2w_local, color, scale=args.frustum_scale)
        if args.flip_y:
            fr.vertices[:, 1] *= -1
        fmeshes.append(fr)

    # build pyrender scene
    scene = pyrender.Scene(bg_color=np.array([1, 1, 1, 1.0]), ambient_light=[0.6, 0.6, 0.6])
    rgb_f = cols.astype(np.float32) / 255.0
    if args.saturation != 1.0:
        gray = rgb_f.mean(axis=1, keepdims=True)
        rgb_f = np.clip(gray + (rgb_f - gray) * args.saturation, 0.0, 1.0)
    alpha = np.ones((len(rgb_f), 1), dtype=np.float32)
    cloud_colors = np.hstack([rgb_f, alpha])
    pcl = pyrender.Mesh.from_points(pts, colors=cloud_colors)
    scene.add(pcl)
    for fr in fmeshes:
        scene.add(pyrender.Mesh.from_trimesh(fr, smooth=False))

    # camera
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(args.fov), znear=0.05, zfar=5000.0)
    if args.cam_pos:
        eye = np.array([float(x) for x in args.cam_pos.split(",")])
    else:
        eye = np.array([pts[:, 0].mean(), pts[:, 1].max() + 100, pts[:, 2].max() + 100])
    if args.cam_target:
        tgt = np.array([float(x) for x in args.cam_target.split(",")])
    else:
        tgt = pts.mean(0)
    pose = _look_at(eye, tgt, up=(0, 1, 0))
    scene.add(cam, pose=pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=pose)

    r = pyrender.OffscreenRenderer(args.width, args.height, point_size=args.point_size)
    color, _ = r.render(scene)
    r.delete()
    img = Image.fromarray(color)
    if not args.no_legend:
        items = [(lbl, col) for lbl, _, col in CAMERA_TYPES if lbl in used_types]
        img = _add_legend(img, items, font_size=args.legend_font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glb", type=Path, default=None,
                   help="Render directly from a previously-exported GLB (fast)")
    p.add_argument("--scene_path", type=Path)
    p.add_argument("--scene_paths", nargs="+", default=None)
    p.add_argument("--dataset_root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--town", default=None)
    p.add_argument("--weather", default=None)
    p.add_argument("--duration", default=None, choices=[None, "short", "mid"])
    p.add_argument("--scene_type", default="dynamic", choices=["static", "dynamic"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--downsample_factor", type=int, default=2)
    p.add_argument("--point_stride", type=int, default=4)
    p.add_argument("--max_depth", type=float, default=200.0)
    p.add_argument("--frustum_scale", type=float, default=2.25)
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument("--all_frustums", action="store_true", default=True)
    p.add_argument("--no_all_frustums", dest="all_frustums", action="store_false")
    p.add_argument("--flip_y", action="store_true", default=True)
    p.add_argument("--no_flip_y", dest="flip_y", action="store_false")
    # render
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1600)
    p.add_argument("--fov", type=float, default=45.0)
    p.add_argument("--point_size", type=float, default=2.0)
    p.add_argument("--cam_pos", default=None, help="x,y,z eye position (post-flip)")
    p.add_argument("--cam_target", default=None, help="x,y,z look-at")
    p.add_argument("--legend_font", type=int, default=28)
    p.add_argument("--saturation", type=float, default=1.5,
                   help="Point-cloud color saturation multiplier (1.0 = no change)")
    p.add_argument("--no_legend", action="store_true")
    main(p.parse_args())
