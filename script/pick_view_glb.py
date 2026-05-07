"""Interactive GLB viewer for picking a render viewpoint.

Loads a .glb into a viser scene; click "Print pose" to dump the current camera
position + look-at in the exact form `render_world_image.py` accepts.

Usage:
    python script/pick_view_glb.py outputs/teaser/T03_ClearNoon_all.glb
    # then open http://localhost:8080 in a browser
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import viser


def main():
    p = argparse.ArgumentParser()
    p.add_argument("glb", type=Path)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--point_size", type=float, default=0.05)
    p.add_argument("--max_points", type=int, default=500_000,
                   help="Random subsample of the cloud for snappy interaction")
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth point before random subsample")
    args = p.parse_args()

    scene = trimesh.load(args.glb)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")

    # add geometries
    geoms = scene.geometry if isinstance(scene, trimesh.Scene) else {"main": scene}
    for name, geom in geoms.items():
        if isinstance(geom, trimesh.PointCloud):
            verts = np.asarray(geom.vertices, dtype=np.float32)
            colors = (geom.colors[:, :3] / 255.0) if geom.colors is not None else np.ones((len(verts), 3))
            colors = np.asarray(colors, dtype=np.float32)
            if args.stride > 1:
                verts = verts[::args.stride]
                colors = colors[::args.stride]
            if args.max_points > 0 and len(verts) > args.max_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(verts), size=args.max_points, replace=False)
                verts = verts[idx]
                colors = colors[idx]
            print(f"  cloud {name}: {len(verts):,} pts (after subsample)")
            server.scene.add_point_cloud(
                f"/pts/{name}",
                points=verts,
                colors=colors,
                point_size=args.point_size,
            )
        elif isinstance(geom, trimesh.path.Path3D):
            for entity in geom.entities:
                pts = np.asarray(geom.vertices[entity.points], dtype=np.float32)
                color = (entity.color[:3] / 255.0) if entity.color is not None else np.array([1, 0, 0])
                server.scene.add_spline_catmull_rom(
                    f"/lines/{name}_{id(entity)}",
                    positions=pts,
                    color=tuple(color),
                    line_width=3.0,
                )
        elif hasattr(geom, "vertices") and hasattr(geom, "faces"):
            colors = None
            if geom.visual is not None and hasattr(geom.visual, "face_colors"):
                colors = np.asarray(geom.visual.face_colors[:, :3] / 255.0, dtype=np.float32)
            server.scene.add_mesh_simple(
                f"/mesh/{name}",
                vertices=np.asarray(geom.vertices, dtype=np.float32),
                faces=np.asarray(geom.faces, dtype=np.int32),
                color=(0.6, 0.6, 0.6) if colors is None else (float(colors[:, 0].mean()), float(colors[:, 1].mean()), float(colors[:, 2].mean())),
            )
        print(f"  added {name}: {type(geom).__name__}")

    pose_label = server.gui.add_text("Last pose", initial_value="(click Print pose)", disabled=True)
    btn = server.gui.add_button("Print pose")

    @btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        if client is None:
            return
        pos = np.asarray(client.camera.position)
        target = np.asarray(client.camera.look_at)
        cli = (
            f"--cam_pos {pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f} "
            f"--cam_target {target[0]:.2f},{target[1]:.2f},{target[2]:.2f}"
        )
        print("\n=== camera pose ===")
        print(cli)
        print("position :", pos.tolist())
        print("look_at  :", target.tolist())
        print("wxyz     :", np.asarray(client.camera.wxyz).tolist())
        print("fov(deg) :", float(np.rad2deg(client.camera.fov)))
        pose_label.value = cli

    print(f"viser listening on http://localhost:{args.port}")
    print("Move the camera, then click 'Print pose' in the GUI.")
    while True:
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
