"""Capture top-down BEV screenshots of CARLA towns for the dataset analyzer.

For each requested town this connects to a running CARLA server, loads the
map, drops an orthographic-style RGB camera high above the map center looking
down, and saves <Town>.png + <Town>.json under --out.

The JSON sidecar holds the world-space extent (CARLA xy, meters) so the
analyzer can place trajectories in the correct frame:
    {"xmin": ..., "xmax": ..., "ymin": ..., "ymax": ...}

Usage (inside an env that has the CARLA python egg / wheel installed):

    # start the server first, e.g.:
    #   /DATA/simulators/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen
    python script/capture_carla_bev.py --out StaDy4D/town_bev \\
        --towns Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD

The default altitude/half-extent works for the StaDy4D maps; tweak with
--altitude / --half-extent if a town doesn't fit.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


DEFAULT_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05",
                 "Town06", "Town07", "Town10HD"]


def town_bbox_from_dataset(stady4d_root: Path, town: str,
                           pad: float = 25.0,
                           durations=("short", "mid"),
                           splits=("train", "test")
                           ) -> tuple[float, float, float, float] | None:
    """Scan every scene that maps to `town` and return CARLA xy bbox padded."""
    xs, ys = [], []
    for dur in durations:
        for sp in splits:
            base = stady4d_root / dur / sp
            if not base.is_dir():
                continue
            for sd in base.iterdir():
                meta_path = sd / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue
                if meta.get("map_name") != town:
                    continue
                cams_root = sd / "static" if (sd / "static").exists() else sd / "dynamic"
                if not cams_root.exists():
                    continue
                for cdir in cams_root.iterdir():
                    ext = cdir / "extrinsics.safetensors"
                    if not ext.exists():
                        continue
                    try:
                        c2w = load_file(str(ext))["c2w"]
                    except Exception:
                        continue
                    # CARLA xy: (c2w[:,2,3], c2w[:,0,3]) — see analyze_dataset
                    xs.append(c2w[:, 2, 3]); ys.append(c2w[:, 0, 3])
    if not xs:
        return None
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    return (float(xs.min() - pad), float(xs.max() + pad),
            float(ys.min() - pad), float(ys.max() + pad))


def capture_town(client, town: str, out_dir: Path, altitude: float,
                 bbox: tuple[float, float, float, float],
                 image_size: int) -> None:
    """bbox = (xmin, xmax, ymin, ymax) in CARLA world coords (meters).

    The capture is squared off (matches longest side) so we can render with
    a single FOV, but the JSON sidecar records the *actual* covered extent.
    """
    import carla
    xmin, xmax, ymin, ymax = bbox
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    print(f"[carla] loading {town}  center=({cx:+.1f}, {cy:+.1f})  half={half:.1f} m")

    world = client.load_world(town)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / 20
    world.apply_settings(settings)
    for _ in range(10):
        world.tick()

    fov_deg = 2 * math.degrees(math.atan(half / altitude))
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(image_size))
    cam_bp.set_attribute("image_size_y", str(image_size))
    cam_bp.set_attribute("fov", f"{fov_deg:.4f}")

    transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=altitude),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    camera = world.spawn_actor(cam_bp, transform)

    captured = {"img": None}
    def _cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[..., :3][..., ::-1]
        captured["img"] = np.ascontiguousarray(arr)
    camera.listen(_cb)

    # tick until we get a frame
    deadline = time.time() + 5
    while captured["img"] is None and time.time() < deadline:
        world.tick()
    camera.stop(); camera.destroy()
    if captured["img"] is None:
        raise RuntimeError(f"no frame returned for {town}")

    # CARLA camera with pitch=-90 / yaw=0 gives image axes:
    #   image-right  <->  +CARLA_y
    #   image-down   <->  +CARLA_x   (because forward of cam is -z; image rows
    #                                 increase along the cam's -x = +CARLA x)
    # so the natural extent is xmin=-half_extent (top), xmax=+half_extent (bot)
    # for vertical axis and ymin=-half_extent (left), ymax=+half_extent (right)
    # for horizontal. We rotate the image so x grows up to match the analyzer's
    # CARLA xy plot convention.
    # CARLA pitch=-90 yaw=0: image-up = world +X, image-right = world +Y.
    # Want array layout for matplotlib origin='lower' with extent (xmin..xmax,
    # ymin..ymax): row -> +Y ascending, col -> +X ascending. That's a CW 90°
    # rotation of the raw frame.
    img = np.rot90(captured["img"], k=-1)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{town}.png"
    json_path = out_dir / f"{town}.json"
    import imageio.v3 as iio
    iio.imwrite(png_path, img)
    json_path.write_text(json.dumps({
        "town": town,
        "xmin": float(cx - half), "xmax": float(cx + half),
        "ymin": float(cy - half), "ymax": float(cy + half),
        "center_x": cx, "center_y": cy,
        "half_extent": half,
        "altitude": altitude, "fov_deg": fov_deg,
        "image_size": image_size,
    }, indent=2))
    print(f"[carla] wrote {png_path}  ({img.shape[1]}x{img.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("StaDy4D/town_bev"))
    ap.add_argument("--towns", nargs="+", default=DEFAULT_TOWNS)
    ap.add_argument("--stady4d-root", type=Path, default=Path("StaDy4D"),
                    help="used to derive each town's center + extent from "
                         "actual scene trajectories")
    ap.add_argument("--bbox-pad", type=float, default=25.0,
                    help="extra meters around the trajectory bbox")
    ap.add_argument("--altitude", type=float, default=600.0,
                    help="camera height above the scene center (m)")
    ap.add_argument("--bbox-override", default=None,
                    help='manual bbox "xmin,xmax,ymin,ymax" (applies to all)')
    ap.add_argument("--image-size", type=int, default=2048)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    override = None
    if args.bbox_override:
        override = tuple(float(v) for v in args.bbox_override.split(","))
        assert len(override) == 4

    for town in args.towns:
        bbox = override or town_bbox_from_dataset(args.stady4d_root, town,
                                                   pad=args.bbox_pad)
        if bbox is None:
            print(f"  !! no scenes found in {args.stady4d_root} for {town}; "
                  f"skipping. Pass --bbox-override or check --stady4d-root.")
            continue
        try:
            capture_town(client, town, args.out, args.altitude, bbox,
                         args.image_size)
        except Exception as e:
            print(f"  !! {town} failed: {e}")


if __name__ == "__main__":
    main()
