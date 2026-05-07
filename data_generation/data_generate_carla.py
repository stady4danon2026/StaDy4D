"""
CARLA Multi-Scene Data Generator v2 — HuggingFace-ready output format.

Changes from v1:
  - "video" → "scene" naming throughout
  - Depth saved as safetensors (stacked [N,H,W] float16) instead of per-frame .npy
  - Extrinsics saved as safetensors (stacked [N,4,4] float32) instead of per-frame .npy
  - Per-frame intrinsic .npy removed (constant per scene → in metadata.json)
  - depth_vis/ dropped (trivially regenerable from depth)
  - RGB saved as lossless MP4 (CRF 0) instead of per-frame PNGs
  - Flat scene indexing (no Town subdirectory in output)
  - Camera labels shortened: camera_XX → cam_XX

Usage:
    python data_generate_carla.py --config configs/config_large_scale-short.json
"""

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from safetensors.numpy import save_file
from scipy.spatial.transform import Rotation as R

import carla
from camera_trajectory_generators import (
    generate_car_forward_trajectory,
    generate_cctv_trajectory,
    generate_drone_forward_trajectory,
    generate_orbit_building_trajectory,
    generate_orbit_crossroad_trajectory,
    generate_pedestrian_trajectory,
    generate_round_building_trajectory,
)

# Camera types that use actor-attached cameras (dynamic-first)
ATTACHED_TYPES = {"car_forward", "pedestrian"}


def town_prefix(town: str) -> str:
    """Town01 → T01, Town10HD → T10HD"""
    return town.replace("Town", "T").replace("HD", "")


# ============================================================
# Configuration
# ============================================================
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def save_progress(output_dir, map_name, scene_idx):
    progress_file = Path(output_dir) / "progress.json"
    with open(progress_file, "w") as f:
        json.dump({"last_map": map_name, "last_scene": scene_idx, "timestamp": time.time()}, f, indent=2)


def load_progress(output_dir):
    progress_file = Path(output_dir) / "progress.json"
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    return None


def reset_world_sync_state(world):
    if world is None:
        return
    try:
        settings = world.get_settings()
        if settings.synchronous_mode or settings.fixed_delta_seconds is not None:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        world.tick()
    except RuntimeError as e:
        print(f"  Warning: Failed to reset world: {e}")


# ============================================================
# Transform / Math Helpers
# ============================================================
def depth_to_meters(depth_img):
    depth_img = depth_img.astype(np.int32)
    encoded = depth_img[:, :, 2] + depth_img[:, :, 1] * 256 + depth_img[:, :, 0] * 256 * 256
    depth_norm = encoded / float(256 * 256 * 256 - 1)
    return (depth_norm * 1000.0).astype(np.float16)


def compute_intrinsics(w, h, fov_deg):
    f = w / (2 * np.tan(fov_deg * np.pi / 360))
    return {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "width": w, "height": h, "fov_deg": fov_deg}


def transform_to_dict(t):
    loc = t.location
    rot = t.rotation
    return {
        "location": {"x": loc.x, "y": loc.y, "z": loc.z},
        "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
    }


def transform_to_matrix(t):
    """Convert CARLA transform to 4x4 c2w matrix (CV convention: X=right, Y=down, Z=forward)."""
    loc = t.location
    rot = t.rotation
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [rot.pitch, rot.yaw, rot.roll], degrees=True).as_matrix()
    T[:3, 3] = [loc.y, loc.z, loc.x]
    T[1, 3] *= -1
    return T


def intrinsics_to_matrix(d):
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]])


def pose_to_transform(pose):
    """Convert a trajectory pose tuple to carla.Transform."""
    if len(pose) == 4:
        x, y, z, yaw = pose
        pitch, roll = -20, 0
    else:
        x, y, z, pitch, yaw, roll = pose
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


# ============================================================
# Camera Helpers
# ============================================================
def spawn_camera(world, width=640, height=360, fov=70):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(width))
    depth_bp.set_attribute("image_size_y", str(height))
    depth_bp.set_attribute("fov", str(fov))
    cam = world.spawn_actor(cam_bp, carla.Transform())
    depth_cam = world.spawn_actor(depth_bp, carla.Transform())
    return cam, depth_cam


def setup_camera_buffers(cam_rgb, cam_depth):
    """Attach listen callbacks and return buffer dicts."""
    rgb_buf, depth_buf = {}, {}

    def make_rgb_cb(buf):
        def cb(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            buf["img"] = arr[:, :, :3].copy()
        return cb

    def make_depth_cb(buf):
        def cb(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            buf["img"] = arr.copy()
        return cb

    cam_rgb.listen(make_rgb_cb(rgb_buf))
    cam_depth.listen(make_depth_cb(depth_buf))
    return rgb_buf, depth_buf


def make_canvas(h, w, n, dtype=np.uint8):
    rows = math.ceil(n / 3)
    cols = min(n, 3)
    return np.zeros((rows * h, cols * w, 3), dtype=dtype)


def update_canvas(canvas, idx, image, h, w):
    r, c = idx // 3, idx % 3
    canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = image


# ============================================================
# Map / Spawn Points
# ============================================================
def get_categorized_spawn_points(world):
    vehicle_sps = world.get_map().get_spawn_points()
    random.shuffle(vehicle_sps)

    pedestrian_sps = []
    for _ in vehicle_sps:
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            pedestrian_sps.append(carla.Transform(loc))

    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    traffic_sps = [tl.get_transform() for tl in traffic_lights]
    random.shuffle(traffic_sps)

    return {
        "vehicles": vehicle_sps,
        "pedestrians": pedestrian_sps,
        "drones": vehicle_sps,
        "traffic": traffic_sps,
    }


# ============================================================
# Trajectory Generation
# ============================================================
TRAJECTORY_GENERATORS = {
    "car_forward": generate_car_forward_trajectory,
    "drone_forward": generate_drone_forward_trajectory,
    "orbit_building": generate_orbit_building_trajectory,
    "orbit_crossroad": generate_orbit_crossroad_trajectory,
    "cctv": generate_cctv_trajectory,
    "pedestrian": generate_pedestrian_trajectory,
    "round_building": generate_round_building_trajectory,
}


def generate_single_video_trajectory(video_idx, frames, traj_type, spawn_points=None):
    gen = TRAJECTORY_GENERATORS.get(traj_type)
    if gen is None:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    return gen(video_idx, frames, spawn_points=spawn_points)


# ============================================================
# Ego Actor Spawning (attached camera types)
# ============================================================
def _vehicle_camera_transform(bb, mount_type):
    roof_z = bb.location.z + bb.extent.z + 0.1
    if mount_type == "forward":
        return carla.Transform(
            carla.Location(x=bb.location.x + bb.extent.x + 0.2, z=roof_z),
            carla.Rotation(pitch=-3),
        )
    elif mount_type == "back":
        return carla.Transform(
            carla.Location(x=bb.location.x - bb.extent.x - 0.2, z=roof_z),
            carla.Rotation(pitch=-3, yaw=180),
        )
    elif mount_type == "left":
        return carla.Transform(
            carla.Location(y=-bb.extent.y - 0.2, z=roof_z),
            carla.Rotation(pitch=-3, yaw=-90),
        )
    elif mount_type == "right":
        return carla.Transform(
            carla.Location(y=bb.extent.y + 0.2, z=roof_z),
            carla.Rotation(pitch=-3, yaw=90),
        )
    return carla.Transform(
        carla.Location(x=bb.location.x + bb.extent.x + 0.2, z=roof_z),
        carla.Rotation(pitch=-3),
    )


VEHICLE_MOUNT_TYPES = ["forward", "forward", "back", "left", "right"]


def spawn_ego_vehicles(world, bp_lib, spawn_points, n, cam_cfg, tm_port):
    vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                   if not bp.has_attribute("number_of_wheels")
                   or int(bp.get_attribute("number_of_wheels")) >= 4]
    egos = []
    shuffled_spawns = list(spawn_points)
    random.shuffle(shuffled_spawns)
    for i in range(min(n, len(shuffled_spawns))):
        vehicle = world.try_spawn_actor(random.choice(vehicle_bps), shuffled_spawns[i])
        if vehicle is None:
            continue
        vehicle.set_autopilot(True, tm_port)

        bb = vehicle.bounding_box
        mount_type = VEHICLE_MOUNT_TYPES[len(egos) % len(VEHICLE_MOUNT_TYPES)]
        cam_t = _vehicle_camera_transform(bb, mount_type)

        cam_rgb_bp = bp_lib.find("sensor.camera.rgb")
        cam_rgb_bp.set_attribute("image_size_x", str(cam_cfg["width"]))
        cam_rgb_bp.set_attribute("image_size_y", str(cam_cfg["height"]))
        cam_rgb_bp.set_attribute("fov", str(cam_cfg["fov"]))
        cam_depth_bp = bp_lib.find("sensor.camera.depth")
        cam_depth_bp.set_attribute("image_size_x", str(cam_cfg["width"]))
        cam_depth_bp.set_attribute("image_size_y", str(cam_cfg["height"]))
        cam_depth_bp.set_attribute("fov", str(cam_cfg["fov"]))

        cam_rgb = world.spawn_actor(cam_rgb_bp, cam_t, attach_to=vehicle)
        cam_depth = world.spawn_actor(cam_depth_bp, cam_t, attach_to=vehicle)
        rgb_buf, depth_buf = setup_camera_buffers(cam_rgb, cam_depth)

        egos.append({
            "actor": vehicle, "cam_rgb": cam_rgb, "cam_depth": cam_depth,
            "rgb_buffer": rgb_buf, "depth_buffer": depth_buf, "type": "vehicle",
        })
    return egos


def spawn_ego_walkers(world, bp_lib, n, cam_cfg):
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    ctrl_bp = bp_lib.find("controller.ai.walker")
    egos = []
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
        if walker is None:
            continue
        world.tick()
        try:
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        except Exception:
            walker.destroy()
            continue
        ctrl.start()
        dest = world.get_random_location_from_navigation()
        if dest:
            ctrl.go_to_location(dest)
        ctrl.set_max_speed(1.2 + random.random() * 0.8)

        bb = walker.bounding_box
        cam_t = carla.Transform(
            carla.Location(x=0.2, z=bb.location.z + bb.extent.z * 0.85),
            carla.Rotation(pitch=-5),
        )

        cam_rgb_bp = bp_lib.find("sensor.camera.rgb")
        cam_rgb_bp.set_attribute("image_size_x", str(cam_cfg["width"]))
        cam_rgb_bp.set_attribute("image_size_y", str(cam_cfg["height"]))
        cam_rgb_bp.set_attribute("fov", str(cam_cfg["fov"]))
        cam_depth_bp = bp_lib.find("sensor.camera.depth")
        cam_depth_bp.set_attribute("image_size_x", str(cam_cfg["width"]))
        cam_depth_bp.set_attribute("image_size_y", str(cam_cfg["height"]))
        cam_depth_bp.set_attribute("fov", str(cam_cfg["fov"]))

        cam_rgb = world.spawn_actor(cam_rgb_bp, cam_t, attach_to=walker)
        cam_depth = world.spawn_actor(cam_depth_bp, cam_t, attach_to=walker)
        rgb_buf, depth_buf = setup_camera_buffers(cam_rgb, cam_depth)

        egos.append({
            "actor": walker, "controller": ctrl, "cam_rgb": cam_rgb, "cam_depth": cam_depth,
            "rgb_buffer": rgb_buf, "depth_buffer": depth_buf, "type": "walker",
        })
    return egos


# ============================================================
# Background Traffic
# ============================================================
def spawn_background_traffic(world, bp_lib, n_vehicles, n_walkers, tm_port):
    actors = []
    walker_controllers = []

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicle_bps = bp_lib.filter("vehicle.*")

    for i in range(min(n_vehicles, len(spawn_points))):
        actor = world.try_spawn_actor(random.choice(vehicle_bps), spawn_points[i])
        if actor:
            actors.append(actor)
            actor.set_autopilot(True, tm_port)

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    ctrl_bp = bp_lib.find("controller.ai.walker")

    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
        if walker is None:
            continue
        actors.append(walker)
        world.tick()
        try:
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            actors.append(ctrl)
            walker_controllers.append((walker, ctrl))
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.4 + random.random() * 1.0)
        except Exception:
            continue

    print(f"  Traffic: {len(actors)} actors")
    return actors, walker_controllers


# ============================================================
# Cleanup
# ============================================================
def cleanup_actors(actors, walker_controllers=None):
    if walker_controllers is None:
        walker_controllers = []
    ctrl_ids = set()
    for _, ctrl in walker_controllers:
        ctrl_ids.add(ctrl.id)
        if ctrl.is_alive:
            ctrl.stop()
            ctrl.destroy()
    for actor in actors:
        if actor.id in ctrl_ids:
            continue
        if actor.is_alive:
            if actor.type_id.startswith("vehicle."):
                actor.set_autopilot(False)
            actor.destroy()


def cleanup_egos(egos):
    for ego in egos:
        ego["cam_rgb"].stop()
        ego["cam_depth"].stop()
        ego["cam_rgb"].destroy()
        ego["cam_depth"].destroy()
        if "controller" in ego:
            ego["controller"].stop()
            ego["controller"].destroy()
        if ego["actor"].is_alive:
            if ego["type"] == "vehicle":
                ego["actor"].set_autopilot(False)
            ego["actor"].destroy()


# ============================================================
# 3D Actor State Recording
# ============================================================
def snapshot_actor_states(world, ego_actor_ids):
    result = []
    for actor in world.get_actors():
        tid = actor.type_id
        if not (tid.startswith("vehicle.") or tid.startswith("walker.pedestrian.")):
            continue
        t = actor.get_transform()
        v = actor.get_velocity()
        bb = actor.bounding_box
        result.append({
            "id": actor.id, "type_id": tid, "is_ego": actor.id in ego_actor_ids,
            "location": {"x": round(t.location.x, 4), "y": round(t.location.y, 4), "z": round(t.location.z, 4)},
            "rotation": {"pitch": round(t.rotation.pitch, 4), "yaw": round(t.rotation.yaw, 4), "roll": round(t.rotation.roll, 4)},
            "velocity": {"x": round(v.x, 4), "y": round(v.y, 4), "z": round(v.z, 4)},
            "bbox_extent": {"x": round(bb.extent.x, 4), "y": round(bb.extent.y, 4), "z": round(bb.extent.z, 4)},
            "bbox_center": {"x": round(bb.location.x, 4), "y": round(bb.location.y, 4), "z": round(bb.location.z, 4)},
        })
    return result


# ============================================================
# Save helpers — safetensors for depth & extrinsics
# ============================================================
def save_camera_safetensors(cam_dir: Path, depth_list: list, extrinsic_list: list, intrinsic_list: list):
    """Save accumulated depth, extrinsics, and intrinsics as safetensors."""
    if depth_list:
        depth_stack = np.stack(depth_list, axis=0)  # [N, H, W] float16
        save_file({"depth": depth_stack}, str(cam_dir / "depth.safetensors"))

    if extrinsic_list:
        ext_stack = np.stack(extrinsic_list, axis=0).astype(np.float32)  # [N, 4, 4]
        save_file({"c2w": ext_stack}, str(cam_dir / "extrinsics.safetensors"))

    if intrinsic_list:
        intr_stack = np.stack(intrinsic_list, axis=0).astype(np.float32)  # [N, 3, 3]
        save_file({"K": intr_stack}, str(cam_dir / "intrinsics.safetensors"))


def save_rgb_mp4(cam_dir: Path, rgb_list: list, fps: int = 10):
    """Save accumulated RGB frames as lossless MP4 (CRF 0, BGR24)."""
    if not rgb_list:
        return
    h, w = rgb_list[0].shape[:2]
    mp4_path = cam_dir / "rgb.mp4"

    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-crf", "0",
        "-preset", "ultrafast",
        "-pix_fmt", "bgr24",
        "-y",
        str(mp4_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in rgb_list:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        print(f"    WARNING: ffmpeg failed for {mp4_path}: {proc.stderr.read().decode()[-200:]}")


# ============================================================
# Phase 1: Dynamic Collection
# ============================================================
def collect_dynamic(world, cameras, num_frames, ego_actor_ids, scene_dir, cam_cfg):
    """
    Collect dynamic scene. Saves:
      - RGB accumulated → rgb.mp4 (lossless CRF 0) at end
      - Depth accumulated → depth.safetensors at end
      - Extrinsics accumulated → extrinsics.safetensors at end
    """
    for cam in cameras:
        label = cam["label"].replace("camera_", "cam_")
        d = scene_dir / "dynamic" / label
        cam["out_dir"] = d
        d.mkdir(parents=True, exist_ok=True)

        w = int(cam["cam_rgb"].attributes["image_size_x"])
        h = int(cam["cam_rgb"].attributes["image_size_y"])
        fov = float(cam["cam_rgb"].attributes["fov"])
        cam["intrinsics"] = compute_intrinsics(w, h, fov)
        cam["collected_frames"] = 0
        cam["rgb_accum"] = []
        cam["depth_accum"] = []
        cam["ext_accum"] = []
        cam["intr_accum"] = []

    canvas = make_canvas(cam_cfg["height"], cam_cfg["width"], len(cameras))
    actor_frames = []

    for idx in range(num_frames):
        # Move free cameras before tick
        for cam in cameras:
            if not cam["is_attached"] and idx < len(cam["trajectory"]):
                cam["cam_rgb"].set_transform(cam["trajectory"][idx])
                cam["cam_depth"].set_transform(cam["trajectory"][idx])

        world.tick()
        actor_frames.append(snapshot_actor_states(world, ego_actor_ids))

        for ci, cam in enumerate(cameras):
            if cam["is_attached"]:
                current_t = cam["cam_rgb"].get_transform()
                cam["trajectory"].append(current_t)
            else:
                if idx >= len(cam["trajectory"]):
                    continue
                current_t = cam["trajectory"][idx]

            if "img" not in cam["rgb_buffer"] or "img" not in cam["depth_buffer"]:
                continue

            rgb = cam["rgb_buffer"]["img"]
            depth = depth_to_meters(cam["depth_buffer"]["img"])

            cam["rgb_accum"].append(rgb.copy())
            update_canvas(canvas, ci, rgb, cam_cfg["height"], cam_cfg["width"])

            # Accumulate depth, extrinsics, and intrinsics
            cam["depth_accum"].append(depth)
            cam["ext_accum"].append(transform_to_matrix(current_t))
            cam["intr_accum"].append(intrinsics_to_matrix(cam["intrinsics"]))

            cam["collected_frames"] += 1

        if idx % 10 == 0 or idx == num_frames - 1:
            print(f"\r    Frame {idx + 1}/{num_frames}", end="", flush=True)
    print()

    # Save accumulated data
    for cam in cameras:
        save_rgb_mp4(cam["out_dir"], cam["rgb_accum"], fps=cam_cfg.get("fps", 10))
        save_camera_safetensors(cam["out_dir"], cam["depth_accum"], cam["ext_accum"], cam["intr_accum"])
        print(f"    {cam['label']}: {cam['collected_frames']} frames")
        del cam["rgb_accum"], cam["depth_accum"], cam["ext_accum"], cam["intr_accum"]

    # Build trajectory dict
    all_trajectories = {cam["label"]: list(cam["trajectory"]) for cam in cameras}
    intrinsics_map = {cam["label"]: cam["intrinsics"] for cam in cameras}

    # Build actor tracks
    tracks = {}
    for fi, frame_actors in enumerate(actor_frames):
        for a in frame_actors:
            aid = a["id"]
            if aid not in tracks:
                tracks[aid] = {
                    "type_id": a["type_id"], "is_ego": a["is_ego"],
                    "bbox_extent": a["bbox_extent"], "bbox_center": a["bbox_center"],
                    "frames": [],
                }
            tracks[aid]["frames"].append({
                "frame": fi, "location": a["location"],
                "rotation": a["rotation"], "velocity": a["velocity"],
            })

    actor_trajectories = {"num_frames": num_frames, "num_actors": len(tracks), "tracks": list(tracks.values())}

    with open(scene_dir / "actors.json", "w") as f:
        json.dump(actor_trajectories, f, indent=2)
    print(f"    Recorded {len(tracks)} actor trajectories")

    return all_trajectories, intrinsics_map, actor_trajectories


# ============================================================
# Phase 2: Static Replay
# ============================================================
def collect_static(world, all_trajectories, intrinsics_map, scene_dir, cam_cfg):
    """Replay all camera trajectories on empty world."""
    labels = list(all_trajectories.keys())
    num_frames = max(len(t) for t in all_trajectories.values())

    cameras = []
    for label in labels:
        cam_rgb, cam_depth = spawn_camera(world, cam_cfg["width"], cam_cfg["height"], cam_cfg["fov"])
        rgb_buf, depth_buf = setup_camera_buffers(cam_rgb, cam_depth)

        short_label = label.replace("camera_", "cam_")
        d = scene_dir / "static" / short_label
        d.mkdir(parents=True, exist_ok=True)

        cam = {
            "label": label, "cam_rgb": cam_rgb, "cam_depth": cam_depth,
            "rgb_buffer": rgb_buf, "depth_buffer": depth_buf,
            "out_dir": d, "intrinsics": intrinsics_map[label],
            "collected_frames": 0, "rgb_accum": [], "depth_accum": [], "ext_accum": [], "intr_accum": [],
        }
        cameras.append(cam)

    canvas = make_canvas(cam_cfg["height"], cam_cfg["width"], len(cameras))

    # Warmup tick
    for cam in cameras:
        ts = all_trajectories[cam["label"]]
        if ts:
            cam["cam_rgb"].set_transform(ts[0])
            cam["cam_depth"].set_transform(ts[0])
    world.tick()

    for idx in range(num_frames):
        for ci, cam in enumerate(cameras):
            ts = all_trajectories[cam["label"]]
            if idx < len(ts):
                cam["cam_rgb"].set_transform(ts[idx])
                cam["cam_depth"].set_transform(ts[idx])
                if ci == 0:
                    world.get_spectator().set_transform(ts[idx])

        world.tick()

        for ci, cam in enumerate(cameras):
            ts = all_trajectories[cam["label"]]
            if idx >= len(ts):
                continue
            if "img" not in cam["rgb_buffer"] or "img" not in cam["depth_buffer"]:
                continue

            rgb = cam["rgb_buffer"]["img"]
            depth = depth_to_meters(cam["depth_buffer"]["img"])

            cam["rgb_accum"].append(rgb.copy())
            update_canvas(canvas, ci, rgb, cam_cfg["height"], cam_cfg["width"])

            cam["depth_accum"].append(depth)
            cam["ext_accum"].append(transform_to_matrix(ts[idx]))
            cam["intr_accum"].append(intrinsics_to_matrix(cam["intrinsics"]))
            cam["collected_frames"] += 1

        if idx % 10 == 0 or idx == num_frames - 1:
            print(f"\r    Frame {idx + 1}/{num_frames}", end="", flush=True)
    print()

    for cam in cameras:
        save_rgb_mp4(cam["out_dir"], cam["rgb_accum"], fps=cam_cfg.get("fps", 10))
        save_camera_safetensors(cam["out_dir"], cam["depth_accum"], cam["ext_accum"], cam["intr_accum"])
        print(f"    {cam['label']}: {cam['collected_frames']} frames")
        cam["cam_rgb"].stop()
        cam["cam_depth"].stop()
        cam["cam_rgb"].destroy()
        cam["cam_depth"].destroy()


# ============================================================
# Metadata
# ============================================================
def save_scene_metadata(scene_dir, cameras, weather_name, config, map_name, flat_idx, scene_seed):
    scene_dir.mkdir(parents=True, exist_ok=True)
    intr = cameras[0]["intrinsics"]

    initial_extrinsics = []
    for cam in cameras:
        if not cam["trajectory"]:
            continue
        t = cam["trajectory"][0]
        initial_extrinsics.append({
            "camera": cam["label"].replace("camera_", "cam_"),
            "traj_type": cam["traj_type"],
            "is_attached": cam["is_attached"],
            "location": {"x": float(t.location.x), "y": float(t.location.y), "z": float(t.location.z)},
            "rotation": {"pitch": float(t.rotation.pitch), "yaw": float(t.rotation.yaw), "roll": float(t.rotation.roll)},
            "c2w_matrix": transform_to_matrix(t).tolist(),
        })

    metadata = {
        "scene_name": scene_dir.name,
        "map_name": map_name,
        "town": town_prefix(map_name),
        "scene_idx": flat_idx,
        "scene_seed": scene_seed,
        "num_cameras": len(cameras),
        "num_frames": len(cameras[0]["trajectory"]) if cameras else 0,
        "fps": config["video_generation"]["fps"],
        "resolution": {"width": intr["width"], "height": intr["height"]},
        "fov_deg": intr["fov_deg"],
        "n_vehicles": config["actors"]["n_vehicles"],
        "n_walkers": config["actors"]["n_walkers"],
        "weather": weather_name,
        "camera_types": [cam["traj_type"] for cam in cameras],
        # Intrinsic parameters (shared across all cameras in this scene)
        "intrinsic": intr,
        "intrinsic_matrix": [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "initial_extrinsics": initial_extrinsics,
    }

    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ============================================================
# GIF Creation
# ============================================================
def create_gif_for_scene(base_dir, scene_dir, fps=10, scale=0.5):
    base_path = Path(base_dir)
    preview_dir = base_path.parent / "preview" / base_path.name
    preview_dir.mkdir(parents=True, exist_ok=True)
    gifs_created = 0

    for mp4_file in sorted(Path(scene_dir).rglob("rgb.mp4")):
        cam_dir = mp4_file.parent
        rel = cam_dir.relative_to(base_path)
        gif_name = str(rel).replace("/", "_").replace("\\", "_") + ".gif"
        gif_path = preview_dir / gif_name

        vf_scale = f"scale=iw*{scale}:ih*{scale}:flags=bilinear," if scale != 1.0 else ""
        vf_filters = f"{vf_scale}split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        cmd = [
            "ffmpeg",
            "-i", str(mp4_file),
            "-vf", vf_filters, "-loop", "0",
            "-y", str(gif_path),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print(f"    [GIF] {gif_name}")
                gifs_created += 1
            else:
                print(f"    [GIF] FAIL {gif_name}")
        except FileNotFoundError:
            print("    [GIF] ffmpeg not found")
            return 0

    return gifs_created


# ============================================================
# Parquet Index
# ============================================================
def build_index(output_root: Path):
    """Build a flat parquet index over all generated scenes."""
    rows = []
    for meta_file in sorted(output_root.rglob("metadata.json")):
        scene_dir = meta_file.parent
        with open(meta_file) as f:
            meta = json.load(f)

        n_cameras = 0
        n_frames = 0
        for render_type in ("dynamic", "static"):
            rd = scene_dir / render_type
            if rd.is_dir():
                cams = [d for d in rd.iterdir() if d.is_dir()]
                n_cameras = max(n_cameras, len(cams))
                for cam in cams:
                    mp4 = cam / "rgb.mp4"
                    if mp4.exists():
                        try:
                            cap = cv2.VideoCapture(str(mp4))
                            n_frames = max(n_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                            cap.release()
                        except Exception:
                            pass

        rel = scene_dir.relative_to(output_root)
        rows.append({
            "scene_name": meta.get("scene_name", scene_dir.name),
            "path": str(rel),
            "town": meta.get("town", meta.get("map_name", "unknown")),
            "weather": meta.get("weather", "unknown"),
            "n_cameras": n_cameras,
            "n_frames": n_frames,
            "n_vehicles": meta.get("n_vehicles", 0),
            "n_walkers": meta.get("n_walkers", 0),
            "width": meta.get("resolution", {}).get("width", 0),
            "height": meta.get("resolution", {}).get("height", 0),
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(output_root / "index.parquet", index=False)
        print(f"\nIndex: {len(df)} scenes → {output_root / 'index.parquet'}")
        print(df.groupby("town").size().to_string())


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CARLA Multi-Scene Data Generator v2")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    args = parser.parse_args()

    print("=" * 60)
    print("CARLA Multi-Scene Data Generator v2 (HF-ready format)")
    print("  Dynamic-first | Attached + Free cameras | Weather x Scenes")
    print("=" * 60)
    config = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Description: {config.get('description', 'N/A')}")

    seed = config.get("random_seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    maps = config["maps"]
    scenes_per_map = config["video_generation"]["videos_per_map"]
    fps = config["video_generation"]["fps"]
    frames_per_scene = config["video_generation"]["video_duration_sec"] * fps
    output_base = Path(config["output"]["base_dir"])
    weather_list = config["weather"] if isinstance(config["weather"], list) else [config["weather"]]
    multi_camera_types = config.get("multi_camera_types", ["car_forward", "drone_forward", "pedestrian"])
    cam_cfg = config["camera"]

    total_per_map = scenes_per_map * len(weather_list)
    n_ego_vehicles = sum(1 for t in multi_camera_types if t == "car_forward")
    n_ego_walkers = sum(1 for t in multi_camera_types if t == "pedestrian")

    print(f"\n  Maps: {len(maps)} | Scenes/map: {scenes_per_map} x {len(weather_list)} weathers = {total_per_map}")
    print(f"  Duration: {config['video_generation']['video_duration_sec']}s @ {fps} FPS = {frames_per_scene} frames")
    print(f"  Cameras: {len(multi_camera_types)} ({n_ego_vehicles} attached vehicles, {n_ego_walkers} attached walkers, {len(multi_camera_types) - n_ego_vehicles - n_ego_walkers} free)")
    print(f"  Types: {', '.join(multi_camera_types)}")
    print(f"  Weathers: {', '.join(weather_list)}")
    print(f"  Traffic: {config['actors']['n_vehicles']}v + {config['actors']['n_walkers']}w")
    print(f"  Resolution: {cam_cfg['width']}x{cam_cfg['height']} | Output: {output_base}/")
    print(f"  Format: MP4 lossless (rgb) + safetensors (depth, extrinsics, intrinsics) + JSON (metadata)")
    print("=" * 60)

    # Resume
    disable_resume = config["execution"].get("disable_resume", False)
    resume_from_map, resume_from_scene = None, None
    if not disable_resume:
        progress = load_progress(output_base)
        resume_from_map = config["execution"].get("resume_from_map") or (progress and progress.get("last_map"))
        resume_from_scene = config["execution"].get("resume_from_video") or (progress and progress.get("last_scene", progress.get("last_video", -1)))
        if resume_from_map:
            print(f"\nRESUME: map='{resume_from_map}', scene>{resume_from_scene}")

    # Connect
    carla_cfg = config["carla"]
    client = carla.Client(carla_cfg["host"], carla_cfg["port"])
    client.set_timeout(carla_cfg["timeout"])
    print(f"\nConnected to CARLA at {carla_cfg['host']}:{carla_cfg['port']}")

    total_scenes = 0
    skip_mode = resume_from_map is not None
    world = None

    for map_idx, map_name in enumerate(maps):
        if world is not None:
            reset_world_sync_state(world)

        if skip_mode:
            if map_name != resume_from_map:
                print(f"\nSkipping {map_name} (resume)")
                continue
            else:
                skip_mode = False

        print(f"\n{'=' * 60}\nMAP {map_idx + 1}/{len(maps)}: {map_name}\n{'=' * 60}")
        tm_port = 8000 + map_idx

        try:
            world = client.load_world(map_name)
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / fps
            world.apply_settings(settings)
            tm = client.get_trafficmanager(tm_port)
            tm.set_synchronous_mode(True)
            world.tick()
            print(f"  {map_name} loaded")
        except Exception as e:
            print(f"  ERROR loading {map_name}: {e}")
            continue

        bp_lib = world.get_blueprint_library()
        spawn_points = get_categorized_spawn_points(world)

        start_scene = 0
        if map_name == resume_from_map and resume_from_scene is not None:
            start_scene = resume_from_scene + 1

        for scene_idx in range(scenes_per_map):
            for weather_idx, weather_name in enumerate(weather_list):
                flat_idx = scene_idx * len(weather_list) + weather_idx

                if flat_idx < start_scene:
                    continue

                scene_name = f"scene_{town_prefix(map_name)}_{flat_idx:03d}"
                print(f"\n--- {map_name} / {scene_name} (seed={scene_idx}, weather={weather_name}) ---")

                weather = getattr(carla.WeatherParameters, weather_name, carla.WeatherParameters.ClearNoon)
                world.set_weather(weather)

                scene_dir = output_base / scene_name

                # Skip existing
                if config["execution"]["skip_existing"]:
                    static_ok = (scene_dir / "static").exists() and any((scene_dir / "static").glob("cam_*"))
                    dynamic_ok = (scene_dir / "dynamic").exists() and any((scene_dir / "dynamic").glob("cam_*"))
                    if static_ok and dynamic_ok:
                        print(f"  Already exists, skipping...")
                        continue

                # Build unified camera list
                ego_vehicles = spawn_ego_vehicles(world, bp_lib, spawn_points["vehicles"], n_ego_vehicles, cam_cfg, tm_port)
                ego_walkers = spawn_ego_walkers(world, bp_lib, n_ego_walkers, cam_cfg)
                print(f"  Ego: {len(ego_vehicles)} vehicles + {len(ego_walkers)} walkers")

                cameras = []
                v_idx, w_idx = 0, 0
                for ci, traj_type in enumerate(multi_camera_types):
                    label = f"camera_{ci:02d}_{traj_type}"

                    if traj_type == "car_forward" and v_idx < len(ego_vehicles):
                        ego = ego_vehicles[v_idx]
                        v_idx += 1
                        cameras.append({
                            "label": label, "traj_type": traj_type, "is_attached": True,
                            "cam_rgb": ego["cam_rgb"], "cam_depth": ego["cam_depth"],
                            "rgb_buffer": ego["rgb_buffer"], "depth_buffer": ego["depth_buffer"],
                            "trajectory": [],
                        })
                    elif traj_type == "pedestrian" and w_idx < len(ego_walkers):
                        ego = ego_walkers[w_idx]
                        w_idx += 1
                        cameras.append({
                            "label": label, "traj_type": traj_type, "is_attached": True,
                            "cam_rgb": ego["cam_rgb"], "cam_depth": ego["cam_depth"],
                            "rgb_buffer": ego["rgb_buffer"], "depth_buffer": ego["depth_buffer"],
                            "trajectory": [],
                        })
                    else:
                        cam_rgb, cam_depth = spawn_camera(world, cam_cfg["width"], cam_cfg["height"], cam_cfg["fov"])
                        rgb_buf, depth_buf = setup_camera_buffers(cam_rgb, cam_depth)
                        traj_seed = flat_idx * 100 + ci
                        traj = generate_single_video_trajectory(traj_seed, frames_per_scene, traj_type, spawn_points)
                        transforms = [pose_to_transform(p) for p in traj]
                        cameras.append({
                            "label": label, "traj_type": traj_type, "is_attached": False,
                            "cam_rgb": cam_rgb, "cam_depth": cam_depth,
                            "rgb_buffer": rgb_buf, "depth_buffer": depth_buf,
                            "trajectory": transforms,
                        })

                if not cameras:
                    print("  No cameras spawned, skipping...")
                    continue

                traffic_actors, traffic_ctrls = spawn_background_traffic(
                    world, bp_lib, config["actors"]["n_vehicles"], config["actors"]["n_walkers"], tm_port)

                for cam in cameras:
                    if not cam["is_attached"] and cam["trajectory"]:
                        cam["cam_rgb"].set_transform(cam["trajectory"][0])
                        cam["cam_depth"].set_transform(cam["trajectory"][0])
                print("  Settling...")
                for _ in range(30):
                    world.tick()

                # PHASE 1: DYNAMIC
                ego_ids = {ego["actor"].id for ego in ego_vehicles + ego_walkers}
                print(f"  [DYNAMIC] {frames_per_scene} frames, {len(cameras)} cameras...")
                all_trajectories, intrinsics_map, _ = collect_dynamic(
                    world, cameras, frames_per_scene, ego_ids, scene_dir, cam_cfg)

                save_scene_metadata(scene_dir, cameras, weather_name, config, map_name, flat_idx, scene_idx)

                # Cleanup all actors
                for cam in cameras:
                    if not cam["is_attached"]:
                        cam["cam_rgb"].stop()
                        cam["cam_depth"].stop()
                        cam["cam_rgb"].destroy()
                        cam["cam_depth"].destroy()
                cleanup_egos(ego_vehicles + ego_walkers)
                cleanup_actors(traffic_actors, traffic_ctrls)

                for _ in range(10):
                    world.tick()

                # PHASE 2: STATIC
                print(f"  [STATIC] Replaying {len(all_trajectories)} trajectories on empty world...")
                collect_static(world, all_trajectories, intrinsics_map, scene_dir, cam_cfg)

                print(f"  {scene_name} complete")
                total_scenes += 1

                create_gif_for_scene(output_base, scene_dir, fps=fps, scale=0.5)
                save_progress(output_base, map_name, flat_idx)

        print(f"\n{map_name} complete")

    reset_world_sync_state(world)

    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE! {total_scenes} scenes")
    print(f"{'=' * 60}")
    print(f"  {output_base}/")
    print(f"    scene_T<NN>_<IDX>/")
    print(f"      metadata.json, actors.json")
    print(f"      dynamic/cam_XX_type/  rgb.mp4  depth.safetensors  extrinsics.safetensors  intrinsics.safetensors")
    print(f"      static/cam_XX_type/   rgb.mp4  depth.safetensors  extrinsics.safetensors  intrinsics.safetensors")
    print(f"  static & dynamic share the SAME camera trajectory (paired data)")
    print(f"  car_forward & pedestrian use actor-attached cameras")
    print("=" * 60)

    build_index(output_base)


if __name__ == "__main__":
    main()
