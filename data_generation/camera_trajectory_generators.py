import numpy as np
import carla

# ============================================================
# Camera trajectory generators
# ============================================================
def generate_car_forward_trajectory(video_idx, frames_per_video=150, speed=0.8, spawn_points=None):
    """
    1. Car Forward: Move forward above the road just like a vehicle
    Camera positioned at vehicle height (2-3m), moving straight along road

    Args:
        video_idx: Video index for different paths and viewing angles
        frames_per_video: Number of frames
        speed: Movement speed in meters per frame
        spawn_points: Dictionary with 'vehicles', 'pedestrians', 'drones' spawn points
    """
    poses = []

    # Different horizontal viewing angles for variety (dashcam perspectives)
    # Cycles through: center, slight right, slight left, right, left, wide right, wide left
    view_angles = [
        0.0,
        0.0,
        0.0,
        90.0,
        180.0,
        270.0,
    ]
    view_angle_offset = view_angles[video_idx % len(view_angles)]

    # Extract vehicle spawn points from dictionary
    vehicle_spawns = spawn_points['vehicles'] if spawn_points and isinstance(spawn_points, dict) else (spawn_points if spawn_points else [])

    start_x, start_y, start_z, start_yaw = 0, 0, 2.5, 90
    # Use spawn points if available (much better than map center!)
    if vehicle_spawns and len(vehicle_spawns) > 0:
        # Select a spawn point based on video_idx
        spawn_point = vehicle_spawns[video_idx % len(vehicle_spawns)]
        start_x = spawn_point.location.x
        start_y = spawn_point.location.y
        start_z = 2.5  # Vehicle camera height
        start_yaw = spawn_point.rotation.yaw  # Use the recommended direction!

    for i in range(frames_per_video):
        t = i / frames_per_video

        # Straight forward movement (minimal lateral offset for realistic driving)
        forward_dist = i * speed
        lateral_offset = 0.5 * np.sin(t * np.pi * 1)  # Very slight lane variation

        # Calculate position based on starting direction
        yaw_rad = np.radians(start_yaw)
        x = start_x + forward_dist * np.cos(yaw_rad) + lateral_offset * np.cos(yaw_rad + np.pi / 2)
        y = start_y + forward_dist * np.sin(yaw_rad) + lateral_offset * np.sin(yaw_rad + np.pi / 2)

        # Constant vehicle height (minor bumps)
        z = start_z + 0.1 * np.sin(t * np.pi * 6)

        # Yaw follows road direction with minimal turning + view angle offset
        yaw = start_yaw + view_angle_offset + 3.0 * np.sin(t * np.pi * 1)

        # Slight forward pitch (natural vehicle angle)
        pitch = -3.0 + 1.0 * np.sin(t * np.pi * 4)

        poses.append((x, y, z, pitch, yaw, 0.0))

    return poses


def generate_drone_forward_trajectory(video_idx, frames_per_video=150, speed=0.6, spawn_points=None):
    """
    2. Drone Forward: Move forward above the road just like a drone
    Smooth aerial movement at 10-20m height

    Args:
        video_idx: Video index for different paths and viewing angles
        frames_per_video: Number of frames
        speed: Movement speed in meters per frame
        spawn_points: Dictionary with 'vehicles', 'pedestrians', 'drones' spawn points
    """
    poses = []

    # Different horizontal viewing angles for variety (drone camera perspectives)
    # Cycles through: forward, angled right, angled left, side right, side left
    view_angles = [
        0.0,
        0.0,
        0.0,
        90.0,
        180.0,
        270.0,
    ]
    view_angle_offset = view_angles[video_idx % len(view_angles)]

    # Extract drone spawn points from dictionary
    drone_spawns = spawn_points['drones'] if spawn_points and isinstance(spawn_points, dict) else (spawn_points if spawn_points else [])

    start_x, start_y, start_z, start_yaw = 0, 0, 15.0, 90
    if drone_spawns and len(drone_spawns) > 0:
        spawn_point = drone_spawns[video_idx % len(drone_spawns)]
        start_x = spawn_point.location.x
        start_y = spawn_point.location.y
        start_z = 15.0  # Drone altitude above spawn point
        start_yaw = spawn_point.rotation.yaw  # Follow road direction

    for i in range(frames_per_video):
        t = i / frames_per_video

        # Smooth forward flight with gentle lateral drift
        forward_dist = i * speed
        lateral_offset = 2.0 * np.sin(t * np.pi * 1.5)  # Smooth side-to-side drift

        # Calculate position
        yaw_rad = np.radians(start_yaw)
        x = start_x + forward_dist * np.cos(yaw_rad) + lateral_offset * np.cos(yaw_rad + np.pi / 2)
        y = start_y + forward_dist * np.sin(yaw_rad) + lateral_offset * np.sin(yaw_rad + np.pi / 2)

        # Smooth altitude variation (drone floating)
        z = start_z + 1.5 * np.sin(t * np.pi * 2)

        # Smooth yaw changes (drone panning) + view angle offset
        yaw = start_yaw + view_angle_offset + 10.0 * np.sin(t * np.pi * 1.5)

        # Slight downward pitch to see the road
        pitch = -25.0 + 5.0 * np.sin(t * np.pi * 2)

        poses.append((x, y, z, pitch, yaw, 0.0))

    return poses

def generate_orbit_building_trajectory(video_idx, frames_per_video=150, spawn_points=None):
    """
    3. Orbit Building: On top of building looking at road, panning left to right
    Camera at rooftop height (30-40m) with horizontal panning motion

    Args:
        video_idx: Video index for different building positions
        frames_per_video: Number of frames
        map_bounds: Dictionary with x_min, x_max, y_min, y_max
    """
    poses = []

    # Extract drone spawn points from dictionary
    drone_spawns = spawn_points['drones'] if spawn_points and isinstance(spawn_points, dict) else (spawn_points if spawn_points else [])

    x_center, y_center = 0, 0
    x_range, y_range = 25, 25
    # Use spawn points if available (much better than map center!)
    if drone_spawns and len(drone_spawns) > 0:
        # Select a spawn point based on video_idx
        spawn_point = drone_spawns[video_idx % len(drone_spawns)]
        x_center = spawn_point.location.x
        y_center = spawn_point.location.y

    # Different building positions overlooking roads
    building_positions = [
        (x_center + x_range, y_center + y_range * 0.6, 48, 180),  # Building south, looking north
        (x_center - x_range * 0.8, y_center + y_range * 0.8, 36, 55),  # Building northeast, looking southwest
        (x_center + x_range, y_center - y_range * 0.6, 50, 270),  # Building west, looking east
        (x_center - x_range, y_center - y_range * 0.7, 47, 90),  # Building east, looking west
        (x_center, y_center + y_range, 45, 180),  # Building south, looking north
    ]

    x_fixed, y_fixed, z_fixed, base_yaw = building_positions[video_idx % len(building_positions)]

    for i in range(frames_per_video):
        t = i / frames_per_video

        # Pan from left to right (or right to left)
        # Total pan range: -60 to +60 degrees (120 degree sweep)
        yaw_pan = -60.0 + t * 120.0
        yaw = base_yaw + yaw_pan

        # Fixed downward angle to view road
        pitch = -45.0

        poses.append((x_fixed, y_fixed, z_fixed, pitch, yaw, 0.0))

    return poses

def generate_orbit_crossroad_trajectory(video_idx, frames_per_video=150, spawn_points=None):
    """
    4. Orbit Crossroad: At crossroad looking at intersection, panning left to right
    Camera at street level (3-5m) positioned at intersection

    Args:
        video_idx: Video index for different crossroad positions
        frames_per_video: Number of frames
        map_bounds: Dictionary with x_min, x_max, y_min, y_max
    """
    poses = []

    # Extract drone spawn points from dictionary
    drone_spawns = spawn_points['drones'] if spawn_points and isinstance(spawn_points, dict) else (spawn_points if spawn_points else [])

    x_center, y_center = 0, 0
    offset = 15
    # Use spawn points if available (much better than map center!)
    if drone_spawns and len(drone_spawns) > 0:
        # Select a spawn point based on video_idx
        spawn_point = drone_spawns[video_idx % len(drone_spawns)]
        x_center = spawn_point.location.x
        y_center = spawn_point.location.y

    # Different crossroad positions and viewing angles
    crossroad_positions = [
        (x_center + offset, y_center + offset, 4.0, 225),  # Northeast corner, looking southwest
        (x_center - offset, y_center + offset, 4.5, 315),  # Northwest corner, looking southeast
        (x_center + offset, y_center - offset, 4.2, 135),  # Southeast corner, looking northwest
        (x_center - offset, y_center - offset, 4.3, 45),  # Southwest corner, looking northeast
        (x_center + offset * 1.3, y_center, 4.0, 180),  # East side, looking west
    ]

    x_fixed, y_fixed, z_fixed, base_yaw = crossroad_positions[video_idx % len(crossroad_positions)]

    for i in range(frames_per_video):
        t = i / frames_per_video

        # Pan across the intersection (left to right sweep)
        # Pan range: -50 to +50 degrees (100 degree sweep)
        yaw_pan = -50.0 + t * 100.0
        yaw = base_yaw + yaw_pan

        # Slight downward angle to see intersection
        pitch = -10.0

        poses.append((x_fixed, y_fixed, z_fixed, pitch, yaw, 0.0))

    return poses

def generate_cctv_trajectory(video_idx, frames_per_video=150, map_bounds=None, spawn_points=None):
    """
    5. CCTV: Static camera mounted near traffic lights looking at intersection
    Completely fixed position, no movement (like real surveillance camera)

    Args:
        video_idx: Video index for different CCTV positions
        frames_per_video: Number of frames
        map_bounds: Dictionary with x_min, x_max, y_min, y_max
        spawn_points: Dictionary with 'traffic' positions (traffic lights)
    """
    poses = []

    # Extract traffic light spawn points from dictionary
    traffic_spawns = spawn_points['traffic'] if spawn_points and isinstance(spawn_points, dict) else (spawn_points if spawn_points else [])

    # Use traffic light positions if available (realistic CCTV mounting locations!)
    if traffic_spawns and len(traffic_spawns) > 0:
        # Select a traffic light position based on video_idx
        spawn_point = traffic_spawns[video_idx % len(traffic_spawns)]
        x_center = spawn_point.location.x
        y_center = spawn_point.location.y
        base_yaw = spawn_point.rotation.yaw  # Face the direction of the traffic light
    else:
        # Fallback to default position
        x_center, y_center = 0, 0
        base_yaw = 180

    # Different CCTV mounting positions near traffic light
    # Position camera slightly offset from traffic light to get good view of intersection
    cctv_positions = [
        (x_center + 3, y_center + 3, 8, -50, base_yaw),  # Offset position 1
        (x_center - 3, y_center + 3, 9, -55, base_yaw + 45),  # Offset position 2
        (x_center + 3, y_center - 3, 7, -52, base_yaw + 90),  # Offset position 3
        (x_center - 3, y_center - 3, 8, -48, base_yaw + 135),  # Offset position 4
        (x_center, y_center, 6, -45, base_yaw + 180),  # Directly at traffic light, higher
    ]

    x_fixed, y_fixed, z_fixed, pitch_fixed, yaw_fixed = cctv_positions[video_idx % len(cctv_positions)]

    # CCTV scanning: pick ONE axis — either pan (yaw) or tilt (pitch)
    # Single slow sweep from -half to +half over the entire video (~2°/s)
    scan_mode = "pan" if (video_idx % 2 == 0) else "tilt"
    total_sweep = 22.5 if scan_mode == "pan" else 12  # total degrees across the whole video

    for i in range(frames_per_video):
        # Linear sweep from -half to +half
        t = i / max(frames_per_video - 1, 1)
        offset = total_sweep * (t - 0.5)  # range: -half → +half

        if scan_mode == "pan":
            poses.append((x_fixed, y_fixed, z_fixed, pitch_fixed, yaw_fixed + offset, 0.0))
        else:
            poses.append((x_fixed, y_fixed, z_fixed, pitch_fixed + offset, yaw_fixed, 0.0))

    return poses



def generate_pedestrian_trajectory(video_idx, frames_per_video=150, speed=0.15, spawn_points=None):
    """
    6. Pedestrian: Camera at person height walking on sidewalk looking at road
    Human perspective at 1.5-1.8m height, walking along sidewalk

    Args:
        video_idx: Video index for different pedestrian paths and viewing angles
        frames_per_video: Number of frames
        speed: Walking speed in meters per frame (~1.5 m/s realistic walking)
        spawn_points: Dictionary with 'vehicles', 'pedestrians', 'drones' spawn points
    """
    poses = []

    # Different horizontal viewing angles for variety (pedestrian head orientations)
    # Cycles through: forward, looking at road (right), looking away (left), looking right, looking left
    view_angles = [
        0.0,
        0.0,
        0.0,
        90.0,
        180.0,
        270.0,
    ]
    view_angle_offset = view_angles[video_idx % len(view_angles)]

    # Extract pedestrian spawn points from dictionary
    pedestrian_spawns = spawn_points['pedestrians'] if spawn_points and isinstance(spawn_points, dict) else (spawn_points if spawn_points else [])

    # Use spawn points if available (pedestrian on sidewalk near roads)
    start_x, start_y, start_z, start_yaw = 0, 0, 1.65, 90
    if pedestrian_spawns and len(pedestrian_spawns) > 0:
        spawn_point = pedestrian_spawns[video_idx % len(pedestrian_spawns)]
        # Pedestrian spawn points already include sidewalk offset
        start_x = spawn_point.location.x
        start_y = spawn_point.location.y
        start_z = 1.65  # Pedestrian eye height
        start_yaw = spawn_point.rotation.yaw  # Walk along the road

    for i in range(frames_per_video):
        t = i / frames_per_video

        # Straight walking with very slight side-to-side (natural human gait)
        forward_dist = i * speed
        lateral_offset = 0.15 * np.sin(t * np.pi * 8)  # Slight walking sway

        # Calculate position
        yaw_rad = np.radians(start_yaw)
        x = start_x + forward_dist * np.cos(yaw_rad) + lateral_offset * np.cos(yaw_rad + np.pi / 2)
        y = start_y + forward_dist * np.sin(yaw_rad) + lateral_offset * np.sin(yaw_rad + np.pi / 2)

        # Slight bobbing (natural walking motion)
        z = start_z + 0.01 * np.sin(t * np.pi * 4)  # Head bobbing while walking

        # View angle offset + slight head turning (looking around)
        yaw = start_yaw + view_angle_offset + 4.0 * np.sin(t * np.pi * 3)  # Look left/right occasionally

        # Mostly level with slight pitch variation (natural head movement)
        pitch = 0.0 + 1.0 * np.sin(t * np.pi * 3)

        poses.append((x, y, z, pitch, yaw, 0.0))

    return poses


def generate_round_building_trajectory(video_idx, frames_per_video=150, spawn_points=None):
    """
    7. Round Building (Arc Shot): Camera orbits around a focal point while
    always keeping focus on it.  The orbit radius, height, and number of
    full revolutions vary per video_idx for diversity.

    Args:
        video_idx: Video index for variety in radius / height / direction
        frames_per_video: Number of frames
        spawn_points: Dictionary with spawn point categories
    """
    poses = []

    # Pick a focal point from vehicle spawn points (roads / interesting areas)
    drone_spawns = (
        spawn_points["drones"]
        if spawn_points and isinstance(spawn_points, dict)
        else (spawn_points if spawn_points else [])
    )

    cx, cy = 0.0, 0.0
    if drone_spawns and len(drone_spawns) > 0:
        sp = drone_spawns[video_idx % len(drone_spawns)]
        cx = sp.location.x
        cy = sp.location.y

    # Orbit parameters — cycle through different configurations
    orbit_configs = [
        # (radius, height, revolutions, pitch, direction)
        (20.0, 15.0, 0.50, -30.0, 1),    # Medium orbit, moderate height
        (30.0, 25.0, 0.40, -35.0, -1),   # Wide orbit, higher, reverse
        (15.0, 10.0, 0.70, -25.0, 1),    # Tight orbit, lower
        (25.0, 20.0, 0.50, -40.0, -1),   # Medium-wide, steeper look-down
        (12.0, 8.0, 0.80, -20.0, 1),     # Very tight, low
    ]

    radius, height, revolutions, pitch, direction = orbit_configs[
        video_idx % len(orbit_configs)
    ]

    # Starting angle offset so each video_idx starts at a different position
    angle_offset = (video_idx * 47) % 360  # arbitrary spread

    for i in range(frames_per_video):
        t = i / frames_per_video
        angle_deg = angle_offset + direction * t * 360.0 * revolutions
        angle_rad = np.radians(angle_deg)

        x = cx + radius * np.cos(angle_rad)
        y = cy + radius * np.sin(angle_rad)
        z = height + 1.5 * np.sin(t * np.pi * 2)  # gentle vertical bob

        # Yaw: always face the center point
        # atan2 gives angle from camera to center
        yaw = np.degrees(np.arctan2(cy - y, cx - x))

        poses.append((x, y, z, pitch, yaw, 0.0))

    return poses


import carla

# ============================================================
# Ego Camera trajectory generators (attached to actors)
# ============================================================

def generate_ego_car_trajectory(video_idx, frames_per_video=150, speed=0.8, spawn_points=None):
    """
    7. Ego Car: Camera attached to actual vehicle with autopilot
    Real vehicle-mounted camera following CARLA's traffic AI

    Args:
        video_idx: Video index for different vehicle types and spawn points
        frames_per_video: Number of frames
        speed: Ignored (vehicle uses autopilot speed)
        spawn_points: Dictionary with 'vehicles' spawn points

    Returns:
        Dictionary with type='ego_car' and metadata for spawning
    """
    # Extract vehicle spawn points
    vehicle_spawns = spawn_points['vehicles'] if spawn_points and isinstance(spawn_points, dict) else []

    # Select spawn point
    if vehicle_spawns and len(vehicle_spawns) > 0:
        spawn_point = vehicle_spawns[video_idx % len(vehicle_spawns)]
    else:
        # Fallback spawn point
        spawn_point = carla.Transform(
            carla.Location(x=0, y=0, z=0.5),
            carla.Rotation(pitch=0, yaw=90, roll=0)
        )

    # Different camera mount positions on vehicle (cycling through different views)
    camera_mounts = [
        # (relative_x, relative_y, relative_z, pitch, yaw, roll)
        (2.0, 0.0, 1.2, -5.0, 0.0, 0.0),    # Hood camera (dashcam view)
        (1.5, 0.0, 1.4, -10.0, 0.0, 0.0),   # Windshield camera (driver view)
        (0.5, 0.0, 1.5, 0.0, 0.0, 0.0),     # Roof center (action cam)
        (2.0, 0.5, 1.2, -5.0, 15.0, 0.0),   # Hood right (passenger side)
        (2.0, -0.5, 1.2, -5.0, -15.0, 0.0), # Hood left (driver side)
        (-1.0, 0.0, 1.5, -10.0, 180.0, 0.0), # Rear camera (backup view)
        (1.0, 1.0, 1.3, -5.0, 90.0, 0.0),   # Right side mirror view
    ]

    mount = camera_mounts[video_idx % len(camera_mounts)]

    return {
        'type': 'ego_car',
        'spawn_point': spawn_point,
        'camera_mount': {
            'x': mount[0],
            'y': mount[1],
            'z': mount[2],
            'pitch': mount[3],
            'yaw': mount[4],
            'roll': mount[5]
        },
        'frames': frames_per_video,
        'use_autopilot': True
    }


def generate_ego_pedestrian_trajectory(video_idx, frames_per_video=150, speed=0.15, spawn_points=None):
    """
    8. Ego Pedestrian: Camera attached to walking pedestrian with AI
    Real pedestrian-mounted camera following CARLA's walker AI

    Args:
        video_idx: Video index for different pedestrian types and spawn points
        frames_per_video: Number of frames
        speed: Walking speed for AI walker (meters per second)
        spawn_points: Dictionary with 'pedestrians' spawn points

    Returns:
        Dictionary with type='ego_pedestrian' and metadata for spawning
    """
    # Extract pedestrian spawn points
    pedestrian_spawns = spawn_points['pedestrians'] if spawn_points and isinstance(spawn_points, dict) else []

    # Select spawn point
    if pedestrian_spawns and len(pedestrian_spawns) > 0:
        spawn_point = pedestrian_spawns[video_idx % len(pedestrian_spawns)]
    else:
        # Fallback spawn point
        spawn_point = carla.Transform(
            carla.Location(x=0, y=0, z=0.5),
            carla.Rotation(pitch=0, yaw=90, roll=0)
        )

    # Different camera positions on pedestrian (head-mounted perspectives)
    camera_mounts = [
        # (relative_x, relative_y, relative_z, pitch, yaw, roll)
        (0.3, 0.0, 1.6, 0.0, 0.0, 0.0),     # Eye level forward (natural view)
        (0.3, 0.0, 1.6, -10.0, 0.0, 0.0),   # Eye level down (watching ground)
        (0.3, 0.0, 1.6, 5.0, 0.0, 0.0),     # Eye level up (looking ahead)
        (0.3, 0.0, 1.6, 0.0, 30.0, 0.0),    # Looking right (checking traffic)
        (0.3, 0.0, 1.6, 0.0, -30.0, 0.0),   # Looking left (checking traffic)
        (0.3, 0.0, 1.6, 0.0, 60.0, 0.0),    # Strong right (peripheral view)
        (0.3, 0.0, 1.6, 0.0, -60.0, 0.0),   # Strong left (peripheral view)
    ]

    mount = camera_mounts[video_idx % len(camera_mounts)]

    # Convert speed from m/s to CARLA walker speed
    walker_speed = speed * 30.0  # Approximate conversion (frame speed to m/s)

    return {
        'type': 'ego_pedestrian',
        'spawn_point': spawn_point,
        'camera_mount': {
            'x': mount[0],
            'y': mount[1],
            'z': mount[2],
            'pitch': mount[3],
            'yaw': mount[4],
            'roll': mount[5]
        },
        'frames': frames_per_video,
        'walker_speed': walker_speed,
        'use_ai_walker': True
    }
