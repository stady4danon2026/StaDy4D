# CARLA Multi-Scene Data Generator

Comprehensive data generation tool for CARLA simulator with support for multiple maps, configurable parameters, and automatic resume capability.

## Features

- **Multi-Map Support**: Generate data across all CARLA maps (12+ different environments)
- **Fully Configurable**: JSON-based configuration for all parameters
- **Resume Capability**: Automatic progress tracking and resume after interruptions
- **Multiple Trajectories**: Support for CCTV, drone, BEV, forward, orbit, and mixed camera paths
- **Dual Datasets**: Generates both static and dynamic scene versions
- **Automatic Video Creation**: Optional MP4 video generation using ffmpeg

## Quick Start

### 1. Basic Usage

```bash
# Quick test (2 videos on Town10HD)
python data_generate_carla.py --config configs/config_test.json

# Test all 6 trajectory types (6 videos on Town10HD)
python data_generate_carla.py --config configs/config_test.json

# Standard run (12 maps × 5 videos = 60 videos)
python data_generate_carla.py --config configs/config_large_scale-short.json

# Large-scale production (12 maps × 20 videos = 240 videos, 15s each)
python data_generate_carla.py --config configs/config_large_scale-long.json
```

**Recommended workflow:**
1. Start with `configs/config_test.json` to verify setup works
2. Test all trajectories with `configs/config_test.json` to see each type
3. Run full production with `config_large_scale.json`

### 2. Configuration Files

Four example configurations are provided:

- **`configs/config_test.json`**: Quick test on Town10HD, 2 videos (cctv + car_forward)
- **`configs/config_test.json`**: Test all 6 trajectory types on Town10HD
- **`config.json`**: Standard configuration with all 12 maps, 5 videos each
- **`config_large_scale.json`**: Production-scale with 12 maps × 20 videos (15s each)

## Configuration Options

### Full Configuration Schema

```json
{
  "description": "Configuration description",

  "carla": {
    "host": "localhost",       // CARLA server host
    "port": 2000,               // CARLA server port
    "timeout": 10.0             // Connection timeout (seconds)
  },

  "maps": [                     // List of CARLA maps to use
    "Town01", "Town02", ...
  ],

  "video_generation": {
    "videos_per_map": 5,        // Number of videos to generate per map
    "video_duration_sec": 2,    // Duration of each video (seconds)
    "fps": 10,                  // Frames per second
    "trajectory_types": [       // Camera trajectory types
      "car_forward",            // Vehicle-level forward motion
      "drone_forward",          // Aerial forward motion
      "orbit_building",         // Building rooftop panning
      "orbit_crossroad",        // Crossroad intersection panning
      "cctv",                   // Static surveillance camera
      "pedestrian",             // Walking at person height
      "mixed"                   // Cycle through all 6 types
    ]
  },

  "camera": {
    "width": 640,               // Image width
    "height": 360,              // Image height
    "fov": 70                   // Field of view (degrees)
  },

  "actors": {
    "n_vehicles": 50,           // Number of vehicles in dynamic scenes
    "n_walkers": 30             // Number of pedestrians in dynamic scenes
  },

  "weather": "ClearNoon",       // Weather preset (e.g., ClearNoon, CloudyNoon, WetNoon, etc.)

  "output": {
    "base_dir": "output",       // Output directory
    "create_videos": true,      // Auto-create MP4 videos with ffmpeg
    "save_rgb": true,           // Save RGB images
    "save_depth": true,         // Save depth data (.npy)
    "save_depth_vis": true,     // Save depth visualizations
    "save_extrinsics": true     // Save camera poses
  },

  "execution": {
    "skip_existing": true,      // Skip already-generated videos
    "resume_from_map": null,    // Resume from specific map (or null)
    "resume_from_video": null   // Resume from specific video index (or null)
  }
}
```

### Available CARLA Maps

- `Town01`: Small town with river
- `Town02`: Residential area with shops
- `Town03`: Urban city with skyscrapers
- `Town04`: Small town with highway
- `Town05`: Multi-level highways
- `Town06`: Low density with forest
- `Town07`: Rural with narrow roads
- `Town10HD`: Downtown area (high detail)
- `Town11`: Industrial district
- `Town12`: Rural residential with farms
- `Town13`: Modern roundabouts
- `Town15`: Tall glass buildings

### Camera Trajectory Types

1. **`car_forward`** - Vehicle Dashcam
   - Height: 2-3m (vehicle roof level)
   - Movement: Straight forward with minimal turning
   - Speed: ~0.8 m/frame (realistic driving)
   - Use case: Dashcam footage, vehicle-mounted cameras

2. **`drone_forward`** - Aerial Drone Shot
   - Height: 10-20m (above road, below buildings)
   - Movement: Smooth forward flight with gentle drift
   - Speed: ~0.6 m/frame
   - Use case: Drone cinematography, aerial surveillance

3. **`orbit_building`** - Building Rooftop Panning
   - Height: 30-40m (rooftop level)
   - Movement: Fixed position, panning left-to-right (120° sweep)
   - Speed: Static position
   - Use case: Cityscape monitoring, wide area views

4. **`orbit_crossroad`** - Intersection Observer
   - Height: 3-5m (street level, elevated position)
   - Movement: Fixed at crossroad, panning across intersection (100° sweep)
   - Speed: Static position
   - Use case: Traffic monitoring, intersection analysis

5. **`cctv`** - Surveillance Camera
   - Height: 30-40m (building rooftop)
   - Movement: Completely static, no movement
   - Speed: 0 (fixed)
   - Use case: Security footage, static monitoring

6. **`pedestrian`** - Human Perspective
   - Height: 1.5-1.8m (eye level)
   - Movement: Walking on sidewalk, looking at road
   - Speed: ~0.15 m/frame (~1.5 m/s walking speed)
   - Use case: Human POV, sidewalk footage

7. **`mixed`** - All Types Combined
   - Automatically cycles through all 6 trajectory types
   - Each video uses a different type

## Resume Capability

The system automatically saves progress after each video. If interrupted, it will resume from where it left off.

### Manual Resume

Edit your config file:

```json
"execution": {
  "resume_from_map": "Town05",
  "resume_from_video": 3
}
```

This will skip all maps before Town05 and start from video 4 on Town05.

### Automatic Resume

The system saves `progress.json` in the output directory. Just re-run with the same config:

```bash
python data_generate_carla.py --config configs/config_large_scale-long.json
```

It will automatically detect and resume from the last completed video.

## Output Structure

```
output/
├── progress.json                 # Resume tracking
├── Town01/
│   ├── video_00/
│   │   ├── intrinsics.json       # Camera intrinsics
│   │   ├── metadata.json         # Video metadata
│   │   ├── static/
│   │   │   ├── rgb/              # RGB images (rgb_XXXX.png)
│   │   │   ├── depth/            # Depth data (depth_XXXX.npy)
│   │   │   ├── depth_vis/        # Depth visualizations
│   │   │   └── extrinsic/        # Camera poses (JSON)
│   │   ├── dynamic/
│   │   │   ├── rgb/
│   │   │   ├── depth/
│   │   │   ├── depth_vis/
│   │   │   └── extrinsic/
│   │   ├── static_rgb.mp4        # Compiled video (if enabled)
│   │   ├── static_depth.mp4
│   │   ├── dynamic_rgb.mp4
│   │   └── dynamic_depth.mp4
│   ├── video_01/
│   └── ...
├── Town02/
└── ...
```

## Large-Scale Data Generation

### Single Run vs Multiple Runs - ANSWER: Use Single Run!

**RECOMMENDED: Use SINGLE RUN for large datasets**

**Why single run is better:**
- ✓ Automatic progress tracking after each video
- ✓ Auto-resume if interrupted (crash, Ctrl+C, connection loss)
- ✓ Skip already-generated videos automatically
- ✓ Unified output structure (all data in one place)
- ✓ Less manual intervention and monitoring
- ✓ Robust error handling (continues if one map/video fails)

**Multiple runs approach (NOT recommended):**
- ✗ Manual tracking of what's been generated
- ✗ Need to manually split configs
- ✗ Risk of duplicate data or gaps
- ✗ More complex output management

### Example Workflow

**Single run that processes everything:**

```bash
# Start large-scale generation (12 maps × 20 videos = 240 videos)
python data_generate_carla.py --config configs/config_large_scale-long.json

# If interrupted at any point, just re-run the same command
python data_generate_carla.py --config configs/config_large_scale-long.json
# ↑ Automatically resumes from last completed video!

# Or manually resume from specific point:
# Edit config_large_scale.json:
# "resume_from_map": "Town07",
# "resume_from_video": 10
python data_generate_carla.py --config configs/config_large_scale-long.json
```

### Time Estimates

With `config_large_scale.json` (15 second videos):
- Per video: ~30-40 seconds (static + dynamic scenes)
- Per map: 20 videos × 35s = ~12 minutes
- **Total: 12 maps × 12 min = ~2.5 hours**

Runs completely unattended with auto-resume!

### Production Recommendations

For generating large amounts of data:

1. **Enable skip_existing**:
   ```json
   "skip_existing": true
   ```

2. **Use robust error handling**: The script continues even if one map/video fails

3. **Monitor progress**: Check `output/progress.json` to see current state

4. **Run in screen/tmux** for long-running sessions:
   ```bash
   screen -S carla_gen
   python data_generate_carla.py --config configs/config_large_scale-long.json
   # Detach: Ctrl+A then D
   # Reattach: screen -r carla_gen
   ```

5. **Estimate time and data volume**:

   **For `config_large_scale.json`:**
   - Time: ~30-40s per video (static + dynamic)
   - Total: 12 maps × 20 videos × 35s = **~2.5 hours**
   - Data: 12 maps × 20 videos × 150 frames × 2 scenes = **57,600 images**
   - Storage: ~640×360 images ≈ **15-20 GB** (with depth maps)

## Download Pre-generated Dataset

Instead of generating, download from HuggingFace:

```bash
pip install huggingface_hub
python download_dataset.py --repo username/StaDy4D
```

See [DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md) for details.

---

## Requirements

- CARLA 0.9.16 running on localhost:2000 (only needed for generation, not for using downloaded dataset)
- Python 3.7+
- Dependencies: `numpy`, `opencv-python`, `carla` (for generation)
- Dependencies: `huggingface_hub` (for downloading dataset)
- Optional: `ffmpeg` for video creation

## Data Volume Summary

### Quick Comparison of Configs

| Config | Maps | Videos/Map | Duration | Total Videos | Frames | Est. Time | Est. Storage |
|--------|------|------------|----------|--------------|--------|-----------|--------------|
| quick_test | 1 | 2 | 5s | 2 | 100 | ~2 min | ~50 MB |
| each_trajectory | 1 | 6 | 5s | 6 | 300 | ~5 min | ~150 MB |
| config.json | 12 | 5 | 2s | 60 | 1,200 | ~40 min | ~3 GB |
| large_scale | 12 | 20 | 15s | 240 | 57,600 | ~2.5 hrs | ~15-20 GB |

*Note: Each video generates 2 scenes (static + dynamic), doubling the data volume*

### Customize Your Own

Want different data volume? Edit any config:

```json
// Small dataset for testing
"videos_per_map": 3,
"video_duration_sec": 3,  // 30 frames per video

// Medium dataset
"videos_per_map": 10,
"video_duration_sec": 10,  // 100 frames per video

// Large dataset
"videos_per_map": 50,
"video_duration_sec": 20,  // 200 frames per video
```

## Tips

- Start with `configs/config_test.json` to verify everything works
- Test all trajectory types with `configs/config_test.json`
- Adjust `n_vehicles` and `n_walkers` based on your GPU performance
- Lower resolution for faster generation (e.g., `320x180`)
- Use `"create_videos": false` if you only need raw frames
- Each map has different characteristics - test a few maps before full run

## Troubleshooting

**CARLA connection error:**
- Ensure CARLA server is running: `./CarlaUE4.sh`
- Check host/port in config match your CARLA instance

**Map loading fails:**
- Some maps may not be available in your CARLA version
- Remove unavailable maps from config

**Out of memory:**
- Reduce `n_vehicles` and `n_walkers`
- Lower resolution
- Process fewer maps at once

**Resume not working:**
- Check `output/progress.json` exists
- Ensure same `base_dir` in config
- Manually set `resume_from_map` in config


ffmpeg -i StaDy4D/sample/Town04/video_00/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0400_dy.gif
ffmpeg -i StaDy4D/sample/Town04/video_00/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0400_st.gif

ffmpeg -i StaDy4D/sample/Town04/video_01/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0401_dy.gif
ffmpeg -i StaDy4D/sample/Town04/video_01/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0401_st.gif

ffmpeg -i scene_check/Town04/video_02/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0402_dy.gif
ffmpeg -i scene_check/Town04/video_02/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0402_st.gif

ffmpeg -i scene_check/Town04/video_03/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0403_dy.gif
ffmpeg -i scene_check/Town04/video_03/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0403_st.gif

----

ffmpeg -i StaDy4D/sample/Town05/video_00/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0500_dy.gif
ffmpeg -i StaDy4D/sample/Town05/video_00/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0500_st.gif

ffmpeg -i StaDy4D/sample/Town05/video_01/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0501_dy.gif
ffmpeg -i StaDy4D/sample/Town05/video_01/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0501_st.gif

ffmpeg -i StaDy4D/sample/Town06/video_00/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0600_dy.gif
ffmpeg -i StaDy4D/sample/Town06/video_00/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0600_st.gif

ffmpeg -i StaDy4D/sample/Town06/video_01/dynamic_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0601_dy.gif
ffmpeg -i StaDy4D/sample/Town06/video_01/static_rgb.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 quick_view/vis0601_st.gif



====


Output Combination

Maps: 8 towns — Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD

Videos set per map: 9

camera_types (9)
  - car_forward × 3
  - drone_forward
  - orbit_building
  - orbit_crossroad
  - cctv
  - pedestrian × 2

weather (6)
  - ClearNoon × 3
  - HardRainNoon
  - MidRainSunset
  - WetCloudyNoon

Total videos: 8 maps × 9 videos × 9 trajectory x 6 weather = 3888 videos

Per video, 6 cameras record simultaneously (from multi_camera_types):
1. car_forward — car-level forward driving
2. drone_forward — elevated aerial forward
3. orbit_building — orbiting around a building
4. orbit_crossroad — orbiting a crossroad
5. cctv — fixed traffic-light mounted
6. pedestrian — sidewalk-level walking

Per video, 2 scene variants:
- static — no actors (empty world)
- dynamic — 80 vehicles + 50 walkers with AI

Per camera per scene, saved outputs:
- rgb/ — PNG frames
- depth/ — .npy depth maps (meters)
- depth_vis/ — grayscale depth visualization PNGs
- extrinsics/ — 4×4 c2w matrix .npy per frame
- intrinsics/ — 3×3 K matrix .npy per frame

Frames per video: 5s × 10 FPS = 50 frames

Weather: cycles per video index — [ClearNoon, HardRainNoon, ClearNoon, MidRainSunset, ClearNoon, WetCloudyNoon], so video 0→ClearNoon, 1→HardRainNoon, 2→ClearNoon,
3→MidRainSunset, 4→ClearNoon, 5→WetCloudyNoon.

Grand total per run:
- 48 videos × 6 cameras × 2 scenes = 576 camera recordings
- 576 × 50 frames = 28,800 RGB frames (+ same for depth)

---
Will scenes be the same across different runs?

No, they will NOT be identical. Several sources of randomness:

1. Spawn points — get_categorized_spawn_points() calls random.shuffle() on vehicle spawn points (line 240) and traffic light spawn points (line 259) with no fixed seed.
2. Pedestrian spawn — world.get_random_location_from_navigation() (line 244) is non-deterministic.
3. Trajectory generation — generate_collision_free_trajectory varies by attempt count, and collision checks are timing-dependent.
4. Dynamic actors — np.random.choice() for vehicle/walker blueprints (lines 938, 951), random walker destinations/speeds.
5. No random.seed() or np.random.seed() is set anywhere in main().

If you want reproducibility, you'd need to seed random and np.random at the start.

---
Quick image folder → GIF

The fastest method is ffmpeg (you already have it since the code uses it for MP4):

# Single folder:
ffmpeg -framerate 10 -i rgb_%04d.png -vf "scale=320:-1" -loop 0 preview.gif

# All camera rgb folders at once (one-liner):
find FirstScale100T -path "*/rgb" -type d -exec sh -c 'dir="$1"; name=$(echo "$dir" | sed "s|/|_|g"); ffmpeg -y -framerate 10 -i "$dir/rgb_%04d.png" -vf "scale=320:-1"
-loop 0 "quick_view/${name}.gif"' _ {} \;


Key flags:
- -vf "scale=320:-1" — downscale to 320px wide (makes GIFs small and fast)
- -loop 0 — infinite loop
- -framerate 10 — matches your 10 FPS config

Want me to write a small script that batch-converts all the rgb folders to GIFs in a quick_view/ directory?