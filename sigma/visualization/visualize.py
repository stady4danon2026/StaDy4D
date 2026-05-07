import argparse
import json
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import viser
import viser.transforms as tf
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


# ======================================================================
# StaDy4DLoader — loads raw StaDy4D dataset (safetensors + mp4)
# ======================================================================

class StaDy4DLoader:
    """Loader for the StaDy4D raw dataset format (rgb.mp4 + safetensors).

    When loaded from a ``static/`` camera directory and a matching
    ``dynamic/`` camera exists, the dynamic scene is available via
    :meth:`get_dynamic_points` (enables the "With Dynamic" toggle in the
    visualizer — accumulated static background + rolling dynamic overlay).
    """

    def __init__(self, camera_dir: Path, scene_dir: Path | None = None):
        from safetensors.numpy import load_file

        self.camera_dir = camera_dir
        # scene_dir is two levels up from camera_dir: .../scene/dynamic/cam_*/
        self.scene_dir = scene_dir or camera_dir.parent.parent

        # Load scene metadata
        metadata_path = self.scene_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.scene_dir}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        resolution = metadata.get("resolution", {})
        self.width_val = resolution.get("width", 640)
        self.height_val = resolution.get("height", 480)
        self.scene_scale: float = 1.0  # Raw CARLA data is metric

        # Load safetensors
        self._depths = load_file(str(camera_dir / "depth.safetensors"))["depth"]       # (N, H, W)
        self._c2w = load_file(str(camera_dir / "extrinsics.safetensors"))["c2w"]       # (N, 4, 4)
        self._K_all = load_file(str(camera_dir / "intrinsics.safetensors"))["K"]       # (N, 3, 3)

        # Decode video frames
        self._rgb_frames: list[np.ndarray] = []
        cap = cv2.VideoCapture(str(camera_dir / "rgb.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        self._n_frames = min(len(self._rgb_frames), self._depths.shape[0], self._c2w.shape[0])

        # Intrinsics from first frame
        K0 = self._K_all[0]
        self.fx = float(K0[0, 0])
        self.fy = float(K0[1, 1])
        self.cx = float(K0[0, 2])
        self.cy = float(K0[1, 2])
        self.K = K0.copy().astype(np.float64)

        self.use_per_frame_intrinsics = True

        self._to_opengl = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)

        # ----- Dynamic pair detection -----
        # If this camera is under static/, look for matching dynamic/ camera.
        self._dyn_depths: np.ndarray | None = None
        self._dyn_c2w: np.ndarray | None = None
        self._dyn_K: np.ndarray | None = None
        self._dyn_rgb: list[np.ndarray] | None = None

        parent_type = camera_dir.parent.name  # "static" or "dynamic"
        cam_name = camera_dir.name
        if parent_type == "static":
            dynamic_cam = camera_dir.parent.parent / "dynamic" / cam_name
            if dynamic_cam.exists() and (dynamic_cam / "rgb.mp4").exists():
                self._load_dynamic_pair(dynamic_cam, load_file)

    def _load_dynamic_pair(self, dynamic_dir: Path, load_file) -> None:
        """Load the dynamic/ camera as overlay data."""
        self._dyn_depths = load_file(str(dynamic_dir / "depth.safetensors"))["depth"]
        self._dyn_c2w = load_file(str(dynamic_dir / "extrinsics.safetensors"))["c2w"]
        self._dyn_K = load_file(str(dynamic_dir / "intrinsics.safetensors"))["K"]

        self._dyn_rgb = []
        cap = cv2.VideoCapture(str(dynamic_dir / "rgb.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._dyn_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    @property
    def has_dynamic(self) -> bool:
        return self._dyn_depths is not None

    @property
    def num_frames(self) -> int:
        return self._n_frames

    @property
    def height(self) -> int:
        return self.height_val

    @property
    def width(self) -> int:
        return self.width_val

    def _unproject(self, depth: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                   rgb: np.ndarray, downsample_factor: int):
        if downsample_factor > 1:
            depth = depth[::downsample_factor, ::downsample_factor]
            rgb = rgb[::downsample_factor, ::downsample_factor]

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        h, w = depth.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u, v = u.flatten(), v.flatten()
        d = depth.flatten()

        fx_d = fx / downsample_factor
        fy_d = fy / downsample_factor
        cx_d = cx / downsample_factor
        cy_d = cy / downsample_factor

        x_cam = (u - cx_d) * d / fx_d
        y_cam = (v - cy_d) * d / fy_d
        z_cam = d

        cam_coords = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=-1)
        world_coords = (cam_coords @ c2w.T)[:, :3]
        rgb_flat = rgb.reshape(-1, 3)
        return world_coords, rgb_flat

    def display_rotation(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation to OpenGL for viser (Z-forward → Z-backward)."""
        return R

    def get_frame_data(self, index: int, downsample_factor: int = 1, min_confidence: float = 0.0):
        rgb_img = self._rgb_frames[index].astype(np.float16) / 255.0
        depth_img = self._depths[index].astype(np.float32)
        cam_c2w = self._c2w[index].astype(np.float64).copy()

        world_coords, rgb_flat = self._unproject(
            depth_img, cam_c2w, self._K_all[index], rgb_img, downsample_factor,
        )
        if downsample_factor > 1:
            rgb_img = rgb_img[::downsample_factor, ::downsample_factor]
        # Return raw c2w — caller uses display_rotation() for viser frustums
        return world_coords, rgb_flat, None, cam_c2w, rgb_img

    def get_dynamic_points(self, index: int, downsample_factor: int = 1):
        """Return the full dynamic scene point cloud for frame ``index``.

        When loaded from ``static/``, this returns the ``dynamic/`` camera's
        frame (scene with moving objects).  The visualizer shows only this
        frame at timestep *t* while accumulating the static background.
        """
        empty = (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
        if not self.has_dynamic or index >= self._dyn_depths.shape[0]:
            return empty

        depth = self._dyn_depths[index].astype(np.float32)
        c2w = self._dyn_c2w[index].astype(np.float64).copy()
        rgb = self._dyn_rgb[index].astype(np.float32) / 255.0

        world_coords, rgb_flat = self._unproject(depth, c2w, self._dyn_K[index], rgb, downsample_factor)
        return world_coords.astype(np.float32), rgb_flat.astype(np.float32)

    def load_depth(self, index: int) -> np.ndarray:
        return self._depths[index].astype(np.float32)


# ======================================================================
# FolderLoader — loads pipeline output (per-frame files)
# ======================================================================

class FolderLoader:
    """Loader for SIGMA pipeline output directories (per-frame PNGs/NPYs)."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.rgb_dir = data_path / "rgb"
        self.depth_dir = data_path / "depth"
        self.confidence_dir = data_path / "confidence"
        self.extrinsic_dir = data_path / "extrinsics"

        # Load metadata if available for width/height
        metadata_path = data_path / "metadata.json"
        if not metadata_path.exists():
            metadata_path = data_path.parent.parent / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        # Width/height are nested under "resolution"
        resolution = metadata.get("resolution", {})
        self.width_val = resolution.get("width")
        self.height_val = resolution.get("height")
        # Per-method scale: VGGT outputs in normalised units (~42x smaller than
        # metric), Pi3X is approximately metric.  Written by the pipeline runner.
        self.scene_scale: float = float(metadata.get("scene_scale", 1.0))

        # Check for global intrinsics.json first
        global_intrinsics_path = data_path.parent / "intrinsic.json"
        if global_intrinsics_path.exists():
            # Use global intrinsics
            self.use_per_frame_intrinsics = False
            with open(global_intrinsics_path, "r") as f:
                intrinsics = json.load(f)

            # Store default intrinsics
            self.fx = intrinsics["fx"]
            self.fy = intrinsics["fy"]
            self.cx = intrinsics["cx"]
            self.cy = intrinsics["cy"]

            self.intrinsic_files = None
        else:
            # Check for per-frame intrinsics directory
            self.intrinsic_dir = data_path / "intrinsics"
            self.use_per_frame_intrinsics = self.intrinsic_dir.exists() and self.intrinsic_dir.is_dir()

            if self.use_per_frame_intrinsics:
                # Load intrinsics files (support both .json and .npy)
                npy_files = sorted(list(self.intrinsic_dir.glob("*.npy")))

                self.intrinsic_files = npy_files
                self.intrinsic_format = "npy"
                # Load first frame's intrinsics to get default values
                K_mat = np.load(self.intrinsic_files[0])
                self.fx = K_mat[0, 0]
                self.fy = K_mat[1, 1]
                self.cx = K_mat[0, 2]
                self.cy = K_mat[1, 2]

        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.rgb_files = sorted(list(self.rgb_dir.glob("*.png")))
        self.depth_files = sorted(list(self.depth_dir.glob("*.npy")))
        self.confidence_files = sorted(list(self.confidence_dir.glob("*.npy"))) if self.confidence_dir.exists() else []
        self.extrinsic_files = sorted(list(self.extrinsic_dir.glob("*.npy")))

        # 4D: dynamic depth (car region only), produced by Pi3Reconstructor.
        self.dynamic_depth_dir = data_path / "dynamic_depth"
        self.dynamic_depth_files = (
            sorted(list(self.dynamic_depth_dir.glob("*.npy")))
            if self.dynamic_depth_dir.exists()
            else []
        )

        assert len(self.rgb_files) > 0, "No RGB files found"
        assert len(self.rgb_files) == len(self.depth_files) == len(self.extrinsic_files), "Mismatch in file counts"

        if self.use_per_frame_intrinsics:
            assert len(self.rgb_files) == len(self.intrinsic_files), "Mismatch between RGB and intrinsic file counts"

    @property
    def num_frames(self) -> int:
        return len(self.rgb_files)

    @property
    def height(self) -> int:
        return self.height_val

    @property
    def width(self) -> int:
        return self.width_val

    @property
    def has_dynamic(self) -> bool:
        """True when dynamic_depth files exist (Pi3 4D output)."""
        return len(self.dynamic_depth_files) > 0

    def get_dynamic_points(self, index: int, downsample_factor: int = 1):
        """Unproject the dynamic (car-only) depth into world-space points.

        Returns:
            (world_coords, rgb_flat) arrays — both shape (N, 3).  Empty arrays
            are returned when no dynamic depth is available for this frame.
        """
        empty = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
        if not self.has_dynamic or index >= len(self.dynamic_depth_files):
            return empty

        dynamic_depth = np.load(self.dynamic_depth_files[index])
        if downsample_factor > 1:
            dynamic_depth = dynamic_depth[::downsample_factor, ::downsample_factor]

        # Intrinsics for this frame.
        if getattr(self, "use_per_frame_intrinsics", False) and getattr(self, "intrinsic_files", None):
            K_mat = np.load(self.intrinsic_files[index])
            fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        # Extrinsic -> cam_c2w.
        extrinsic_mat = np.load(self.extrinsic_files[index])
        if extrinsic_mat.shape == (3, 4):
            cam_c2w = np.eye(4)
            cam_c2w[:3, :3] = extrinsic_mat[:3, :3]
            cam_c2w[:3, 3] = extrinsic_mat[:3, 3]
        elif extrinsic_mat.shape == (4, 4):
            cam_c2w = extrinsic_mat.copy()
        else:
            return empty

        h, w = dynamic_depth.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u, v = u.flatten(), v.flatten()
        z = dynamic_depth.flatten()
        valid = z > 0
        if not np.any(valid):
            return empty

        fx_d = fx / downsample_factor
        fy_d = fy / downsample_factor
        cx_d = cx / downsample_factor
        cy_d = cy / downsample_factor

        z_valid = z[valid] * self.scene_scale
        cam_c2w[:3, 3] *= self.scene_scale

        x = (u[valid] - cx_d) * z_valid / fx_d
        y = (v[valid] - cy_d) * z_valid / fy_d
        cam_coords = np.stack([x, y, z_valid, np.ones(valid.sum())], axis=-1)
        world_coords = (cam_coords @ cam_c2w.T)[:, :3]

        # Use the composited RGB (car visible) if available, otherwise plain RGB.
        rgb_img = iio.imread(self.rgb_files[index]).astype(np.float32) / 255.0
        if downsample_factor > 1:
            rgb_img = rgb_img[::downsample_factor, ::downsample_factor]
        rgb_flat = rgb_img.reshape(-1, 3)[valid]

        return world_coords.astype(np.float32), rgb_flat.astype(np.float32)

    def get_frame_data(self, index: int, downsample_factor: int = 1, min_confidence: float = 0.0):
        # Load RGB
        rgb_img = iio.imread(self.rgb_files[index])
        rgb_img = rgb_img.astype(np.float16) / 255.0

        # Load Depth
        depth_img = np.load(self.depth_files[index])

        # Load Confidence
        confidence_img = None
        if index < len(self.confidence_files):
            confidence_img = np.load(self.confidence_files[index])

        # Load intrinsics (per-frame or global)
        if self.use_per_frame_intrinsics:
            if self.intrinsic_format == "npy":
                K_mat = np.load(self.intrinsic_files[index])
                fx = K_mat[0, 0]
                fy = K_mat[1, 1]
                cx = K_mat[0, 2]
                cy = K_mat[1, 2]
        else:
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy

        # Load Extrinsic
        # Expected format: camera-to-world (c2w) transformation matrix
        # Camera coordinate system convention: X right, Y down, Z forward
        # The c2w matrix transforms points from camera space to world space
        extrinsic_mat = np.load(self.extrinsic_files[index])

        # Check if extrinsic is in matrix format (3x4 [R|t]) or 4x4
        if extrinsic_mat.shape == (3, 4):
            # [R | t] format (camera-to-world)
            # R: rotation from camera to world coordinates
            # t: camera position in world coordinates
            R_mat = extrinsic_mat[:3, :3]
            t_vec = extrinsic_mat[:3, 3]

            cam_c2w = np.eye(4)
            cam_c2w[:3, :3] = R_mat
            cam_c2w[:3, 3] = t_vec
        elif extrinsic_mat.shape == (4, 4):
            # Full 4x4 matrix (camera-to-world)
            cam_c2w = extrinsic_mat.copy()
        elif extrinsic_mat.shape == (6,):
            T = extrinsic_mat[[2, 0, 1]]
            # Assuming XYZ Euler angles in degrees
            r = R.from_euler("xyz", [extrinsic_mat[3:]], degrees=True)
            R_mat = r.as_matrix()
            cam_c2w = np.eye(4)
            cam_c2w[:3, :3] = R_mat
            cam_c2w[:3, 3] = T
        else:
            raise ValueError(f"Unexpected extrinsic matrix shape: {extrinsic_mat.shape}")

        # Downsample
        if downsample_factor > 1:
            rgb_img = rgb_img[::downsample_factor, ::downsample_factor]
            depth_img = depth_img[::downsample_factor, ::downsample_factor]
            if confidence_img is not None:
                confidence_img = confidence_img[::downsample_factor, ::downsample_factor]

        h, w = depth_img.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        u = u.flatten()
        v = v.flatten()
        z = depth_img.flatten()

        # Unproject
        # Note: cx, cy, fx, fy need to be scaled by downsample_factor
        fx_d = fx / downsample_factor
        fy_d = fy / downsample_factor
        cx_d = cx / downsample_factor
        cy_d = cy / downsample_factor

        z = z * self.scene_scale
        cam_c2w[:3, 3] *= self.scene_scale

        x = (u - cx_d) * z / fx_d
        y = (v - cy_d) * z / fy_d

        cam_coords = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        world_coords = (cam_coords @ cam_c2w.T)[:, :3]

        rgb_flat = rgb_img.reshape(-1, 3)

        # Filter by confidence
        if confidence_img is not None and min_confidence > 0:
            mask = confidence_img.flatten() >= min_confidence
            world_coords = world_coords[mask]
            rgb_flat = rgb_flat[mask]

        return world_coords, rgb_flat, None, cam_c2w, rgb_img

    def display_rotation(self, R: np.ndarray) -> np.ndarray:
        """Identity — pipeline outputs already match viser convention."""
        return R

    def load_depth(self, index: int) -> np.ndarray:
        return np.load(self.depth_files[index])


# ======================================================================
# Auto-detect loader
# ======================================================================

def create_loader(data_path: Path) -> FolderLoader | StaDy4DLoader:
    """Auto-detect data format and return the appropriate loader.

    - If ``data_path`` contains ``rgb.mp4`` -> StaDy4DLoader (raw StaDy4D)
    - Otherwise -> FolderLoader (pipeline output with per-frame files)
    """
    if (data_path / "rgb.mp4").exists():
        return StaDy4DLoader(data_path)
    return FolderLoader(data_path)


def _render_points_to_image(
    points: np.ndarray,
    colors: np.ndarray,
    cam_c2w: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project a point cloud into an image from a given camera pose."""
    if points.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    cam_R = cam_c2w[:3, :3]
    cam_t = cam_c2w[:3, 3]
    cam_coords = (points - cam_t) @ cam_R

    z = cam_coords[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)

    cam_coords = cam_coords[valid]
    z = z[valid]
    proj_colors = colors[valid]

    u = (cam_coords[:, 0] * fx / z) + cx
    v = (cam_coords[:, 1] * fy / z) + cy

    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Use floor + clipping to avoid rounding outside the image bounds.
    u = np.floor(u[valid]).astype(np.int32)
    v = np.floor(v[valid]).astype(np.int32)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    z = z[valid].astype(np.float32)
    proj_colors = proj_colors[valid].astype(np.float32)

    lin = v * width + u
    depth_flat = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(depth_flat, lin, z)

    hit_mask = np.isclose(depth_flat[lin], z, rtol=1e-3, atol=1e-6)
    color_flat = np.zeros((width * height, 3), dtype=np.float32)
    color_flat[lin[hit_mask]] = proj_colors[hit_mask]

    image = color_flat.reshape(height, width, 3)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def record_accumulate_gif(
    positions: list[np.ndarray],
    colors: list[np.ndarray],
    cam_c2w: np.ndarray | list[np.ndarray],
    gif_path: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fps: float,
) -> None:
    """Record an accumulate-mode GIF from preloaded point clouds."""
    if gif_path is None:
        return

    gif_path.parent.mkdir(parents=True, exist_ok=True)

    total_points = sum(len(p) for p in positions)
    if total_points == 0:
        print("No points available for GIF recording.")
        return

    all_points = np.empty((total_points, 3), dtype=np.float32)
    all_colors = np.empty((total_points, 3), dtype=np.float32)
    prefixes: list[int] = []
    cursor = 0
    for pts, cols in zip(positions, colors):
        n = len(pts)
        all_points[cursor : cursor + n] = pts.astype(np.float32, copy=False)
        all_colors[cursor : cursor + n] = cols.astype(np.float32, copy=False)
        cursor += n
        prefixes.append(cursor)

    # Handle single camera pose vs list of poses
    if isinstance(cam_c2w, list):
        cam_c2w_list = cam_c2w
    else:
        cam_c2w_list = [cam_c2w] * len(prefixes)

    frames: list[np.ndarray] = []
    for idx, prefix in enumerate(tqdm(prefixes, desc="Recording accumulate GIF")):
        # Use the corresponding camera pose for this frame, or the last one if we run out
        current_cam_c2w = cam_c2w_list[min(idx, len(cam_c2w_list) - 1)]

        frame_img = _render_points_to_image(
            all_points[:prefix], all_colors[:prefix], current_cam_c2w, width, height, fx, fy, cx, cy
        )
        frames.append(frame_img)

    if frames:
        iio.imwrite(gif_path, frames, duration=1.0 / max(fps, 1e-3), loop=0)
        print(f"Saved accumulate GIF to {gif_path}")


def main(
    data_path: Path,
    downsample_factor: int = 1,
    max_frames: int = 300,
    share: bool = False,
    point_downsample: float = 1.0,
    point_stride: int = 1,
    min_confidence: float = 0.0,
    record_gif_path: Path | None = None,
    record_gif_fps: float = 10.0,
    zoom_scale: float = 0.7,
    num_rotate_frames: int = 60,
    total_rotation_deg: float = 60.0,
) -> None:
    server = viser.ViserServer()
    server.scene.set_up_direction("-y")  # scene Y is inverted (up in world = -Y in rendered coords)
    if share:
        server.request_share_url()

    print("Loading frames!")
    loader = create_loader(data_path)

    num_frames = min(max_frames, loader.num_frames)
    first_position, first_color, first_track, initial_cam_c2w, first_rgb = loader.get_frame_data(
        0, downsample_factor, min_confidence
    )

    cam_origin = initial_cam_c2w[:3, 3]
    dists = np.linalg.norm(first_position - cam_origin, axis=1)
    valid_mask = np.isfinite(first_position).all(axis=1) & (dists < 1000)
    recenter_offset = first_position[valid_mask].mean(axis=0) if np.any(valid_mask) else np.zeros(3)
    first_position = first_position - recenter_offset
    initial_cam_c2w = initial_cam_c2w.copy()
    initial_cam_c2w[:3, 3] = initial_cam_c2w[:3, 3] - recenter_offset

    mid_index = num_frames // 2
    mid_position, _, _, mid_cam_c2w, _ = loader.get_frame_data(mid_index, downsample_factor, min_confidence)
    mid_position = mid_position - recenter_offset
    mid_cam_c2w = mid_cam_c2w.copy()
    mid_cam_c2w[:3, 3] = mid_cam_c2w[:3, 3] - recenter_offset

    depth_img_mid = loader.load_depth(mid_index)
    if downsample_factor > 1:
        depth_img_mid = depth_img_mid[::downsample_factor, ::downsample_factor]
    hm, wm = depth_img_mid.shape[:2]
    center_depth_mid = depth_img_mid[hm // 2, wm // 2]
    if not np.isfinite(center_depth_mid) or center_depth_mid <= 0:
        valid_depths_mid = depth_img_mid[np.isfinite(depth_img_mid) & (depth_img_mid > 0)]
        center_depth_mid = float(np.median(valid_depths_mid)) if valid_depths_mid.size > 0 else 1.0

    forward_world = mid_cam_c2w[:3, 2]  # camera forward direction in world
    record_cam_c2w = mid_cam_c2w.copy()
    record_cam_c2w[:3, 3] = mid_cam_c2w[:3, 3] - forward_world * (zoom_scale * center_depth_mid)
    offset_position = record_cam_c2w[:3, 3]

    # Intrinsics for recording
    fx_rec = float(loader.fx)
    fy_rec = float(loader.fy)
    cx_rec = float(loader.cx)
    cy_rec = float(loader.cy)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.0001 * point_stride,
            max=0.003 * point_stride,
            step=0.0001 * point_stride,
            initial_value=0.003 * point_stride,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=10)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))
        gui_show_all_frames = server.gui.add_checkbox("Show All Frames", False)
        gui_accumulate_mode = server.gui.add_checkbox("Accumulate Mode", False)
        gui_accumulate_dynamic = server.gui.add_checkbox("  With Dynamic", False, disabled=True)

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    # ------------------------------------------------------------------
    # Visibility helpers
    # ------------------------------------------------------------------

    def _set_static_visibility(mode: str, t: int) -> None:
        """mode: "current" | "past" | "all_except" | "all"

        current     – only frame i==t
        past        – frames 0..t-1 (Normal Acc)
        all_except  – all frames except i==t (Dynamic Acc background)
        all         – every frame
        """
        for i, fn in enumerate(frame_nodes):
            if mode == "current":
                fn.visible = i == t
            elif mode == "past":
                fn.visible = i < t
            elif mode == "all_except":
                fn.visible = i != t
            else:  # "all"
                fn.visible = True

    def _set_dynamic_visibility(mode: str, t: int = 0) -> None:
        """mode: "none" | "rolling"

        none    – all hidden
        rolling – only frame i==t visible
        """
        for i, dyn in enumerate(dynamic_frame_nodes):
            if dyn is not None:
                dyn.visible = (mode == "rolling") and (i == t)

    def _set_frustum_visibility(t: int) -> None:
        """Only the current timestep's camera frustum is shown."""
        for i, fn in enumerate(frustum_nodes):
            fn.visible = i == t

    def _apply_mode(t: int) -> None:
        """Apply static + dynamic + frustum visibility for the current UI state."""
        if gui_show_all_frames.value:
            # Show full static scene + all camera frustums, no dynamic
            _set_static_visibility("all", t)
            _set_dynamic_visibility("none", t)
            for fn in frustum_nodes:
                fn.visible = True
            return
        elif gui_accumulate_mode.value:
            if gui_accumulate_dynamic.value:
                # Dynamic Acc: static = all except t, dynamic = t only
                _set_static_visibility("all_except", t)
                _set_dynamic_visibility("rolling", t)
            else:
                # Normal Acc: static = 0..t-1, no dynamic
                _set_static_visibility("past", t)
                _set_dynamic_visibility("none", t)
        else:
            # Normal mode: static scene only at t, no dynamic object
            _set_static_visibility("current", t)
            _set_dynamic_visibility("none", t)
        _set_frustum_visibility(t)

    # ------------------------------------------------------------------
    # GUI callbacks
    # ------------------------------------------------------------------

    @gui_show_all_frames.on_update
    def _(_) -> None:
        with server.atomic():
            if gui_show_all_frames.value:
                gui_playing.value = False
                gui_playing.disabled = True
                gui_accumulate_mode.disabled = True
                gui_accumulate_dynamic.disabled = True
            else:
                gui_playing.disabled = False
                gui_accumulate_mode.disabled = False
                gui_accumulate_dynamic.disabled = not gui_accumulate_mode.value
            _apply_mode(gui_timestep.value)

    @gui_accumulate_mode.on_update
    def _(_) -> None:
        with server.atomic():
            if gui_accumulate_mode.value:
                gui_show_all_frames.disabled = True
                gui_accumulate_dynamic.disabled = False
            else:
                gui_show_all_frames.disabled = False
                gui_accumulate_dynamic.disabled = True
            _apply_mode(gui_timestep.value)

    @gui_accumulate_dynamic.on_update
    def _(_) -> None:
        with server.atomic():
            _apply_mode(gui_timestep.value)

    prev_timestep = gui_timestep.value

    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        with server.atomic():
            _apply_mode(gui_timestep.value)
        prev_timestep = gui_timestep.value
        server.flush()

    offset_position = record_cam_c2w[:3, 3]

    # Set viewer camera to match first frame's orientation, positioned beyond the camera
    # This needs to be done via the client connect callback
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        cam_pos = initial_cam_c2w[:3, 3]
        client.camera.position = tuple(cam_pos)
        client.camera.look_at = tuple(np.zeros(3))  # look at scene center

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        show_axes=False,
    )

    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    frustum_nodes: list[viser.CameraFrustumHandle] = []
    positions_for_record: list[np.ndarray] = []
    colors_for_record: list[np.ndarray] = []

    # Separate root for camera frustums so they can be controlled independently
    # from point-cloud frame nodes.
    server.scene.add_frame("/cameras", show_axes=False)

    # 4D dynamic nodes — one per frame, always shows only the current timestep.
    server.scene.add_frame("/dynamics", show_axes=False)
    dynamic_frame_nodes: list[viser.FrameHandle | None] = []
    dynamic_point_nodes: list[viser.PointCloudHandle | None] = []

    for i in tqdm(range(num_frames)):
        if i == 0:
            position, color, track, cam_c2w, frustum_img = (
                first_position,
                first_color,
                first_track,
                initial_cam_c2w,
                first_rgb,
            )
        else:
            position, color, track, cam_c2w, frustum_img = loader.get_frame_data(i, downsample_factor, min_confidence)
            position = position - recenter_offset
            cam_c2w = cam_c2w.copy()
            cam_c2w[:3, 3] = cam_c2w[:3, 3] - recenter_offset

        # Apply spatial downsampling
        if point_downsample < 1.0:
            num_points = len(position)
            num_sampled = max(1, int(num_points * point_downsample))
            indices = np.random.choice(num_points, num_sampled, replace=False)
            position = position[indices]
            color = color[indices]

        # Apply stride-based downsampling
        if point_stride > 1:
            position = position[::point_stride]
            color = color[::point_stride]

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        positions_for_record.append(position)
        colors_for_record.append(color)

        # Place the point cloud in the frame.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=position,
                colors=color,
                point_size=0.01,
                point_shape="rounded",
            )
        )

        if track is not None:
            r = np.linspace(0, 1, track.shape[0])
            g = np.linspace(1, 0, track.shape[0])
            b = np.linspace(0.5, 1, track.shape[0])

            colors = np.stack([r, g, b], axis=-1)  # shape (100, 3)
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/some_points",
                points=track,
                colors=colors,
                point_size=0.003,
                point_shape="rounded",
            )

        # Place the frustum under /cameras (independent of point-cloud frames).
        fov = 2 * np.arctan2(loader.height / 2, loader.K[0, 0])
        aspect = loader.width / loader.height
        frustum_nodes.append(server.scene.add_camera_frustum(
            f"/cameras/t{i}",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frustum_img,
            wxyz=tf.SO3.from_matrix(loader.display_rotation(cam_c2w[:3, :3])).wxyz,
            position=cam_c2w[:3, 3],
        ))
        server.scene.add_frame(
            f"/cameras/t{i}/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

        # 4D: dynamic (car-only) point cloud for this frame.
        if loader.has_dynamic:
            dyn_pos, dyn_col = loader.get_dynamic_points(i, downsample_factor)
            dyn_pos = dyn_pos - recenter_offset
            if point_stride > 1:
                dyn_pos = dyn_pos[::point_stride]
                dyn_col = dyn_col[::point_stride]
            if len(dyn_pos) > 0:
                dyn_frame = server.scene.add_frame(f"/dynamics/t{i}", show_axes=False)
                dyn_node = server.scene.add_point_cloud(
                    name=f"/dynamics/t{i}/point_cloud",
                    points=dyn_pos,
                    colors=dyn_col,
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                )
                dynamic_frame_nodes.append(dyn_frame)
                dynamic_point_nodes.append(dyn_node)
            else:
                dynamic_frame_nodes.append(None)
                dynamic_point_nodes.append(None)
        else:
            dynamic_frame_nodes.append(None)
            dynamic_point_nodes.append(None)

    _apply_mode(gui_timestep.value)  # initial visibility (normal mode)

    if record_gif_path is not None:
        gui_accumulate_mode.value = True
        fx_d = fx_rec / downsample_factor
        fy_d = fy_rec / downsample_factor
        cx_d = cx_rec / downsample_factor
        cy_d = cy_rec / downsample_factor
        img_h, img_w = first_rgb.shape[:2]

        # Generate orbital camera path
        cam_c2w_list = []
        num_accum_frames = len(positions_for_record)

        # Part 1: Accumulation phase (static camera)
        for _ in range(num_accum_frames):
            cam_c2w_list.append(record_cam_c2w)

        # Part 2: Rotation phase (0 -> total_rotation_deg -> 0 degrees)
        # Extend positions and colors with empty data to keep the accumulated point cloud
        for _ in range(num_rotate_frames):
            positions_for_record.append(np.zeros((0, 3), dtype=np.float32))
            colors_for_record.append(np.zeros((0, 3), dtype=np.float32))

        for i in range(num_rotate_frames):
            # Sine wave pattern for 0 -> Max -> 0
            progress = i / max(1, num_rotate_frames - 1)
            angle_deg = total_rotation_deg * np.sin(progress * np.pi)
            
            r_orbit = R.from_euler("y", angle_deg, degrees=True).as_matrix()

            # Apply rotation to the camera position (orbiting around 0,0,0)
            # and to the camera orientation
            new_c2w = record_cam_c2w.copy()
            new_c2w[:3, 3] = r_orbit @ record_cam_c2w[:3, 3]
            new_c2w[:3, :3] = r_orbit @ record_cam_c2w[:3, :3]

            cam_c2w_list.append(new_c2w)

        record_accumulate_gif(
            positions_for_record,
            colors_for_record,
            cam_c2w_list,
            record_gif_path,
            img_w,
            img_h,
            fx_d,
            fy_d,
            cx_d,
            cy_d,
            record_gif_fps,
        )

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[(gui_timestep.value + 1) % num_frames].point_size = gui_point_size.value

        # Update dynamic node point size for current timestep.
        dyn_node = dynamic_point_nodes[gui_timestep.value]
        if dyn_node is not None:
            dyn_node.point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True, help="Path to pipeline output dir or StaDy4D camera dir")
    parser.add_argument("--downsample_factor", type=int, default=1, help="Image downsampling factor")
    parser.add_argument("--max_frames", type=int, default=300, help="Maximum number of frames to load")
    parser.add_argument("--share", action="store_true", help="Share the visualization via URL")
    parser.add_argument(
        "--point_downsample", type=float, default=1.0, help="Point cloud downsampling ratio (0.0-1.0, 1.0 = all points)"
    )
    parser.add_argument(
        "--point_stride",
        type=int,
        default=1,
        help="Point cloud stride-based downsampling (e.g., 10 = every 10th point)",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.4,
        help="Minimum confidence threshold for point cloud visualization",
    )
    parser.add_argument("--record_gif_path", type=Path, default=None, help="Path to save accumulate-mode GIF")
    parser.add_argument("--record_gif_fps", type=float, default=10.0, help="FPS for the accumulate GIF")
    parser.add_argument("--zoom_scale", type=float, default=0.7, help="Zoom scale for camera positioning (default: 0.7)")
    parser.add_argument("--num_rotate_frames", type=int, default=60, help="Number of frames for rotation phase in GIF (default: 60)")
    parser.add_argument("--total_rotation_deg", type=float, default=60.0, help="Total rotation degrees for orbital camera path (default: 60.0)")
    args = parser.parse_args()

    main(
        args.data_path,
        args.downsample_factor,
        args.max_frames,
        args.share,
        args.point_downsample,
        args.point_stride,
        args.min_confidence,
        args.record_gif_path,
        args.record_gif_fps,
        args.zoom_scale,
        args.num_rotate_frames,
        args.total_rotation_deg,
    )
