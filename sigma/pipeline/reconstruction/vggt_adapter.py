"""VGGT-based reconstruction implementation."""

from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.reconstruction.aggregator import SceneAggregator, SceneFragment
from sigma.pipeline.reconstruction.base_recon import (
    BaseReconstructor,
    CameraParameters,
    ReconstructionOutputs,
    SceneReconstruction,
    depth_to_pointcloud,
    estimate_normals,
    extract_frames_from_records,
)
from sigma.pipeline.reconstruction.vggt_utils import preprocess_frame_dict_for_vggt, rescale_vggt_prediction, rescale_vggt_intrinsic
from sigma.utils.log_messages import log_config, log_step, log_success

LOGGER = logging.getLogger(__name__)


# Helper functions for quaternion operations (used by online reconstructor)
def rotation_matrix_to_quaternion(R_mat: np.ndarray) -> np.ndarray:
    rot = R.from_matrix(R_mat)
    return rot.as_quat()


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    rot = R.from_quat(q)
    return rot.as_matrix()


def compute_rotation_offset(q_target: np.ndarray, q_source: np.ndarray) -> np.ndarray:
    """Compute rotation offset from source to target quaternion.

    Args:
        q_target: Target quaternion [x, y, z, w]
        q_source: Source quaternion [x, y, z, w]

    Returns:
        Offset quaternion q_hat such that q_target = q_hat * q_source
    """
    rot_target = R.from_quat(q_target)
    rot_source = R.from_quat(q_source)
    rot_offset = rot_target * rot_source.inv()
    return rot_offset.as_quat()


def average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    """Average multiple quaternions using eigenvalue decomposition.

    Args:
        quaternions: List of quaternions [x, y, z, w]

    Returns:
        Average quaternion [x, y, z, w]
    """
    # Stack quaternions into matrix
    Q = np.array(quaternions)  # Shape: (N, 4)

    # Compute the average using the method from Markley et al. 2007
    # Build the 4x4 matrix M = sum(q_i * q_i^T)
    M = np.zeros((4, 4))
    for q in Q:
        M += np.outer(q, q)
    M /= len(Q)

    # The average quaternion is the eigenvector corresponding to the largest eigenvalue
    _, eigenvectors = np.linalg.eigh(M)
    avg_quat = eigenvectors[:, -1]  # Eigenvector with largest eigenvalue

    return avg_quat


def build_intrinsics_from_fov(
    fov_h: torch.Tensor, fov_w: torch.Tensor, image_size_hw: tuple
) -> torch.Tensor:
    """Build intrinsic matrix from field of view parameters.

    Args:
        fov_h: Vertical field of view in radians
        fov_w: Horizontal field of view in radians
        image_size_hw: Image size as (height, width)

    Returns:
        Intrinsic matrix [3, 3]
    """
    H, W = image_size_hw
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)

    intrinsics = torch.zeros(fov_h.shape + (3, 3), device=fov_h.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0

    return intrinsics

def _composite_moving_object(
    scene_frame: np.ndarray,
    origin_frame: np.ndarray | None,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    """Blend the moving object from origin_frame back onto the inpainted scene_frame."""
    if origin_frame is None or mask is None:
        return None

    import cv2

    H, W = scene_frame.shape[:2]
    if origin_frame.shape[:2] != (H, W):
        origin_frame = cv2.resize(origin_frame, (W, H))
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    result = scene_frame.copy()
    moving = mask > 0
    if moving.any():
        result[moving] = origin_frame[moving]
    return result


class VGGTOfflineReconstructor(BaseReconstructor):
    """VGGT (Visual Geometry Grounded Transformer) 3D reconstruction.

    This reconstructor takes multiple frames and produces:
    - Camera intrinsics and extrinsics for each view
    - Dense depth maps per frame
    - 3D point clouds with confidence scores
    - Aggregated global scene reconstruction

    Args:
        checkpoint: Path or identifier for VGGT model checkpoint.
        min_confidence: Minimum confidence threshold for point filtering (0-1).
        device: Device to use for reconstruction ("cuda" or "cpu").
        use_gpu: Whether to use GPU acceleration for reconstruction (deprecated, use device).
        max_frames_batch: Maximum number of frames to process in a single batch.
        estimate_normals: Whether to estimate surface normals from depth.
        depth_scale: Scale factor for depth values (meters per unit).
    """

    name = "vggt_reconstructor"
    run_deferred = True  # Run once on all frames after inpainting, not per-frame

    def __init__(
        self,
        checkpoint: str = "facebook/VGGT-1B",
        min_confidence: float = 0.4,
        device: str | None = None,
        max_frames: int = 5,
        estimate_normals: bool = True,
        depth_scale: float = 1.0,
        exclude_moving_objects: bool = True,
        use_origin_frames: bool = True,
        **kwargs: Any,
    ) -> None:
        aggregator = kwargs.get("aggregator") or SceneAggregator()
        super().__init__(aggregator=aggregator)
        self.checkpoint = checkpoint
        self.min_confidence = min_confidence

        # Support both new 'device' parameter and legacy 'use_gpu'
        self.max_frames = max_frames
        self.device = device
        self.estimate_normals = estimate_normals
        self.depth_scale = depth_scale
        self.exclude_moving_objects = exclude_moving_objects
        self.use_origin_frames = use_origin_frames
        self.model = None
        self.initialized = False

    def setup(self) -> None:
        """Load VGGT model and initialize reconstruction pipeline."""
        log_step(LOGGER, "Loading VGGT model", self.checkpoint)

        from vggt.models.vggt import VGGT

        # Use device from config directly (simpler than parent)
        self.model = VGGT.from_pretrained(self.checkpoint).to(self.device)
        self.initialized = True

        log_success(LOGGER, "VGGT reconstructor initialized")

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Run VGGT reconstruction on input frames.

        Args:
            frame_records: Dictionary of stored :class:`FrameIORecord` objects.
            frame_idx: Current timestep index.

        Returns:
            StageResult with comprehensive reconstruction outputs including camera parameters,
            depth maps, 3D point clouds, and aggregated scene.
        """
        if not self.initialized:
            raise RuntimeError("VGGTReconstructor.setup() must be called first.")

        frames_dict = extract_frames_from_records(frame_records, frame_idx, max_frames=self.max_frames)
        frame_index = frame_idx

        # Run VGGT reconstruction on all available frames
        reconstruction_result = self._run_vggt_reconstruction(frames_dict)

        # Build current scene for the latest frame
        current_scene = self._build_current_scene(
            reconstruction_result, frame_index
        )

        # Add to aggregator
        if current_scene.points_3d is not None:
            fragment = SceneFragment(
                points=current_scene.points_3d,
                confidence=current_scene.confidence,
                metadata={
                    "timestep": frame_index,
                    "num_points": len(current_scene.points_3d),
                    "colors": current_scene.colors,
                    "camera": current_scene.camera,
                },
            )
            self.aggregator.add_fragment(fragment)
        current_record = frame_records.get(frame_index)
        if current_record is not None:
            current_record.depth = current_scene.depth_map
            if current_scene.camera is not None:
                current_record.extrinsic = current_scene.camera.extrinsic
                current_record.intrinsic = current_scene.camera.intrinsic

        # Build aggregated scene from all fragments
        aggregated_scene = self.aggregator.build_scene()

        # Create comprehensive outputs
        recon_outputs = ReconstructionOutputs(
            frame_idx=frame_idx,
            depth_map=current_scene.depth_map.astype(np.float16),
            aggregated_scene=aggregated_scene,
            per_view_cameras=reconstruction_result["per_view_cameras"],
            confidence_map=current_scene.confidence,
            current_scene=current_scene, 
            current_frame=reconstruction_result.get("current_frame"), # for visualization
            point_cloud=current_scene.points_3d, # for saving ply
            point_color=current_scene.colors, # for saving ply
            global_point_cloud=reconstruction_result.get("global_points"), # for saving gif
            global_point_color=reconstruction_result.get("global_colors"), # for saving gif
            reconstruction_metadata={
                "num_views": len(frames_dict),
                "current_timestep": frame_index,
                "min_confidence": self.min_confidence,
                "total_points": reconstruction_result.get("total_points", 0),
            },
            extrinsic=current_scene.camera.extrinsic,
            intrinsic=current_scene.camera.intrinsic,
        )
        vis_assets = asdict(recon_outputs)
        if reconstruction_result.get("all_data") is not None:
            vis_assets.update(reconstruction_result["all_data"])

        return StageResult(
            data={"reconstruction": recon_outputs, "frame_idx": frame_index, "frame_size": reconstruction_result.get("current_frame").shape[:2]},
            visualization_assets=vis_assets,
        )

    def finalize(self, frame_records: Dict[int, FrameIORecord]) -> List[Tuple[int, StageResult]]:
        """Run VGGT once on all inpainted frames after inpainting is complete.

        Args:
            frame_records: All collected frame records with inpainted images.

        Returns:
            List of (frame_idx, StageResult) for each processed frame.
        """
        if not self.initialized:
            raise RuntimeError("VGGTOfflineReconstructor.setup() must be called first.")

        frames_dict = extract_frames_from_records(
            frame_records, max(frame_records), max_frames=self.max_frames
        )
        timesteps = sorted(frames_dict.keys())
        if not timesteps:
            return []

        # Collect masks & originals for filtering / compositing (mirrors Pi3 pattern).
        masks_dict: Dict[int, np.ndarray] = {}
        origin_dict: Dict[int, np.ndarray] = {}
        for idx, record in frame_records.items():
            if idx in frames_dict:
                if record.mask is not None:
                    masks_dict[idx] = record.mask
                if record.origin_image is not None:
                    origin_dict[idx] = record.origin_image

        # When use_origin_frames: feed VGGT the original (unmasked) frames so it
        # can reconstruct depth for the dynamic object region too.
        if self.use_origin_frames:
            vggt_input_dict: Dict[int, np.ndarray] = {}
            for idx in timesteps:
                record = frame_records.get(idx)
                frame = (
                    record.origin_image
                    if record is not None and record.origin_image is not None
                    else frames_dict.get(idx)
                )
                if frame is not None:
                    vggt_input_dict[idx] = frame
        else:
            vggt_input_dict = frames_dict

        log_step(LOGGER, f"Running VGGT offline on {len(timesteps)} frames")
        reconstruction_result = self._run_vggt_reconstruction(vggt_input_dict)

        results: List[Tuple[int, StageResult]] = []
        for frame_idx in timesteps:
            # Use inpainted frame for color (even when origin frames were fed to VGGT).
            reconstruction_result["current_frame"] = frames_dict[frame_idx]
            current_scene = self._build_current_scene(reconstruction_result, frame_idx)

            # ---- Apply moving-object mask (mirrors Pi3._build_frame_result) ----
            import cv2 as _cv2
            dynamic_depth: np.ndarray | None = None
            mask_frame = masks_dict.get(frame_idx)
            if mask_frame is not None:
                depth_map = current_scene.depth_map
                H_d, W_d = depth_map.shape[:2]
                moving = (mask_frame > 0).astype(np.uint8)
                if moving.shape != (H_d, W_d):
                    moving = _cv2.resize(moving, (W_d, H_d), interpolation=_cv2.INTER_NEAREST)
                moving_bool = moving.astype(bool)

                # Squeeze trailing channel dim for clean np.where broadcast.
                depth_2d = depth_map.squeeze() if depth_map.ndim == 3 else depth_map
                dynamic_depth = np.where(moving_bool, depth_2d, 0.0).astype(np.float32)

                if self.exclude_moving_objects:
                    static_depth = np.where(moving_bool, 0.0, depth_2d).astype(np.float32)
                    current_scene.depth_map = static_depth
                    if current_scene.confidence is not None:
                        conf = current_scene.confidence
                        if conf.ndim == 1:
                            conf = conf.reshape(H_d, W_d)
                        conf = np.where(moving_bool, 0.0, conf)
                        current_scene.confidence = conf.flatten()

            # ---- Composited frame: moving object blended back onto inpainted ----
            composited = _composite_moving_object(
                frames_dict[frame_idx],
                origin_dict.get(frame_idx),
                mask_frame,
            )

            # Update frame record with depth and camera parameters
            record = frame_records.get(frame_idx)
            if record is not None:
                record.depth = current_scene.depth_map
                if current_scene.camera is not None:
                    record.extrinsic = current_scene.camera.extrinsic
                    record.intrinsic = current_scene.camera.intrinsic

            # Accumulate into aggregator
            if current_scene.points_3d is not None:
                fragment = SceneFragment(
                    points=current_scene.points_3d,
                    confidence=current_scene.confidence,
                    metadata={
                        "timestep": frame_idx,
                        "num_points": len(current_scene.points_3d),
                        "colors": current_scene.colors,
                        "camera": current_scene.camera,
                    },
                )
                self.aggregator.add_fragment(fragment)

            aggregated_scene = self.aggregator.build_scene()

            recon_outputs = ReconstructionOutputs(
                frame_idx=frame_idx,
                depth_map=current_scene.depth_map.astype(np.float16),
                aggregated_scene=aggregated_scene,
                per_view_cameras=reconstruction_result["per_view_cameras"],
                confidence_map=current_scene.confidence,
                current_scene=current_scene,
                # Show composited view (car blended back) for visualization; depth
                # and confidence still reflect background-only reconstruction.
                current_frame=composited if composited is not None else frames_dict[frame_idx],
                point_cloud=current_scene.points_3d,
                point_color=current_scene.colors,
                global_point_cloud=None,   # populated in last result only (below)
                global_point_color=None,
                reconstruction_metadata={
                    "num_views": len(frames_dict),
                    "current_timestep": frame_idx,
                    "min_confidence": self.min_confidence,
                    "total_points": reconstruction_result.get("total_points", 0),
                },
                extrinsic=current_scene.camera.extrinsic,
                intrinsic=current_scene.camera.intrinsic,
            )
            vis_assets = asdict(recon_outputs)
            if composited is not None:
                vis_assets["composited_frame"] = composited
            if dynamic_depth is not None:
                vis_assets["dynamic_depth"] = dynamic_depth

            results.append((frame_idx, StageResult(
                data={
                    "reconstruction": recon_outputs,
                    "frame_idx": frame_idx,
                    "frame_size": frames_dict[frame_idx].shape[:2],
                },
                visualization_assets=vis_assets,
            )))

        # Populate global_point_cloud in the very last result so the pipeline
        # runner can pass it to save_reconstruction_summary() for PLY/GIF export.
        if results:
            final_scene = self.aggregator.build_scene()
            _, last_result = results[-1]
            last_result.visualization_assets["global_point_cloud"] = final_scene.get("merged_points")
            last_result.visualization_assets["global_point_color"] = final_scene.get("merged_colors")

        return results

    def _run_vggt_reconstruction(self, frames: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Run VGGT model on input frames to produce 3D reconstruction.

        Args:
            frames: Dictionary of RGB frames {timestep: (H, W, 3)}.

        Returns:
            Dictionary containing reconstruction outputs (cameras, depths, points, etc.).
        """
        if self.model is None:
            raise RuntimeError("VGGT model not initialized. Call setup() first.")

        timesteps = sorted(frames.keys())
        return self._run_vggt_model(frames, timesteps)

    def _run_vggt_model(self, frames: Dict[int, np.ndarray], timesteps: list) -> Dict[str, Any]:
        """Run actual VGGT model inference.

        Args:
            frames: Dictionary of RGB frames.
            timesteps: Sorted list of timesteps.

        Returns:
            Dictionary containing reconstruction outputs.
        """

        # Preprocess frames for VGGT model
        images_tensor, images_info = preprocess_frame_dict_for_vggt(frames, timesteps, device=self.device)

        # Run VGGT inference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                predictions = self.model(images_tensor)
                predictions = rescale_vggt_prediction(predictions, images_info)

        # Extract predictions
        # VGGT returns: pose_enc, pose_enc_list, depth, depth_conf, world_points, world_points_conf
        depth_maps, confidences = {}, {}
        per_view_cameras, point_clouds = {}, {}

        all_extrinsic, all_intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_info['padded_size'])
        all_intrinsic = rescale_vggt_intrinsic(all_intrinsic, images_info)

        # Convert world-to-camera (VGGT output) to camera-to-world (visualization format)
        # VGGT outputs [R_w2c | t_w2c], we need [R_c2w | t_c2w]
        # Do conversion in torch, then convert to numpy
        R_w2c = all_extrinsic[:, :, :, :3]  # [B, S, 3, 3]
        t_w2c = all_extrinsic[:, :, :, 3]   # [B, S, 3]
        
        # Convert to camera-to-world using torch operations
        R_c2w = R_w2c.transpose(-2, -1)  # [B, S, 3, 3] - transpose rotation
        t_c2w = -torch.matmul(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)  # [B, S, 3]
        
        # Reconstruct extrinsic matrix in c2w format
        all_extrinsic = torch.cat([R_c2w, t_c2w.unsqueeze(-1)], dim=-1)  # [B, S, 3, 4]
        
        # Now convert to numpy
        predictions_depth = predictions["depth"][0].cpu().numpy()
        predictions_depth_conf = predictions["depth_conf"][0].cpu().numpy()
        all_intrinsic = all_intrinsic.cpu().numpy()
        all_extrinsic = all_extrinsic.cpu().numpy()
        
        for idx, timestep in enumerate(timesteps):
            intrinsic = all_intrinsic[0, idx]
            extrinsic_c2w = all_extrinsic[0, idx]
            per_view_cameras[timestep] = CameraParameters(intrinsic=intrinsic, extrinsic=extrinsic_c2w, distortion=None)

            # Extract depth map
            depth_map = predictions_depth[idx]
            depth_maps[timestep] = depth_map * self.depth_scale

            # Compute point cloud from depth map
            # depth_to_pointcloud now accepts camera-to-world extrinsics directly
            points_3d, colors = depth_to_pointcloud(
                depth_maps[timestep], frames[timestep], intrinsic, extrinsic_c2w
            )

            confidence = predictions_depth_conf[idx].flatten()

            point_clouds[timestep] = points_3d
            confidences[timestep] = confidence

        # Save all camera-to-world extrinsics
        all_data = dict(all_data=dict(all_extrinsic=all_extrinsic[0], 
                        all_intrinsic=all_intrinsic[0], 
                        predictions_depth=predictions_depth, 
                        predictions_depth_conf=predictions_depth_conf, 
                        images_tensor=frames))
        # Merge point clouds
        current_data = self._merge_point_clouds(
            point_clouds, confidences, per_view_cameras, depth_maps, timesteps, frames
        )
        current_data.update(all_data)
        return current_data

    def _build_current_scene(
        self, reconstruction_result: Dict[str, Any], timestep: int
    ) -> SceneReconstruction:
        """Build SceneReconstruction for current timestep.

        Args:
            reconstruction_result: Output from VGGT reconstruction.
            timestep: Current frame timestep.
            frame: Current RGB frame.

        Returns:
            SceneReconstruction for the current view.
        """
        point_clouds = reconstruction_result.get("point_clouds", {})
        depth_maps = reconstruction_result.get("depth_maps", {})
        confidences = reconstruction_result.get("confidences", {})
        per_view_cameras = reconstruction_result.get("per_view_cameras", {})
        current_frame = reconstruction_result.get("current_frame")

        points_3d = point_clouds.get(timestep)
        depth_map = depth_maps.get(timestep)
        confidence = confidences.get(timestep)
        camera = per_view_cameras.get(timestep)

        # Extract colors from frame
        colors = current_frame.reshape(-1, 3)[: len(points_3d)]

        return SceneReconstruction(
            points_3d=points_3d,
            colors=colors,
            depth_map=depth_map,
            confidence=confidence,
            camera=camera,
            metadata={
                "timestep": timestep,
                "reconstruction_method": "vggt",
                "checkpoint": self.checkpoint,
            },
        )

    def _merge_point_clouds(
        self,
        point_clouds: Dict[int, np.ndarray],
        confidences: Dict[int, np.ndarray],
        per_view_cameras: Dict[int, CameraParameters],
        depth_maps: Dict[int, np.ndarray],
        timesteps: list,
        frames: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Merge point clouds from multiple views.

        Args:
            point_clouds: Per-view point clouds.
            confidences: Per-view confidence scores.
            per_view_cameras: Per-view camera parameters.
            depth_maps: Per-view depth maps.
            timesteps: List of timesteps.
            frames: Reshaped frames.

        Returns:
            Dictionary with merged reconstruction outputs.
        """
        all_points = [pc for pc in point_clouds.values() if pc is not None]
        all_confidences = [conf for conf in confidences.values() if conf is not None]

        global_points = np.concatenate(all_points, axis=0)
        global_confidences = np.concatenate(all_confidences)

        # Extract colors from frames
        all_colors = []
        for timestep in timesteps:
            frame = frames[timestep]
            num_points_in_frame = len(point_clouds[timestep])
            frame_colors = frame.reshape(-1, 3)[: num_points_in_frame]
            all_colors.append(frame_colors)
        global_colors = np.concatenate(all_colors) if all_colors else None
        # Filter by confidence
        mask = global_confidences >= self.min_confidence
        global_points = global_points[mask]
        global_colors = global_colors[mask] if global_colors is not None else None

        return {
            "per_view_cameras": per_view_cameras,
            "depth_maps": depth_maps,
            "point_clouds": point_clouds,
            "confidences": confidences,
            "global_points": global_points,
            "global_colors": global_colors,
            "total_points": len(global_points) if global_points is not None else 0,
            "current_frame": frames[timesteps[-1]],
        }

    def teardown(self) -> None:
        """Release model resources."""
        LOGGER.info("Tearing down VGGT reconstructor")
        self.model = None
        self.initialized = False


class VGGTOnlineReconstructor(VGGTOfflineReconstructor):
    """VGGT online reconstructor with pose and depth alignment.

    Extends VGGTOfflineReconstructor to add:
    - Incremental processing with sliding window of frames
    - Pose alignment across frames using previous extrinsics
    - Depth scale alignment using historical statistics
    - Point cloud history accumulation

    Args:
        checkpoint: Path or identifier for VGGT model checkpoint.
        min_confidence: Minimum confidence threshold for point filtering (0-1).
        device: Device to use for reconstruction ("cuda" or "cpu").
        max_frames: Maximum number of frames to use for reconstruction (default: 5).
                   Uses current frame and up to (max_frames - 1) previous frames.
        estimate_normals: Whether to estimate surface normals from depth.
        depth_scale: Scale factor for depth values (meters per unit).
    """

    run_deferred = False  # Processes frames incrementally, cannot be deferred

    def __init__(
        self,
        checkpoint: str = "facebook/VGGT-1B",
        min_confidence: float = 0.4,
        device: str | None = None,
        max_frames: int = 5,
        estimate_normals: bool = True,
        depth_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,
            min_confidence=min_confidence,
            device=device,
            max_frames=max_frames,
            estimate_normals=estimate_normals,
            depth_scale=depth_scale,
            **kwargs,
        )

        # Store previous poses for alignment, depth statistics for affine alignment
        self.prev_poses: Dict[int, Dict[str, np.ndarray]] = {}  # timestep -> {'t': translation, 'q': quaternion}
        self.prev_depth_stats: Dict[int, Dict[str, float]] = {}  # timestep -> {'mean': mean, 'std': std}

        # History point cloud accumulation
        self.history_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.history_colors: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.history_confidences: np.ndarray = np.empty((0,), dtype=np.float32)


    def _refine_pose_encoding(
        self,
        pose_encoding: torch.Tensor,
        timesteps: list,
    ) -> torch.Tensor:
        """Refine pose encoding using alignment with the first (oldest) frame.

        Calibrates frame T's pose using the overlap with frame T-N (first in timesteps).

        Args:
            pose_encoding: Pose encoding [batch, num_frames, 9] containing [t_w2c, q_w2c, fov_h, fov_w]
            timesteps: List of timesteps corresponding to frames

        Returns:
            Refined pose encoding with aligned rotation and translation
        """
        if len(self.prev_poses) < 1 or len(timesteps) < 2:
            return pose_encoding

        # Use the first (oldest) frame for calibration
        first_timestep = timesteps[0]
        if first_timestep not in self.prev_poses:
            return pose_encoding

        LOGGER.debug(f"Refining pose using first frame timestep {first_timestep}")

        # Extract current predictions (in w2c format from VGGT)
        T_w2c_curr = pose_encoding[0, :, :3].cpu().numpy()  # [num_frames, 3]
        q_w2c_curr = pose_encoding[0, :, 3:7].cpu().numpy()  # [num_frames, 4]

        # Get previous pose for the first timestep (stored in c2w format)
        q_c2w_prev = self.prev_poses[first_timestep]['q']
        t_c2w_prev = self.prev_poses[first_timestep]['t']

        # Convert previous c2w pose back to w2c for comparison with VGGT output
        rot_c2w_prev = R.from_quat(q_c2w_prev)
        rot_w2c_prev = rot_c2w_prev.inv()
        q_w2c_prev = rot_w2c_prev.as_quat()
        R_w2c_prev = rot_w2c_prev.as_matrix()
        t_w2c_prev = -R_w2c_prev @ t_c2w_prev

        # Current prediction for the first timestep (in w2c format)
        q_w2c_current = q_w2c_curr[0]
        t_w2c_current = T_w2c_curr[0]

        # Compute rotation offset: q_hat = q_w2c_prev * q_w2c_current^(-1)
        q_hat = compute_rotation_offset(q_w2c_prev, q_w2c_current)

        # Compute translation offset: t_offset = t_w2c_prev - R(q_hat) @ t_w2c_current
        R_hat = quaternion_to_rotation_matrix(q_hat)
        t_offset = t_w2c_prev - R_hat @ t_w2c_current

        # Apply alignment to the newest frame (last in timesteps)
        q_newest = q_w2c_curr[-1]
        t_newest = T_w2c_curr[-1]

        # Apply rotation offset: q_aligned = q_hat * q_newest
        rot_offset = R.from_quat(q_hat)
        rot_newest = R.from_quat(q_newest)
        rot_aligned = rot_offset * rot_newest
        q_aligned = rot_aligned.as_quat()

        # Apply translation offset: t_aligned = R(q_hat) @ t_newest + t_offset
        t_aligned = R_hat @ t_newest + t_offset

        # Update pose encoding for newest frame
        refined_pose = pose_encoding.clone()
        refined_pose[0, -1, :3] = torch.from_numpy(t_aligned).to(pose_encoding.device)
        refined_pose[0, -1, 3:7] = torch.from_numpy(q_aligned).to(pose_encoding.device)

        LOGGER.debug(f"Applied pose alignment for timestep {timesteps[-1]} using reference frame {first_timestep}")

        return refined_pose

    def _refine_depth_scale(
        self,
        depth_maps: torch.Tensor,
        timesteps: list,
    ) -> torch.Tensor:
        """Refine depth using affine transformation with the first (oldest) frame.

        Calibrates frame T's depth using affine model: depth_new = a * depth_old + b
        Uses the overlap with frame T-N (first in timesteps).

        Args:
            depth_maps: Depth maps [batch, num_frames, H, W, 1]
            timesteps: List of timesteps corresponding to frames

        Returns:
            Refined depth maps with affine alignment
        """
        if len(self.prev_depth_stats) < 1 or len(timesteps) < 2:
            return depth_maps

        # Use the first (oldest) frame for calibration
        first_timestep = timesteps[0]
        if first_timestep not in self.prev_depth_stats:
            return depth_maps

        LOGGER.debug(f"Refining depth using first frame timestep {first_timestep}")

        # Extract current depth predictions
        depth_maps_np = depth_maps[0].cpu().numpy()  # [num_frames, H, W, 1]

        # Previous depth statistics for the first timestep
        mean_prev = self.prev_depth_stats[first_timestep]['mean']
        std_prev = self.prev_depth_stats[first_timestep]['std']

        # Current prediction depth statistics for the first timestep
        depth_current = depth_maps_np[0, ..., 0]
        mean_current = depth_current.mean()
        std_current = depth_current.std()

        # Compute affine parameters: depth_new = scale * depth_old + offset
        # Using mean and std matching:
        # scale = std_prev / std_current (avoid division by zero)
        # offset = mean_prev - scale * mean_current
        if std_current < 1e-6:  # Avoid division by zero
            return depth_maps

        scale = std_prev / std_current
        offset = mean_prev - scale * mean_current

        LOGGER.debug(
            f"Depth affine: mean_prev={mean_prev:.4f}, std_prev={std_prev:.4f}, "
            f"mean_curr={mean_current:.4f}, std_curr={std_current:.4f}, "
            f"scale={scale:.4f}, offset={offset:.4f}"
        )

        # Apply affine transformation to the newest frame
        refined_depth = depth_maps.clone()
        refined_depth[0, -1] = refined_depth[0, -1] * scale + offset

        LOGGER.debug(f"Applied depth affine (scale={scale:.4f}, offset={offset:.4f}) to timestep {timesteps[-1]}")

        return refined_depth

    def _run_vggt_model(self, frames: Dict[int, np.ndarray], timesteps: list) -> Dict[str, Any]:
        """Run actual VGGT model inference with pose and depth refinement.

        Args:
            frames: Dictionary of RGB frames.
            timesteps: Sorted list of timesteps.

        Returns:
            Dictionary containing reconstruction outputs.
        """

        # Preprocess frames for VGGT model
        images_tensor, images_info = preprocess_frame_dict_for_vggt(
            frames, timesteps, mode="pad", target_size=518, device=self.device
        )

        # Run VGGT inference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                predictions = self.model(images_tensor)
                predictions = rescale_vggt_prediction(predictions, images_info)

        # Extract predictions
        # VGGT returns: pose_enc, pose_enc_list, depth, depth_conf, world_points, world_points_conf
        depth_maps, confidences = {}, {}
        per_view_cameras, point_clouds = {}, {}

        # Refine pose encoding with alignment
        pose_encoding = predictions["pose_enc"]
        refined_pose_encoding = self._refine_pose_encoding(pose_encoding, timesteps)

        # Refine depth maps with scale alignment
        raw_depth = predictions["depth"]
        refined_depth = self._refine_depth_scale(raw_depth, timesteps)

        # Build intrinsics from FOV
        fov_h = refined_pose_encoding[..., 7]
        fov_w = refined_pose_encoding[..., 8]
        all_intrinsic = build_intrinsics_from_fov(fov_h, fov_w, images_info['resized_size'])

        # Only process the last (newest) frame for reconstruction
        current_timestep = timesteps[-1]
        current_idx = len(timesteps) - 1

        # Extract refined pose for current frame (in w2c format from VGGT)
        t_w2c = refined_pose_encoding[0, current_idx, :3].cpu().numpy()
        q_w2c = refined_pose_encoding[0, current_idx, 3:7].cpu().numpy()

        # Convert from w2c to c2w format
        # q_c2w = inverse(q_w2c)
        rot_w2c = R.from_quat(q_w2c)
        rot_c2w = rot_w2c.inv()
        quat = rot_c2w.as_quat()

        # t_c2w = -R_c2w @ t_w2c
        R_c2w = rot_c2w.as_matrix()
        T = -R_c2w @ t_w2c

        # Build extrinsic matrix in c2w format for CameraParameters
        extrinsic = np.zeros((3, 4))
        extrinsic[:, :3] = R_c2w
        extrinsic[:, 3] = T

        intrinsic = all_intrinsic[0, current_idx].cpu().numpy()
        per_view_cameras[current_timestep] = CameraParameters(intrinsic=intrinsic, extrinsic=extrinsic, distortion=None)

        # Extract depth map for current frame (after refinement)
        depth_map = refined_depth[0, current_idx].cpu().numpy()
        depth_maps[current_timestep] = depth_map * self.depth_scale

        # Compute point cloud from depth map
        points_3d, colors = depth_to_pointcloud(
            depth_maps[current_timestep], frames[current_timestep], intrinsic, extrinsic
        )
        confidence = predictions["depth_conf"][0, current_idx].flatten().cpu().numpy()

        point_clouds[current_timestep] = points_3d
        confidences[current_timestep] = confidence

        # Store pose components in c2w format for future alignment
        self.prev_poses[current_timestep] = {'t': T, 'q': quat}

        # Store depth statistics for current frame (for future alignment)
        self.prev_depth_stats[current_timestep] = {
            'mean': float(depth_map.mean()),
            'std': float(depth_map.std())
        }

        # Store all pose components for reference (even though we only reconstruct the last frame)
        for idx, timestep in enumerate(timesteps):
            if timestep != current_timestep:
                # Extract w2c pose from VGGT output
                t_w2c_i = refined_pose_encoding[0, idx, :3].cpu().numpy()
                q_w2c_i = refined_pose_encoding[0, idx, 3:7].cpu().numpy()

                # Convert w2c to c2w for storage
                rot_w2c_i = R.from_quat(q_w2c_i)
                rot_c2w_i = rot_w2c_i.inv()
                q_c2w_i = rot_c2w_i.as_quat()
                t_c2w_i = -rot_c2w_i.as_matrix() @ t_w2c_i

                # self.prev_poses[timestep] = {'t': t_c2w_i, 'q': q_c2w_i}

                # Store depth statistics for this timestep
                depth_i = refined_depth[0, idx, ..., 0].cpu().numpy()
                self.prev_depth_stats[timestep] = {
                    'mean': float(depth_i.mean()),
                    'std': float(depth_i.std())
                }

        # Update history and merge point clouds
        return self._merge_point_clouds(
            point_clouds, confidences, per_view_cameras, depth_maps, timesteps, frames
        )

    def _merge_point_clouds(
        self,
        point_clouds: Dict[int, np.ndarray],
        confidences: Dict[int, np.ndarray],
        per_view_cameras: Dict[int, CameraParameters],
        depth_maps: Dict[int, np.ndarray],
        timesteps: list,
        frames: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        # Get current frame data (only one frame reconstructed)
        current_timestep = timesteps[-1]
        current_points = point_clouds[current_timestep]
        current_confidence = confidences[current_timestep]
        current_frame = frames[current_timestep]
        current_colors = current_frame.reshape(-1, 3)[: len(current_points)]

        # Filter current frame by confidence
        mask = current_confidence >= self.min_confidence
        filtered_points = current_points[mask]
        filtered_colors = current_colors[mask]
        filtered_confidence = current_confidence[mask]

        self.history_points = np.vstack([self.history_points, filtered_points])
        self.history_colors = np.vstack([self.history_colors, filtered_colors])
        self.history_confidences = np.concatenate([self.history_confidences, filtered_confidence])

        LOGGER.debug(f"History point cloud size: {len(self.history_points)} points")
        return {
            "per_view_cameras": per_view_cameras,
            "extrinsics": per_view_cameras[current_timestep].extrinsic,
            "intrinsics": per_view_cameras[current_timestep].intrinsic,
            "depth_maps": depth_maps,
            "point_clouds": point_clouds,
            "confidences": confidences,
            "global_points": self.history_points,
            "global_colors": self.history_colors,
            "total_points": len(self.history_points) if self.history_points is not None else 0,
            "current_frame": frames[timesteps[-1]],
        }
