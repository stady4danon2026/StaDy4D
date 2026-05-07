"""Pi3/Pi3X-based reconstruction implementation."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.reconstruction.aggregator import SceneAggregator, SceneFragment
from sigma.pipeline.reconstruction.base_recon import (
    BaseReconstructor,
    CameraParameters,
    ReconstructionOutputs,
    SceneReconstruction,
    extract_frames_from_records,
)
from sigma.utils.log_messages import log_step, log_success

LOGGER = logging.getLogger(__name__)


def _compute_target_size(H: int, W: int, pixel_limit: int = 255000) -> tuple[int, int]:
    """Compute Pi3-compatible image dimensions (multiples of 14, within pixel_limit).

    Returns:
        (TARGET_W, TARGET_H)
    """
    scale = math.sqrt(pixel_limit / max(W * H, 1))
    W_f, H_f = W * scale, H * scale
    k, m = round(W_f / 14), round(H_f / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_f / H_f:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _resize_map(arr: np.ndarray, H: int, W: int, mode: str = "bilinear") -> np.ndarray:
    """Resize a (H_src, W_src) numpy float array to (H, W)."""
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    kwargs: dict = {"align_corners": False} if mode == "bilinear" else {}
    out = F.interpolate(t, size=(H, W), mode=mode, **kwargs)
    return out.squeeze().numpy()


def _resize_point_map(pts: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Resize a (H_src, W_src, 3) point map tensor to (H, W, 3) numpy array."""
    t = pts.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H_src, W_src)
    out = F.interpolate(t, size=(H, W), mode="nearest")
    return out.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)


def _composite_moving_object(
    scene_frame: np.ndarray,
    origin_frame: np.ndarray | None,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    """Blend the moving object from origin_frame back onto the inpainted scene_frame.

    Args:
        scene_frame: Background-only (inpainted) frame, uint8 (H, W, 3).
        origin_frame: Original frame with moving object, uint8 (H, W, 3).
        mask: Foreground mask (H, W); 1 = moving object, 0 = background.

    Returns:
        Composited frame (H, W, 3) or None when compositing is not possible.
    """
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


class Pi3Reconstructor(BaseReconstructor):
    """Pi3X-based static-scene reconstruction (offline / deferred mode).

    All frames are forwarded to Pi3X in a single batch after inpainting is
    complete, matching the VGGTOfflineReconstructor pattern.

    The reconstructed scene uses **inpainted** frames (moving object removed)
    so that depth and point-cloud outputs describe only the static background
    (Todo 2).  When ``exclude_moving_objects=True`` (default), pixels inside
    the segmentation mask are additionally zeroed in the depth / confidence
    maps to prevent any residual foreground contribution.

    A composited visualisation frame (moving object blended back onto the
    inpainted scene) is attached to each per-frame result as ``composited_frame``
    inside the visualisation assets (Todo 4).

    Args:
        checkpoint: HuggingFace model ID or local path to Pi3X weights.
        min_confidence: Minimum sigmoid-confidence for filtering points (0–1).
        device: Compute device ("cuda" or "cpu").
        max_frames: Maximum number of frames processed in one Pi3X call.
        exclude_moving_objects: Zero depth/confidence inside the per-frame
            segmentation mask (requires ``FrameIORecord.mask`` to be set).
        pixel_limit: Maximum H×W fed to Pi3X (controls internal image resize).
    """

    name = "pi3_reconstructor"
    run_deferred = True  # batch-process all frames in finalize()

    def __init__(
        self,
        checkpoint: str = "yyfz233/Pi3X",
        min_confidence: float = 0.1,
        device: str = "cuda",
        max_frames: int = 50,
        exclude_moving_objects: bool = True,
        pixel_limit: int = 255000,
        use_origin_frames: bool = True,
        calibrate_poses: bool = True,
        pose_window_size: int = 5,
        finalize_batch_size: int | None = None,
        fill_masked_depth: bool = False,
        mask_dilate_px: int = 3,
        fill_confidence: float = 0.5,
        lora_checkpoint: str | None = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        **kwargs: Any,
    ) -> None:
        aggregator = kwargs.get("aggregator") or SceneAggregator()
        super().__init__(aggregator=aggregator)
        self.checkpoint = checkpoint
        self.min_confidence = min_confidence
        self.device = device
        self.max_frames = max_frames
        self.exclude_moving_objects = exclude_moving_objects
        self.pixel_limit = pixel_limit
        self.use_origin_frames = use_origin_frames
        self.calibrate_poses = calibrate_poses
        self.pose_window_size = pose_window_size
        self.finalize_batch_size = finalize_batch_size
        self.fill_masked_depth = fill_masked_depth
        self.mask_dilate_px = mask_dilate_px
        self.fill_confidence = fill_confidence
        self.lora_checkpoint = lora_checkpoint
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.model = None
        self.initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load Pi3X model from HuggingFace or a local checkpoint path.
        Reuses a shared instance if another stage (e.g. motion) already loaded it.
        """
        log_step(LOGGER, "Loading Pi3X model (shared cache)", self.checkpoint)
        from sigma.pipeline._model_registry import get_or_load_pi3

        self.model = get_or_load_pi3(self.checkpoint, self.device)

        if self.lora_checkpoint:
            self._attach_and_load_lora(self.lora_checkpoint)
            log_success(LOGGER, f"Loaded LoRA checkpoint: {self.lora_checkpoint}")

        self.initialized = True
        log_success(LOGGER, "Pi3 reconstructor initialized")

    def _attach_and_load_lora(self, ckpt_path: str) -> None:
        """Inject LoRA adapters and load saved deltas.

        Mirrors the wiring done at training time (see script/train_pi3_lora.py).
        """
        import math
        import torch.nn as nn

        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt.get("config", {"rank": self.lora_rank,
                                    "alpha": self.lora_alpha,
                                    "targets": ["decoder", "point_decoder"],
                                    "last_n": 8})

        class LoRALinear(nn.Module):
            def __init__(self, base: nn.Linear, r: int, a: float):
                super().__init__()
                self.base = base
                for p in self.base.parameters():
                    p.requires_grad_(False)
                self.r = r; self.scale = a / r
                self.lora_A = nn.Linear(base.in_features, r, bias=False)
                self.lora_B = nn.Linear(r, base.out_features, bias=False)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

            def forward(self, x):  # type: ignore[override]
                return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

        # Resolve target names from training cfg
        targets = list(cfg.get("targets") or [])
        last_n = int(cfg.get("last_n", 0))
        if last_n > 0:
            targets = [k for k in targets if k != "decoder"]
            n_blocks = len(self.model.decoder)
            first = max(0, n_blocks - last_n)
            targets += [f"decoder.{i}." for i in range(first, n_blocks)]

        rank = int(cfg.get("rank", self.lora_rank))
        alpha = float(cfg.get("alpha", self.lora_alpha))

        n_attached = 0
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(kw in name for kw in targets):
                continue
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, rank, alpha).to(self.device))
            n_attached += 1

        # Load LoRA-only state dict
        lora_state = ckpt.get("lora_state", ckpt)
        missing, unexpected = self.model.load_state_dict(lora_state, strict=False)
        # Most keys are missing-from-ckpt (the frozen base); that's expected.
        if unexpected:
            LOGGER.warning("Unexpected LoRA keys: %s", unexpected[:5])
        log_step(LOGGER, f"LoRA: attached {n_attached} layers, "
                          f"loaded {sum(1 for k in lora_state if 'lora_' in k)} param tensors")

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Not used — Pi3Reconstructor is deferred; call finalize() instead."""
        raise RuntimeError(
            "Pi3Reconstructor is deferred (run_deferred=True). "
            "The runner should call finalize() after all inpainting is done."
        )

    def finalize(
        self, frame_records: Dict[int, FrameIORecord]
    ) -> List[Tuple[int, StageResult]]:
        """Run Pi3X once on all inpainted frames and return per-frame results.

        Args:
            frame_records: All collected FrameIORecord objects (with inpainted
                images filled in by the inpainting stage).

        Returns:
            Ordered list of (frame_idx, StageResult) tuples — one per frame.
        """
        if not self.initialized:
            raise RuntimeError("Pi3Reconstructor.setup() must be called first.")

        # Prefer inpainted frame; fall back to origin if inpainting was skipped.
        frames_dict = extract_frames_from_records(
            frame_records, max(frame_records), max_frames=self.max_frames
        )
        timesteps = sorted(frames_dict.keys())
        if not timesteps:
            return []

        log_step(LOGGER, f"Running Pi3X offline on {len(timesteps)} frames")

        # Collect masks & originals needed for filtering / compositing.
        masks_dict: Dict[int, np.ndarray] = {}
        origin_dict: Dict[int, np.ndarray] = {}
        for idx, record in frame_records.items():
            if idx in frames_dict:
                if record.mask is not None:
                    masks_dict[idx] = record.mask
                if self.use_origin_frames and record.origin_image is not None:
                    origin_dict[idx] = record.origin_image

        # When use_origin_frames: feed Pi3 the original (unmasked) frames so it can
        # see and reconstruct depth for the dynamic object too.
        # Inpainted frames are still kept in frames_dict for static point-cloud colors.
        if self.use_origin_frames:
            pi3_input_dict: Dict[int, np.ndarray] = {}
            for idx in timesteps:
                record = frame_records.get(idx)
                frame = (
                    record.origin_image
                    if record is not None and record.origin_image is not None
                    else frames_dict.get(idx)
                )
                if frame is not None:
                    pi3_input_dict[idx] = frame
        else:
            pi3_input_dict = frames_dict

        # Route to chunked finalize if batch_size is set and we have enough frames.
        if self.finalize_batch_size and len(timesteps) > self.finalize_batch_size:
            return self._finalize_chunked(
                frame_records, frames_dict, pi3_input_dict,
                masks_dict, origin_dict, timesteps,
            )

        # --- Full-batch path (original) ---
        return self._finalize_full_batch(
            frame_records, frames_dict, pi3_input_dict,
            masks_dict, origin_dict, timesteps,
        )

    def _finalize_full_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        frames_dict: Dict[int, np.ndarray],
        pi3_input_dict: Dict[int, np.ndarray],
        masks_dict: Dict[int, np.ndarray],
        origin_dict: Dict[int, np.ndarray],
        timesteps: list,
    ) -> List[Tuple[int, StageResult]]:
        """Full-batch finalize.

        If ``use_origin_frames=True`` (default for NEW_*), reuse cached Pi3 outputs
        populated by an earlier stage (e.g. Sam3TttUnion). Otherwise the inpainted
        frames differ from origin and we must re-run Pi3 on the inpainted ones.
        """
        from sigma.pipeline._pi3_cache import has_pi3_outputs, ensure_pi3_outputs, stack_pi3_outputs

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        if self.use_origin_frames:
            # Cache is keyed on origin frames — safe to reuse.
            ensure_pi3_outputs(frame_records, self.model, device=self.device,
                               pixel_limit=self.pixel_limit, capture_conf_features=False)
            res = stack_pi3_outputs(frame_records, indices=timesteps)
            H_pi3 = frame_records[timesteps[0]].pi3_input_h
            W_pi3 = frame_records[timesteps[0]].pi3_input_w
            H_orig, W_orig = frame_records[timesteps[0]].origin_image.shape[:2]
        else:
            # Inpainted-frame path: run Pi3 here on the inpainted images.
            imgs_tensor, resize_info = self._preprocess_frames(pi3_input_dict, timesteps)
            H_pi3, W_pi3 = resize_info["pi3_size"]
            H_orig, W_orig = resize_info["orig_size"]
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = self.model(imgs_tensor)

        # Sliding-window pose calibration.
        calibrated_poses: Dict[int, np.ndarray] = {}
        if self.calibrate_poses and len(timesteps) > self.pose_window_size:
            calibrated_poses = self._calibrate_poses(pi3_input_dict, timesteps, dtype)

        results = self._build_all_frame_results(
            frame_records, res, frames_dict, masks_dict, origin_dict,
            timesteps, H_pi3, W_pi3, H_orig, W_orig, calibrated_poses,
        )
        self._fill_masked_depth_from_global(frame_records, results, masks_dict)
        return results

    def _finalize_chunked(
        self,
        frame_records: Dict[int, FrameIORecord],
        frames_dict: Dict[int, np.ndarray],
        pi3_input_dict: Dict[int, np.ndarray],
        masks_dict: Dict[int, np.ndarray],
        origin_dict: Dict[int, np.ndarray],
        timesteps: list,
    ) -> List[Tuple[int, StageResult]]:
        """Chunked finalize: process frames in overlapping windows to avoid OOM.

        Splits the sequence into chunks of ``finalize_batch_size`` frames with
        50% overlap.  Each chunk is run through Pi3X independently, and the
        per-chunk results are stitched via Umeyama Sim(3) pose alignment on
        overlapping frames (same algorithm as ``_calibrate_poses``).
        """
        B = self.finalize_batch_size
        N = len(timesteps)
        step = max(1, B // 2)  # 50% overlap

        log_step(LOGGER, f"Chunked finalize: {N} frames, chunk_size={B}, step={step}")

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # ---- Phase 1: Run Pi3X on each chunk, collect raw poses + per-frame data ----
        chunk_records: list[tuple[list, dict]] = []  # (chunk_timesteps, pi3x_result)
        chunk_poses: list[dict[int, np.ndarray]] = []  # local poses per chunk
        chunk_resize_info: list[dict] = []

        starts = list(range(0, N, step))
        # Ensure we cover the last frames.
        if starts[-1] + B < N:
            starts.append(N - B)

        for start in starts:
            end = min(start + B, N)
            chunk_ts = timesteps[start:end]
            chunk_input = {t: pi3_input_dict[t] for t in chunk_ts}

            imgs_tensor, resize_info = self._preprocess_frames(chunk_input, chunk_ts)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = self.model(imgs_tensor)

            # Extract local poses for Sim(3) alignment.
            local_poses: dict[int, np.ndarray] = {}
            for local_i, t in enumerate(chunk_ts):
                cam_pose = res["camera_poses"][0, local_i]
                local_poses[t] = cam_pose.cpu().float().numpy()

            chunk_records.append((chunk_ts, res))
            chunk_poses.append(local_poses)
            chunk_resize_info.append(resize_info)

            LOGGER.debug(
                "Chunk [%d:%d] processed (%d frames)", start, end, len(chunk_ts)
            )

        # ---- Phase 2: Chain chunk poses via Umeyama Sim(3) ----
        global_poses: dict[int, np.ndarray] = {}

        for chunk_idx, (chunk_ts, local_poses) in enumerate(
            zip([cr[0] for cr in chunk_records], chunk_poses)
        ):
            if not global_poses:
                global_poses.update(local_poses)
                continue

            overlap_ts = [t for t in chunk_ts if t in global_poses]

            if len(overlap_ts) < 2:
                if overlap_ts:
                    t0 = overlap_ts[0]
                    delta = global_poses[t0][:3, 3] - local_poses[t0][:3, 3]
                    for t in chunk_ts:
                        if t not in global_poses:
                            p = local_poses[t].copy()
                            p[:3, 3] += delta
                            global_poses[t] = p
                continue

            src_pts = np.stack([local_poses[t][:3, 3] for t in overlap_ts])
            tgt_pts = np.stack([global_poses[t][:3, 3] for t in overlap_ts])
            scale, R_align, t_align = self._umeyama_sim3(src_pts, tgt_pts)

            for t in chunk_ts:
                if t not in global_poses:
                    p = local_poses[t].copy()
                    p[:3, 3] = scale * (R_align @ p[:3, 3]) + t_align
                    global_poses[t] = p

        # ---- Phase 3: Build per-frame results using each chunk's Pi3X output ----
        # Assign each timestep to the chunk that contains it (prefer the chunk where
        # it is closest to the center for best depth quality).
        ts_to_chunk: dict[int, int] = {}
        for chunk_idx, (chunk_ts, _) in enumerate(chunk_records):
            for t in chunk_ts:
                if t not in ts_to_chunk:
                    ts_to_chunk[t] = chunk_idx

        results: List[Tuple[int, StageResult]] = []
        for frame_idx in timesteps:
            chunk_idx = ts_to_chunk[frame_idx]
            chunk_ts, res = chunk_records[chunk_idx]
            resize_info = chunk_resize_info[chunk_idx]
            H_pi3, W_pi3 = resize_info["pi3_size"]
            H_orig, W_orig = resize_info["orig_size"]
            local_i = chunk_ts.index(frame_idx)

            current_scene, recon_outputs, composited, dynamic_depth = self._build_frame_result(
                local_i, frame_idx, res,
                frames_dict, masks_dict, origin_dict,
                H_pi3, W_pi3, H_orig, W_orig,
                pose_override=global_poses.get(frame_idx),
            )

            record = frame_records.get(frame_idx)
            if record is not None:
                record.depth = current_scene.depth_map
                if current_scene.camera is not None:
                    record.extrinsic = current_scene.camera.extrinsic
                    record.intrinsic = current_scene.camera.intrinsic

            if current_scene.points_3d is not None and len(current_scene.points_3d) > 0:
                self.aggregator.add_fragment(SceneFragment(
                    points=current_scene.points_3d,
                    confidence=current_scene.confidence,
                    metadata={
                        "timestep": frame_idx,
                        "colors": current_scene.colors,
                        "camera": current_scene.camera,
                    },
                ))

            # Skip per-frame aggregator.build_scene() (was O(N²)) — only the
            # last frame's aggregated_scene is consumed downstream, and it is
            # populated after the loop completes.
            recon_outputs.aggregated_scene = None

            vis_assets = asdict(recon_outputs)
            if composited is not None:
                vis_assets["composited_frame"] = composited
            if dynamic_depth is not None:
                vis_assets["dynamic_depth"] = dynamic_depth

            results.append((
                frame_idx,
                StageResult(
                    data={
                        "reconstruction": recon_outputs,
                        "frame_idx": frame_idx,
                        "frame_size": (H_orig, W_orig),
                    },
                    visualization_assets=vis_assets,
                ),
            ))

        # Multi-view depth fill (uses the fully-aggregated cloud).
        self._fill_masked_depth_from_global(frame_records, results, masks_dict)

        if results:
            final_scene = self.aggregator.build_scene()
            _, last_result = results[-1]
            last_result.data["reconstruction"].aggregated_scene = final_scene
            last_result.visualization_assets["global_point_cloud"] = final_scene.get("merged_points")
            last_result.visualization_assets["global_point_color"] = final_scene.get("merged_colors")

        log_step(LOGGER, f"Chunked finalize complete: {len(results)} frames")
        return results

    def _build_all_frame_results(
        self,
        frame_records: Dict[int, FrameIORecord],
        res: Dict[str, Any],
        frames_dict: Dict[int, np.ndarray],
        masks_dict: Dict[int, np.ndarray],
        origin_dict: Dict[int, np.ndarray],
        timesteps: list,
        H_pi3: int, W_pi3: int, H_orig: int, W_orig: int,
        calibrated_poses: Dict[int, np.ndarray],
    ) -> List[Tuple[int, StageResult]]:
        """Build per-frame StageResults from a single Pi3X run (shared by full-batch path)."""
        results: List[Tuple[int, StageResult]] = []
        for i, frame_idx in enumerate(timesteps):
            current_scene, recon_outputs, composited, dynamic_depth = self._build_frame_result(
                i, frame_idx, res,
                frames_dict, masks_dict, origin_dict,
                H_pi3, W_pi3, H_orig, W_orig,
                pose_override=calibrated_poses.get(frame_idx),
            )

            # Propagate depth / camera parameters back into the frame record.
            record = frame_records.get(frame_idx)
            if record is not None:
                record.depth = current_scene.depth_map
                if current_scene.camera is not None:
                    record.extrinsic = current_scene.camera.extrinsic
                    record.intrinsic = current_scene.camera.intrinsic

            # Add filtered points to the scene aggregator.
            if current_scene.points_3d is not None and len(current_scene.points_3d) > 0:
                self.aggregator.add_fragment(SceneFragment(
                    points=current_scene.points_3d,
                    confidence=current_scene.confidence,
                    metadata={
                        "timestep": frame_idx,
                        "colors": current_scene.colors,
                        "camera": current_scene.camera,
                    },
                ))

            # Skip per-frame aggregator.build_scene() (was O(N²)) — only the
            # last frame's aggregated_scene is consumed downstream, and it is
            # populated after the loop completes.
            recon_outputs.aggregated_scene = None

            vis_assets = asdict(recon_outputs)
            if composited is not None:
                vis_assets["composited_frame"] = composited
            if dynamic_depth is not None:
                vis_assets["dynamic_depth"] = dynamic_depth

            results.append((
                frame_idx,
                StageResult(
                    data={
                        "reconstruction": recon_outputs,
                        "frame_idx": frame_idx,
                        "frame_size": (H_orig, W_orig),
                    },
                    visualization_assets=vis_assets,
                ),
            ))

        # Populate global_point_cloud in the very last result so the pipeline
        # runner can pass it to save_reconstruction_summary() for PLY/GIF export.
        if results:
            final_scene = self.aggregator.build_scene()
            _, last_result = results[-1]
            last_result.data["reconstruction"].aggregated_scene = final_scene
            last_result.visualization_assets["global_point_cloud"] = final_scene.get("merged_points")
            last_result.visualization_assets["global_point_color"] = final_scene.get("merged_colors")

        return results

    # ------------------------------------------------------------------
    # Multi-view depth fill (replaces RGB inpainting)
    # ------------------------------------------------------------------

    def _fill_masked_depth_from_global(
        self,
        frame_records: Dict[int, FrameIORecord],
        results: List[Tuple[int, StageResult]],
        masks_dict: Dict[int, np.ndarray],
    ) -> None:
        """Fill per-frame masked-depth holes by back-projecting the aggregated
        global static cloud through each frame's camera.

        Per frame: take the merged global cloud (built from all frames'
        non-mask static points), transform into camera frame using
        ``inv(c2w)``, project with ``K``, z-buffer to ``(H, W)``, and write
        the resulting depth into pixels where ``mask>0`` and the projection
        landed a point.  Updates ``recon_outputs.depth_map``,
        ``confidence_map``, ``visualization_assets["depth_map"]`` and the
        backing ``FrameIORecord.depth`` so downstream consumers see the fill.
        """
        if not self.fill_masked_depth or self.aggregator is None:
            return
        scene = self.aggregator.build_scene()
        global_pts = scene.get("merged_points")
        if global_pts is None or len(global_pts) == 0:
            return

        import cv2

        # Move global cloud to GPU once. Per-frame projection is a single
        # batched matmul + scatter_reduce (z-buffer); this avoids the
        # O(N×F) numpy matmul that was the pipeline bottleneck.
        device = self.device if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        pts = torch.as_tensor(np.asarray(global_pts, dtype=np.float32), device=device)
        N = pts.shape[0]
        ones = torch.ones((N, 1), device=device, dtype=torch.float32)
        pts_h = torch.cat([pts, ones], dim=1)              # (N, 4)
        pts_h_T = pts_h.T.contiguous()                      # (4, N)

        n_filled_total = 0
        for frame_idx, result in results:
            recon = result.data.get("reconstruction") if result.data else None
            if recon is None or recon.depth_map is None:
                continue
            mask = masks_dict.get(frame_idx)
            if mask is None:
                continue

            H, W = recon.depth_map.shape[:2]

            # 3x4 c2w → 4x4 on GPU.
            c2w = torch.eye(4, device=device, dtype=torch.float32)
            c2w[:3, :] = torch.as_tensor(np.asarray(recon.extrinsic, dtype=np.float32),
                                          device=device)
            try:
                w2c = torch.linalg.inv(c2w)
            except RuntimeError:
                continue

            pts_cam = (w2c @ pts_h_T).T[:, :3]              # (N, 3)
            z = pts_cam[:, 2]
            front = z > 1e-4
            if not front.any():
                continue
            pts_cam = pts_cam[front]
            z = pts_cam[:, 2]

            K = torch.as_tensor(np.asarray(recon.intrinsic, dtype=np.float32),
                                 device=device)
            u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
            u_int = torch.round(u).long()
            v_int = torch.round(v).long()
            in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
            if not in_bounds.any():
                continue
            u_int = u_int[in_bounds]
            v_int = v_int[in_bounds]
            z_in = z[in_bounds].contiguous()

            # Z-buffer via scatter_reduce_('amin') on a flat (H*W,) tensor.
            INF = torch.tensor(float("inf"), device=device, dtype=torch.float32)
            depth_buf = INF.expand(H * W).clone()
            flat_idx = v_int * W + u_int
            depth_buf.scatter_reduce_(0, flat_idx, z_in, reduce="amin", include_self=True)
            depth_buf = depth_buf.view(H, W)
            depth_buf = torch.where(torch.isfinite(depth_buf), depth_buf,
                                     torch.zeros_like(depth_buf))
            depth_buf_np = depth_buf.cpu().numpy()

            moving = (mask > 0).astype(np.uint8)
            if moving.shape != (H, W):
                moving = cv2.resize(moving, (W, H), interpolation=cv2.INTER_NEAREST)
            if self.mask_dilate_px > 0:
                k = self.mask_dilate_px
                kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
                moving = cv2.dilate(moving, kernel)

            fill = (moving > 0) & (depth_buf_np > 0)
            n_fill = int(fill.sum())
            if n_fill == 0:
                continue
            n_filled_total += n_fill

            depth_f32 = recon.depth_map.astype(np.float32, copy=True)
            depth_f32[fill] = depth_buf_np[fill]
            recon.depth_map = depth_f32.astype(recon.depth_map.dtype)
            result.visualization_assets["depth_map"] = recon.depth_map

            if recon.confidence_map is not None:
                conf_f32 = np.asarray(recon.confidence_map, dtype=np.float32).copy()
                if conf_f32.shape == fill.shape:
                    conf_f32[fill] = max(self.min_confidence, self.fill_confidence)
                    recon.confidence_map = conf_f32
                    result.visualization_assets["confidence_map"] = conf_f32

            rec = frame_records.get(frame_idx)
            if rec is not None:
                rec.depth = depth_f32

        if n_filled_total > 0:
            log_step(
                LOGGER,
                f"Multi-view depth fill: {n_filled_total} pixels filled across "
                f"{len(results)} frames",
            )

    def teardown(self) -> None:
        """Release model resources."""
        LOGGER.info("Tearing down Pi3 reconstructor")
        self.model = None
        self.initialized = False

    # ------------------------------------------------------------------
    # Sliding-window pose calibration + per-frame result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _umeyama_sim3(
        src: np.ndarray,  # (N, 3) positions in source frame
        tgt: np.ndarray,  # (N, 3) corresponding positions in target frame
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Umeyama (1991) Sim(3) alignment.

        Finds scale *s*, rotation *R* (3×3) and translation *t* (3,) such that
        ``s * R @ src[i] + t ≈ tgt[i]`` in the least-squares sense.

        Returns:
            (scale, R_3x3, t_3)
        """
        n = len(src)
        src_mean = src.mean(0)
        tgt_mean = tgt.mean(0)
        src_c = src - src_mean
        tgt_c = tgt - tgt_mean

        src_var = float((src_c ** 2).sum()) / n
        if src_var < 1e-10:
            # Degenerate (all points coincide) — fall back to pure translation.
            return 1.0, np.eye(3), tgt_mean - src_mean

        H = (tgt_c.T @ src_c) / n          # (3, 3)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0, 1.0, d])
        R = U @ D @ Vt
        scale = float((S * D.diagonal()).sum() / src_var)
        t = tgt_mean - scale * (R @ src_mean)
        return scale, R, t

    def _calibrate_poses(
        self,
        pi3_input_dict: Dict[int, np.ndarray],
        timesteps: list,
        dtype: torch.dtype,
    ) -> Dict[int, np.ndarray]:
        """Estimate camera poses using overlapping sliding windows + Sim(3) chaining.

        Pi3X run on all N frames at once normalises the whole scene to a unit
        cube, compressing the camera translation.  Running it on small windows
        of ``pose_window_size`` frames each gives more accurate *local* relative
        poses; chaining consecutive windows via Umeyama alignment on their
        overlapping frames recovers the correct global trajectory scale.

        Strategy
        --------
        - Window stride  = ``pose_window_size // 2``  (50 % overlap).
        - Depth maps are **not** re-estimated here; they come from the full-batch
          run in ``finalize()`` so depth quality is unaffected.
        - Only camera translations are used for the Umeyama alignment; the
          rotation part of the transform is also applied to keep orientations
          consistent.

        Returns:
            Dict mapping ``frame_idx`` → calibrated 4×4 c2w numpy array.
        """
        W = self.pose_window_size
        step = max(1, W // 2)           # 50 % overlap
        N = len(timesteps)

        if N <= W:
            # All frames fit in a single window — no chaining needed.
            LOGGER.debug("calibrate_poses: N=%d ≤ window=%d, skipping calibration", N, W)
            return {}

        log_step(LOGGER, f"Calibrating poses: {N} frames, window={W}, step={step}")

        # ------------------------------------------------------------------
        # Phase 1: run Pi3X on each window, collect local poses.
        # ------------------------------------------------------------------
        window_records: list[tuple[list, dict[int, np.ndarray]]] = []

        starts = list(range(0, N, step))
        # Always include a window ending at the last frame.
        if starts[-1] + W < N:
            starts.append(N - W)

        for start in starts:
            end = min(start + W, N)
            win_ts = timesteps[start:end]
            win_input = {t: pi3_input_dict[t] for t in win_ts}

            imgs_tensor, _ = self._preprocess_frames(win_input, win_ts)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    win_res = self.model(imgs_tensor)

            win_poses: dict[int, np.ndarray] = {}
            for local_i, t in enumerate(win_ts):
                cam_pose = win_res["camera_poses"][0, local_i]   # (4, 4) c2w tensor
                win_poses[t] = cam_pose.cpu().float().numpy()    # → numpy (4, 4)

            window_records.append((win_ts, win_poses))

        # ------------------------------------------------------------------
        # Phase 2: chain windows via Umeyama Sim(3) on overlapping frames.
        # ------------------------------------------------------------------
        global_poses: dict[int, np.ndarray] = {}

        for win_ts, win_poses in window_records:
            if not global_poses:
                global_poses.update(win_poses)
                continue

            overlap_ts = [t for t in win_ts if t in global_poses]

            if len(overlap_ts) < 2:
                # Too little overlap — fall back to translation-only shift.
                if overlap_ts:
                    t0 = overlap_ts[0]
                    delta = global_poses[t0][:3, 3] - win_poses[t0][:3, 3]
                    for t in win_ts:
                        if t not in global_poses:
                            p = win_poses[t].copy()
                            p[:3, 3] += delta
                            global_poses[t] = p
                continue

            src_pts = np.stack([win_poses[t][:3, 3] for t in overlap_ts])
            tgt_pts = np.stack([global_poses[t][:3, 3] for t in overlap_ts])

            scale, R_align, t_align = self._umeyama_sim3(src_pts, tgt_pts)

            for t in win_ts:
                if t not in global_poses:
                    p = win_poses[t].copy()
                    # Apply Sim(3) scale+translation to position only.
                    # R_align is NOT applied to orientation: for a forward-moving
                    # camera, overlap points are nearly collinear so the SVD
                    # rotation is degenerate (free to spin around the motion axis),
                    # causing CW/CCW oscillation.  Pi3X's relative orientations
                    # are already accurate; only the translation magnitude needs
                    # scale correction.
                    p[:3, 3] = scale * (R_align @ p[:3, 3]) + t_align
                    global_poses[t] = p

        log_step(LOGGER, f"Pose calibration complete: {len(global_poses)} frames calibrated")
        return global_poses

    def _build_frame_result(
        self,
        local_idx: int,
        frame_idx: int,
        res: Dict[str, Any],
        frames_dict: Dict[int, np.ndarray],
        masks_dict: Dict[int, np.ndarray],
        origin_dict: Dict[int, np.ndarray],
        H_pi3: int,
        W_pi3: int,
        H_orig: int,
        W_orig: int,
        pose_override: "np.ndarray | None" = None,
        local_scale: float = 1.0,
    ) -> Tuple[SceneReconstruction, ReconstructionOutputs, "np.ndarray | None", "np.ndarray | None"]:
        """Extract per-frame SceneReconstruction and ReconstructionOutputs from Pi3X output.

        Args:
            pose_override: Optional calibrated 4×4 c2w matrix (numpy float32).
                When provided it replaces Pi3X's own camera pose and the
                world-space point cloud is recomputed from camera-frame
                ``local_points`` transformed by this pose, so depth and
                point geometry stay consistent with the calibrated trajectory.
            local_scale: Scale factor (Sim(3)) to apply to ``local_points``
                before transforming to world.  Online mode runs Pi3X on a
                different sliding window per frame, each normalised to its
                own unit cube — so depth values across calls have
                inconsistent scale.  Passing the per-window Sim(3) scale
                from incremental pose chaining brings depths to a global
                metric frame.  Defaults to 1.0 (no rescale).

        Returns:
            (current_scene, recon_outputs, composited_frame, dynamic_depth)
            dynamic_depth is a (H, W) float32 array with depth only in the car region
            (background zeroed), or None when no mask is available.
        """
        # Pi3X outputs for this frame (Pi3 internal resolution).
        local_pts = res["local_points"][0, local_idx]   # (H_pi3, W_pi3, 3) cam-frame pts
        global_pts = res["points"][0, local_idx]         # (H_pi3, W_pi3, 3) world pts
        conf_logits = res["conf"][0, local_idx, :, :, 0] # (H_pi3, W_pi3) raw logits
        camera_pose = res["camera_poses"][0, local_idx]  # (4, 4) c2w matrix

        # ---- Confidence: sigmoid + depth-edge filter ----
        conf_score = torch.sigmoid(conf_logits)
        try:
            from pi3.utils.geometry import depth_edge
            non_edge = ~depth_edge(local_pts[..., 2].unsqueeze(0), rtol=0.03).squeeze(0)
            conf_valid_pi3 = (conf_score >= self.min_confidence) & non_edge
        except Exception:
            conf_valid_pi3 = conf_score >= self.min_confidence

        # ---- Depth: z-component of local_points (metric), resize to orig ----
        depth_pi3 = local_pts[..., 2].cpu().float().numpy()   # (H_pi3, W_pi3)
        if local_scale != 1.0:
            depth_pi3 = depth_pi3 * float(local_scale)
        depth_orig = _resize_map(depth_pi3, H_orig, W_orig, mode="bilinear")

        # ---- Confidence map at original resolution ----
        conf_float = conf_score.cpu().float().numpy()
        conf_orig = _resize_map(conf_float, H_orig, W_orig, mode="bilinear")

        # ---- Apply moving-object mask ----
        mask_frame = masks_dict.get(frame_idx)
        dynamic_depth: np.ndarray | None = None
        if mask_frame is not None:
            import cv2
            moving = (mask_frame > 0).astype(np.uint8)
            if moving.shape != depth_orig.shape:
                moving = cv2.resize(moving, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            moving_bool = moving.astype(bool)
            # Dynamic depth: car region only (captured before static zeroing).
            dynamic_depth = np.where(moving_bool, depth_orig, 0.0).astype(np.float32)
            if self.exclude_moving_objects:
                # Static depth: background only (Todo 2 — zero car region).
                depth_orig = np.where(moving_bool, 0.0, depth_orig)
                conf_orig = np.where(moving_bool, 0.0, conf_orig)

        # ---- Intrinsics derived from Pi3X predicted local_points ----
        intrinsic_pi3 = self._derive_intrinsics(local_pts.cpu(), H_pi3, W_pi3)
        # Scale to original image resolution.
        intrinsic_orig = intrinsic_pi3.copy()
        intrinsic_orig[0, 0] *= W_orig / W_pi3   # fx
        intrinsic_orig[0, 2] *= W_orig / W_pi3   # cx
        intrinsic_orig[1, 1] *= H_orig / H_pi3   # fy
        intrinsic_orig[1, 2] *= H_orig / H_pi3   # cy

        # ---- Extrinsic: use calibrated pose when provided ----
        if pose_override is not None:
            extrinsic = pose_override[:3, :].astype(np.float64)   # (3, 4)
        else:
            extrinsic = camera_pose[:3, :].cpu().float().numpy()   # (3, 4)

        # ---- World-space point cloud resampled to original resolution ----
        if pose_override is not None:
            # Recompute world pts from camera-frame local_points + calibrated pose
            # so the point cloud is consistent with the corrected trajectory.
            # When local_scale != 1.0 (online), also rescale local_pts so depth
            # values match the global metric frame implied by pose_override.
            local_pts_np = local_pts.cpu().float().numpy()         # (H_pi3, W_pi3, 3)
            if local_scale != 1.0:
                local_pts_np = local_pts_np * float(local_scale)
            local_resized = _resize_point_map(
                torch.from_numpy(local_pts_np), H_orig, W_orig
            )
            pts_cam = local_resized.reshape(-1, 3)                 # (H*W, 3) cam frame
            R_c2w = pose_override[:3, :3].astype(np.float32)
            t_c2w = pose_override[:3, 3].astype(np.float32)
            pts_flat = (R_c2w @ pts_cam.T).T + t_c2w              # (H*W, 3) world frame
        else:
            global_resized = _resize_point_map(global_pts.cpu().float(), H_orig, W_orig)
            pts_flat = global_resized.reshape(-1, 3)
        inpainted_frame = frames_dict[frame_idx]          # (H_orig, W_orig, 3)
        colors_flat = inpainted_frame.reshape(-1, 3)

        conf_mask = conf_orig.flatten() >= self.min_confidence
        pts_filtered = pts_flat[conf_mask]
        colors_filtered = colors_flat[conf_mask]
        conf_filtered = conf_orig.flatten()[conf_mask]

        camera = CameraParameters(intrinsic=intrinsic_orig, extrinsic=extrinsic, distortion=None)

        current_scene = SceneReconstruction(
            points_3d=pts_filtered,
            colors=colors_filtered,
            depth_map=depth_orig.astype(np.float32),
            confidence=conf_filtered,
            camera=camera,
            metadata={
                "timestep": frame_idx,
                "reconstruction_method": "pi3x",
                "checkpoint": self.checkpoint,
            },
        )

        # ---- Composited frame: moving object blended back (Todo 4) ----
        origin_frame = origin_dict.get(frame_idx)
        composited = _composite_moving_object(inpainted_frame, origin_frame, mask_frame)

        recon_outputs = ReconstructionOutputs(
            frame_idx=frame_idx,
            depth_map=depth_orig.astype(np.float16),
            current_scene=current_scene,
            aggregated_scene=None,   # filled after aggregation in finalize()
            per_view_cameras={frame_idx: camera},
            confidence_map=conf_orig,
            # current_frame is the composited view (scene + moving object) for visualization.
            # Depth/confidence still reflect the background-only reconstruction.
            current_frame=composited if composited is not None else inpainted_frame,
            point_cloud=pts_filtered,
            point_color=colors_filtered,
            global_point_cloud=None,
            global_point_color=None,
            extrinsic=extrinsic,
            intrinsic=intrinsic_orig,
            reconstruction_metadata={
                "reconstruction_method": "pi3x",
                "checkpoint": self.checkpoint,
                "min_confidence": self.min_confidence,
                "num_views": 1,
                "current_timestep": frame_idx,
            },
        )

        return current_scene, recon_outputs, composited, dynamic_depth

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_frames(
        self,
        frames_dict: Dict[int, np.ndarray],
        timesteps: list,
    ) -> Tuple[torch.Tensor, Dict]:
        """Resize frames to a Pi3-compatible size and stack as (1, N, 3, H, W).

        Pi3 requires images in [0, 1] with H and W both divisible by 14.
        """
        from PIL import Image
        from torchvision import transforms

        H_orig, W_orig = frames_dict[timesteps[0]].shape[:2]
        TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, self.pixel_limit)

        to_tensor = transforms.ToTensor()
        tensor_list = []
        for t in timesteps:
            frame = frames_dict[t]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil = Image.fromarray(frame).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            tensor_list.append(to_tensor(pil))   # (3, H, W) in [0, 1]

        imgs = torch.stack(tensor_list, dim=0).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        LOGGER.debug(
            "Pi3X input: %d frames at %dx%d (original %dx%d)",
            len(timesteps), TARGET_H, TARGET_W, H_orig, W_orig,
        )
        return imgs, {
            "orig_size": (H_orig, W_orig),
            "pi3_size": (TARGET_H, TARGET_W),
            "timesteps": timesteps,
        }

    @staticmethod
    def _derive_intrinsics(
        local_pts: torch.Tensor, H: int, W: int
    ) -> np.ndarray:
        """Estimate camera intrinsics from Pi3X predicted local_points.

        Pi3X stores local_points[h, w] = [x, y, z] in camera frame where::

            x = (u - cx) * z / fx  →  x/z = (u - cx) / fx
            y = (v - cy) * z / fy  →  y/z = (v - cy) / fy

        We assume cx = W/2, cy = H/2 (standard pinhole), then recover
        fx and fy via median over non-central pixels with positive depth.
        """
        cx, cy = W / 2.0, H / 2.0
        z = local_pts[:, :, 2]
        x = local_pts[:, :, 0]
        y = local_pts[:, :, 1]

        u = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
        v = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)

        valid_z = z > 0.01
        mask_fx = valid_z & ((u - cx).abs() > W * 0.1)
        mask_fy = valid_z & ((v - cy).abs() > H * 0.1)

        default_fx = W / (2 * math.tan(math.radians(60)))
        default_fy = H / (2 * math.tan(math.radians(60)))
        try:
            fx = (
                float(torch.median((u[mask_fx] - cx) / (x[mask_fx] / z[mask_fx])).item())
                if mask_fx.sum() > 10
                else default_fx
            )
            fy = (
                float(torch.median((v[mask_fy] - cy) / (y[mask_fy] / z[mask_fy])).item())
                if mask_fy.sum() > 10
                else default_fy
            )
        except Exception:
            fx, fy = default_fx, default_fy

        # Clamp to physically plausible range (roughly FOV 12°–140°).
        fx = float(np.clip(fx, W * 0.2, W * 5.0))
        fy = float(np.clip(fy, H * 0.2, H * 5.0))
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


class Pi3OnlineReconstructor(Pi3Reconstructor):
    """Pi3X-based reconstruction with a sliding-window online mode.

    Processes frames incrementally as they arrive (``run_deferred = False``).
    Each call to ``process()`` runs Pi3X on a window of the most recent
    ``max_frames`` frames, extracts the result for the *current* (newest) frame,
    and accumulates an ever-growing global point cloud in the scene aggregator —
    mirroring the behaviour of :class:`VGGTOnlineReconstructor`.

    **Pose calibration** is performed incrementally via Umeyama Sim(3) chaining:
    each window overlaps ``max_frames - 1`` frames with the previous call, giving
    a well-constrained alignment that recovers the correct forward-motion scale.
    Pi3X normalises each window independently, which compresses the translation;
    the alignment corrects this by mapping the window's local frame onto the
    accumulated global frame.

    Args:
        checkpoint: HuggingFace model ID or local path to Pi3X weights.
        min_confidence: Minimum sigmoid-confidence for filtering points (0–1).
        device: Compute device ("cuda" or "cpu").
        max_frames: Sliding-window size (number of frames fed to Pi3X per call).
        exclude_moving_objects: Zero depth/confidence inside the per-frame mask.
        pixel_limit: Maximum H×W fed to Pi3X (controls internal image resize).
        use_origin_frames: Feed Pi3X original frames (with moving object) so it
            can produce depth for the dynamic region too.
    """

    name = "pi3_online_reconstructor"
    run_deferred = False  # per-frame incremental processing

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Accumulated calibrated global poses: frame_idx → 4×4 c2w (numpy).
        # Reset at each setup() / teardown() cycle via setup().
        self._global_poses: Dict[int, np.ndarray] = {}

    def setup(self) -> None:
        super().setup()
        self._global_poses = {}   # clear between runs
        # When the first full window arrives we replay results for the
        # warmup frames using that call's multi-view Pi3X output, so
        # their depth + pose are in the same coord frame as everything
        # after.  eval_batch.post_process_fill picks this up and rewrites
        # the corresponding entries in online_results before save.
        self._pending_warmup: Dict[int, StageResult] = {}

    def process(
        self, frame_records: Dict[int, FrameIORecord], frame_idx: int
    ) -> StageResult:
        """Run Pi3X on a sliding window ending at *frame_idx* and return the
        current frame's reconstruction result.

        Args:
            frame_records: All accumulated :class:`FrameIORecord` objects so far.
            frame_idx: Index of the frame to reconstruct (newest in the window).

        Returns:
            :class:`StageResult` for the current frame.
        """
        if not self.initialized:
            raise RuntimeError("Pi3OnlineReconstructor.setup() must be called first.")

        # Build sliding-window dict of inpainted frames (up to max_frames).
        frames_dict = extract_frames_from_records(
            frame_records, frame_idx, max_frames=self.max_frames
        )
        timesteps = sorted(frames_dict.keys())
        if not timesteps:
            raise RuntimeError(f"No frames available for frame_idx={frame_idx}.")

        # Collect mask / origin only for frames in the current window.
        masks_dict: Dict[int, np.ndarray] = {}
        origin_dict: Dict[int, np.ndarray] = {}
        for idx in timesteps:
            record = frame_records.get(idx)
            if record is not None:
                if record.mask is not None:
                    masks_dict[idx] = record.mask
                if self.use_origin_frames and record.origin_image is not None:
                    origin_dict[idx] = record.origin_image

        # Optionally feed Pi3X the original (unmasked) frames for full depth coverage.
        if self.use_origin_frames:
            pi3_input_dict: Dict[int, np.ndarray] = {}
            for idx in timesteps:
                record = frame_records.get(idx)
                frame = (
                    record.origin_image
                    if record is not None and record.origin_image is not None
                    else frames_dict.get(idx)
                )
                if frame is not None:
                    pi3_input_dict[idx] = frame
        else:
            pi3_input_dict = frames_dict

        imgs_tensor, resize_info = self._preprocess_frames(pi3_input_dict, timesteps)
        H_pi3, W_pi3 = resize_info["pi3_size"]
        H_orig, W_orig = resize_info["orig_size"]

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                res = self.model(imgs_tensor)

        # ------------------------------------------------------------------
        # Incremental Sim(3) pose chaining
        # ------------------------------------------------------------------
        # Extract Pi3X local poses for every frame in the current window.
        win_local_poses: Dict[int, np.ndarray] = {
            t: res["camera_poses"][0, local_i].cpu().float().numpy()
            for local_i, t in enumerate(timesteps)
        }

        # local_scale = scale factor mapping this window's Pi3X output to the
        # global frame.  Per-window Pi3X normalises to a unit cube, so each
        # call has its own implicit scale; multiplying local_points by this
        # factor brings depths to a consistent global metric.
        #
        # CRITICAL: Pi3X is a multi-view model — its output for a 1-frame or
        # 2-frame input is meaningless (no triangulation possible).  We must
        # not anchor _global_poses on a window with fewer than `_min_anchor`
        # frames, otherwise every later Sim(3) chain inherits the bad scale.
        _min_anchor = max(3, min(self.max_frames, 5))

        if len(timesteps) < _min_anchor and not self._global_poses:
            # Warmup: window too small for Pi3X to produce reliable poses.
            # Use Pi3X's own pose as a placeholder; do NOT anchor _global_poses.
            # Once a full window arrives, the deferred re-anchor below will
            # backfill all warmup frames in one consistent Sim(3) frame.
            pose_override = win_local_poses[frame_idx]
            local_scale: float = 1.0
            anchor_now = False
        elif not self._global_poses:
            # First call with a full-enough window — anchor all warmup
            # frames in this window to Pi3X's coordinate frame.
            self._global_poses.update(win_local_poses)
            pose_override = win_local_poses[frame_idx]
            local_scale = 1.0
            anchor_now = True

            # Replay warmup frames using this call's multi-view Pi3X output.
            # Each older frame in the window gets its own per-frame result
            # built with the *correct* anchor pose — overwriting the
            # placeholder we returned during warmup.
            for replay_local_idx, replay_idx in enumerate(timesteps):
                if replay_idx == frame_idx:
                    continue  # current frame is built normally below
                replay_pose = win_local_poses[replay_idx]
                r_scene, r_outputs, r_comp, r_dyn = self._build_frame_result(
                    replay_local_idx, replay_idx, res,
                    frames_dict, masks_dict, origin_dict,
                    H_pi3, W_pi3, H_orig, W_orig,
                    pose_override=replay_pose,
                    local_scale=1.0,
                )
                rec = frame_records.get(replay_idx)
                if rec is not None:
                    rec.depth = r_scene.depth_map
                    if r_scene.camera is not None:
                        rec.extrinsic = r_scene.camera.extrinsic
                        rec.intrinsic = r_scene.camera.intrinsic
                # Add to aggregator (these are reliable now).
                if (r_scene.points_3d is not None
                        and len(r_scene.points_3d) > 0):
                    self.aggregator.add_fragment(SceneFragment(
                        points=r_scene.points_3d,
                        confidence=r_scene.confidence,
                        metadata={"timestep": replay_idx,
                                   "colors": r_scene.colors,
                                   "camera": r_scene.camera},
                    ))
                vis = asdict(r_outputs)
                if r_comp is not None:
                    vis["composited_frame"] = r_comp
                if r_dyn is not None:
                    vis["dynamic_depth"] = r_dyn
                self._pending_warmup[replay_idx] = StageResult(
                    data={"reconstruction": r_outputs,
                           "frame_idx": replay_idx,
                           "frame_size": (H_orig, W_orig)},
                    visualization_assets=vis,
                )
        else:
            overlap_ts = [t for t in timesteps if t in self._global_poses]
            if len(overlap_ts) >= 2:
                src_pts = np.stack([win_local_poses[t][:3, 3] for t in overlap_ts])
                tgt_pts = np.stack([self._global_poses[t][:3, 3] for t in overlap_ts])
                scale, R_align, t_align = self._umeyama_sim3(src_pts, tgt_pts)
                p = win_local_poses[frame_idx].copy()
                # Scale+translate position only — see _calibrate_poses for why
                # R_align is not applied to the orientation matrix.
                p[:3, 3] = scale * (R_align @ p[:3, 3]) + t_align
                pose_override = p
                local_scale = float(scale)
            elif len(overlap_ts) == 1:
                # Only one overlap point — translation-only correction with
                # default scale.  Single-overlap scale is unidentifiable.
                t0 = overlap_ts[0]
                delta = self._global_poses[t0][:3, 3] - win_local_poses[t0][:3, 3]
                p = win_local_poses[frame_idx].copy()
                p[:3, 3] += delta
                pose_override = p
                local_scale = 1.0
            else:
                # No overlap (shouldn't occur in normal online usage).
                pose_override = win_local_poses[frame_idx]
                local_scale = 1.0

            self._global_poses[frame_idx] = pose_override
            anchor_now = False

        # Use only the current (newest) frame's output from the window.
        current_local_idx = len(timesteps) - 1
        current_scene, recon_outputs, composited, dynamic_depth = self._build_frame_result(
            current_local_idx, frame_idx, res,
            frames_dict, masks_dict, origin_dict,
            H_pi3, W_pi3, H_orig, W_orig,
            pose_override=pose_override,
            local_scale=local_scale,
        )

        # During warmup (window too small for reliable Pi3X output), zero out
        # depth / points and skip aggregator addition so the warmup frames
        # don't contaminate the global cloud.
        is_warmup = (len(timesteps) < _min_anchor and not anchor_now and
                      not self._global_poses)
        if is_warmup:
            zero_depth = np.zeros_like(current_scene.depth_map, dtype=current_scene.depth_map.dtype)
            current_scene.depth_map = zero_depth
            recon_outputs.depth_map = zero_depth.astype(np.float16)

        # Propagate depth / camera back into the frame record.
        record = frame_records.get(frame_idx)
        if record is not None:
            record.depth = current_scene.depth_map
            if current_scene.camera is not None:
                record.extrinsic = current_scene.camera.extrinsic
                record.intrinsic = current_scene.camera.intrinsic

        # Accumulate into the scene aggregator (skip warmup).
        if (not is_warmup
                and current_scene.points_3d is not None
                and len(current_scene.points_3d) > 0):
            self.aggregator.add_fragment(SceneFragment(
                points=current_scene.points_3d,
                confidence=current_scene.confidence,
                metadata={
                    "timestep": frame_idx,
                    "colors": current_scene.colors,
                    "camera": current_scene.camera,
                },
            ))

        aggregated_scene = self.aggregator.build_scene()
        recon_outputs.aggregated_scene = aggregated_scene

        # Expose the running global point cloud so the pipeline runner can
        # write the final PLY/GIF using the last frame's result.
        vis_assets = asdict(recon_outputs)
        vis_assets["global_point_cloud"] = aggregated_scene.get("merged_points")
        vis_assets["global_point_color"] = aggregated_scene.get("merged_colors")
        if composited is not None:
            vis_assets["composited_frame"] = composited
        if dynamic_depth is not None:
            vis_assets["dynamic_depth"] = dynamic_depth

        return StageResult(
            data={
                "reconstruction": recon_outputs,
                "frame_idx": frame_idx,
                "frame_size": (H_orig, W_orig),
            },
            visualization_assets=vis_assets,
        )

    def finalize(
        self, frame_records: Dict[int, FrameIORecord]
    ) -> List[Tuple[int, StageResult]]:
        """Not used — Pi3OnlineReconstructor processes frames incrementally."""
        raise RuntimeError(
            "Pi3OnlineReconstructor is online (run_deferred=False). "
            "The runner calls process() per frame; finalize() is not supported."
        )

    def post_process_fill(
        self,
        frame_records: Dict[int, FrameIORecord],
        results: List[Tuple[int, StageResult]],
        masks_dict: Dict[int, np.ndarray],
    ) -> None:
        """Apply multi-view depth fill across all per-frame online results.

        Steps:
          1. Replay any warmup-frame results captured at the first anchor
             call — these overwrite the zero-depth placeholders we emitted
             during warmup.
          2. Run the offline-style multi-view depth fill on every frame.
        """
        # Step 1: warmup replay
        if self._pending_warmup:
            replaced = 0
            for i, (fidx, _) in enumerate(results):
                if fidx in self._pending_warmup:
                    results[i] = (fidx, self._pending_warmup[fidx])
                    replaced += 1
            if replaced > 0:
                log_step(LOGGER, f"Online: replayed {replaced} warmup frame(s) "
                                  f"with anchor-coord results")
            self._pending_warmup.clear()

        # Step 2: multi-view fill
        self._fill_masked_depth_from_global(frame_records, results, masks_dict)
