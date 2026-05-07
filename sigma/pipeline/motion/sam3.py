"""SAM3 motion estimation using text-prompted segmentation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator
from sigma.pipeline.sam3_base import SAM3Base
from sigma.data.frame_record import FrameIORecord

LOGGER = logging.getLogger(__name__)


class SAM3MotionEstimator(BaseMotionEstimator, SAM3Base):
    """Motion estimation using SAM3 for text-prompted foreground detection.

    Unlike GroundedSAMMotionEstimator which requires separate Grounding DINO +
    SAM2 models, SAM3 combines open-vocabulary detection and segmentation in a
    single model.
    """

    name = "sam3_motion"

    def __init__(
        self,
        prompt: str = "person. car. bus. truck. bike.",
        confidence_threshold: float = 0.5,
        model_id: str | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.prompt = prompt
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        self.device = device

    def setup(self) -> None:
        """Load SAM3 model and processor."""
        self._setup_sam3(
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------
    # Motion-specific helpers
    # ------------------------------------------------------------------

    def _warp_prev_frame(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp previous frame to current frame coordinates using optical flow."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        h, w = prev_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            prev_frame,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return warped, flow

    def _flow_to_color(self, flow: np.ndarray) -> np.ndarray:
        """Convert optical flow to RGB image using HSV color space."""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(
        self,
        frame_records: Dict[int, FrameIORecord],
        frame_idx: int,
    ) -> StageResult:
        """Process frame pair to detect foreground objects and generate motion data.

        Args:
            frame_records: Dictionary of frame records indexed by frame number.
            frame_idx: Current frame index.

        Returns:
            StageResult containing motion data and visualization assets.
        """
        curr_record = frame_records[frame_idx]
        curr_frame = curr_record.origin_image

        # First frame — return empty motion
        if frame_idx == 0 or frame_idx - 1 not in frame_records:
            motion_payload = {
                "curr_mask": np.zeros(curr_frame.shape[:2], dtype=np.uint8),
                "warped_prev_to_curr": curr_frame,
                "optical_flow": np.zeros((*curr_frame.shape[:2], 2), dtype=np.float32),
                "static_curr": curr_frame,
                "fundamental_matrix": None,
            }
            return StageResult(data={"motion": motion_payload}, visualization_assets={})

        prev_record = frame_records[frame_idx - 1]
        prev_frame = prev_record.origin_image

        # 1. Detect and segment with SAM3
        moving_curr, boxes_np, scores_np = self._detect_and_segment(curr_frame, self.prompt)

        if moving_curr.sum() == 0:
            LOGGER.info(f"No objects detected in frame {frame_idx}")
            motion_payload = {
                "curr_mask": np.zeros(curr_frame.shape[:2], dtype=np.uint8),
                "warped_prev_to_curr": curr_frame,
                "optical_flow": np.zeros((*curr_frame.shape[:2], 2), dtype=np.float32),
                "static_curr": curr_frame,
                "fundamental_matrix": None,
            }
            return StageResult(data={"motion": motion_payload}, visualization_assets={})

        # 2. Compute optical flow and warp
        warped_prev, flow = self._warp_prev_frame(prev_frame, curr_frame)

        # 3. Create static background composite
        mask_curr_3d = moving_curr.astype(np.float32)[..., None]
        static_curr = (
            warped_prev.astype(np.float32) * mask_curr_3d
            + curr_frame.astype(np.float32) * (1.0 - mask_curr_3d)
        ).astype(curr_frame.dtype)

        # Motion payload
        motion_payload = {
            "curr_mask": moving_curr,
            "warped_prev_to_curr": warped_prev,
            "optical_flow": flow,
            "static_curr": static_curr,
            "fundamental_matrix": None,
        }

        # Visualization assets
        bbox_vis = self._draw_boxes(curr_frame, boxes_np, scores_np)
        mask_overlay = self._overlay_mask(curr_frame, moving_curr)
        flow_vis = self._flow_to_color(flow)

        visualization_assets = {
            "bbox": bbox_vis,
            "mask_overlay": mask_overlay,
            "mask_curr": moving_curr * 255,
            "warped_prev_to_curr": warped_prev,
            "motion_vector": flow_vis,
        }

        return StageResult(data={"motion": motion_payload}, visualization_assets=visualization_assets)

    def teardown(self) -> None:
        """Clean up models and free memory."""
        self._teardown_sam3()
