"""Grounded SAM motion estimation using object detection and segmentation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model

from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator
from sigma.data.frame_record import FrameIORecord

LOGGER = logging.getLogger(__name__)


class GroundedSAMMotionEstimator(BaseMotionEstimator):
    """Motion estimation using Grounding DINO and SAM 2 for foreground detection."""

    name = "grounded_sam_motion"

    def __init__(
        self,
        grounding_dino_model: str = "IDEA-Research/grounding-dino-base",
        sam2_checkpoint: str = "facebook/sam2.1-hiera-large",
        prompt: str = "person. car. bus. truck. bike. chair. table.",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        margin_left: float = 0.05,
        margin_right: float = 0.05,
        margin_top: float = 0.05,
        margin_bottom: float = 0.05,
        use_rectangular_mask: bool = False,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.grounding_dino_model_id = grounding_dino_model
        self.prompt = prompt
        self.sam2_checkpoint = sam2_checkpoint
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.use_rectangular_mask = use_rectangular_mask
        self.device = device
        
        self.grounding_model = None
        self.processor = None
        self.sam2_processor = None
        self.sam2_model = None
        self.ready = False

    def setup(self) -> None:
        """Load Grounding DINO and SAM 2 models."""
        LOGGER.info(f"Loading Grounding DINO: {self.grounding_dino_model_id}")
        
        # Load Grounding DINO
        try:
            self.processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.grounding_dino_model_id
            ).to(self.device)
        except Exception as e:
            raise ImportError(
                f"Failed to load Grounding DINO from {self.grounding_dino_model_id}. "
                f"Ensure transformers is up to date."
            ) from e

        LOGGER.info(f"Loading SAM 2: {self.sam2_checkpoint}")
        
        # Load SAM 2
        try:    
            self.sam2_processor = Sam2Processor.from_pretrained(self.sam2_checkpoint)
            self.sam2_model = Sam2Model.from_pretrained(self.sam2_checkpoint).to(self.device)
        except Exception as e:
            raise ImportError(f"Failed to load SAM 2: {e}") from e

        self.ready = True

    def _detect_objects(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Detect objects using Grounding DINO.
        
        Args:
            image: RGB image (H, W, 3).
            
        Returns:
            boxes: Array of bounding boxes (N, 4) in [x1, y1, x2, y2] format.
            labels: List of detected object labels.
            scores: Array of confidence scores (N,).
        """
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        labels = results.get("text_labels", [])
        scores = results["scores"].cpu().numpy()
        
        return boxes, labels, scores

    def _segment_with_sam(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray
    ) -> np.ndarray:
        """Segment objects using SAM 2.
        
        Args:
            image: RGB image (H, W, 3).
            boxes: Array of bounding boxes (N, 4).
            
        Returns:
            Binary mask (H, W) where 1 is foreground.
        """
        pil_image = Image.fromarray(image)
        input_boxes = [boxes.tolist()]
        
        inputs = self.sam2_processor(
            images=pil_image, 
            input_boxes=input_boxes, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam2_model(**inputs)
        
        masks = self.sam2_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"]
        )[0]

        # masks: (N, 3, H, W) -> (N, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        
        # Combine all object masks: (N, H, W) -> (H, W)
        foreground_mask = masks.any(dim=0).cpu().numpy().astype(np.uint8)
        
        return foreground_mask

    def _boxes_to_mask(
        self,
        image_shape: Tuple[int, int],
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a binary mask from bounding boxes with percentage-based margins.

        Args:
            image_shape: Shape of the image (H, W).
            boxes: Array of bounding boxes with shape (N, 4) in format [x1, y1, x2, y2].

        Returns:
            mask: Binary mask (H, W) where 1 indicates the region within the boxes.
            expanded_boxes: Array of expanded bounding boxes (N, 4) after applying margins.
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        expanded_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            # Calculate bbox dimensions
            box_width = x2 - x1
            box_height = y2 - y1

            # Calculate pixel margins based on percentages
            left_px = int(box_width * self.margin_left)
            right_px = int(box_width * self.margin_right)
            top_px = int(box_height * self.margin_top)
            bottom_px = int(box_height * self.margin_bottom)

            # Apply margins with clipping to image boundaries
            x1_expanded = max(0, x1 - left_px)
            y1_expanded = max(0, y1 - top_px)
            x2_expanded = min(image_shape[1], x2 + right_px)
            y2_expanded = min(image_shape[0], y2 + bottom_px)

            mask[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 1
            expanded_boxes.append([x1_expanded, y1_expanded, x2_expanded, y2_expanded])

        return mask, np.array(expanded_boxes)

    def _draw_boxes(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray, 
        labels: List[str], 
        scores: np.ndarray
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        vis_img = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            label = labels[i] if i < len(labels) else "unknown"
            score = scores[i] if i < len(scores) else 0.0
            
            # Draw box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label_text = f"{label} {score:.2f}"
            cv2.putText(
                vis_img, label_text, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        return vis_img

    def _overlay_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create visualization overlay of mask on frame."""
        binary = (mask > 0).astype(np.uint8) * 255
        masked = cv2.bitwise_and(frame, frame, mask=binary)
        return masked

    def _warp_prev_frame(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
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
            prev_frame, map_x, map_y, 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT
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
    # Batched detection
    # ------------------------------------------------------------------

    def _detect_objects_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8,
    ) -> Tuple[List[np.ndarray], List[List[str]], List[np.ndarray]]:
        """Detect objects in multiple images using chunked DINO forward passes.

        Args:
            images: List of RGB images (H, W, 3).
            batch_size: Max images per forward pass to avoid OOM.

        Returns:
            Tuple of (boxes_list, labels_list, scores_list) — one entry per image.
        """
        all_boxes: List[np.ndarray] = []
        all_labels: List[List[str]] = []
        all_scores: List[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            pil_images = [Image.fromarray(img) for img in chunk]
            target_sizes = [img.size[::-1] for img in pil_images]

            inputs = self.processor(
                images=pil_images,
                text=[self.prompt] * len(pil_images),
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )

            for result in results:
                all_boxes.append(result["boxes"].cpu().numpy())
                all_labels.append(result.get("text_labels", []))
                all_scores.append(result["scores"].cpu().numpy())

        return all_boxes, all_labels, all_scores

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn=None,
    ) -> Dict[int, StageResult]:
        """Batch process all frames with batched DINO detection + per-frame SAM + flow.

        The heavy DINO forward pass is batched across all frames. SAM segmentation
        and optical flow are still per-frame (different box counts, pairwise flow).
        """
        if not self.ready:
            raise RuntimeError("Call setup() before process_batch().")

        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}

        # First frame: empty motion (no previous frame).
        first_idx = sorted_indices[0]
        first_frame = frame_records[first_idx].origin_image
        h, w = first_frame.shape[:2]
        results[first_idx] = StageResult(
            data={"motion": {
                "curr_mask": np.zeros((h, w), dtype=np.uint8),
                "warped_prev_to_curr": first_frame,
                "optical_flow": np.zeros((h, w, 2), dtype=np.float32),
                "static_curr": first_frame,
                "fundamental_matrix": None,
            }},
            visualization_assets={},
        )
        if progress_fn:
            progress_fn(1, total)

        frames_to_detect = sorted_indices[1:]
        if not frames_to_detect:
            return results

        # Batched DINO detection across all frames.
        batch_images = [frame_records[idx].origin_image for idx in frames_to_detect]
        LOGGER.info("Running batched DINO detection on %d frames", len(batch_images))
        batch_boxes, batch_labels, batch_scores = self._detect_objects_batch(batch_images)

        # Per-frame: SAM segmentation + optical flow + compositing.
        for i, frame_idx in enumerate(frames_to_detect):
            curr_frame = frame_records[frame_idx].origin_image
            prev_frame = frame_records[frame_idx - 1].origin_image
            boxes = batch_boxes[i]
            labels = batch_labels[i]
            scores = batch_scores[i]

            if len(boxes) == 0:
                results[frame_idx] = StageResult(
                    data={"motion": {
                        "curr_mask": np.zeros(curr_frame.shape[:2], dtype=np.uint8),
                        "warped_prev_to_curr": curr_frame,
                        "optical_flow": np.zeros((*curr_frame.shape[:2], 2), dtype=np.float32),
                        "static_curr": curr_frame,
                        "fundamental_matrix": None,
                    }},
                    visualization_assets={},
                )
                if progress_fn:
                    progress_fn(i + 2, total)
                continue

            # SAM segmentation (per-frame — variable box count).
            if self.use_rectangular_mask:
                moving_curr, expanded_boxes = self._boxes_to_mask(curr_frame.shape[:2], boxes)
                bbox_vis = self._draw_boxes(curr_frame, expanded_boxes, labels, scores)
            else:
                _, expanded_boxes = self._boxes_to_mask(curr_frame.shape[:2], boxes)
                moving_curr = self._segment_with_sam(curr_frame, expanded_boxes)
                bbox_vis = self._draw_boxes(curr_frame, expanded_boxes, labels, scores)

            # Optical flow (pairwise).
            warped_prev, flow = self._warp_prev_frame(prev_frame, curr_frame)

            # Static background composite.
            mask_curr_3d = moving_curr.astype(np.float32)[..., None]
            static_curr = (
                warped_prev.astype(np.float32) * mask_curr_3d
                + curr_frame.astype(np.float32) * (1.0 - mask_curr_3d)
            ).astype(curr_frame.dtype)

            motion_payload = {
                "curr_mask": moving_curr,
                "warped_prev_to_curr": warped_prev,
                "optical_flow": flow,
                "static_curr": static_curr,
                "fundamental_matrix": None,
            }

            mask_overlay = self._overlay_mask(curr_frame, moving_curr)
            flow_vis = self._flow_to_color(flow)
            visualization_assets = {
                "bbox": bbox_vis,
                "mask_overlay": mask_overlay,
                "mask_curr": moving_curr * 255,
                "warped_prev_to_curr": warped_prev,
                "motion_vector": flow_vis,
            }

            results[frame_idx] = StageResult(
                data={"motion": motion_payload},
                visualization_assets=visualization_assets,
            )
            if progress_fn:
                progress_fn(i + 2, total)

        return results

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Process frame pair to detect foreground objects and generate motion data.
        
        Args:
            frame_records: Dictionary of frame records indexed by frame number.
            frame_idx: Current frame index.
            
        Returns:
            StageResult containing motion data and visualization assets.
        """
        if not self.ready:
            raise RuntimeError("Call setup() before process().")
        
        # Extract current frame
        curr_record = frame_records[frame_idx]
        curr_frame = curr_record.origin_image
        
        # First frame - return empty motion
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

        # 1. Detect objects using Grounding DINO
        boxes, labels, scores = self._detect_objects(curr_frame)
        
        if len(boxes) == 0:
            LOGGER.info(f"No objects detected in frame {frame_idx}")
            # No objects detected - return empty mask
            motion_payload = {
                "curr_mask": np.zeros(curr_frame.shape[:2], dtype=np.uint8),
                "warped_prev_to_curr": curr_frame,
                "optical_flow": np.zeros((*curr_frame.shape[:2], 2), dtype=np.float32),
                "static_curr": curr_frame,
                "fundamental_matrix": None,
            }
            return StageResult(data={"motion": motion_payload}, visualization_assets={})
        
        # 2. Generate mask (either rectangular or SAM segmentation)
        if self.use_rectangular_mask:
            moving_curr, expanded_boxes = self._boxes_to_mask(curr_frame.shape[:2], boxes)
            # Visualize expanded boxes (after applying margins)
            bbox_vis = self._draw_boxes(curr_frame, expanded_boxes, labels, scores)
        else:
            _, expanded_boxes = self._boxes_to_mask(curr_frame.shape[:2], boxes)
            moving_curr = self._segment_with_sam(curr_frame, expanded_boxes)
            # For SAM segmentation, visualize original boxes
            bbox_vis = self._draw_boxes(curr_frame, expanded_boxes, labels, scores)

        # Invert mask: 1 for background (keep), 0 for foreground (inpaint)
        # moving_curr_inverted = 1 - moving_curr

        # 3. Compute optical flow and warp
        warped_prev, flow = self._warp_prev_frame(prev_frame, curr_frame)

        # 4. Create static background composite (use non-inverted for blending)
        mask_curr_3d = moving_curr.astype(np.float32)[..., None]
        static_curr = (
            warped_prev.astype(np.float32) * mask_curr_3d +
            curr_frame.astype(np.float32) * (1.0 - mask_curr_3d)
        ).astype(curr_frame.dtype)

        # Prepare motion payload (use inverted mask)
        motion_payload = {
            "curr_mask": moving_curr,
            "warped_prev_to_curr": warped_prev,
            "optical_flow": flow,
            "static_curr": static_curr,
            "fundamental_matrix": None,  # Not computed in this method
        }

        # Visualization assets
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
        self.grounding_model = None
        self.processor = None
        self.sam2_processor = None
        self.sam2_model = None
        self.ready = False
