"""Grounded SAM 2 preprocessor for foreground removal."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sigma.pipeline.preprocess.base_preprocess import BasePreprocessor
from sigma.pipeline.base_stage import StageResult
from sigma.utils.log_messages import log_step

LOGGER = logging.getLogger(__name__)


class GroundedSAMPreprocessor(BasePreprocessor):
    """Preprocessor that uses Grounding DINO and SAM 2 to detect and mask foreground objects."""

    name = "grounded_sam_preprocessor"

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
        self.sam2_predictor = None
        self.ready = False

    def setup(self) -> None:
        """Load Grounding DINO and SAM 2 models."""
        log_step(LOGGER, "Loading Grounding DINO", self.grounding_dino_model_id)
        
        # Load Grounding DINO
        try:
            self.processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to(self.device)
        except Exception as e:
            raise ImportError(f"Failed to load Grounding DINO from {self.grounding_dino_model_id}. Ensure transformers is up to date.") from e

        log_step(LOGGER, "Loading SAM 2", self.sam2_checkpoint)
        
        # Load SAM 2
        try:    
            self.sam2_processor = Sam2Processor.from_pretrained(self.sam2_checkpoint)
            self.sam2_model = Sam2Model.from_pretrained(self.sam2_checkpoint).to(self.device)
        except Exception as e:
            raise ImportError(f"Failed to load SAM 2: {e}") from e

        self.ready = True
    
    def _boxes_to_mask(self, image_shape: tuple, boxes: np.ndarray) -> tuple:
        """
        Generate a binary mask from bounding boxes with percentage-based margins.

        Args:
            image_shape: Shape of the image (H, W) or (H, W, C).
            boxes: Array of bounding boxes with shape (N, 4) in format [x1, y1, x2, y2].

        Returns:
            mask: Binary mask (H, W) where 1 indicates the region within the boxes.
            expanded_boxes: Array of expanded bounding boxes (N, 4) after applying margins.
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
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
    
    def _draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: List[str], scores: np.ndarray) -> np.ndarray:
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
            cv2.putText(vis_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis_img

    def process(self, image: np.ndarray) -> np.ndarray:
        """Detect and segment objects.

        Args:
            image: RGB image (H, W, 3).

        Returns:
            Binary mask (H, W) where 1 is foreground.
        """
        if not self.ready:
            raise RuntimeError("GroundedSAMPreprocessor.setup() must be called before process().")

        # 1. Grounding DINO Detection
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

        if len(boxes) == 0:
            LOGGER.info("No objects detected by Grounding DINO.")
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            empty_viz_asset = {
                'bbox': image.copy(),
                'mask_overlay': image.copy(),
            }
            return empty_mask, empty_viz_asset

        if self.use_rectangular_mask:
            background_mask, expanded_boxes = self._boxes_to_mask(image.shape, boxes)
            # Visualize expanded boxes (after applying margins)
            predicted_bbox_image = self._draw_boxes(image, expanded_boxes, labels, scores)
        else:
            # For SAM segmentation, visualize original boxes
            predicted_bbox_image = self._draw_boxes(image, boxes, labels, scores)
            _, expanded_boxes = self._boxes_to_mask(image.shape, boxes)

            input_boxes = [expanded_boxes.tolist()]

            inputs = self.sam2_processor(images=pil_image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sam2_model(**inputs)
            masks = self.sam2_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

            # masks: (N, 3, H, W) -> (N, H, W)
            if masks.ndim == 4:
                masks = masks[:, 0, :, :]

            # Combine all object masks
            # masks: (N, H, W) -> (H, W)
            background_mask = masks.any(dim=0).cpu().numpy().astype(np.uint8)
        
        # blend mask with image for visualization
        # Create a red overlay for foreground
        mask_overlay = image.copy()
        
        # Create colored mask (Red)
        colored_mask = np.zeros_like(image)
        colored_mask[background_mask == 1] = [0, 0, 255] 
        
        # Blend where mask is present
        alpha = 0.5
        mask_indices = background_mask == 1
        if np.any(mask_indices):
            mask_overlay[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)

        viz_asset = {
            'bbox': predicted_bbox_image,
            'mask_overlay': mask_overlay,
        }
        
        return background_mask, viz_asset

    def teardown(self) -> None:
        self.grounding_model = None
        self.processor = None
        self.sam2_predictor = None
        self.ready = False
