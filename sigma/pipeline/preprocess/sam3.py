"""SAM3 preprocessor for text-prompted foreground segmentation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from sigma.pipeline.preprocess.base_preprocess import BasePreprocessor
from sigma.pipeline.sam3_base import SAM3Base

LOGGER = logging.getLogger(__name__)


class SAM3Preprocessor(BasePreprocessor, SAM3Base):
    """Preprocessor using SAM3 for open-vocabulary foreground detection.

    Replaces the two-model Grounding DINO + SAM2 pipeline with a single
    SAM3 model that handles both detection and segmentation.
    """

    name = "sam3_preprocessor"

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
        """Load SAM3 model."""
        self._setup_sam3(
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            model_id=self.model_id,
        )

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Detect and segment foreground objects.

        Args:
            image: RGB image ``(H, W, 3)`` uint8.

        Returns:
            Tuple of (mask, viz_assets) where mask is ``(H, W)`` uint8 with
            1=foreground, and viz_assets contains bbox and mask overlay images.
        """
        mask, boxes, scores = self._detect_and_segment(image, self.prompt)

        # Bounding-box visualization
        bbox_vis = self._draw_boxes(image, boxes, scores)

        # Red-tinted mask overlay
        mask_overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = [0, 0, 255]

        alpha = 0.5
        fg = mask == 1
        if np.any(fg):
            mask_overlay[fg] = cv2.addWeighted(
                image[fg], 1 - alpha, colored_mask[fg], alpha, 0
            )

        viz_assets = {
            "bbox": bbox_vis,
            "mask_overlay": mask_overlay,
        }

        return mask, viz_assets

    def teardown(self) -> None:
        """Release SAM3 model."""
        self._teardown_sam3()
