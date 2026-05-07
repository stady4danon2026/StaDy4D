"""Shared SAM3 base mixin for detection and segmentation.

Uses the HuggingFace transformers SAM3 implementation (facebook/sam3).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Silence httpx HTTP request logs produced by transformers' hub probing
logging.getLogger("httpx").setLevel(logging.WARNING)

_DEFAULT_MODEL_ID = "facebook/sam3"

# Module-level cache: (model_id, device) -> (model, processor)
# Avoids reloading weights when both SAM3Preprocessor and SAM3MotionEstimator
# are instantiated for the same model.
_SAM3_CACHE: Dict[tuple, tuple] = {}


class SAM3Base:
    """Mixin providing SAM3 model loading, detection/segmentation, and visualization.

    Classes using this mixin should call ``_setup_sam3`` in their ``setup()``
    and ``_teardown_sam3`` in their ``teardown()``.
    """

    # Set by _setup_sam3
    _sam3_model: Any = None
    _sam3_processor: Any = None
    _sam3_device: str = "cuda"
    _sam3_threshold: float = 0.5
    _sam3_mask_threshold: float = 0.5
    _sam3_ready: bool = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _setup_sam3(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        model_id: str | None = None,
    ) -> None:
        """Load SAM3 model and processor from HuggingFace.

        Args:
            device: Torch device string.
            confidence_threshold: Score threshold for instance segmentation.
            mask_threshold: Mask binarization threshold.
            model_id: HuggingFace model ID (defaults to ``facebook/sam3``).
        """
        from transformers import Sam3Model, Sam3Processor

        model_id = model_id or _DEFAULT_MODEL_ID
        cache_key = (model_id, device)

        if cache_key in _SAM3_CACHE:
            LOGGER.info("Reusing cached SAM3 model for %s on %s", model_id, device)
            self._sam3_model, self._sam3_processor = _SAM3_CACHE[cache_key]
        else:
            LOGGER.info("Loading SAM3 model from %s", model_id)
            processor = Sam3Processor.from_pretrained(model_id)
            model = Sam3Model.from_pretrained(model_id).to(device)
            _SAM3_CACHE[cache_key] = (model, processor)
            self._sam3_model = model
            self._sam3_processor = processor
            LOGGER.info("SAM3 model loaded successfully")

        self._sam3_device = device
        self._sam3_threshold = confidence_threshold
        self._sam3_mask_threshold = mask_threshold
        self._sam3_ready = True

    def _teardown_sam3(self) -> None:
        """Release this instance's SAM3 references (shared cache is kept)."""
        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_ready = False

    # ------------------------------------------------------------------
    # Detection / segmentation
    # ------------------------------------------------------------------

    def _detect_and_segment(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SAM3 text-prompted detection and segmentation.

        Args:
            image_rgb: RGB image as ``(H, W, 3)`` uint8 array.
            prompt: Text prompt for open-vocabulary detection.

        Returns:
            mask: Union foreground mask ``(H, W)`` uint8 (1=foreground).
            boxes: Detected bounding boxes ``(N, 4)`` float32.
            scores: Detection confidence scores ``(N,)`` float32.
        """
        if not self._sam3_ready:
            raise RuntimeError("Call _setup_sam3() before _detect_and_segment().")

        h, w = image_rgb.shape[:2]
        pil_image = Image.fromarray(image_rgb)

        inputs = self._sam3_processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self._sam3_device)

        with torch.no_grad():
            outputs = self._sam3_model(**inputs)

        results = self._sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=self._sam3_threshold,
            mask_threshold=self._sam3_mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        seg_masks = results.get("masks")    # list of (H, W) bool tensors
        seg_scores = results.get("scores")  # list of float

        if seg_masks is None or len(seg_masks) == 0:
            return (
                np.zeros((h, w), dtype=np.uint8),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # Stack instance masks and compute union
        if isinstance(seg_masks, torch.Tensor):
            masks_np = seg_masks.cpu().numpy()
        else:
            masks_np = np.stack([
                m.cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
                for m in seg_masks
            ])
        # masks_np: (N, H, W)
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        mask = masks_np.any(axis=0).astype(np.uint8)

        # Derive bounding boxes from masks
        boxes_list = []
        for m in masks_np:
            ys, xs = np.where(m)
            if len(ys) == 0:
                boxes_list.append([0, 0, 0, 0])
            else:
                boxes_list.append([xs.min(), ys.min(), xs.max(), ys.max()])
        boxes_np = np.array(boxes_list, dtype=np.float32)

        # Scores
        if isinstance(seg_scores, torch.Tensor):
            scores_np = seg_scores.cpu().numpy().astype(np.float32)
        else:
            scores_np = np.array(seg_scores, dtype=np.float32)

        return mask, boxes_np, scores_np

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Draw bounding boxes with confidence scores on the image."""
        vis_img = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            score = scores[i] if i < len(scores) else 0.0
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return vis_img

    def _overlay_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create visualization overlay of mask on frame."""
        binary = (mask > 0).astype(np.uint8) * 255
        masked = cv2.bitwise_and(frame, frame, mask=binary)
        return masked
