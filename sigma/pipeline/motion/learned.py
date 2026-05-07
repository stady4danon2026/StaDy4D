"""Learning-based motion estimation powered by SAM2 and GMM segmenters."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import sys
# sys.path.append("/data/chw143/experiments/sam2") #TODO: modify this
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    build_sam2 = None
    SAM2ImagePredictor = None

from omegaconf import OmegaConf
from hydra.utils import instantiate
from huggingface_hub import hf_hub_download
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator, MotionOutputs


def _ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _normalize_probability(diff: np.ndarray) -> np.ndarray:
    diff = diff.astype(np.float32)
    if diff.size == 0:
        return diff
    min_val = float(diff.min())
    max_val = float(diff.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(diff, dtype=np.float32)
    return (diff - min_val) / (max_val - min_val)


def _diff_binary(prev_frame: np.ndarray, curr_frame: np.ndarray, blur: int) -> np.ndarray:
    prev_gray = _ensure_gray(prev_frame)
    curr_gray = _ensure_gray(curr_frame)
    diff = cv2.absdiff(prev_gray, curr_gray)
    blur = blur if blur % 2 == 1 else blur + 1
    diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


class SAM2Segmenter:
    """Wrap SAM2ImagePredictor and return a foreground mask guided by motion cues."""

    def __init__(self, device: str | None = None) -> None:
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed or not found. Please install it to use SAM2Segmenter.")
        # Initialize model and predictor directly as in the notebook
        sam2_checkpoint = hf_hub_download(repo_id="facebook/sam2-hiera-large", filename="sam2_hiera_large.pt")
        model_cfg = "configs/sam2_hiera_l.yaml"
        
        # Load config manually to bypass Hydra search path issues
        cfg = OmegaConf.load(model_cfg)
        
        # Apply overrides expected by SAM2
        overrides = OmegaConf.from_dotlist([
            "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98"
        ])
        cfg = OmegaConf.merge(cfg, overrides)
        
        sam2_model = instantiate(cfg.model, _recursive_=True)
        
        # Load checkpoint
        if sam2_checkpoint is not None:
            state_dict = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)["model"]
            sam2_model.load_state_dict(state_dict)
            
        sam2_model.to(device)
        sam2_model.eval()
        
        self.predictor = SAM2ImagePredictor(sam2_model)

    def _get_motion_points(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Blur and threshold to get binary motion mask
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, diff_binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # Min area threshold
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])
        
        # Fallback if no motion detected
        if not points:
            h, w = curr_frame.shape[:2]
            points.append([w // 2, h // 2])
            
        return np.array(points, dtype=np.float32)

    def __call__(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Set image for predictor
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(curr_rgb)
        
        # 2. Generate prompts from motion
        input_points = self._get_motion_points(prev_frame, curr_frame)
        input_labels = np.ones(len(input_points), dtype=np.int32)  # Foreground labels
        
        # 3. Predict masks
        # Using multimask_output=True to get best candidates, then picking the best score
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Select the mask with the highest score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Prepare output
        probability = best_mask.astype(np.float32)
        segmentation = (probability > 0.5).astype(np.uint8)
        
        return probability, segmentation


class GMMSegmenter:
    """Gaussian Mixture based background subtractor wrapper."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        history: int = 50,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.model = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def __call__(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Adapt to the previous frame first and then segment the current frame
        self.model.apply(prev_frame, learningRate=1.0)
        fg_mask = self.model.apply(curr_frame, learningRate=0.0)
        probability = _normalize_probability(fg_mask)
        segmentation = (probability >= self.confidence_threshold).astype(np.uint8)
        return probability, segmentation

class LearnedMotionEstimator(BaseMotionEstimator):
    """Learning-based motion estimator leveraging SAM2 or GMM segmentation."""

    name = "learned_motion"

    def __init__(
        self,
        backbone: str = "sam2",
        confidence_threshold: float = 0.5,
        device: str | None = None,
        visualization: bool = True,
        mask_model: str | None = None,
        gmm_history: int = 40,
        gmm_var_threshold: float = 12.0,
        gmm_detect_shadows: bool = False,
        **_: Any,
    ) -> None:
        self.backbone = backbone.lower()
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.visualization = visualization
        self.mask_model = mask_model.lower() if mask_model else None
        self.gmm_history = gmm_history
        self.gmm_var_threshold = gmm_var_threshold
        self.gmm_detect_shadows = gmm_detect_shadows

        self.initialized = False
        self.ready = False
        self.models: Dict[str, Any] | None = None

    def setup(self) -> None:
        self._init_models()
        self.initialized = True
        self.ready = True

    def _init_models(self) -> None:
        if self.backbone == "sam2":
            self.models = {
                "segmenter": SAM2Segmenter(device=self.device),
                "type": "sam2",
            }
        elif self.backbone == "gmm":
            self.models = {
                "segmenter": GMMSegmenter(
                    confidence_threshold=self.confidence_threshold,
                    history=self.gmm_history,
                    var_threshold=self.gmm_var_threshold,
                    detect_shadows=self.gmm_detect_shadows,
                ),
                "type": "gmm",
            }
        else:
            raise ValueError(f"Unsupported backbone '{self.backbone}'. Expected 'sam2' or 'gmm'.")

    def _run_model(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Models not initialized. Call setup() before process().")
        segmenter = self.models.get("segmenter")
        if segmenter is None:
            raise RuntimeError("Missing segmenter in initialized models.")
        probability_map, segmentation = segmenter(prev_frame, curr_frame)

        return probability_map, segmentation

    @staticmethod
    def _overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if frame.ndim != 3:
            return frame
        tint = np.zeros_like(frame)
        tint[..., 2] = 255
        binary = (mask > 0).astype(np.uint8)
        tinted = cv2.addWeighted(frame, 0.4, tint, 0.6, 0)
        overlay = frame.copy()
        overlay[binary.astype(bool)] = tinted[binary.astype(bool)]
        return overlay

    def process(self, inputs: Dict[str, Any]) -> StageResult:
        """Produce motion cues purely from the learned backbone."""
        if not self.initialized or not self.ready:
            raise RuntimeError("LearnedMotionEstimator.setup() must be called before process().")

        prev_frame = inputs.get("prev_frame")
        curr_frame = inputs.get("curr_frame")
        if prev_frame is None or curr_frame is None:
            raise ValueError("prev_frame and curr_frame must be provided.")

        probability_map, segmentation = self._run_model(prev_frame, curr_frame)
        foreground_mask = segmentation.astype(np.uint8)

        motion_outputs = MotionOutputs(
            background_mask=foreground_mask,
            motion_vectors=None,
            fundamental_matrix=None,
        )
        vis_assets = {
            "probability_map": probability_map,
            "segmentation": foreground_mask,
        }
        if self.visualization:
            overlay = self._overlay_mask(curr_frame, foreground_mask)
            # cv2.imwrite("/data/chw143/experiments/SIGMA/outputs/visuals/motion/debug_overlay.jpg", overlay)
            vis_assets["overlay"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return StageResult(
            data={
                "motion": motion_outputs,
                "inputs": inputs,
                "foreground_mask": foreground_mask,
            },
            visualization_assets=vis_assets,
        )

    def teardown(self) -> None:
        self.initialized = False
        self.ready = False
        self.models = None

