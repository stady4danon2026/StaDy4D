"""Hybrid geometric + learning-based motion estimation placeholder."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator, MotionOutputs


##########################################
# 0. Model init
##########################################


def init_models(device: str | None = None) -> Dict[str, Any]:
    """Load DISK, LightGlue, and LoFTR on the selected device."""
    if device is None:
        resolved = K.utils.get_cuda_or_mps_device_if_available()
    else:
        resolved = torch.device(device)
    disk = KF.DISK.from_pretrained("depth").to(resolved).eval()
    lightglue = KF.LightGlue("disk").to(resolved).eval()
    loftr = KF.LoFTR(pretrained="outdoor").to(resolved).eval()
    return {"device": resolved, "disk": disk, "lightglue": lightglue, "loftr": loftr}


def _cv2_to_torch_rgb(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


##########################################
# 1. LightGlue (DISK) → sparse F
##########################################


def compute_F_lightglue(img1_bgr: np.ndarray, img2_bgr: np.ndarray, models: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = models["device"]
    disk = models["disk"]
    lg = models["lightglue"]
    img1 = _cv2_to_torch_rgb(img1_bgr, device)
    img2 = _cv2_to_torch_rgb(img2_bgr, device)
    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)
    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = disk(inp, n=None, pad_if_not_divisible=True)
        kps1, desc1 = features1.keypoints, features1.descriptors
        kps2, desc2 = features2.keypoints, features2.descriptors
        image0 = {
            "keypoints": kps1[None],
            "descriptors": desc1[None],
            "image_size": hw1.flip(0).view(1, 2),
        }
        image1 = {
            "keypoints": kps2[None],
            "descriptors": desc2[None],
            "image_size": hw2.flip(0).view(1, 2),
        }
        out = lg({"image0": image0, "image1": image1})
        idxs = out["matches"][0]
    if idxs.numel() == 0:
        raise RuntimeError("LightGlue returned zero matches.")
    idxs_np = idxs.detach().cpu().numpy()
    mkpts1 = kps1[idxs_np[:, 0]].detach().cpu().numpy()
    mkpts2 = kps2[idxs_np[:, 1]].detach().cpu().numpy()
    F, inliers = cv2.findFundamentalMat(
        mkpts1,
        mkpts2,
        cv2.USAC_MAGSAC,
        1.0,
        0.999,
        100000,
    )
    if F is None or inliers is None:
        raise RuntimeError("Fundamental matrix estimation failed")
    inliers = (inliers > 0).ravel()
    return F, mkpts1[inliers], mkpts2[inliers]


##########################################
# 2. LoFTR → semi-dense correspondences
##########################################


def compute_loftr_matches(img1_bgr: np.ndarray, img2_bgr: np.ndarray, models: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    device = models["device"]
    loftr = models["loftr"]
    img1 = _cv2_to_torch_rgb(img1_bgr, device)
    img2 = _cv2_to_torch_rgb(img2_bgr, device)
    img1 = K.geometry.resize(img1, (img1.shape[2], img1.shape[3]), antialias=True)
    img2 = K.geometry.resize(img2, (img2.shape[2], img2.shape[3]), antialias=True)
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }
    with torch.inference_mode():
        corr = loftr(input_dict)
    mkpts0 = corr["keypoints0"].cpu().numpy()
    mkpts1 = corr["keypoints1"].cpu().numpy()
    return mkpts0, mkpts1


##########################################
# 3. Epipolar residual map from LoFTR matches
##########################################


def _compute_match_residuals(F: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric epipolar residuals for both images."""
    if mkpts0.size == 0 or mkpts1.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty
    x1 = np.concatenate([mkpts0, np.ones((mkpts0.shape[0], 1))], axis=1)
    x2 = np.concatenate([mkpts1, np.ones((mkpts1.shape[0], 1))], axis=1)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    residual_curr = np.abs(np.sum(x2 * Fx1, axis=1)) / (np.sqrt(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2) + 1e-6)
    residual_prev = np.abs(np.sum(x1 * Ftx2, axis=1)) / (np.sqrt(Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2) + 1e-6)
    return residual_prev.astype(np.float32), residual_curr.astype(np.float32)


def _scatter_residual_map(points: np.ndarray, residuals: np.ndarray, shape: Tuple[int, int], blur_ksize: int) -> np.ndarray:
    H, W = shape[:2]
    if points.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    residual_map = np.zeros((H, W), dtype=np.float32)
    for (u, v), r in zip(points, residuals):
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < W and 0 <= vi < H:
            residual_map[vi, ui] = max(residual_map[vi, ui], r)
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        residual_map = cv2.GaussianBlur(residual_map, (k, k), 0)
    return residual_map


##########################################
# 4. Foreground mask from residual map
##########################################


def generate_fg_mask(residual: np.ndarray, k: float = 2.5, top_percent: float | None = None):
    if top_percent is not None:
        vals = residual[residual > 0]
        if vals.size == 0:
            th = np.inf
        else:
            th = np.quantile(vals, 1.0 - top_percent)
    else:
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) + 1e-6
        th = med + k * mad

    mask = (residual > th).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, th


##########################################
# 5. Main pipeline: LightGlue + LoFTR
##########################################


def segment_foreground_lightglue_loftr(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    models: Dict[str, Any],
    residual_k: float = 2.5,
    blur_ksize: int = 5,
    max_input_edge: int | None = None,
    top_percent: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    H, W = img1_bgr.shape[:2]
    H_curr, W_curr = img2_bgr.shape[:2]
    scale = 1.0
    proc_img1 = img1_bgr
    proc_img2 = img2_bgr
    if max_input_edge is not None:
        max_dim = max(H, W, H_curr, W_curr)
        if max_dim > max_input_edge:
            scale = max_input_edge / float(max_dim)
            new_w = max(16, int(round(W * scale)))
            new_h = max(16, int(round(H * scale)))
            proc_img1 = cv2.resize(img1_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            proc_img2 = cv2.resize(img2_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1) sparse F from LightGlue (DISK)
    F_scaled, mkpts_sparse0, mkpts_sparse1 = compute_F_lightglue(proc_img1, proc_img2, models)
    # 2) semi-dense matches from LoFTR
    mkpts0, mkpts1 = compute_loftr_matches(proc_img1, proc_img2, models)
    # 3) symmetric epipolar residual maps
    residual_prev_vals, residual_curr_vals = _compute_match_residuals(F_scaled, mkpts0, mkpts1)
    residual_prev_map = _scatter_residual_map(mkpts0, residual_prev_vals, proc_img1.shape[:2], blur_ksize)
    residual_curr_map = _scatter_residual_map(mkpts1, residual_curr_vals, proc_img2.shape[:2], blur_ksize)
    # 4) foreground masks
    mask_prev, th_prev = generate_fg_mask(residual_prev_map, residual_k, top_percent)
    mask_curr, th_curr = generate_fg_mask(residual_curr_map, residual_k, top_percent)

    if scale != 1.0:
        inv_scale = 1.0 / scale
        mask_prev = cv2.resize(mask_prev, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_curr = cv2.resize(mask_curr, (W_curr, H_curr), interpolation=cv2.INTER_NEAREST)
        residual_prev_map = cv2.resize(residual_prev_map, (W, H), interpolation=cv2.INTER_LINEAR)
        residual_curr_map = cv2.resize(residual_curr_map, (W_curr, H_curr), interpolation=cv2.INTER_LINEAR)
        mkpts0 *= inv_scale
        mkpts_sparse0 *= inv_scale
        mkpts1 *= inv_scale
        mkpts_sparse1 *= inv_scale
        S = np.diag([scale, scale, 1.0])
        F = S.T @ F_scaled @ S
    else:
        F = F_scaled
    return (
        mask_prev,
        mask_curr,
        residual_prev_map,
        residual_curr_map,
        (th_prev, th_curr),
        F,
        (mkpts0, mkpts1),
        (mkpts_sparse0, mkpts_sparse1),
    )


def _overlay_mask(frame: np.ndarray, dynamic_mask: np.ndarray) -> np.ndarray:
    """Tint background regions (mask == 0) in Cyan."""
    if frame.ndim != 3:
        return frame
    
    # Create Cyan color layer (B=255, G=255, R=0)
    color = np.zeros_like(frame)
    color[..., 0] = 255 # B
    color[..., 1] = 255 # G
    color[..., 2] = 0   # R

    # Invert mask: tint background (where mask is 0)
    binary = (dynamic_mask == 0).astype(np.uint8)
    
    # Blend frame with color: 0.6 frame + 0.4 color
    tinted = cv2.addWeighted(frame, 0.6, color, 0.4, 0)
    
    overlay = frame.copy()
    # Apply tint only to masked regions (background)
    overlay[binary.astype(bool)] = tinted[binary.astype(bool)]
    
    return overlay


def _clear_border(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return mask
    trimmed = mask.copy()
    trimmed[:width, :] = 0
    trimmed[-width:, :] = 0
    trimmed[:, :width] = 0
    trimmed[:, -width:] = 0
    return trimmed


def _suppress_noise(mask: np.ndarray, min_ratio: float, min_pixels: int) -> np.ndarray:
    area = float(np.count_nonzero(mask))
    if area == 0:
        return mask
    if min_pixels > 0 and area < float(min_pixels):
        return np.zeros_like(mask)
    if min_ratio > 0 and area / mask.size < min_ratio:
        return np.zeros_like(mask)
    return mask


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if num_labels <= 1:
        return mask
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_pixels:
            filtered[labels == label] = 1
    return filtered


def _warp_prev_frame(prev_frame: np.ndarray, curr_frame: np.ndarray, moving_mask: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    """Warp previous frame to current frame coordinates using optical flow."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
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
    warped = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if moving_mask is not None:
        background = (1 - moving_mask.astype(np.float32))[..., None]
        warped = (warped * background + curr_frame * (1 - background)).astype(prev_frame.dtype)
    return warped, flow


def _flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow to RGB image using HSV color space."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


class HybridMotionEstimator(BaseMotionEstimator):
    """Stub for SuperGlue + fundamental matrix workflow."""

    name = "hybrid_motion"

    def __init__(
        self,
        device: str | None = None,
        residual_k: float = 2.5,
        blur_ksize: int = 5,
        max_input_edge: int | None = 1024,
        border_clear: int = 8,
        min_mask_ratio: float = 1,
        min_mask_pixels: int = 10,
        min_component_pixels: int = 800,
        top_percent: float | None = None,
        **_: Any,
    ) -> None:
        self.device = device
        self.residual_k = residual_k
        self.blur_ksize = blur_ksize
        self.max_input_edge = max_input_edge
        self.border_clear = border_clear
        self.min_mask_ratio = min_mask_ratio
        self.min_mask_pixels = min_mask_pixels
        self.min_component_pixels = min_component_pixels
        self.top_percent = top_percent
        self.models: Dict[str, Any] | None = None
        self.ready = False

    def setup(self) -> None:
        self.models = init_models(self.device)
        self.loaded = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Fuse learned matching with geometric verification."""
        if not self.loaded:
            raise RuntimeError("HybridMotionEstimator.setup() must be called first.")

        # Extract frames from frame_records
        curr_record = frame_records[frame_idx]
        curr_frame = curr_record.origin_image
        
        # Get previous frame
        if frame_idx == 0 or frame_idx - 1 not in frame_records:
            # First frame, return empty motion
            motion_outputs = MotionOutputs(
                background_mask=None,
                motion_vectors=None,
                fundamental_matrix=None,
            )
            return StageResult(data={"motion": motion_outputs}, visualization_assets={})
        
        prev_record = frame_records[frame_idx - 1]
        prev_frame = prev_record.origin_image

        (
            dynamic_prev,
            dynamic_curr,
            residual_prev,
            residual_curr,
            _,
            F,
            loftr_tracks,
            sparse_tracks,
        ) = segment_foreground_lightglue_loftr(
            prev_frame,
            curr_frame,
            self.models,
            self.residual_k,
            self.blur_ksize,
            self.max_input_edge,
            self.top_percent,
        )

        moving_prev = _clear_border(dynamic_prev.astype(np.uint8), self.border_clear)
        moving_curr = _clear_border(dynamic_curr.astype(np.uint8), self.border_clear)
        moving_prev = _remove_small_components(moving_prev, self.min_component_pixels)
        moving_curr = _remove_small_components(moving_curr, self.min_component_pixels)
        moving_prev = _suppress_noise(moving_prev, self.min_mask_ratio, self.min_mask_pixels)
        moving_curr = _suppress_noise(moving_curr, self.min_mask_ratio, self.min_mask_pixels)
        warped_prev, flow = _warp_prev_frame(prev_frame, curr_frame, moving_prev)

        motion_outputs = MotionOutputs(
            background_mask=moving_curr,
            motion_vectors=flow,
            fundamental_matrix=F,
        )
        
        vis_assets = {
            "mask_curr": cv2.cvtColor(_overlay_mask(curr_frame, moving_curr), cv2.COLOR_BGR2RGB),
            "warped_prev": warped_prev,
            "motion_vector": _flow_to_color(flow),
        }
        
        return StageResult(data={"motion": motion_outputs}, visualization_assets=vis_assets)

    def teardown(self) -> None:
        self.loaded = False
