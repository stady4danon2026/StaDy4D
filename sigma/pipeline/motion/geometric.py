"""Geometric motion estimation via SIFT/ORB + Optical Flow."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import time
import os

from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator


##########################################
# 1. Sparse F via SIFT/ORB
##########################################


def compute_F_geometric(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Fundamental matrix using SIFT keypoints and RANSAC."""
    # Initialize SIFT detector
    # Note: SIFT is patent-free in opencv-contrib-python >= 4.3
    sift = cv2.SIFT_create()

    # Detect and compute
    kp1, des1 = sift.detectAndCompute(img1_bgr, None)
    kp2, des2 = sift.detectAndCompute(img2_bgr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("Not enough SIFT features found.")

    # Match features using FLANN or BFMatcher
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    if len(pts1) < 8:
        raise RuntimeError("Not enough good matches after ratio test.")

    # Compute F
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)

    if F is None or inliers is None:
        raise RuntimeError("Fundamental matrix estimation failed")
    
    inliers = (inliers > 0).ravel()
    return F, pts1[inliers], pts2[inliers]


##########################################
# 2. Dense Flow & Residuals
##########################################


def _compute_flow_residuals(F: np.ndarray, flow: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute epipolar residuals based on dense optical flow."""
    H, W = shape
    # Grid of coordinates
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    
    # Points in frame 1: (x, y)
    pts1 = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    
    # Points in frame 2: (x + u, y + v)
    flow_flat = flow.reshape(-1, 2)
    pts2 = pts1 + flow_flat

    # Homogeneous coordinates
    x1 = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
    x2 = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)

    # Epipolar lines
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T

    # Symmetric epipolar distance
    # d(x2, Fx1) + d(x1, Ftx2)
    # Sampson approximation is often used, but symmetric distance is fine too.
    # Using the same formula as hybrid.py for consistency:
    # residual_curr = |x2^T F x1| / sqrt((Fx1)_1^2 + (Fx1)_2^2)
    
    # Numerator: x2^T F x1
    # We can compute dot product of x2 and Fx1
    num = np.abs(np.sum(x2 * Fx1, axis=1))
    
    den1 = np.sqrt(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2) + 1e-6
    den2 = np.sqrt(Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2) + 1e-6
    
    residual_curr = (num / den1).reshape(H, W).astype(np.float32)
    residual_prev = (num / den2).reshape(H, W).astype(np.float32)
    
    return residual_prev, residual_curr


##########################################
# 3. Helpers (Copied/Adapted from hybrid.py)
##########################################


def generate_fg_mask(residual: np.ndarray, top_percent: float = 0.05):
    # med = np.median(residual)
    # mad = np.median(np.abs(residual - med)) + 1e-6
    # th = med + k * mad
    # mask = (residual > th).astype(np.uint8)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # return mask, th
    flat = residual.reshape(-1)
    th = np.quantile(flat, 1.0 - top_percent)
    mask = (residual < th).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, th


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
    
    # Convert BGR to RGB for visualization
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


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


def postprocess_dynamic_mask(mask: np.ndarray,
                             close_ksize: int = 11,
                             min_hole_pixels: int = 1000) -> np.ndarray:
    """Fill small holes and smooth foreground regions."""
    mask = mask.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    inv = 1 - closed
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv)
    filled = closed.copy()
    h, w = mask.shape

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        x0 = stats[lab, cv2.CC_STAT_LEFT]
        y0 = stats[lab, cv2.CC_STAT_TOP]
        x1 = x0 + stats[lab, cv2.CC_STAT_WIDTH]
        y1 = y0 + stats[lab, cv2.CC_STAT_HEIGHT]

        touches_border = (x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h)

        if (not touches_border) and area <= min_hole_pixels:
            filled[labels == lab] = 1

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k_small)

    return filled.astype(np.uint8)


def _warp_prev_frame(prev_frame: np.ndarray, curr_frame: np.ndarray, moving_mask: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
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
    warped = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # User requested "static scene warp", so we do NOT paste the current frame's foreground on top.
    # if moving_mask is not None:
    #     background = (1 - moving_mask.astype(np.float32))[..., None]
    #     warped = (warped * background + curr_frame * (1 - background)).astype(prev_frame.dtype)
    return warped, flow


def _flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow to RGB image using HSV color space."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def squares_from_mask(
    mask: np.ndarray,
    min_size: int = 16,          # Minimum square side length (pixels)
    coverage: float = 0.95,      # Target coverage ratio
    max_squares: int = 512,      # Maximum number of squares
    pad: int = 0,                # Extra padding around each square
    inside_ratio: float = 0.9    # Minimum ratio inside remaining mask
) -> np.ndarray:
    """
    Approximate arbitrary binary mask with squares using distance transform.
    Returns: rect_mask (0/255 uint8)
    """
    assert mask.ndim == 2, "mask must be HxW"
    H, W = mask.shape
    # Convert to 0/1
    work = (mask > 0).astype(np.uint8)
    total = int(work.sum())
    if total == 0:
        return np.zeros_like(mask, np.uint8)

    squares = []
    filled = 0

    while work.any():
        # Chessboard distance (suitable for inscribed squares)
        dt = cv2.distanceTransform(work, cv2.DIST_C, 3)
        r = int(dt.max())  # Radius (actual side ~2r+1)
        if r * 2 + 1 < min_size:
            break

        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        half = r
        x1 = max(0, x - half - pad)
        y1 = max(0, y - half - pad)
        x2 = min(W, x + half + 1 + pad)
        y2 = min(H, y + half + 1 + pad)

        # Shrink if extending too far outside
        tile = work[y1:y2, x1:x2]
        if tile.size == 0:
            work[y, x] = 0
            continue

        while tile.mean() < inside_ratio and (x2 - x1) >= min_size + 2 and (y2 - y1) >= min_size + 2:
            x1 += 1
            y1 += 1
            x2 -= 1
            y2 -= 1
            tile = work[y1:y2, x1:x2]

        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            # Clear peak to avoid infinite loop
            work[y, x] = 0
            continue

        squares.append((x1, y1, x2, y2))
        work[y1:y2, x1:x2] = 0  # Remove covered area
        filled = total - int(work.sum())

        if filled / total >= coverage or len(squares) >= max_squares:
            break

    # Rasterize output
    rect_mask = np.zeros_like(mask, np.uint8)
    for x1, y1, x2, y2 in squares:
        rect_mask[y1:y2, x1:x2] = 1

    return rect_mask


##########################################
# 4. Main Pipeline
##########################################


def segment_foreground_geometric(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    top_percent: float = 0.05,
    blur_ksize: int = 5,
    max_input_edge: int | None = None,
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

    # 1) Sparse F from SIFT
    F_scaled, mkpts_sparse0, mkpts_sparse1 = compute_F_geometric(proc_img1, proc_img2)
    
    # 2) Dense flow: curr -> prev
    prev_gray = cv2.cvtColor(proc_img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(proc_img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,          # Changed from prev_gray
        curr_gray,          # Changed from curr_gray
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    
    # 3) Warp prev to current frame coordinates
    # Flow is prev->curr, so to warp prev to curr, we need to find where each curr pixel came from
    # For each curr pixel (x,y), we look at prev pixel (x-fx, y-fy)
    h, w = curr_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Subtract flow to find source position in prev frame
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    warped_prev = cv2.remap(
        proc_img1,                    # prev image
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    
    # 4) Photometric difference on current frame coords
    diff = cv2.absdiff(proc_img2, warped_prev)   # curr - warped_prev
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Optional blur
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        diff_gray = cv2.GaussianBlur(diff_gray, (k, k), 0)

    # 5) Threshold photometric difference to get mask_curr
    flat_diff = diff_gray.reshape(-1)
    th_diff = np.quantile(flat_diff, 1.0 - top_percent)
    mask_curr = (diff_gray > th_diff).astype(np.uint8)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask_curr = cv2.morphologyEx(mask_curr, cv2.MORPH_CLOSE, kernel)
    mask_curr = cv2.morphologyEx(mask_curr, cv2.MORPH_OPEN, kernel)
    
    # mask_prev is not needed
    mask_prev = np.zeros_like(mask_curr)
    
    # Residual maps for visualization (reuse photometric difference)
    residual_prev_map = np.zeros_like(diff_gray, dtype=np.float32)
    residual_curr_map = diff_gray.astype(np.float32)
    
    th_prev = 0.0
    th_curr = float(th_diff)

    if scale != 1.0:
        inv_scale = 1.0 / scale
        mask_prev = cv2.resize(mask_prev, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_curr = cv2.resize(mask_curr, (W_curr, H_curr), interpolation=cv2.INTER_NEAREST)
        residual_prev_map = cv2.resize(residual_prev_map, (W, H), interpolation=cv2.INTER_LINEAR)
        residual_curr_map = cv2.resize(residual_curr_map, (W_curr, H_curr), interpolation=cv2.INTER_LINEAR)
        # Sparse points need scaling
        mkpts_sparse0 *= inv_scale
        mkpts_sparse1 *= inv_scale
        
        S = np.diag([scale, scale, 1.0])
        F = S.T @ F_scaled @ S
    else:
        F = F_scaled

    # Dynamic Flow Visualization
    # Resize flow to original resolution
    if scale != 1.0:
        flow_full = cv2.resize(flow, (W_curr, H_curr), interpolation=cv2.INTER_LINEAR)
        flow_full *= (1.0 / scale)
    else:
        flow_full = flow

    vis_img = img2_bgr.copy()
    
    # Draw flow vectors for masked pixels
    step = 3
    y_grid, x_grid = np.mgrid[0:H_curr:step, 0:W_curr:step]
    
    # Filter by mask
    mask_down = mask_curr[0:H_curr:step, 0:W_curr:step]
    y_grid = y_grid[mask_down > 0]
    x_grid = x_grid[mask_down > 0]
    
    if len(x_grid) > 0:
        fx = flow_full[0:H_curr:step, 0:W_curr:step, 0][mask_down > 0]
        fy = flow_full[0:H_curr:step, 0:W_curr:step, 1][mask_down > 0]
        
        # Color by magnitude
        mag_viz = np.sqrt(fx**2 + fy**2)
        mag_norm = np.clip(mag_viz * 5, 0, 255).astype(np.uint8)
        
        for x, y, u, v, m in zip(x_grid, y_grid, fx, fy, mag_norm):
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + u), int(y + v)
            
            # Color: Green to Red based on magnitude
            color = (0, 255 - int(m), int(m))
            
            # Draw line
            cv2.line(vis_img, (x1, y1), (x2, y2), color, 1)
            # Draw start point
            cv2.circle(vis_img, (x1, y1), 1, color, -1)
            # Draw end point
            cv2.circle(vis_img, (x2, y2), 1, color, -1)

    # Save visualization
    timestamp = int(time.time() * 1000)
    vis_dir = "outputs/visuals/motion"
    os.makedirs(vis_dir, exist_ok=True)
    cv2.imwrite(os.path.join(vis_dir, f"flow_dynamic_{timestamp}.png"), vis_img)

    # Dense matches return empty
    mkpts0 = np.zeros((0, 2), dtype=np.float32)
    mkpts1 = np.zeros((0, 2), dtype=np.float32)

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


class GeometricMotionEstimator(BaseMotionEstimator):
    """Geometric motion stage powered by SIFT + Farneback Flow."""

    name = "geometric_motion"

    def __init__(
        self,
        top_percent: float = 0.05,
        blur_ksize: int = 5,
        max_input_edge: int | None = 1024,
        border_clear: int = 8,
        min_mask_ratio: float = 0.0,
        min_mask_pixels: int = 10,
        min_component_pixels: int = 100,
        rectangle_mask_padding: int = 10,
        **_: Any,
    ) -> None:
        self.top_percent = top_percent
        self.blur_ksize = blur_ksize
        self.max_input_edge = max_input_edge
        self.border_clear = border_clear
        self.min_mask_ratio = min_mask_ratio
        self.min_mask_pixels = min_mask_pixels
        self.min_component_pixels = min_component_pixels
        self.rectangle_mask_padding = rectangle_mask_padding
        self.ready = False

    def setup(self) -> None:
        # No heavy models to load
        self.ready = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        if not self.ready:
            raise RuntimeError("Call setup() before process().")
        
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
            dense_tracks, # Empty for geometric
            sparse_tracks,
        ) = segment_foreground_geometric(
            prev_frame,
            curr_frame,
            self.top_percent,
            self.blur_ksize,
            self.max_input_edge,
        )

        moving_prev = dynamic_prev.astype(np.uint8)  # Skip border clearing
        moving_curr = dynamic_curr.astype(np.uint8)  # Skip border clearing
        moving_prev = _remove_small_components(moving_prev, self.min_component_pixels)
        moving_curr = _remove_small_components(moving_curr, self.min_component_pixels)
        moving_prev = _suppress_noise(moving_prev, self.min_mask_ratio, self.min_mask_pixels)
        moving_curr = _suppress_noise(moving_curr, self.min_mask_ratio, self.min_mask_pixels)
        
        moving_prev = postprocess_dynamic_mask(moving_prev)
        moving_curr = postprocess_dynamic_mask(moving_curr)
        
        # Save original mask before square conversion
        original_mask_curr = moving_curr.copy()
        
        # Convert masks to squares using distance transform
        moving_curr_rect = squares_from_mask(
            moving_curr,
            min_size=16,           # Minimum square size
            coverage=0.95,         # Cover 95% of original mask
            max_squares=512,       # Max number of squares
            pad=self.rectangle_mask_padding,  # Padding around squares
            inside_ratio=0.85      # Must be 85% inside mask
        )
        
        # Use square mask for inpainting (convert to 0/1)
        moving_curr = (moving_curr_rect > 0).astype(np.uint8)
        
        # Re-compute flow for warping (or reuse if we passed it out, but for simplicity and matching hybrid structure, we call _warp_prev_frame)
        warped_prev, flow = _warp_prev_frame(prev_frame, curr_frame, moving_prev)

        mask_curr_3d = moving_curr.astype(np.float32)[..., None]
        static_curr = (warped_prev.astype(np.float32) * mask_curr_3d + curr_frame.astype(np.float32) * (1.0 - mask_curr_3d)).astype(
            curr_frame.dtype
        )

        # Invert mask for inpainting: mask=1 means area to inpaint (moving objects to remove)
        moving_curr_inverted = 1 - moving_curr
        
        motion_payload = {
            "curr_mask": moving_curr_inverted,
            "warped_prev_to_curr": warped_prev,
            "optical_flow": flow,
            "static_curr": static_curr,
            "fundamental_matrix": F,
        }

        visualization_assets = {
            "original_mask_curr": original_mask_curr * 255,      # Original binary mask
            "original_overlay_curr": _overlay_mask(curr_frame, original_mask_curr),  # Original mask on image
            "rectangular_mask_curr": moving_curr * 255,          # Rectangular mask
            "rectangular_overlay_curr": _overlay_mask(curr_frame, moving_curr),  # Rectangle mask on image
            "warped_prev_to_curr": warped_prev,
            "motion_vector": _flow_to_color(flow),
        }

        return StageResult(data={"motion": motion_payload}, visualization_assets=visualization_assets)

    def teardown(self) -> None:
        self.ready = False
