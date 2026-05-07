"""Structured representation of the per-frame pipeline artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class FrameIORecord:
    """Bundle every tensor we track for a single frame index."""

    frame_idx: int
    origin_image: Any | None = None
    inpainted_image: Any | None = None
    warped_image: Any | None = None
    mask: Any | None = None
    depth: Any | None = None
    extrinsic: Any | None = None
    intrinsic: Any | None = None
    motion_vector: Any | None = None
    fundamental_matrix: Any | None = None
    # Cached Pi3 outputs (populated by whichever stage runs Pi3 first; lets
    # later stages skip redundant forwards). One slot per Pi3 output type.
    pi3_local_points: Any | None = None
    pi3_world_points: Any | None = None
    pi3_camera_pose: Any | None = None
    pi3_conf_logits: Any | None = None
    pi3_metric: Any | None = None
    pi3_conf_features: Any | None = None  # conf_decoder hidden states (post-special-token slice)
    pi3_patch_h: int | None = None
    pi3_patch_w: int | None = None
    pi3_input_h: int | None = None  # Pi3-input image H (after _compute_target_size resize)
    pi3_input_w: int | None = None

    def to_payload(self) -> Dict[str, Any]:
        """Return a plain dictionary representation."""
        return asdict(self)
