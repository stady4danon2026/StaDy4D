"""Scene aggregation utilities for combining per-frame reconstructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class SceneFragment:
    """Stores a reconstructed point cloud fragment and metadata.

    Attributes:
        points: 3D point cloud (N, 3) in world coordinates.
        confidence: Confidence scores for each point (N,).
        metadata: Additional information about the fragment (timestep, camera, colors, etc.).
    """

    points: Any  # shape (N, 3)
    confidence: Any  # shape (N,)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneAggregator:
    """Aggregate and merge point cloud fragments from multiple views.

    This class maintains a history of scene fragments and can merge them into
    a unified global scene. It supports confidence-based filtering and prioritization
    of high-quality points (e.g., from real observations vs. inpainted regions).

    Args:
        prioritize_real: If True, prioritize points from real (non-inpainted) regions.
        history_size: Maximum number of fragments to keep in memory.
        min_confidence: Minimum confidence threshold for including points in merged scene.
        voxel_size: Optional voxel size for downsampling merged point cloud.
    """

    def __init__(
        self,
        prioritize_real: bool = True,
        history_size: int = 100,
        min_confidence: float = 0.3,
        voxel_size: float | None = None,
    ) -> None:
        self.prioritize_real = prioritize_real
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.voxel_size = voxel_size
        self.fragments: List[SceneFragment] = []

    def add_fragment(self, fragment: SceneFragment) -> None:
        """Add a new scene fragment to the aggregator.

        Args:
            fragment: SceneFragment containing points, confidence, and metadata.
        """
        if fragment.points is not None and len(fragment.points) > 0:
            self.fragments.append(fragment)
            if len(self.fragments) > self.history_size:
                self.fragments.pop(0)

    def build_scene(self) -> Dict[str, Any]:
        """Merge all fragments into a unified global scene.

        Returns:
            Dictionary containing:
                - merged_points: Global point cloud (M, 3)
                - merged_colors: Colors for global points (M, 3)
                - merged_confidence: Confidence for each point (M,)
                - num_fragments: Number of fragments used
                - num_points: Total number of points in merged scene
                - statistics: Additional aggregation statistics
        """
        if not self.fragments:
            return {
                "merged_points": None,
                "merged_colors": None,
                "merged_confidence": None,
                "num_fragments": 0,
                "num_points": 0,
                "statistics": {},
            }

        # Collect all points and metadata
        all_points = []
        all_confidences = []
        all_colors = []

        for fragment in self.fragments:
            if fragment.points is None or len(fragment.points) == 0:
                continue

            points = fragment.points
            confidence = fragment.confidence if fragment.confidence is not None else np.ones(len(points))

            # Filter by confidence threshold
            mask = confidence >= self.min_confidence
            filtered_points = points[mask]
            filtered_confidence = confidence[mask]

            if len(filtered_points) == 0:
                continue

            all_points.append(filtered_points)
            all_confidences.append(filtered_confidence)

            # Extract colors from metadata if available
            colors = fragment.metadata.get("colors")
            if colors is not None and len(colors) == len(points):
                all_colors.append(colors[mask])

        if not all_points:
            return {
                "merged_points": None,
                "merged_colors": None,
                "merged_confidence": None,
                "num_fragments": len(self.fragments),
                "num_points": 0,
                "statistics": {},
            }

        # Merge all points
        merged_points = np.vstack(all_points)
        merged_confidence = np.concatenate(all_confidences)
        merged_colors = np.vstack(all_colors) if all_colors else None

        # Optional: Downsample using voxel grid
        if self.voxel_size is not None and self.voxel_size > 0:
            merged_points, merged_colors, merged_confidence = self._voxel_downsample(
                merged_points, merged_colors, merged_confidence
            )

        # Compute statistics
        statistics = {
            "mean_confidence": float(np.mean(merged_confidence)),
            "std_confidence": float(np.std(merged_confidence)),
            "min_confidence": float(np.min(merged_confidence)),
            "max_confidence": float(np.max(merged_confidence)),
            "point_density": len(merged_points) / len(self.fragments) if self.fragments else 0,
        }

        return {
            "merged_points": merged_points,
            "merged_colors": merged_colors,
            "merged_confidence": merged_confidence,
            "num_fragments": len(self.fragments),
            "num_points": len(merged_points),
            "statistics": statistics,
            "prioritize_real": self.prioritize_real,
        }

    def _voxel_downsample(
        self, points: np.ndarray, colors: np.ndarray | None, confidence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Downsample point cloud using voxel grid filtering.

        Args:
            points: Point cloud (N, 3).
            colors: Colors (N, 3) or None.
            confidence: Confidence values (N,).

        Returns:
            Tuple of (downsampled_points, downsampled_colors, downsampled_confidence).
        """
        if self.voxel_size is None or len(points) == 0:
            return points, colors, confidence

        # Compute voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)

        # Create unique voxel keys
        voxel_keys = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_keys:
                voxel_keys[key] = []
            voxel_keys[key].append(i)

        # Average points within each voxel
        downsampled_points = []
        downsampled_colors = []
        downsampled_confidence = []

        for indices in voxel_keys.values():
            # Average point positions weighted by confidence
            weights = confidence[indices]
            weights = weights / (np.sum(weights) + 1e-8)

            avg_point = np.average(points[indices], axis=0, weights=weights)
            downsampled_points.append(avg_point)

            if colors is not None:
                avg_color = np.average(colors[indices], axis=0, weights=weights)
                downsampled_colors.append(avg_color)

            avg_confidence = np.mean(confidence[indices])
            downsampled_confidence.append(avg_confidence)

        downsampled_points = np.array(downsampled_points)
        downsampled_colors = np.array(downsampled_colors) if downsampled_colors else None
        downsampled_confidence = np.array(downsampled_confidence)

        return downsampled_points, downsampled_colors, downsampled_confidence

    def clear(self) -> None:
        """Clear all stored fragments."""
        self.fragments.clear()

    def get_num_fragments(self) -> int:
        """Get the current number of stored fragments."""
        return len(self.fragments)
