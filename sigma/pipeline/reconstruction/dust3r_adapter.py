"""DUSt3R reconstruction placeholder."""

from __future__ import annotations

from typing import Any, Dict

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.reconstruction.aggregator import SceneAggregator, SceneFragment
from sigma.pipeline.reconstruction.base_recon import BaseReconstructor, ReconstructionOutputs


class DUSt3RReconstructor(BaseReconstructor):
    """Stub for DUSt3R 3D reconstruction."""

    name = "dust3r_reconstructor"

    def __init__(self, checkpoint: str = "dust3r-xs", min_confidence: float = 0.3, **kwargs: Any) -> None:
        aggregator = kwargs.get("aggregator") or SceneAggregator()
        super().__init__(aggregator=aggregator)
        self.checkpoint = checkpoint
        self.min_confidence = min_confidence
        self.initialized = False

    def setup(self) -> None:
        self.initialized = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Run DUSt3R on the inputs and propagate the aggregated scene."""
        if not self.initialized:
            raise RuntimeError("DUSt3RReconstructor.setup() must be invoked before process().")

        fragment = SceneFragment(points=None, confidence=None)
        self.aggregator.add_fragment(fragment)
        outputs = ReconstructionOutputs(current_scene=None, aggregated_scene=self.aggregator.build_scene())
        vis_assets = {"depth": None}
        return StageResult(data={"reconstruction": outputs, "frame_idx": frame_idx}, visualization_assets=vis_assets)

    def teardown(self) -> None:
        self.initialized = False
