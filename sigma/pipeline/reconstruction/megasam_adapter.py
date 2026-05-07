"""MegaSAM reconstruction placeholder."""

from __future__ import annotations

from typing import Any, Dict

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.reconstruction.aggregator import SceneAggregator, SceneFragment
from sigma.pipeline.reconstruction.base_recon import BaseReconstructor, ReconstructionOutputs


class MegaSAMReconstructor(BaseReconstructor):
    """Stub for MegaSAM reconstruction flows."""

    name = "megasam_reconstructor"

    def __init__(self, checkpoint: str = "megasam-large", min_confidence: float = 0.35, **kwargs: Any) -> None:
        aggregator = kwargs.get("aggregator") or SceneAggregator()
        super().__init__(aggregator=aggregator)
        self.checkpoint = checkpoint
        self.min_confidence = min_confidence
        self.ready = False

    def setup(self) -> None:
        self.ready = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Invoke MegaSAM reconstructor and append the resulting fragment."""
        if not self.ready:
            raise RuntimeError("MegaSAMReconstructor.setup() must run before process().")

        fragment = SceneFragment(points=None, confidence=None)
        self.aggregator.add_fragment(fragment)
        outputs = ReconstructionOutputs(current_scene=None, aggregated_scene=self.aggregator.build_scene())
        vis_assets = {"mesh": None}
        return StageResult(data={"reconstruction": outputs, "frame_idx": frame_idx}, visualization_assets=vis_assets)

    def teardown(self) -> None:
        self.ready = False
