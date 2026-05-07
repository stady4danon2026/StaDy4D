"""Pipeline orchestration logic for SIGMA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from sigma.data import FrameDataModule, FrameSequenceConfig
from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.inpainting import GeometricInpainter
from sigma.pipeline.inpainting.base_inpainting import InpaintingOutputs
from sigma.pipeline.motion import GeometricMotionEstimator
from sigma.pipeline.motion.base_motion import MotionOutputs
from sigma.pipeline.reconstruction import VGGTOfflineReconstructor
from sigma.pipeline.reconstruction.base_recon import ReconstructionOutputs
from sigma.utils import ensure_dir, setup_logging
from sigma.utils.fancy_progress import FancyProgressDisplay
from sigma.utils.log_messages import log_progress, log_section, log_step
from sigma.visualization.exporters import dump_stage_assets, finalize_reconstruction_export, save_reconstruction_summary

LOGGER = logging.getLogger(__name__)
DEFAULT_STAGE_ORDER: Sequence[str] = ("preprocess", "motion", "inpainting", "reconstruction")


@dataclass
class RunnerConfig:
    """Highlighted dataclass representation of the ``run`` Hydra section."""

    name: str
    timestamp: str
    output_dir: Path
    visualization_dir: Path
    save_every_n_frames: int
    demo_video: str
    test_modules: List[str] = field(default_factory=list)
    mock_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "RunnerConfig":
        """Build a :class:`RunnerConfig` from an OmegaConf node."""
        modules = list(cfg.get("test_modules") or [])
        mock_inputs = OmegaConf.to_container(cfg.get("mock_inputs", {}), resolve=True) or {}
        return cls(
            name=str(cfg.name),
            timestamp=str(cfg.timestamp),
            output_dir=Path(cfg.output_dir),
            visualization_dir=Path(cfg.visualization_dir),
            save_every_n_frames=int(cfg.save_every_n_frames),
            demo_video=str(cfg.demo_video),
            test_modules=modules,
            mock_inputs=mock_inputs,
        )


class PipelineRunner:
    """Create each stage and execute the pipeline per-frame."""

    def __init__(self, cfg: DictConfig) -> None:
        """Store configuration, resolve dataclasses, and prepare stage objects."""
        self.cfg = cfg
        self.run_cfg = RunnerConfig.from_cfg(cfg.run)
        self.output_dir = self.run_cfg.output_dir
        self.vis_dir = self.run_cfg.visualization_dir

        self.save_gifs = cfg.visualization.get("save_gifs", True)
        self.save_ply = cfg.visualization.get("save_ply", True)

        setup_logging(level=cfg.logging.level, save_dir=cfg.logging.save_dir)

        self.data_module = self._build_data_module(cfg)
        self.motion_stage = self._build_stage(cfg.pipeline.motion, default_cls=GeometricMotionEstimator)
        self.inpainter = self._build_stage(cfg.pipeline.inpainting, default_cls=GeometricInpainter)
        self.reconstructor: VGGTOfflineReconstructor = self._build_stage(cfg.pipeline.reconstruction, default_cls=VGGTOfflineReconstructor)
        
        self.preprocessor = None
        if "preprocess" in cfg.pipeline:
             # We assume the implementation is specified in the config
             self.preprocessor = self._build_stage(cfg.pipeline.preprocess, default_cls=None)

        self.stages = {
            "motion": self.motion_stage,
            "inpainting": self.inpainter,
            "reconstruction": self.reconstructor,
        }
        if self.preprocessor:
            self.stages["preprocess"] = self.preprocessor
        self.active_modules = self._resolve_active_modules()
        self.frame_records: Dict[int, FrameIORecord] = {}

    def _build_data_module(self, cfg: DictConfig) -> FrameDataModule:
        """Instantiate the frame iterator datamodule."""
        dm_cfg = FrameSequenceConfig(
            frames_dir=Path(cfg.data.frames_dir),
            frame_stride=cfg.data.frame_stride,
            max_frames=cfg.data.max_frames,
            data_format=cfg.data.get("data_format", "dynamic"),
            camera=cfg.data.get("camera", ""),
            load_metadata=cfg.data.get("load_metadata", True),
        )
        return FrameDataModule(dm_cfg)

    def _build_stage(self, stage_cfg: DictConfig, default_cls: Any):
        """Hydra-style instantiation for a stage, falling back to defaults."""
        implementation = stage_cfg.get("implementation")
        params = {k: v for k, v in stage_cfg.items() if k not in {"name", "implementation"}}
        cls = default_cls
        if implementation:
            try:
                cls = get_class(implementation)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ImportError(f"Failed to import '{implementation}'") from exc
        if cls is None or cls is type(None):
            return None
        return cls(**params)

    def _resolve_active_modules(self) -> Sequence[str]:
        """Decide which modules to run for quick testing."""
        if not self.run_cfg.test_modules:
            return DEFAULT_STAGE_ORDER

        requested = []
        for module in self.run_cfg.test_modules:
            if module not in DEFAULT_STAGE_ORDER:
                raise ValueError(f"Unknown module '{module}'. Expected one of {DEFAULT_STAGE_ORDER}.")
            requested.append(module)
        return tuple(requested)

    def _get_stages_config(self) -> dict[str, str]:
        """Get configuration for fancy progress display.

        Returns:
            Dictionary mapping stage names to their method names
        """
        stages_config = {}
        for stage_name in DEFAULT_STAGE_ORDER:
            if stage_name in self.active_modules:
                stage_obj = self.stages.get(stage_name)
                if stage_obj:
                    # Extract method name from class name
                    method_name = self._extract_method_name(stage_obj)
                    stages_config[stage_name] = method_name
        return stages_config

    def _extract_method_name(self, stage_obj: Any) -> str:
        """Extract a readable method name from a stage object.

        Args:
            stage_obj: The stage object instance

        Returns:
            Human-readable method name
        """
        class_name = stage_obj.__class__.__name__.lower()

        # Map class names to readable method names
        if "geometric" in class_name:
            return "geometric"
        elif "learned" in class_name:
            return "learned"
        elif "hybrid" in class_name:
            return "hybrid"
        elif "vggt" in class_name:
            return "vggt"
        elif "dust3r" in class_name:
            return "dust3r"
        elif "megasam" in class_name:
            return "megasam"
        else:
            # Default: use the class name without common suffixes
            for suffix in ["estimator", "inpainter", "reconstructor"]:
                if suffix in class_name:
                    return class_name.replace(suffix, "").strip()
            return class_name

    def _mock_data_for_stage(self, stage_name: str) -> Dict[str, Any]:
        """Return mock data blocks for a stage when upstream modules are disabled."""
        stage_mock = dict(self.run_cfg.mock_inputs.get(stage_name, {}))
        if stage_name == "motion" and "motion" in stage_mock:
            motion_payload = stage_mock["motion"] or {}
            stage_mock["motion"] = MotionOutputs(
                background_mask=motion_payload.get("background_mask"),
                motion_vectors=motion_payload.get("motion_vectors"),
                fundamental_matrix=motion_payload.get("fundamental_matrix"),
                warped_image=motion_payload.get("warped_image"),
            )
        if stage_name == "inpainting" and "inpainting" in stage_mock:
            inpaint_payload = stage_mock["inpainting"] or {}
            stage_mock["inpainting"] = InpaintingOutputs(
                inpainted_image=inpaint_payload.get("inpainted_image"),
                confidence_map=inpaint_payload.get("confidence_map"),
            )
        if stage_name == "reconstruction" and "reconstruction" in stage_mock:
            recon_payload = stage_mock["reconstruction"] or {}
            stage_mock["reconstruction"] = ReconstructionOutputs(
                current_scene=recon_payload.get("current_scene"),
                aggregated_scene=recon_payload.get("aggregated_scene"),
            )
        return stage_mock

    def run(self) -> None:
        """Run the pipeline in phase-based batch mode.

        Instead of processing one frame at a time through all stages, we:

        1. **Load** all frames into memory at once.
        2. **Batch motion** — run detection models on all frames in one forward
           pass (GroundedSAM/SAM3), then per-frame optical flow.
        3. **Batch inpainting** — process all frames (parallel for BlankInpainter,
           sequential for SDXL which needs the previous inpainted frame).
        4. **Batch reconstruction** — deferred reconstructors already batch all
           frames; online reconstructors still run per-frame with a sliding window.
        """
        log_section(LOGGER, "SIGMA Pipeline Starting")
        log_step(LOGGER, f"Run: {self.run_cfg.name} | Timestamp: {self.run_cfg.timestamp}")
        log_step(LOGGER, f"Output directory: {self.output_dir}")
        log_step(LOGGER, "Setting up data module and stages")
        self.data_module.setup()
        for name in self.active_modules:
            self.stages[name].setup()
        ensure_dir(self.output_dir)
        ensure_dir(self.vis_dir)

        # ── Phase 1: Load all frames ─────────────────────────────────────
        all_frames = self.data_module.load_all_frames()
        total_frames = len(all_frames)
        log_step(LOGGER, f"Loaded {total_frames} frames into memory")

        for i, frame in enumerate(all_frames):
            self.frame_records[i] = FrameIORecord(frame_idx=i, origin_image=frame)

        # Initialize fancy progress display.
        stages_config = self._get_stages_config()
        fancy_progress = FancyProgressDisplay(
            total_frames=total_frames,
            stages_config=stages_config,
        )

        # ── Phase 2: Preprocess first frame (optional) ───────────────────
        if self.preprocessor and "preprocess" in self.active_modules:
            log_step(LOGGER, "Preprocessing first frame")
            first_frame = all_frames[0]

            background_mask, preprocess_viz_asset = self.preprocessor.process(first_frame)

            self.frame_records[0].mask = background_mask

            # Inpaint frame 0 so downstream stages see it as done.
            inpaint_result = self.inpainter.process(self.frame_records, 0)
            self._update_frame_record_from_inpainting(
                self.frame_records[0], inpaint_result.data.get("inpainting")
            )

            vis_assets = {
                "origin": first_frame,
                "mask": background_mask,
                "inpainted": self.frame_records[0].inpainted_image,
            }
            vis_assets.update(preprocess_viz_asset)
            self._handle_visualizations("preprocess", vis_assets, 0)

        # ── Phase 3: Batch motion estimation ──────────────────────────────
        if "motion" in self.active_modules:
            log_step(LOGGER, f"Running batch motion estimation ({total_frames} frames)")
            fancy_progress.update(0, "motion")

            def _motion_progress(current: int, total: int) -> None:
                fancy_progress.update(current, "motion")

            motion_results = self.motion_stage.process_batch(
                self.frame_records, progress_fn=_motion_progress,
            )
            for frame_idx in sorted(motion_results):
                # Skip frame 0 if preprocessor already set its mask
                if frame_idx == 0 and self.frame_records[0].mask is not None:
                    continue
                result = motion_results[frame_idx]
                self._update_frame_record_from_motion(
                    self.frame_records[frame_idx], result.data.get("motion")
                )
                self._handle_visualizations(
                    "motion", result.visualization_assets, frame_idx
                )

        # ── Phase 4: Batch inpainting ─────────────────────────────────────
        if "inpainting" in self.active_modules:
            log_step(LOGGER, f"Running batch inpainting ({total_frames} frames)")
            fancy_progress.update(0, "inpainting")

            def _inpaint_progress(current: int, total: int) -> None:
                fancy_progress.update(current, "inpainting")

            inpaint_results = self.inpainter.process_batch(
                self.frame_records, progress_fn=_inpaint_progress,
            )
            for frame_idx in sorted(inpaint_results):
                result = inpaint_results[frame_idx]
                self._update_frame_record_from_inpainting(
                    self.frame_records[frame_idx], result.data.get("inpainting")
                )
                self._handle_visualizations(
                    "inpainting", result.visualization_assets, frame_idx
                )

        # ── Phase 5: Reconstruction ───────────────────────────────────────
        last_frame_size: tuple | None = None
        last_recon_result = None

        if "reconstruction" in self.active_modules:
            fancy_progress.update(0, "reconstruction")

            if self.reconstructor.run_deferred:
                log_step(LOGGER, "Running deferred reconstruction (batch)")
                per_frame_results = self.reconstructor.finalize(self.frame_records)
                for frame_idx, recon_result in per_frame_results:
                    self._handle_visualizations(
                        "reconstruction", recon_result.visualization_assets, frame_idx
                    )
                    last_frame_size = recon_result.data["frame_size"]
                    last_recon_result = recon_result
            else:
                log_step(LOGGER, "Running online reconstruction (per-frame)")
                for frame_idx in sorted(self.frame_records.keys()):
                    fancy_progress.update(frame_idx + 1, "reconstruction")
                    recon_result = self.reconstructor.process(
                        self.frame_records, frame_idx
                    )
                    self._handle_visualizations(
                        "reconstruction", recon_result.visualization_assets, frame_idx
                    )
                    last_frame_size = recon_result.data["frame_size"]
                    last_recon_result = recon_result

            fancy_progress.update(total_frames, "reconstruction")

        # ── Finalize: metadata, PLY, GIF ──────────────────────────────────
        if "reconstruction" in self.active_modules and last_frame_size is not None:
            finalize_reconstruction_export(
                self.vis_dir,
                total_frames,
                last_frame_size,
                fps=30,
                reconstruction_method=getattr(self.reconstructor, "name", None),
            )
            if last_recon_result is not None:
                vis = last_recon_result.visualization_assets
                save_reconstruction_summary(
                    self.vis_dir,
                    global_points=vis.get("global_point_cloud"),
                    global_colors=vis.get("global_point_color"),
                    save_ply=self.save_ply,
                    save_gifs=self.save_gifs,
                )

        fancy_progress.finish()
        log_section(LOGGER, "Pipeline Complete")
        log_step(LOGGER, "Tearing down stages")
        self.data_module.teardown()
        for name in self.active_modules:
            self.stages[name].teardown()
        
    def _handle_visualizations(self, stage_name: str, assets: Dict[str, Any], frame_idx: int) -> None:
        """Persist visualization assets if a stage produces any."""
        stage_dir = self.vis_dir / stage_name / f"frame_{frame_idx:05d}"
        dump_stage_assets(
            stage_name,
            assets,
            stage_dir,
            save_gifs=self.save_gifs,
            save_ply=self.save_ply,
        )

    def _update_frame_record_from_motion(self, record: FrameIORecord, motion_data: Any) -> None:
        """Fill in per-frame motion fields regardless of the stage implementation."""
        if motion_data is None:
            return
        if isinstance(motion_data, MotionOutputs):
            record.mask = motion_data.background_mask
            record.motion_vector = motion_data.motion_vectors
            record.fundamental_matrix = motion_data.fundamental_matrix
            record.warped_image = record.warped_image
            return
        if isinstance(motion_data, dict):
            record.mask = motion_data.get("curr_mask", record.mask)
            record.warped_image = motion_data.get("warped_prev_to_curr", record.warped_image)
            record.motion_vector = motion_data.get("optical_flow", record.motion_vector)
            record.fundamental_matrix = motion_data.get("fundamental_matrix", record.fundamental_matrix)

    def _update_frame_record_from_inpainting(self, record: FrameIORecord, inpainting_data: Any) -> None:
        """Attach the newest inpainted frame onto the record."""
        if isinstance(inpainting_data, InpaintingOutputs):
            record.inpainted_image = inpainting_data.inpainted_image
            return
        if isinstance(inpainting_data, dict):
            record.inpainted_image = inpainting_data.get("inpainted_frame", record.inpainted_image)
            record.warped_image = inpainting_data.get("warped_prev_to_curr", record.warped_image)
