"""Hydra entrypoint for the SIGMA pipeline."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

# Suppress noisy third-party output before any HF/diffusers imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")
warnings.filterwarnings("ignore", message=".*model of type sam2_video.*")
warnings.filterwarnings("ignore", message=".*No LoRA keys.*")
warnings.filterwarnings("ignore", message=".*not expected and will be ignored.*")

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

from sigma.runners.pipeline_runner import PipelineRunner

LOGGER = logging.getLogger(__name__)


def _ensure_output_dirs(cfg: DictConfig) -> None:
    """Create output directories declared in the config, if they do not exist."""
    output_dir = Path(cfg.run.output_dir)
    visualization_dir = Path(cfg.run.visualization_dir)
    logs_dir = Path(cfg.logging.save_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_file = output_dir / "config.yaml"
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    LOGGER.info(f"Saved resolved config to {config_file}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
    """Entry point triggered by Hydra."""
    LOGGER.info("Launching SIGMA pipeline with config")
    _ensure_output_dirs(cfg)

    # Allow `run.runner` to be either:
    # - absent (use the built-in PipelineRunner)
    # - a DictConfig/dict with a `_target_` (Hydra instantiation)
    # - a string import path to a class (e.g. "sigma.runners.pipeline_runner:PipelineRunner")
    runner_cfg = cfg.get("runner", None)
    if runner_cfg is None:
        runner = PipelineRunner(cfg)
    elif isinstance(runner_cfg, (dict, type(cfg))):
        # Hydra-style config with `_target_`
        runner = instantiate(runner_cfg, cfg)
    elif isinstance(runner_cfg, str):
        # String path to a class
        cls = get_class(runner_cfg)
        runner = cls(cfg)
    else:
        raise TypeError("Unsupported type for 'runner' config: %r" % type(runner_cfg))

    runner.run()


from hydra.core.global_hydra import GlobalHydra

if __name__ == "__main__":
    # Add this line to clear the global hydra instance for sam2 import
    GlobalHydra.instance().clear()
    run_pipeline()
