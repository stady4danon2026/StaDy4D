"""Logging helpers."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from sigma.utils.io import ensure_dir

# Custom theme for beautiful logging
SIGMA_THEME = Theme({
    "logging.level.info": "cyan",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
    "logging.level.debug": "dim cyan",
})


def setup_logging(level: str = "INFO", save_dir: str | None = None) -> None:
    """Configure python logging with optional file handler."""
    # Clear any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a beautiful console with custom theme
    console = Console(theme=SIGMA_THEME)

    # Configure Rich logging with beautiful format
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        show_level=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        omit_repeated_times=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[rich_handler],
        format="%(message)s",
        force=True,  # Force reconfiguration to override any existing handlers
    )

    # Suppress noisy third-party library logs
    third_party_loggers = [
        "dinov2",
        "matplotlib",
        "PIL",
        "torch",
        "transformers",
        "transformers.utils.loading_report",
        "huggingface_hub",
        "diffusers",
        "peft",
        "filelock",
        "accelerate",
    ]
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Suppress HuggingFace/diffusers print-level noise via env vars
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Suppress Python warnings from third-party libs
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
    warnings.filterwarnings("ignore", message=".*fast processor.*")
    warnings.filterwarnings("ignore", message=".*model of type sam2_video.*")
    warnings.filterwarnings("ignore", message=".*No LoRA keys.*")
    warnings.filterwarnings("ignore", message=".*not expected and will be ignored.*")
    warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

    # File handler still gets detailed logs
    if save_dir:
        log_dir = Path(save_dir)
        ensure_dir(log_dir)
        file_handler = logging.FileHandler(log_dir / "sigma.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
