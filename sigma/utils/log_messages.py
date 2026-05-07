"""Beautiful log message formatting utilities."""

from __future__ import annotations

import logging
from typing import Any


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a section header with visual separation."""
    logger.info(f"[bold cyan]{'=' * 60}[/bold cyan]")
    logger.info(f"[bold cyan]{title}[/bold cyan]")
    logger.info(f"[bold cyan]{'=' * 60}[/bold cyan]")


def log_step(logger: logging.Logger, step: str, detail: str | None = None) -> None:
    """Log a processing step with optional detail."""
    if detail:
        logger.info(f"[cyan]▶[/cyan] {step}: [white]{detail}[/white]")
    else:
        logger.info(f"[cyan]▶[/cyan] {step}")


def log_success(logger: logging.Logger, message: str) -> None:
    """Log a success message."""
    logger.info(f"[green]✓[/green] {message}")


def log_progress(logger: logging.Logger, current: int, total: int, description: str = "") -> None:
    """Log progress information."""
    percentage = (current / total * 100) if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_length - filled)
    msg = f"[cyan]{bar}[/cyan] {current}/{total} ({percentage:.1f}%)"
    if description:
        msg = f"{description}: {msg}"
    logger.info(msg)


def log_config(logger: logging.Logger, key: str, value: Any) -> None:
    """Log a configuration parameter."""
    logger.info(f"[dim]  {key}:[/dim] [white]{value}[/white]")


def log_warning_box(logger: logging.Logger, message: str) -> None:
    """Log a warning in a box for visibility."""
    logger.warning(f"[yellow]⚠[/yellow]  {message}")


def log_error_box(logger: logging.Logger, message: str) -> None:
    """Log an error in a box for visibility."""
    logger.error(f"[red]✗[/red] {message}")
