"""Fancy terminal progress display for SIGMA pipeline."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Literal

StageType = Literal["motion", "inpainting", "reconstruction"]

# Suppress logging during fancy progress display
LOGGER = logging.getLogger(__name__)


class Color:
    """ANSI color codes."""

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RESET = "\033[0m"


class Style:
    """ANSI style codes."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class BoxChars:
    """Unicode box-drawing characters."""

    # Heavy borders (outer box)
    HEAVY_HORIZONTAL = "═"
    HEAVY_VERTICAL = "║"
    HEAVY_TOP_LEFT = "╔"
    HEAVY_TOP_RIGHT = "╗"
    HEAVY_BOTTOM_LEFT = "╚"
    HEAVY_BOTTOM_RIGHT = "╝"

    # Light borders (stage boxes)
    LIGHT_HORIZONTAL = "─"
    LIGHT_VERTICAL = "│"
    LIGHT_TOP_LEFT = "┌"
    LIGHT_TOP_RIGHT = "┐"
    LIGHT_BOTTOM_LEFT = "└"
    LIGHT_BOTTOM_RIGHT = "┘"

    # Symbols
    ARROW_RIGHT = "→"
    FILLED_BLOCK = "█"
    LIGHT_SHADE = "░"


@dataclass
class PipelineStage:
    """Configuration for a pipeline stage."""

    name: str
    method: str
    active: bool = False


class TerminalRenderer:
    """Handles low-level terminal rendering operations."""

    @staticmethod
    def clear_lines(count: int) -> None:
        """Clear the specified number of lines from terminal.

        Args:
            count: Number of lines to clear
        """
        for _ in range(count):
            sys.stdout.write("\033[F\033[K")  # Move up and clear line

    @staticmethod
    def print_lines(lines: list[str]) -> None:
        """Print lines to terminal with flush.

        Args:
            lines: Lines to print
        """
        output = "\n".join(lines)
        print(output, flush=True)


class ProgressBar:
    """Renders a progress bar."""

    def __init__(self, width: int) -> None:
        """Initialize progress bar.

        Args:
            width: Width of the progress bar in characters
        """
        self.width = width

    def render(self, current: int, total: int) -> str:
        """Render the progress bar.

        Args:
            current: Current progress value
            total: Total value

        Returns:
            Formatted progress bar string with colors
        """
        percentage = self._calculate_percentage(current, total)
        filled_width = int((percentage / 100) * self.width)
        empty_width = self.width - filled_width

        return (
            f"{Color.GREEN}{BoxChars.FILLED_BLOCK * filled_width}"
            f"{Color.GRAY}{BoxChars.LIGHT_SHADE * empty_width}{Color.RESET}"
        )

    @staticmethod
    def _calculate_percentage(current: int, total: int) -> int:
        """Calculate percentage, handling edge cases.

        Args:
            current: Current value
            total: Total value

        Returns:
            Percentage as integer (0-100)
        """
        if total == 0:
            return 0
        return min(100, int((current / total) * 100))


class StageBox:
    """Renders a single pipeline stage box."""

    def __init__(self, width: int) -> None:
        """Initialize stage box.

        Args:
            width: Width of the box in characters
        """
        self.width = width
        self.max_text_width = width - 4  # Account for borders and padding

    def render(self, stage: PipelineStage) -> str:
        """Render a stage box.

        Args:
            stage: Stage configuration

        Returns:
            Formatted stage box string with colors
        """
        color, style = self._get_style(stage.active)
        text = self._format_text(stage)

        return (
            f"{color}{style}"
            f"{BoxChars.LIGHT_TOP_LEFT}{text:^{self.width - 2}}{BoxChars.LIGHT_TOP_RIGHT}"
            f"{Color.RESET}"
        )

    def _format_text(self, stage: PipelineStage) -> str:
        """Format stage text with truncation if needed.

        Args:
            stage: Stage configuration

        Returns:
            Formatted text string
        """
        stage_text = stage.name.capitalize()
        method_text = f"({stage.method})"

        # Truncate stage name if combined text is too long
        if len(stage_text) + len(method_text) + 1 > self.max_text_width:
            stage_text = stage_text[:self.max_text_width - len(method_text) - 4] + "..."

        return f"{stage_text} {method_text}"

    @staticmethod
    def _get_style(active: bool) -> tuple[str, str]:
        """Get color and style based on active state.

        Args:
            active: Whether the stage is currently active

        Returns:
            Tuple of (color, style)
        """
        if active:
            return Color.GREEN, Style.BOLD
        return Color.GRAY, Style.DIM


class PipelineStagesRenderer:
    """Renders the pipeline stages line with data flow arrows."""

    STAGE_ORDER = ["motion", "inpainting", "reconstruction"]
    ARROW_SPACING = 3  # " → "

    def __init__(self, width: int) -> None:
        """Initialize pipeline stages renderer.

        Args:
            width: Available width for the stages line
        """
        self.width = width

    def render(self, stages: dict[str, PipelineStage]) -> str:
        """Render the pipeline stages line.

        Args:
            stages: Dictionary of stage configurations

        Returns:
            Formatted stages line with arrows
        """
        stages_to_show = self._get_ordered_stages(stages)

        if not stages_to_show:
            return " " * self.width

        box_width = self._calculate_box_width(len(stages_to_show))
        stage_boxes = self._render_stage_boxes(stages_to_show, box_width)

        return self._join_with_arrows(stage_boxes)

    def _get_ordered_stages(self, stages: dict[str, PipelineStage]) -> list[PipelineStage]:
        """Get stages in correct order.

        Args:
            stages: Dictionary of stage configurations

        Returns:
            Ordered list of stages
        """
        return [stages[name] for name in self.STAGE_ORDER if name in stages]

    def _calculate_box_width(self, num_stages: int) -> int:
        """Calculate width for each stage box.

        Args:
            num_stages: Number of stages to display

        Returns:
            Width per box
        """
        total_arrow_space = self.ARROW_SPACING * (num_stages - 1)
        available_width = self.width - total_arrow_space
        return available_width // num_stages

    def _render_stage_boxes(self, stages: list[PipelineStage], box_width: int) -> list[str]:
        """Render all stage boxes.

        Args:
            stages: List of stages
            box_width: Width per box

        Returns:
            List of rendered box strings
        """
        renderer = StageBox(box_width)
        return [renderer.render(stage) for stage in stages]

    def _join_with_arrows(self, boxes: list[str]) -> str:
        """Join boxes with arrow symbols.

        Args:
            boxes: List of box strings

        Returns:
            Joined string with arrows
        """
        arrow = f" {Color.YELLOW}{BoxChars.ARROW_RIGHT}{Color.RESET} "
        return arrow.join(boxes)


class BorderRenderer:
    """Renders box borders."""

    def __init__(self, width: int) -> None:
        """Initialize border renderer.

        Args:
            width: Width of the box
        """
        self.width = width

    def top_border(self) -> str:
        """Render top border."""
        return (
            f"{BoxChars.HEAVY_TOP_LEFT}"
            f"{BoxChars.HEAVY_HORIZONTAL * (self.width - 2)}"
            f"{BoxChars.HEAVY_TOP_RIGHT}"
        )

    def bottom_border(self) -> str:
        """Render bottom border."""
        return (
            f"{BoxChars.HEAVY_BOTTOM_LEFT}"
            f"{BoxChars.HEAVY_HORIZONTAL * (self.width - 2)}"
            f"{BoxChars.HEAVY_BOTTOM_RIGHT}"
        )

    def separator(self) -> str:
        """Render separator line."""
        return (
            f"{BoxChars.HEAVY_VERTICAL}"
            f"{BoxChars.HEAVY_HORIZONTAL * (self.width - 2)}"
            f"{BoxChars.HEAVY_VERTICAL}"
        )

    def wrap_line(self, content: str) -> str:
        """Wrap content in vertical borders.

        Args:
            content: Content to wrap

        Returns:
            Wrapped line
        """
        return f"{BoxChars.HEAVY_VERTICAL} {content} {BoxChars.HEAVY_VERTICAL}"


class FancyProgressDisplay:
    """Fancy terminal display with boxes showing pipeline progress."""

    DEFAULT_WIDTH = 80

    def __init__(self, total_frames: int, stages_config: dict[str, str], suppress_logs: bool = True) -> None:
        """Initialize the fancy progress display.

        Args:
            total_frames: Total number of frames to process
            stages_config: Dictionary mapping stage names to their methods
                          e.g., {"motion": "geometric", "inpainting": "learned", "reconstruction": "vggt"}
            suppress_logs: Whether to suppress logging during rendering (default: True)
        """
        self.total_frames = total_frames
        self.current_frame = 0
        self.current_stage: str | None = None
        self.suppress_logs = suppress_logs

        # Initialize stages
        self.stages: dict[str, PipelineStage] = {
            name: PipelineStage(name=name, method=method)
            for name, method in stages_config.items()
        }

        # Initialize renderers
        self.width = self.DEFAULT_WIDTH
        self.border_renderer = BorderRenderer(self.width)
        self.progress_bar = ProgressBar(self.width - 4)
        self.stages_renderer = PipelineStagesRenderer(self.width - 4)
        self.terminal = TerminalRenderer()

        self._last_lines_count = 0
        self._is_active = False

        # Setup logging suppression if requested
        if self.suppress_logs:
            self._setup_logging_filter()

    def update(self, frame: int, stage: StageType | None = None) -> None:
        """Update the display with current frame and stage.

        Args:
            frame: Current frame number
            stage: Current active stage name
        """
        self.current_frame = frame

        if stage is not None:
            self._update_active_stage(stage)

        self._render()

    def finish(self) -> None:
        """Finalize the display."""
        self.current_frame = self.total_frames
        self._deactivate_all_stages()
        self._render()
        print()  # Add a newline at the end
        self._is_active = False

        # Remove logging filter if it was set up
        if self.suppress_logs:
            self._remove_logging_filter()

    def _update_active_stage(self, stage: str) -> None:
        """Update which stage is currently active.

        Args:
            stage: Stage name to activate
        """
        self.current_stage = stage

        # Deactivate all stages
        for s in self.stages.values():
            s.active = False

        # Activate current stage
        if stage in self.stages:
            self.stages[stage].active = True

    def _deactivate_all_stages(self) -> None:
        """Deactivate all stages."""
        for stage in self.stages.values():
            stage.active = False

    def _render(self) -> None:
        """Render the complete display."""
        # Clear previous output
        if self._last_lines_count > 0:
            self.terminal.clear_lines(self._last_lines_count)

        # Build display lines
        lines = self._build_display_lines()

        # Render to terminal
        self.terminal.print_lines(lines)
        self._last_lines_count = len(lines)

    def _build_display_lines(self) -> list[str]:
        """Build all display lines.

        Returns:
            List of formatted lines to display
        """
        lines = []

        # Top border
        lines.append(self.border_renderer.top_border())

        # Progress section
        lines.extend(self._build_progress_section())

        # Separator
        lines.append(self.border_renderer.separator())

        # Pipeline stages section
        lines.extend(self._build_stages_section())

        # Bottom border
        lines.append(self.border_renderer.bottom_border())

        return lines

    def _build_progress_section(self) -> list[str]:
        """Build the progress section lines.

        Returns:
            List of progress section lines
        """
        lines = []

        # Progress text
        progress_text = f" Frame Progress: {self.current_frame}/{self.total_frames} "
        formatted_text = f"{Style.BOLD}{progress_text:^{self.width - 4}}{Color.RESET}"
        lines.append(self.border_renderer.wrap_line(formatted_text))

        # Progress bar
        progress_bar = self.progress_bar.render(self.current_frame, self.total_frames)
        lines.append(self.border_renderer.wrap_line(progress_bar))

        return lines

    def _build_stages_section(self) -> list[str]:
        """Build the pipeline stages section lines.

        Returns:
            List of stages section lines
        """
        stages_line = self.stages_renderer.render(self.stages)
        return [self.border_renderer.wrap_line(stages_line)]

    def _setup_logging_filter(self) -> None:
        """Setup logging filter to suppress logs during fancy progress display."""
        # Get the root logger
        root_logger = logging.getLogger()

        # Create a custom filter that suppresses logs when fancy progress is active
        class FancyProgressFilter(logging.Filter):
            def __init__(self, progress_display):
                super().__init__()
                self.progress_display = progress_display

            def filter(self, record):
                # Only suppress if the display is active and it's not from our own logger
                if self.progress_display._is_active and record.name != __name__:
                    return False
                return True

        self._log_filter = FancyProgressFilter(self)
        root_logger.addFilter(self._log_filter)
        self._is_active = True

    def _remove_logging_filter(self) -> None:
        """Remove the logging filter."""
        if hasattr(self, '_log_filter'):
            root_logger = logging.getLogger()
            root_logger.removeFilter(self._log_filter)
            del self._log_filter
