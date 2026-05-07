"""Base class for preprocessing stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from sigma.pipeline.base_stage import BaseStage, StageResult


class BasePreprocessor(BaseStage, ABC):
    """Abstract base class for preprocessors."""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process a single image and return a mask.

        Args:
            image: Input RGB image (H, W, 3).

        Returns:
            Binary mask (H, W) where 1 indicates foreground/region to inpaint.
        """
        pass
