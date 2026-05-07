"""Capture Pi3X conf_decoder features via a forward hook.

Usage:
    extractor = Pi3FeatureExtractor(pi3_model)
    with extractor.capture():
        out = pi3_model(imgs)        # normal Pi3 forward
    feats = extractor.features        # (B*N, num_patches, 1024)
    patch_h, patch_w = extractor.patch_hw

The captured tensor is the input to a downstream DynHead.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class Pi3FeatureExtractor:
    """Hook the conf_decoder output (post special-token slice)."""

    def __init__(self, pi3_model: nn.Module) -> None:
        self.model = pi3_model
        self.features: torch.Tensor | None = None
        self.patch_hw: tuple[int, int] | None = None
        self._handle = None
        self._input_hw: tuple[int, int] | None = None

    @property
    def patch_start_idx(self) -> int:
        return int(getattr(self.model, "patch_start_idx", 5))

    def _hook(self, module, inputs, output):
        # output: (B*N, hw, 1024).  Slice off special tokens to align with
        # how conf_head is fed inside Pi3X.forward_head.
        feats = output[:, self.patch_start_idx :].float().detach()
        self.features = feats

    def _input_hook(self, module, inputs):
        # First positional arg is `imgs` of shape (B, N, 3, H, W).
        imgs = inputs[0]
        H, W = imgs.shape[-2:]
        self._input_hw = (H, W)
        self.patch_hw = (H // 14, W // 14)

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Context manager that installs/removes the hook."""
        if not hasattr(self.model, "conf_decoder"):
            raise AttributeError("Pi3 model has no conf_decoder; expected Pi3X.")

        self.features = None
        self.patch_hw = None
        h_out = self.model.conf_decoder.register_forward_hook(self._hook)
        h_in = self.model.register_forward_pre_hook(self._input_hook)
        try:
            yield
        finally:
            h_out.remove()
            h_in.remove()
