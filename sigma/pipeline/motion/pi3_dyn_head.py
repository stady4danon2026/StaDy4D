"""Pi3 dynamic-mask head.

Clones Pi3X's ``conf_head`` (a MoGe-style ``ConvHead``) to predict a per-pixel
dynamic-object logit map.  Shares Pi3's frozen ``conf_decoder`` features as
input — the head is the only trainable component.

Two-stage training:
  1. Offline pretrain on GSAM pseudo-labels (script/train_pi3_dyn_head.py).
  2. Per-scene photometric TTT at inference time (pi3_dyn_ttt.py).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def build_dyn_head(dec_embed_dim: int = 1024) -> nn.Module:
    """Construct a fresh DynHead matching Pi3X conf_head's architecture.

    Mirrors the ConvHead config from Pi3X.__init__ (num_features=4,
    dim_in=dec_embed_dim, dim_out=[1], dim_proj=1024, dim_upsample=[256,128,64]).
    """
    from pi3.models.layers.conv_head import ConvHead

    return ConvHead(
        num_features=4,
        dim_in=dec_embed_dim,
        projects=nn.Identity(),
        dim_out=[1],
        dim_proj=1024,
        dim_upsample=[256, 128, 64],
        dim_times_res_block_hidden=2,
        num_res_blocks=2,
        res_block_norm="group_norm",
        last_res_blocks=0,
        last_conv_channels=32,
        last_conv_size=1,
        using_uv=True,
    )


class DynHead(nn.Module):
    """Per-pixel dynamic-mask head riding on Pi3's conf_decoder features."""

    def __init__(self, dec_embed_dim: int = 1024) -> None:
        super().__init__()
        self.dec_embed_dim = dec_embed_dim
        self.head = build_dyn_head(dec_embed_dim)

    def forward(
        self,
        conf_features: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """Decode per-pixel dyn logits from cached conf_decoder features.

        Args:
            conf_features: ``(B*N, num_patches, dec_embed_dim)`` — output of
                ``conf_decoder`` *after* slicing off the special tokens
                (``ret_conf[:, patch_start_idx:].float()``).
            patch_h, patch_w: token grid shape (img_h//14, img_w//14).

        Returns:
            ``(B*N, H, W)`` raw logits (sigmoid → probability of dynamic).
        """
        out = self.head(conf_features, patch_h=patch_h, patch_w=patch_w)
        # ConvHead with single dim_out returns a list of length 1.
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.squeeze(1)  # (B*N, H, W)

    @torch.no_grad()
    def predict_mask(
        self,
        conf_features: torch.Tensor,
        patch_h: int,
        patch_w: int,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Sigmoid + threshold convenience wrapper. Returns (B*N, H, W) bool."""
        logits = self.forward(conf_features, patch_h=patch_h, patch_w=patch_w)
        return torch.sigmoid(logits) > threshold

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "dec_embed_dim": self.dec_embed_dim,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "DynHead":
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            head = cls(dec_embed_dim=ckpt.get("dec_embed_dim", 1024))
            head.load_state_dict(ckpt["state_dict"])
        else:
            head = cls()
            head.load_state_dict(ckpt)
        return head
