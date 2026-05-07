"""Tiny U-Net architecture for binary dynamic-segmentation student head.

Takes (B, 3, H, W) RGB in [0, 1], outputs (B, 1, H, W) logits.
~2.5M parameters. Works on inputs that are multiples of 16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class StudentUNet(nn.Module):
    """4-level U-Net with channel widths [32, 64, 128, 256, 256]."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base: int = 32) -> None:
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.enc4 = _DoubleConv(base * 4, base * 8)
        self.bottleneck = _DoubleConv(base * 8, base * 8)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
        self.dec4 = _DoubleConv(base * 16, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
