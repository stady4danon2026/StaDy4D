"""Shared model registry to avoid duplicate loads across pipeline stages.

Pi3X is loaded by both ``Pi3Reconstructor`` (for depth/pose) and
``Sam3TttUnionMotionEstimator`` (for conf-decoder features used by the
TTT head). Loading it twice wastes ~5GB GPU and ~5s per scene.

Stages call ``get_or_load_pi3(checkpoint, device)`` instead of
``Pi3X.from_pretrained()`` directly. First caller loads it; subsequent
callers reuse the same object.
"""
from __future__ import annotations
from typing import Any

_pi3_cache: dict[tuple[str, str], Any] = {}


def get_or_load_pi3(checkpoint: str, device: str):
    """Return a cached Pi3X model; load if not already cached."""
    key = (checkpoint, str(device))
    if key not in _pi3_cache:
        from pi3.models.pi3x import Pi3X
        import torch
        model = Pi3X.from_pretrained(checkpoint).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _pi3_cache[key] = model
    return _pi3_cache[key]


def clear_pi3_cache() -> None:
    """Free all cached Pi3 models. Call at end of run if needed."""
    global _pi3_cache
    import torch
    _pi3_cache.clear()
    torch.cuda.empty_cache()
