"""SIGMA package initialization.

Expose a lazily-imported `run_pipeline` entrypoint. Importing `sigma`
should not execute `sigma.main` (which pulls in Hydra) at package import
time, to avoid import-time side effects and the `runpy` warning.
"""

from typing import Any


def run_pipeline(*args: Any, **kwargs: Any) -> Any:
	"""Lazily import and call the real `run_pipeline` from `sigma.main`.

	This prevents `sigma.main` from being imported when the package is
	imported, avoiding the RuntimeWarning seen when executing modules with
	runpy or similar tooling.
	"""
	from .main import run_pipeline as _run_pipeline

	return _run_pipeline(*args, **kwargs)


__all__ = ["run_pipeline"]
