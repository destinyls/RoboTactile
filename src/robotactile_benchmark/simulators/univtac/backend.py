"""Validated UniVTAC backend and lazy runtime factory."""

from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend

UniVTACBackend = UniVTACIsaacBackend
create_univtac_runtime = launch_univtac_runtime

__all__ = ["UniVTACBackend", "create_univtac_runtime"]
