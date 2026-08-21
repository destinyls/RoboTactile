"""Simulator backends with dependency-light public contracts."""

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    UniVTACTaskRegistry,
    UniVTACTaskSpec,
    build_univtac_backend_config,
    load_univtac_task_registry,
)

__all__ = [
    "UniVTACBackendConfig",
    "UniVTACTaskRegistry",
    "UniVTACTaskSpec",
    "build_univtac_backend_config",
    "load_univtac_task_registry",
]
