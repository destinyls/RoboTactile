"""Dependency-light UniVTAC simulator contracts."""

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    UniVTACTaskRegistry,
    UniVTACTaskSpec,
    build_univtac_backend_config,
    load_univtac_task_registry,
)

UniVTACSimulatorConfig = UniVTACBackendConfig
build_univtac_simulator_config = build_univtac_backend_config

__all__ = [
    "UniVTACSimulatorConfig",
    "UniVTACTaskRegistry",
    "UniVTACTaskSpec",
    "build_univtac_simulator_config",
    "load_univtac_task_registry",
]
