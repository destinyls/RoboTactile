"""Shared UniVTAC simulator integration for all model adapters."""

from robotactile_benchmark.simulators.univtac.backend import (
    UniVTACBackend,
    create_univtac_runtime,
)
from robotactile_benchmark.simulators.univtac.contracts import (
    UniVTACSimulatorConfig,
    UniVTACTaskRegistry,
    UniVTACTaskSpec,
    build_univtac_simulator_config,
    load_univtac_task_registry,
)

__all__ = [
    "UniVTACBackend",
    "UniVTACSimulatorConfig",
    "UniVTACTaskRegistry",
    "UniVTACTaskSpec",
    "build_univtac_simulator_config",
    "create_univtac_runtime",
    "load_univtac_task_registry",
]
