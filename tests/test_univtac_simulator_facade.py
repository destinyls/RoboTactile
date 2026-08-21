"""Stable simulator namespace for the shared UniVTAC backend."""

import sys

from robotactile_benchmark.backends.univtac_contracts import (
    UPSTREAM_COMMIT,
    UniVTACBackendConfig,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.simulators.univtac import (
    UniVTACBackend,
    UniVTACSimulatorConfig,
    build_univtac_simulator_config,
    create_univtac_runtime,
)


def test_simulator_facade_reuses_validated_backend() -> None:
    assert UniVTACBackend is UniVTACIsaacBackend
    assert UniVTACSimulatorConfig is UniVTACBackendConfig
    assert build_univtac_simulator_config is build_univtac_backend_config
    assert create_univtac_runtime is launch_univtac_runtime
    assert UPSTREAM_COMMIT == "05bcd3edb92237107efa40105292a24f1a9fd761"


def test_simulator_facade_import_is_isaac_and_torch_free() -> None:
    assert "isaaclab.app" not in sys.modules
    assert "torch" not in sys.modules
