"""First-class external model integration registry."""

from robotactile_benchmark.integrations.contracts import (
    ModelIntegrationCapabilities,
    ModelIntegrationSpec,
    PolicyAdapter,
)
from robotactile_benchmark.integrations.registry import (
    ModelIntegrationConfig,
    get_model_integration,
    list_model_integrations,
    load_model_integration_config,
)

__all__ = [
    "ModelIntegrationCapabilities",
    "ModelIntegrationConfig",
    "ModelIntegrationSpec",
    "PolicyAdapter",
    "get_model_integration",
    "list_model_integrations",
    "load_model_integration_config",
]
