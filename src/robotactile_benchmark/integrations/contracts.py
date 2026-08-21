"""Public model-integration contracts shared by ACT and N0-TWAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy

PolicyAdapter = ClosedLoopPolicy

_CONDITIONS = frozenset({"clean", "faulted", "no_touch", "restored"})


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ModelIntegrationCapabilities:
    """Observable capabilities used by eligibility and public reporting."""

    consumes_tactile: bool
    structural_absence: bool
    matched_no_touch: bool
    stateful_commit: bool
    action_spec: str
    supported_conditions: Tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "consumes_tactile",
            "structural_absence",
            "matched_no_touch",
            "stateful_commit",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.action_spec != ACTION_SPEC:
            raise ValueError(f"action_spec must be {ACTION_SPEC}")
        if (
            not self.supported_conditions
            or len(set(self.supported_conditions)) != len(self.supported_conditions)
            or not set(self.supported_conditions).issubset(_CONDITIONS)
        ):
            raise ValueError("supported_conditions must be unique benchmark conditions")
        if self.matched_no_touch != ("no_touch" in self.supported_conditions):
            raise ValueError("matched_no_touch disagrees with supported conditions")


@dataclass(frozen=True)
class ModelIntegrationSpec:
    """One statically registered first-class policy integration."""

    integration_id: str
    display_name: str
    adapter_api_version: str
    external_pin_id: str
    factory_path: str
    artifact_schema: str
    qualification_protocol: str
    license_spdx: str
    capabilities: ModelIntegrationCapabilities

    def __post_init__(self) -> None:
        for name in (
            "integration_id",
            "display_name",
            "adapter_api_version",
            "external_pin_id",
            "factory_path",
            "artifact_schema",
            "qualification_protocol",
            "license_spdx",
        ):
            _nonempty(getattr(self, name), name)
        prefix = f"robotactile_benchmark.integrations.{self.integration_id}."
        if (
            not self.factory_path.startswith(prefix)
            or self.factory_path.count(":") != 1
        ):
            raise ValueError("factory_path must name the owned integration package")


__all__ = [
    "ModelIntegrationCapabilities",
    "ModelIntegrationSpec",
    "PolicyAdapter",
]
