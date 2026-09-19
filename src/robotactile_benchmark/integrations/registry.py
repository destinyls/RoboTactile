"""Static first-class model registry with strict checked-in configs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping, Optional, cast

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC, QPOS8_ACTION_SPEC
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC
from robotactile_benchmark.integrations.contracts import (
    ModelIntegrationCapabilities,
    ModelIntegrationSpec,
)

_CONFIG_VERSION = "robotactile-model-integration-config-v1"
_CONFIG_FIELDS = frozenset(
    {"artifact_manifest", "device", "integration_id", "schema_version", "transport"}
)

_ACT = ModelIntegrationSpec(
    integration_id="act",
    display_name="ACT",
    adapter_api_version="robotactile-policy-adapter-v1",
    external_pin_id="univtac",
    factory_path="robotactile_benchmark.integrations.act.factory:load_act_adapter",
    artifact_schema="official_act_artifact_manifest.schema.json",
    qualification_protocol="live-preflight-v1",
    license_spdx="Apache-2.0",
    capabilities=ModelIntegrationCapabilities(
        consumes_tactile=True,
        structural_absence=False,
        matched_no_touch=True,
        stateful_commit=False,
        action_spec=ACTION_SPEC,
        supported_conditions=("clean", "faulted", "no_touch"),
    ),
)
_DREAM_TAC = ModelIntegrationSpec(
    integration_id="dream_tac",
    display_name="Dream-Tac",
    adapter_api_version="robotactile-policy-adapter-v1",
    external_pin_id="dream_tac",
    factory_path=(
        "robotactile_benchmark.integrations.dream_tac.factory:load_dream_tac_adapter"
    ),
    artifact_schema="dream_tac_artifact_manifest.schema.json",
    qualification_protocol="source-bound-http-ee8-v1",
    license_spdx="Apache-2.0",
    capabilities=ModelIntegrationCapabilities(
        consumes_tactile=True,
        structural_absence=False,
        matched_no_touch=False,
        stateful_commit=False,
        action_spec=EE8_ACTION_SPEC,
        supported_conditions=("clean", "faulted"),
    ),
)
_FTP1_POLICY = ModelIntegrationSpec(
    integration_id="ftp1_policy",
    display_name="FTP-1",
    adapter_api_version="robotactile-policy-adapter-v1",
    external_pin_id="ftp1_policy",
    factory_path=(
        "robotactile_benchmark.integrations.ftp1_policy.factory:"
        "load_ftp1_policy_adapter"
    ),
    artifact_schema="ftp1_policy_artifact_manifest.schema.json",
    qualification_protocol="official-zmq-temporal-ensemble-v1",
    license_spdx="Apache-2.0",
    capabilities=ModelIntegrationCapabilities(
        consumes_tactile=True,
        structural_absence=False,
        matched_no_touch=False,
        stateful_commit=True,
        action_spec=QPOS8_ACTION_SPEC,
        supported_conditions=("clean", "faulted"),
    ),
)
_N0_TWAM = ModelIntegrationSpec(
    integration_id="n0_twam",
    display_name="N0-TWAM",
    adapter_api_version="robotactile-policy-adapter-v1",
    external_pin_id="n0_twam",
    factory_path=(
        "robotactile_benchmark.integrations.n0_twam.factory:load_n0_twam_adapter"
    ),
    artifact_schema="robotactile-n0-official-artifact-v3",
    qualification_protocol="official-websocket-grounding-v1",
    license_spdx="CC-BY-NC-SA-4.0",
    capabilities=ModelIntegrationCapabilities(
        consumes_tactile=True,
        structural_absence=False,
        matched_no_touch=False,
        stateful_commit=True,
        action_spec=EE8_ACTION_SPEC,
        supported_conditions=("clean", "faulted"),
    ),
)
_N0_VTLA = ModelIntegrationSpec(
    integration_id="n0_vtla",
    display_name="N0-VTLA",
    adapter_api_version="robotactile-policy-adapter-v1",
    external_pin_id="n0_vtla",
    factory_path=(
        "robotactile_benchmark.integrations.n0_vtla.factory:load_n0_vtla_adapter"
    ),
    artifact_schema="n0_vtla_artifact_manifest.schema.json",
    qualification_protocol="official-zmq-full-chunk-v1",
    license_spdx="CC-BY-SA-4.0",
    capabilities=ModelIntegrationCapabilities(
        consumes_tactile=True,
        structural_absence=False,
        matched_no_touch=False,
        stateful_commit=False,
        action_spec=ACTION_SPEC,
        supported_conditions=("clean", "faulted"),
    ),
)
_REGISTRY = (_ACT, _DREAM_TAC, _FTP1_POLICY, _N0_TWAM, _N0_VTLA)


def list_model_integrations() -> tuple[ModelIntegrationSpec, ...]:
    """Return the exact ordered built-in registry."""

    return _REGISTRY


def get_model_integration(integration_id: str) -> ModelIntegrationSpec:
    """Return one registered model without dynamic imports."""

    matches = tuple(spec for spec in _REGISTRY if spec.integration_id == integration_id)
    if len(matches) != 1:
        raise KeyError(f"unknown model integration: {integration_id}")
    return matches[0]


@dataclass(frozen=True)
class ModelIntegrationConfig:
    """Dependency-light selection of one static integration and artifact."""

    schema_version: str
    integration_id: str
    artifact_manifest: str
    device: str
    transport: str

    def __post_init__(self) -> None:
        if self.schema_version != _CONFIG_VERSION:
            raise ValueError("model integration config version mismatch")
        spec = get_model_integration(self.integration_id)
        if (
            not self.artifact_manifest
            or self.artifact_manifest.strip() != self.artifact_manifest
            or "\x00" in self.artifact_manifest
        ):
            raise ValueError("artifact_manifest must be a non-empty safe path")
        manifest_path = Path(self.artifact_manifest)
        if not manifest_path.is_absolute() and ".." in manifest_path.parts:
            raise ValueError("relative artifact_manifest cannot escape its root")
        if not self.device or self.device.strip() != self.device:
            raise ValueError("device must be a non-empty string")
        expected_transport = {
            "act": "in_process",
            "dream_tac": "official_http",
            "ftp1_policy": "official_zmq",
            "n0_twam": "official_websocket",
            "n0_vtla": "official_zmq",
        }[spec.integration_id]
        if self.transport != expected_transport:
            raise ValueError("transport does not match the registered integration")

    @property
    def spec(self) -> ModelIntegrationSpec:
        return get_model_integration(self.integration_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_manifest": self.artifact_manifest,
            "device": self.device,
            "integration_id": self.integration_id,
            "schema_version": self.schema_version,
            "transport": self.transport,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config_bytes(integration_id: str, path: Optional[Path]) -> bytes:
    relative = f"configs/integrations/{integration_id}.json"
    if path is None:
        packaged = resources.files("robotactile_benchmark").joinpath(relative)
        if packaged.is_file():
            return packaged.read_bytes()
        path = _project_root() / relative
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("model integration config must be a regular file")
    return selected.read_bytes()


def load_model_integration_config(
    integration_id: str, path: Optional[Path] = None
) -> ModelIntegrationConfig:
    """Load one canonical config and bind it to its requested registry entry."""

    get_model_integration(integration_id)
    try:
        value = strict_json_bytes(
            _config_bytes(integration_id, path), f"{integration_id}.json"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("model integration config is not canonical") from error
    if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
        raise ValueError("model integration config fields mismatch")
    if value["integration_id"] != integration_id:
        raise ValueError(
            "model integration config disagrees with requested integration"
        )
    config = ModelIntegrationConfig(**cast(dict[str, str], dict(value)))
    if canonical_json_bytes(config.to_dict()) != _config_bytes(integration_id, path):
        raise ValueError("typed model integration config is not canonical")
    return config


__all__ = [
    "ModelIntegrationConfig",
    "get_model_integration",
    "list_model_integrations",
    "load_model_integration_config",
]
