"""Read-only runtime readiness probe for the official UniVTAC ACT policy."""

from __future__ import annotations

import importlib
import io
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.integrations.runtime_config import (
    ACTRuntimeArtifacts,
    resolve_act_runtime_artifacts,
)
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTLoadRequest,
    load_official_univtac_act_policy,
)

RUNTIME_PROBE_SCHEMA_VERSION = "robotactile-official-act-runtime-probe-v1"
RUNTIME_PROBE_EVIDENCE_LEVEL = "official_act_runtime_probe_no_simulator_execution_v1"
_REQUIRED_DEPENDENCIES = ("numpy", "torch", "torchvision", "IPython")


@dataclass(frozen=True)
class OfficialACTRuntimeProbeResult:
    """Fail-closed readiness result that never represents closed-loop evidence."""

    integration_config_path: Path
    artifact_valid: bool
    dependency_imports_attempted: bool
    dependency_imports: Mapping[str, bool]
    policy_load_requested: bool
    policy_loaded: bool
    policy_closed: bool
    artifact_manifest_path: Optional[Path] = None
    task_id: Optional[str] = None
    profile: Optional[str] = None
    device: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    error_stage: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integration_config_path",
            Path(self.integration_config_path).absolute(),
        )
        imports = dict(self.dependency_imports)
        if self.dependency_imports_attempted:
            if set(imports) != set(_REQUIRED_DEPENDENCIES):
                raise ValueError("ACT dependency probe results are incomplete")
        elif imports:
            raise ValueError("unattempted ACT dependency imports must be empty")
        object.__setattr__(self, "dependency_imports", MappingProxyType(imports))
        for name in (
            "artifact_valid",
            "dependency_imports_attempted",
            "policy_load_requested",
            "policy_loaded",
            "policy_closed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.policy_loaded and not self.policy_load_requested:
            raise ValueError("ACT policy cannot load without explicit opt-in")
        if self.policy_closed and not self.policy_loaded:
            raise ValueError("ACT policy cannot close before it is loaded")
        if self.artifact_valid:
            required = (
                self.artifact_manifest_path,
                self.task_id,
                self.profile,
                self.device,
                self.checkpoint_sha256,
                self.config_sha256,
            )
            if any(value is None for value in required):
                raise ValueError("valid ACT artifact evidence is incomplete")
        error_fields = (self.error_stage, self.error_type, self.error_message)
        if any(value is None for value in error_fields) and any(
            value is not None for value in error_fields
        ):
            raise ValueError("ACT runtime probe error evidence is incomplete")

    @property
    def dependency_imports_valid(self) -> bool:
        return self.dependency_imports_attempted and all(
            self.dependency_imports.values()
        )

    @property
    def passed(self) -> bool:
        policy_ready = not self.policy_load_requested or (
            self.policy_loaded and self.policy_closed
        )
        return (
            self.artifact_valid
            and self.dependency_imports_valid
            and policy_ready
            and self.error_stage is None
        )

    def to_dict(self) -> dict[str, object]:
        error = None
        if self.error_stage is not None:
            error = {
                "message": self.error_message,
                "stage": self.error_stage,
                "type": self.error_type,
            }
        return {
            "artifact_manifest_path": (
                None
                if self.artifact_manifest_path is None
                else str(self.artifact_manifest_path)
            ),
            "artifact_valid": self.artifact_valid,
            "checkpoint_sha256": self.checkpoint_sha256,
            "closed_loop_execution_claimed": False,
            "config_sha256": self.config_sha256,
            "dependency_imports": dict(self.dependency_imports),
            "dependency_imports_attempted": self.dependency_imports_attempted,
            "dependency_imports_valid": self.dependency_imports_valid,
            "device": self.device,
            "environment_modified": False,
            "error": error,
            "evidence_level": RUNTIME_PROBE_EVIDENCE_LEVEL,
            "integration_config_path": str(self.integration_config_path),
            "isaac_sim_started": False,
            "passed": self.passed,
            "policy_closed": self.policy_closed,
            "policy_load_requested": self.policy_load_requested,
            "policy_loaded": self.policy_loaded,
            "profile": self.profile,
            "schema_version": RUNTIME_PROBE_SCHEMA_VERSION,
            "task_id": self.task_id,
        }


def _result(
    *,
    integration_config_path: Path,
    artifact_valid: bool,
    dependency_imports_attempted: bool,
    dependency_imports: Mapping[str, bool],
    policy_load_requested: bool,
    policy_loaded: bool,
    policy_closed: bool,
    runtime: Optional[ACTRuntimeArtifacts] = None,
    error: Optional[Exception] = None,
    error_stage: Optional[str] = None,
) -> OfficialACTRuntimeProbeResult:
    if (error is None) is not (error_stage is None):
        raise ValueError("ACT probe error and stage must be provided together")
    manifest = None if runtime is None else runtime.manifest
    return OfficialACTRuntimeProbeResult(
        integration_config_path=integration_config_path,
        artifact_valid=artifact_valid,
        dependency_imports_attempted=dependency_imports_attempted,
        dependency_imports=dependency_imports,
        policy_load_requested=policy_load_requested,
        policy_loaded=policy_loaded,
        policy_closed=policy_closed,
        artifact_manifest_path=None if runtime is None else runtime.manifest_path,
        task_id=None if manifest is None else manifest.task_id,
        profile=None if manifest is None else manifest.profile.value,
        device=None if runtime is None else runtime.device,
        checkpoint_sha256=(None if manifest is None else manifest.checkpoint_sha256),
        config_sha256=None if manifest is None else manifest.config_sha256,
        error_stage=error_stage,
        error_type=None if error is None else type(error).__name__,
        error_message=None if error is None else str(error),
    )


def _import_dependency(module_name: str) -> None:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        importlib.import_module(module_name)


def _probe_dependency_imports() -> tuple[dict[str, bool], Optional[Exception]]:
    results: dict[str, bool] = {}
    first_error: Optional[Exception] = None
    for module_name in _REQUIRED_DEPENDENCIES:
        try:
            _import_dependency(module_name)
        except Exception as error:  # Dependency import failures are probe evidence.
            results[module_name] = False
            if first_error is None:
                first_error = error
        else:
            results[module_name] = True
    return results, first_error


def _policy_identity(runtime: ACTRuntimeArtifacts) -> PolicyIdentity:
    manifest = runtime.manifest
    consumes_tactile = manifest.profile is OfficialACTProfile.UNIVTAC
    return PolicyIdentity(
        system_id=(
            f"official-univtac-act.{manifest.task_id}.{manifest.profile.value}."
            "policy_last.v1"
        ),
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        action_spec=ACTION_SPEC,
        consumes_tactile=consumes_tactile,
        supports_structural_absence=not consumes_tactile,
    )


def probe_official_act_runtime(
    integration_config_path: Path,
    *,
    load_policy: bool = False,
) -> OfficialACTRuntimeProbeResult:
    """Validate an ACT runtime, optionally load it, and never start Isaac Sim."""

    if type(load_policy) is not bool:
        raise TypeError("load_policy must be bool")
    selected_config = Path(integration_config_path).expanduser().absolute()
    try:
        runtime = resolve_act_runtime_artifacts(selected_config)
    except Exception as error:  # Convert validation failure into one safe receipt.
        return _result(
            integration_config_path=selected_config,
            artifact_valid=False,
            dependency_imports_attempted=False,
            dependency_imports={},
            policy_load_requested=load_policy,
            policy_loaded=False,
            policy_closed=False,
            error=error,
            error_stage="artifact_validation",
        )

    imports, import_error = _probe_dependency_imports()
    if import_error is not None:
        return _result(
            integration_config_path=selected_config,
            artifact_valid=True,
            dependency_imports_attempted=True,
            dependency_imports=imports,
            policy_load_requested=load_policy,
            policy_loaded=False,
            policy_closed=False,
            runtime=runtime,
            error=import_error,
            error_stage="dependency_imports",
        )
    if not load_policy:
        return _result(
            integration_config_path=selected_config,
            artifact_valid=True,
            dependency_imports_attempted=True,
            dependency_imports=imports,
            policy_load_requested=False,
            policy_loaded=False,
            policy_closed=False,
            runtime=runtime,
        )

    manifest = runtime.manifest
    try:
        request = OfficialUniVTACACTLoadRequest(
            manifest=manifest,
            task_id=manifest.task_id,
            profile=manifest.profile,
            device_name=runtime.device,
            live=True,
        )
        identity = _policy_identity(runtime)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            policy = load_official_univtac_act_policy(identity, request)
    except Exception as error:  # Strict loader failures are diagnostic evidence.
        return _result(
            integration_config_path=selected_config,
            artifact_valid=True,
            dependency_imports_attempted=True,
            dependency_imports=imports,
            policy_load_requested=True,
            policy_loaded=False,
            policy_closed=False,
            runtime=runtime,
            error=error,
            error_stage="policy_load",
        )

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            policy.close()
    except Exception as error:  # A loaded policy must also release cleanly.
        return _result(
            integration_config_path=selected_config,
            artifact_valid=True,
            dependency_imports_attempted=True,
            dependency_imports=imports,
            policy_load_requested=True,
            policy_loaded=True,
            policy_closed=False,
            runtime=runtime,
            error=error,
            error_stage="policy_close",
        )
    return _result(
        integration_config_path=selected_config,
        artifact_valid=True,
        dependency_imports_attempted=True,
        dependency_imports=imports,
        policy_load_requested=True,
        policy_loaded=True,
        policy_closed=True,
        runtime=runtime,
    )


def runtime_probe_json_bytes(result: OfficialACTRuntimeProbeResult) -> bytes:
    """Encode one probe result as exactly one canonical JSON line."""

    if type(result) is not OfficialACTRuntimeProbeResult:
        raise TypeError("result must be an exact OfficialACTRuntimeProbeResult")
    return canonical_json_bytes(result.to_dict())


__all__ = [
    "OfficialACTRuntimeProbeResult",
    "RUNTIME_PROBE_EVIDENCE_LEVEL",
    "RUNTIME_PROBE_SCHEMA_VERSION",
    "probe_official_act_runtime",
    "runtime_probe_json_bytes",
]
