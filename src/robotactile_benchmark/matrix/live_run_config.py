"""Strict deployment-time resources for resumable live matrix execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from robotactile_benchmark.closed_loop.artifact_contracts import ArtifactValidationError
from robotactile_benchmark.closed_loop.artifact_io import strict_json_bytes
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.matrix.contracts import MatrixCellSpec
from robotactile_benchmark.matrix.live_executor_contracts import (
    LiveMatrixCellResources,
    LiveMatrixExecutionTemplate,
)
from robotactile_benchmark.matrix.manifest import MatrixManifest

LIVE_MATRIX_RUN_CONFIG_SEMANTIC_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_CONFIG_BYTES = 4 * 1024 * 1024


class LiveMatrixRunConfigError(ValueError):
    """Raised when deployment resources do not exactly cover a matrix."""


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LiveMatrixRunConfigError(f"{name} must be a lowercase SHA256")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LiveMatrixRunConfigError(f"{name} must be a positive integer")
    return value


def _path(value: object, name: str, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise LiveMatrixRunConfigError(f"{name} must be a non-empty path")
    raw = Path(value)
    return (raw if raw.is_absolute() else root / raw).absolute()


def _optional_path(value: object, name: str, root: Path) -> Optional[Path]:
    return None if value is None else _path(value, name, root)


@dataclass(frozen=True)
class LiveMatrixResourceEntry:
    """Path binding for one content-addressed matrix cell."""

    fault_manifest_path: Optional[Path]
    rest_references_path: Optional[Path]
    matched_no_touch_artifact_path: Optional[Path]

    def to_resources(self) -> LiveMatrixCellResources:
        return LiveMatrixCellResources(
            fault_manifest_path=self.fault_manifest_path,
            rest_references_path=self.rest_references_path,
            matched_no_touch_artifact_path=self.matched_no_touch_artifact_path,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fault_manifest_path": (
                None
                if self.fault_manifest_path is None
                else str(self.fault_manifest_path)
            ),
            "rest_references_path": (
                None
                if self.rest_references_path is None
                else str(self.rest_references_path)
            ),
            "matched_no_touch_artifact_path": (
                None
                if self.matched_no_touch_artifact_path is None
                else str(self.matched_no_touch_artifact_path)
            ),
        }


@dataclass(frozen=True)
class LiveMatrixRunConfig:
    """Frozen runtime, official ACT, and per-cell resource bindings."""

    matrix_manifest_sha256: str
    policy_kind: LivePolicyKind
    max_control_cycles: int
    max_observation_steps: int
    execute_action_steps: int
    wall_timeout_s: float
    upstream_root: Path
    runtime_root: Path
    act_device_name: str
    simulator_device: str
    launcher_args: Mapping[str, Any]
    official_act_artifact_root: Path
    stats_sha256: str
    encoder_sha256: str
    resources: Mapping[str, LiveMatrixResourceEntry]
    semantic_version: str = LIVE_MATRIX_RUN_CONFIG_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix_manifest_sha256",
            _sha256(self.matrix_manifest_sha256, "matrix_manifest_sha256"),
        )
        kind = (
            self.policy_kind
            if isinstance(self.policy_kind, LivePolicyKind)
            else LivePolicyKind(self.policy_kind)
        )
        if kind is not LivePolicyKind.ACT:
            raise LiveMatrixRunConfigError("live matrix CLI currently requires ACT")
        object.__setattr__(self, "policy_kind", kind)
        for name in (
            "max_control_cycles",
            "max_observation_steps",
            "execute_action_steps",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.execute_action_steps != 1:
            raise LiveMatrixRunConfigError("ACT execute_action_steps must equal one")
        if isinstance(self.wall_timeout_s, bool) or not isinstance(
            self.wall_timeout_s, (int, float)
        ):
            raise LiveMatrixRunConfigError("wall_timeout_s must be a real number")
        timeout = float(self.wall_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise LiveMatrixRunConfigError("wall_timeout_s must be positive and finite")
        object.__setattr__(self, "wall_timeout_s", timeout)
        for name in ("upstream_root", "runtime_root", "official_act_artifact_root"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
            object.__setattr__(self, name, value.absolute())
        for name in ("act_device_name", "simulator_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise LiveMatrixRunConfigError(f"{name} must be non-empty")
        for name in ("stats_sha256", "encoder_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.launcher_args, Mapping):
            raise TypeError("launcher_args must be a mapping")
        object.__setattr__(self, "launcher_args", freeze_value(self.launcher_args))
        if not isinstance(self.resources, Mapping) or not self.resources:
            raise LiveMatrixRunConfigError("resources must be a non-empty mapping")
        normalized: dict[str, LiveMatrixResourceEntry] = {}
        for key, value in self.resources.items():
            digest = _sha256(key, "resource cell SHA256")
            if type(value) is not LiveMatrixResourceEntry:
                raise TypeError("resource values must be LiveMatrixResourceEntry")
            normalized[digest] = value
        object.__setattr__(self, "resources", MappingProxyType(normalized))
        if self.semantic_version != LIVE_MATRIX_RUN_CONFIG_SEMANTIC_VERSION:
            raise LiveMatrixRunConfigError("live matrix config version mismatch")

    def template(self) -> LiveMatrixExecutionTemplate:
        return LiveMatrixExecutionTemplate(
            policy_kind=self.policy_kind,
            max_control_cycles=self.max_control_cycles,
            max_observation_steps=self.max_observation_steps,
            execute_action_steps=self.execute_action_steps,
            wall_timeout_s=self.wall_timeout_s,
            upstream_root=self.upstream_root,
            runtime_root=self.runtime_root,
            act_device_name=self.act_device_name,
            simulator_device=self.simulator_device,
            launcher_args=self.launcher_args,
        )

    def resources_for(self, cell: MatrixCellSpec) -> LiveMatrixCellResources:
        entry = self.resources.get(cell.sha256)
        if entry is None:
            raise LiveMatrixRunConfigError("matrix cell has no resource entry")
        return entry.to_resources()

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix_manifest_sha256": self.matrix_manifest_sha256,
            "policy_kind": self.policy_kind.value,
            "max_control_cycles": self.max_control_cycles,
            "max_observation_steps": self.max_observation_steps,
            "execute_action_steps": self.execute_action_steps,
            "wall_timeout_s": self.wall_timeout_s,
            "upstream_root": str(self.upstream_root),
            "runtime_root": str(self.runtime_root),
            "act_device_name": self.act_device_name,
            "simulator_device": self.simulator_device,
            "launcher_args": dict(self.launcher_args),
            "official_act_artifact_root": str(self.official_act_artifact_root),
            "stats_sha256": self.stats_sha256,
            "encoder_sha256": self.encoder_sha256,
            "resources": {
                key: value.to_dict() for key, value in self.resources.items()
            },
            "semantic_version": self.semantic_version,
        }


def load_live_matrix_run_config(
    path: Path, manifest: MatrixManifest
) -> LiveMatrixRunConfig:
    """Strict-load and cross-check a deployment resource document."""

    config_path = Path(path).absolute()
    if config_path.is_symlink() or not config_path.is_file():
        raise LiveMatrixRunConfigError("live matrix config must be a regular file")
    raw = config_path.read_bytes()
    if not 1 <= len(raw) <= _MAX_CONFIG_BYTES:
        raise LiveMatrixRunConfigError("live matrix config size is invalid")
    try:
        value = strict_json_bytes(raw, "live matrix run config")
    except ArtifactValidationError as error:
        raise LiveMatrixRunConfigError(str(error)) from error
    fields = set(LiveMatrixRunConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != fields:
        raise LiveMatrixRunConfigError("live matrix config fields mismatch")
    resources = value["resources"]
    if not isinstance(resources, dict):
        raise LiveMatrixRunConfigError("live matrix resources must be an object")
    base = config_path.parent
    entries: dict[str, LiveMatrixResourceEntry] = {}
    for key, item in resources.items():
        resource_fields = set(LiveMatrixResourceEntry.__dataclass_fields__)
        if not isinstance(item, dict) or set(item) != resource_fields:
            raise LiveMatrixRunConfigError("live matrix resource fields mismatch")
        entries[key] = LiveMatrixResourceEntry(
            fault_manifest_path=_optional_path(
                item["fault_manifest_path"], "fault_manifest_path", base
            ),
            rest_references_path=_optional_path(
                item["rest_references_path"], "rest_references_path", base
            ),
            matched_no_touch_artifact_path=_optional_path(
                item["matched_no_touch_artifact_path"],
                "matched_no_touch_artifact_path",
                base,
            ),
        )
    config = LiveMatrixRunConfig(
        matrix_manifest_sha256=value["matrix_manifest_sha256"],
        policy_kind=value["policy_kind"],
        max_control_cycles=value["max_control_cycles"],
        max_observation_steps=value["max_observation_steps"],
        execute_action_steps=value["execute_action_steps"],
        wall_timeout_s=value["wall_timeout_s"],
        upstream_root=_path(value["upstream_root"], "upstream_root", base),
        runtime_root=_path(value["runtime_root"], "runtime_root", base),
        act_device_name=value["act_device_name"],
        simulator_device=value["simulator_device"],
        launcher_args=value["launcher_args"],
        official_act_artifact_root=_path(
            value["official_act_artifact_root"], "official_act_artifact_root", base
        ),
        stats_sha256=value["stats_sha256"],
        encoder_sha256=value["encoder_sha256"],
        resources=entries,
        semantic_version=value["semantic_version"],
    )
    if config.matrix_manifest_sha256 != manifest.sha256:
        raise LiveMatrixRunConfigError("run config does not bind the matrix manifest")
    if set(config.resources) != {cell.sha256 for cell in manifest.cells}:
        raise LiveMatrixRunConfigError(
            "run config must cover every matrix cell exactly"
        )
    for cell in manifest.cells:
        config.resources_for(cell)
    return config
