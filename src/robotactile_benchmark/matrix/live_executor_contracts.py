"""Frozen inputs and exact request materialization for live matrix cells."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.matrix.contracts import MatrixCellSpec
from robotactile_benchmark.trials import Condition


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    return value.absolute()


def _optional_path(value: object, name: str) -> Optional[Path]:
    return None if value is None else _absolute_path(value, name)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class LiveMatrixCellResources:
    """Resolved external files for exactly one matrix condition."""

    fault_manifest_path: Optional[Path]
    rest_references_path: Optional[Path]
    matched_no_touch_artifact_path: Optional[Path]

    def __post_init__(self) -> None:
        for name in (
            "fault_manifest_path",
            "rest_references_path",
            "matched_no_touch_artifact_path",
        ):
            object.__setattr__(self, name, _optional_path(getattr(self, name), name))


class LiveMatrixResourceResolver(Protocol):
    """Resolve immutable files without embedding machine paths in the matrix."""

    def __call__(self, cell: MatrixCellSpec) -> LiveMatrixCellResources: ...


@dataclass(frozen=True)
class LiveMatrixExecutionTemplate:
    """Path/runtime fields shared by all cells in one bounded live run."""

    policy_kind: LivePolicyKind
    max_control_cycles: int
    max_observation_steps: int
    execute_action_steps: int
    wall_timeout_s: float
    upstream_root: Path
    runtime_root: Path
    act_device_name: Optional[str]
    simulator_device: Optional[str]
    launcher_args: Mapping[str, Any] = field(default_factory=dict)
    n0_source_commit: Optional[str] = None
    n0_normalizer_sha256: Optional[str] = None
    n0_serve_bundle_sha256: Optional[str] = None
    n0_prompt_manifest_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        kind = (
            self.policy_kind
            if isinstance(self.policy_kind, LivePolicyKind)
            else LivePolicyKind(self.policy_kind)
        )
        object.__setattr__(self, "policy_kind", kind)
        for name in (
            "max_control_cycles",
            "max_observation_steps",
            "execute_action_steps",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if isinstance(self.wall_timeout_s, bool) or not isinstance(
            self.wall_timeout_s, (int, float)
        ):
            raise TypeError("wall_timeout_s must be a real number")
        timeout = float(self.wall_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("wall_timeout_s must be positive and finite")
        object.__setattr__(self, "wall_timeout_s", timeout)
        for name in ("upstream_root", "runtime_root"):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), name))
        if not isinstance(self.launcher_args, Mapping):
            raise TypeError("launcher_args must be a mapping")
        object.__setattr__(self, "launcher_args", freeze_value(self.launcher_args))


def materialize_live_matrix_request(
    cell: MatrixCellSpec,
    template: LiveMatrixExecutionTemplate,
    resources: LiveMatrixCellResources,
    *,
    output_dir: Path,
) -> LiveUniVTACRunRequest:
    """Build and strict-load a request that must reproduce the matrix cell."""

    if type(cell) is not MatrixCellSpec:
        raise TypeError("cell must be an exact MatrixCellSpec")
    if type(template) is not LiveMatrixExecutionTemplate:
        raise TypeError("template must be an exact LiveMatrixExecutionTemplate")
    if type(resources) is not LiveMatrixCellResources:
        raise TypeError("resources must be exact LiveMatrixCellResources")
    condition = cell.trial.condition
    _validate_resources(condition, resources)
    trial = cell.trial
    request = LiveUniVTACRunRequest(
        task_id=trial.task,
        condition=condition,
        policy_kind=template.policy_kind,
        base_system_id=trial.base_system_id,
        dataset_sha256=trial.dataset_sha256,
        checkpoint_sha256=trial.checkpoint_sha256,
        config_sha256=trial.config_sha256,
        base_system_manifest_sha256=trial.base_system_manifest_sha256,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        max_control_cycles=template.max_control_cycles,
        max_observation_steps=template.max_observation_steps,
        execute_action_steps=template.execute_action_steps,
        wall_timeout_s=template.wall_timeout_s,
        upstream_root=template.upstream_root,
        runtime_dir=template.runtime_root / cell.sha256,
        output_dir=_absolute_path(output_dir, "output_dir"),
        fault_manifest_path=resources.fault_manifest_path,
        rest_references_path=resources.rest_references_path,
        restoration_index=trial.restoration_index,
        restoration_mode=trial.restoration_mode,
        matched_no_touch_system_id=trial.matched_no_touch_system_id,
        matched_no_touch_artifact_path=(
            resources.matched_no_touch_artifact_path
            if condition is Condition.NO_TOUCH
            else None
        ),
        act_device_name=template.act_device_name,
        simulator_device=template.simulator_device,
        launcher_args=template.launcher_args,
        n0_source_commit=template.n0_source_commit,
        n0_normalizer_sha256=template.n0_normalizer_sha256,
        n0_serve_bundle_sha256=template.n0_serve_bundle_sha256,
        n0_prompt_manifest_sha256=template.n0_prompt_manifest_sha256,
    )
    try:
        loaded = load_live_univtac_run(request)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise ValueError(
            "materialized live request failed matrix cell validation"
        ) from error
    if loaded.trial != trial or loaded.trial.sha256 != trial.sha256:
        raise ValueError("materialized live request does not match matrix cell trial")
    if loaded.fault_manifest != cell.fault_manifest:
        raise ValueError("materialized live request does not match matrix cell fault")
    return request


def _validate_resources(
    condition: Condition, resources: LiveMatrixCellResources
) -> None:
    faulted = condition in {Condition.FAULTED, Condition.RESTORED}
    if faulted != (resources.fault_manifest_path is not None):
        raise ValueError("resolved fault manifest does not match matrix condition")
    if condition is Condition.NO_TOUCH:
        if resources.matched_no_touch_artifact_path is None:
            raise ValueError("no-touch cell requires a matched policy artifact")
        if resources.rest_references_path is not None:
            raise ValueError("no-touch cell cannot use rest references")
    elif resources.matched_no_touch_artifact_path is not None:
        raise ValueError("tactile cells cannot use a matched no-touch artifact")
    if not faulted and resources.rest_references_path is not None:
        raise ValueError("rest references require a faulted/restored cell")
