"""Immutable UniVTAC task, runtime, and handshake contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from robotactile_benchmark.action_specs import (
    ACTION_MODE_BY_SPEC,
    QPOS8_ACTION_SPEC,
    validate_action_spec,
)
from robotactile_benchmark.adapters.univtac import (
    DepthPhaseTracker,
    UniVTACAliasManifest,
)
from robotactile_benchmark.contracts import canonical_hash, freeze_value

UPSTREAM_COMMIT = "05bcd3edb92237107efa40105292a24f1a9fd761"
REGISTRY_ID = "robotactile_univtac_tasks_v1"
REGISTRY_SEMANTIC_VERSION = "1.0"
REGISTRY_RESOURCE_SHA256 = (
    "6f8d58b8efce09f1c8d3f1a97a9780ba722b748b0d23086b7051a8bf75272084"
)
BACKEND_ID = "robotactile-univtac-backend-v1"
ACTION_MODE = "qpos"
SIM_HZ = 120
DECIMATION = 1
PHYSICS_STEPS_PER_ACTION = 1
N0_DECIMATION = 1
# N0 was trained from UniVTAC rows recorded every two 120 Hz simulator ticks.
# Its dense EE endpoints must therefore be delivered at 60 Hz; routing every
# endpoint through the upstream variable-length motion-planning loop dilates
# one model action by roughly an order of magnitude.
N0_PHYSICS_STEPS_PER_ACTION = 2
QPOS_ACTION_EXECUTION_CONTRACT = "univtac_stock_qpos_v1"
N0_STOCK_EE_ACTION_EXECUTION_CONTRACT = "univtac_stock_ee_v1"
N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT = "robotactile_fixed_endpoint_v1"
N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT = "robotactile_n0_training_60hz_ee_v1"
N0_EE_ACTION_EXECUTION_CONTRACTS = frozenset(
    {
        N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
        N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
        N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    }
)
FIXED_NATIVE_STEP_CONTRACT = "fixed_physics_steps_per_action_v1"
N0_STOCK_EE_NATIVE_STEP_CONTRACT = "univtac_stock_ee_variable_native_steps_v1"
# These frozen overrides fall through without an explicit return when no stop fires.
EARLY_STOP_NONE_IS_FALSE_TASK_IDS = frozenset({"insert_hole", "insert_tube"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class UniVTACContractError(ValueError):
    """Stable fail-closed error for UniVTAC contract drift."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UniVTACContractError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    normalized = _nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise UniVTACContractError(f"{name} must be a lowercase SHA256")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UniVTACContractError(f"{name} must be a positive integer")
    return int(value)


def validate_n0_ee_action_execution_contract(value: object) -> str:
    """Return one explicit N0 EE execution contract or fail closed."""

    if type(value) is not str or value not in N0_EE_ACTION_EXECUTION_CONTRACTS:
        raise UniVTACContractError("unsupported N0 EE action execution contract")
    return value


@dataclass(frozen=True)
class UniVTACTaskSpec:
    """One source-bound UniVTAC task and its frozen success semantics."""

    task_id: str
    prompt_id: str
    prompt: str
    module_name: str
    class_name: str
    success_predicate_id: str
    task_source_sha256: str
    action_horizon: int
    early_stop_capable: bool
    predicate_note: str

    def __post_init__(self) -> None:
        for field_name in (
            "task_id",
            "prompt_id",
            "prompt",
            "module_name",
            "class_name",
            "success_predicate_id",
            "predicate_note",
        ):
            object.__setattr__(
                self, field_name, _nonempty(getattr(self, field_name), field_name)
            )
        object.__setattr__(
            self,
            "task_source_sha256",
            _sha256(self.task_source_sha256, "task_source_sha256"),
        )
        object.__setattr__(
            self, "action_horizon", _positive_int(self.action_horizon, "action_horizon")
        )
        if type(self.early_stop_capable) is not bool:
            raise UniVTACContractError("early_stop_capable must be bool")
        if self.module_name != f"envs.{self.task_id}" or self.class_name != "Task":
            raise UniVTACContractError("task module/class identity drift")
        if not self.prompt_id.endswith(".v1"):
            raise UniVTACContractError("prompt_id must be explicitly versioned")
        expected_predicate = (
            f"univtac-upstream-05bcd3ed.{self.task_id}.check_success.v1"
        )
        if self.success_predicate_id != expected_predicate:
            raise UniVTACContractError("success predicate identity drift")


@dataclass(frozen=True)
class UniVTACRuntimeHandshake:
    """Runtime facts captured after task construction and before reset."""

    backend_id: str
    config_sha256: str
    registry_resource_sha256: str
    upstream_commit: str
    task_id: str
    task_source_sha256: str
    success_predicate_id: str
    action_horizon: int
    live_joint_names: Tuple[str, ...]
    action_spec: str
    action_mode: str
    force: bool
    sim_hz: int
    decimation: int
    physics_steps_per_action: int

    def __post_init__(self) -> None:
        if _COMMIT.fullmatch(self.upstream_commit) is None:
            raise UniVTACContractError(
                "handshake upstream commit must be lowercase hex"
            )
        for name in ("config_sha256", "registry_resource_sha256", "task_source_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        joint_names = tuple(
            _nonempty(item, "live joint name") for item in self.live_joint_names
        )
        if len(joint_names) != 9 or len(set(joint_names)) != 9:
            raise UniVTACContractError(
                "handshake live joint names must be nine unique names"
            )
        object.__setattr__(self, "live_joint_names", joint_names)

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class UniVTACBackendConfig:
    """Dependency-light configuration consumed by the production backend."""

    task: UniVTACTaskSpec
    registry_resource_sha256: str
    upstream_commit: str
    action_spec: str
    action_mode: str
    force: bool
    sim_hz: int
    decimation: int
    physics_steps_per_action: int
    canonical_joint_names: Tuple[str, ...]
    action_lower_bounds: Tuple[float, ...]
    action_upper_bounds: Tuple[float, ...]
    head_shape: Tuple[int, ...]
    wrist_shape: Tuple[int, ...]
    tactile_rgb_shape: Tuple[int, ...]
    tactile_depth_shape: Tuple[int, ...]
    aliases: UniVTACAliasManifest
    phase_tracker: DepthPhaseTracker

    def __post_init__(self) -> None:
        if self.upstream_commit != UPSTREAM_COMMIT:
            raise UniVTACContractError("upstream commit does not match frozen snapshot")
        action_spec = validate_action_spec(self.action_spec)
        expected_decimation = (
            N0_DECIMATION if action_spec != QPOS8_ACTION_SPEC else DECIMATION
        )
        expected_physics_steps = (
            N0_PHYSICS_STEPS_PER_ACTION
            if action_spec != QPOS8_ACTION_SPEC
            else PHYSICS_STEPS_PER_ACTION
        )
        if (
            self.action_mode != ACTION_MODE_BY_SPEC[action_spec]
            or self.force is not True
            or self.sim_hz != SIM_HZ
            or self.decimation != expected_decimation
            or self.physics_steps_per_action != expected_physics_steps
        ):
            raise UniVTACContractError("runtime constants drift from the v1 contract")
        object.__setattr__(self, "action_spec", action_spec)
        names = tuple(self.canonical_joint_names)
        if len(names) != 9 or len(set(names)) != 9:
            raise UniVTACContractError(
                "canonical joint names must be nine unique names"
            )
        lower = tuple(float(value) for value in self.action_lower_bounds)
        upper = tuple(float(value) for value in self.action_upper_bounds)
        if len(lower) != 8 or len(upper) != 8:
            raise UniVTACContractError("action bounds must each have length eight")
        if not all(math.isfinite(value) for value in lower + upper):
            raise UniVTACContractError("action bounds must be finite")
        if any(low >= high for low, high in zip(lower, upper)):
            raise UniVTACContractError(
                "every action lower bound must be below its upper bound"
            )
        object.__setattr__(self, "canonical_joint_names", names)
        object.__setattr__(self, "action_lower_bounds", lower)
        object.__setattr__(self, "action_upper_bounds", upper)

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    @property
    def action_execution_contract(self) -> str:
        """Return the source-bound action surface used in production."""

        if self.action_spec == QPOS8_ACTION_SPEC:
            return QPOS_ACTION_EXECUTION_CONTRACT
        return N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT

    @property
    def native_step_contract(self) -> str:
        """Describe whether one policy action has a fixed native-step count."""

        return FIXED_NATIVE_STEP_CONTRACT

    @property
    def fixed_physics_steps_per_action(self) -> Optional[int]:
        """Return a production cadence only when the upstream path is fixed."""

        return self.physics_steps_per_action

    def expected_handshake(
        self, live_joint_names: Tuple[str, ...]
    ) -> UniVTACRuntimeHandshake:
        return UniVTACRuntimeHandshake(
            backend_id=BACKEND_ID,
            config_sha256=self.sha256,
            registry_resource_sha256=self.registry_resource_sha256,
            upstream_commit=self.upstream_commit,
            task_id=self.task.task_id,
            task_source_sha256=self.task.task_source_sha256,
            success_predicate_id=self.task.success_predicate_id,
            action_horizon=self.task.action_horizon,
            live_joint_names=live_joint_names,
            action_spec=self.action_spec,
            action_mode=self.action_mode,
            force=self.force,
            sim_hz=self.sim_hz,
            decimation=self.decimation,
            physics_steps_per_action=self.physics_steps_per_action,
        )

    def validate_handshake(self, handshake: UniVTACRuntimeHandshake) -> None:
        expected = self.expected_handshake(handshake.live_joint_names)
        if set(handshake.live_joint_names) != set(self.canonical_joint_names):
            raise UniVTACContractError("handshake joint-name set mismatch")
        if handshake != expected:
            if handshake.config_sha256 != self.sha256:
                raise UniVTACContractError("handshake config hash mismatch")
            raise UniVTACContractError("handshake does not match backend config")


@dataclass(frozen=True)
class UniVTACTaskRegistry:
    """Strict parsed view of the packaged task registry."""

    registry_id: str
    semantic_version: str
    upstream_repository: str
    upstream_commit: str
    resource_sha256: str
    runtime: Mapping[str, Any]
    aliases: Mapping[str, Any]
    phase_tracker: Mapping[str, Any]
    tasks: Tuple[UniVTACTaskSpec, ...]

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if len(tasks) != 8 or len({task.task_id for task in tasks}) != 8:
            raise UniVTACContractError("registry must contain eight unique tasks")
        for field_name in ("runtime", "aliases", "phase_tracker"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise UniVTACContractError(f"registry {field_name} must be a mapping")
            object.__setattr__(self, field_name, freeze_value(value))
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(
            self,
            "resource_sha256",
            _sha256(self.resource_sha256, "resource_sha256"),
        )

    def task(self, task_id: str) -> UniVTACTaskSpec:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"unknown UniVTAC task: {task_id}")


def load_univtac_task_registry(
    document: Optional[Mapping[str, Any]] = None,
) -> UniVTACTaskRegistry:
    """Load the strict source-bound v1 task registry without simulator imports."""

    from robotactile_benchmark.backends.univtac_registry import load_registry

    return load_registry(document)


def build_univtac_backend_config(
    task_id: str, action_spec: str = QPOS8_ACTION_SPEC
) -> UniVTACBackendConfig:
    """Build one typed backend config from the frozen packaged registry."""

    from robotactile_benchmark.backends.univtac_registry import build_config

    return build_config(task_id, action_spec=action_spec)


def validate_packaged_univtac_config(config: UniVTACBackendConfig) -> None:
    """Reject a self-consistent config not rebuilt from the packaged registry."""

    try:
        expected = build_univtac_backend_config(
            config.task.task_id, action_spec=config.action_spec
        )
    except KeyError as error:
        raise UniVTACContractError(
            "backend config does not match the packaged registry"
        ) from error
    if config != expected:
        raise UniVTACContractError(
            "backend config does not match the packaged registry"
        )
