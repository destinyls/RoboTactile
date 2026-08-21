"""Immutable contracts for auditable closed-loop benchmark execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Tuple

import numpy as np

from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    ObservationRecord,
    canonical_hash,
    freeze_array,
    freeze_value,
)

ACTION_SPEC = "qpos8_next_step"
SEMANTIC_VERSION = "1.0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _require_positive_integer(value: Any, name: str) -> int:
    normalized = _require_nonnegative_integer(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_positive_finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _require_action_spec(value: Any) -> str:
    action_spec: str = _require_nonempty_string(value, "action_spec")
    if action_spec != ACTION_SPEC:
        raise ValueError(f"action_spec must be {ACTION_SPEC}")
    return action_spec


def _freeze_action_array(value: Any, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must already have a floating dtype")
    frozen = freeze_array(value, dtype=np.float32)
    if frozen.ndim != 2 or frozen.shape[0] == 0 or frozen.shape[1] != 8:
        raise ValueError(f"{name} must have shape [H, 8] with H > 0")
    if not np.isfinite(frozen).all():
        raise ValueError(f"{name} must be finite")
    return frozen


class BackendSignal(str, Enum):
    """The terminal or in-progress signal emitted by a simulation backend."""

    RUNNING = "running"
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    EARLY_STOP = "early_stop"
    TIMEOUT = "timeout"


def _normalize_signal(value: Any) -> BackendSignal:
    if isinstance(value, BackendSignal):
        return value
    try:
        return BackendSignal(value)
    except (TypeError, ValueError) as error:
        raise ValueError("signal must be a supported BackendSignal") from error


@dataclass(frozen=True)
class ClosedLoopRunSpec:
    """Bounded execution budget and success predicate for one closed-loop run."""

    prompt: str
    success_predicate_id: str
    max_control_cycles: int
    max_observation_steps: int
    execute_action_steps: int
    wall_timeout_s: float
    semantic_version: str = SEMANTIC_VERSION

    def __post_init__(self) -> None:
        prompt = _require_nonempty_string(self.prompt, "prompt")
        predicate_id = _require_nonempty_string(
            self.success_predicate_id, "success_predicate_id"
        )
        max_control_cycles = _require_positive_integer(
            self.max_control_cycles, "max_control_cycles"
        )
        max_observation_steps = _require_positive_integer(
            self.max_observation_steps, "max_observation_steps"
        )
        execute_action_steps = _require_positive_integer(
            self.execute_action_steps, "execute_action_steps"
        )
        if execute_action_steps > max_observation_steps:
            raise ValueError("execute_action_steps cannot exceed max_observation_steps")
        if max_control_cycles > max_observation_steps:
            raise ValueError("max_control_cycles cannot exceed max_observation_steps")
        wall_timeout_s = _require_positive_finite_real(
            self.wall_timeout_s, "wall_timeout_s"
        )
        if self.semantic_version != SEMANTIC_VERSION:
            raise ValueError(f"semantic_version must be {SEMANTIC_VERSION}")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "success_predicate_id", predicate_id)
        object.__setattr__(self, "max_control_cycles", max_control_cycles)
        object.__setattr__(self, "max_observation_steps", max_observation_steps)
        object.__setattr__(self, "execute_action_steps", execute_action_steps)
        object.__setattr__(self, "wall_timeout_s", wall_timeout_s)

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class PolicyEpisodeContext:
    """Policy-visible reset context without evaluator-only provenance."""

    episode_id: str
    task: str
    initial_seed: int
    exogenous_seed: int
    instruction: str
    action_spec: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", _require_nonempty_string(self.episode_id, "episode_id")
        )
        object.__setattr__(self, "task", _require_nonempty_string(self.task, "task"))
        object.__setattr__(
            self,
            "initial_seed",
            _require_nonnegative_integer(self.initial_seed, "initial_seed"),
        )
        object.__setattr__(
            self,
            "exogenous_seed",
            _require_nonnegative_integer(self.exogenous_seed, "exogenous_seed"),
        )
        object.__setattr__(
            self,
            "instruction",
            _require_nonempty_string(self.instruction, "instruction"),
        )
        object.__setattr__(self, "action_spec", _require_action_spec(self.action_spec))

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class PolicyIdentity:
    """Content-addressed identity and capability declaration for a policy."""

    system_id: str
    checkpoint_sha256: str
    config_sha256: str
    action_spec: str
    consumes_tactile: bool
    supports_structural_absence: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "system_id", _require_nonempty_string(self.system_id, "system_id")
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _require_sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self, "config_sha256", _require_sha256(self.config_sha256, "config_sha256")
        )
        object.__setattr__(self, "action_spec", _require_action_spec(self.action_spec))
        for field_name in ("consumes_tactile", "supports_structural_absence"):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be bool")

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class ActionPlan:
    """A policy proposal expressed in the benchmark's canonical action space."""

    action_spec: str
    source_step_index: int
    actions: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_spec", _require_action_spec(self.action_spec))
        object.__setattr__(
            self,
            "source_step_index",
            _require_nonnegative_integer(self.source_step_index, "source_step_index"),
        )
        object.__setattr__(
            self, "actions", _freeze_action_array(self.actions, "actions")
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class ResetReceipt:
    """Identity-bearing receipt returned by a deterministic backend reset."""

    episode_id: str
    initial_seed: int
    exogenous_seed: int
    simulator_state_sha256: str
    native_reset_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", _require_nonempty_string(self.episode_id, "episode_id")
        )
        object.__setattr__(
            self,
            "initial_seed",
            _require_nonnegative_integer(self.initial_seed, "initial_seed"),
        )
        object.__setattr__(
            self,
            "exogenous_seed",
            _require_nonnegative_integer(self.exogenous_seed, "exogenous_seed"),
        )
        object.__setattr__(
            self,
            "simulator_state_sha256",
            _require_sha256(self.simulator_state_sha256, "simulator_state_sha256"),
        )
        object.__setattr__(
            self,
            "native_reset_id",
            _require_nonempty_string(self.native_reset_id, "native_reset_id"),
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class BackendTransition:
    """One clean evaluator record and its native backend execution metadata."""

    clean_record: EvaluationRecord
    signal: BackendSignal
    native_step_id: int
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.clean_record, EvaluationRecord):
            raise TypeError("clean_record must be an EvaluationRecord")
        if any(item.active_fault_ids for item in self.clean_record.provenance):
            raise ValueError("clean_record may not contain active fault IDs")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "signal", _normalize_signal(self.signal))
        object.__setattr__(
            self,
            "native_step_id",
            _require_nonnegative_integer(self.native_step_id, "native_step_id"),
        )
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class ExecutionBatch:
    """A bounded sequence of transitions generated by one backend execution."""

    transitions: Tuple[BackendTransition, ...]
    executed_action_count: int

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if not transitions:
            raise ValueError("transitions must be non-empty")
        if not all(isinstance(item, BackendTransition) for item in transitions):
            raise TypeError("transitions must contain BackendTransition values")
        executed_action_count = _require_positive_integer(
            self.executed_action_count, "executed_action_count"
        )
        if executed_action_count > len(transitions):
            raise ValueError("executed_action_count cannot exceed transition count")
        benchmark_indices = [
            item.clean_record.observation.step_index for item in transitions
        ]
        if any(
            current != previous + 1
            for previous, current in zip(benchmark_indices, benchmark_indices[1:])
        ):
            raise ValueError("transition benchmark step indices must be dense")
        native_ids = [item.native_step_id for item in transitions]
        if any(
            current <= previous for previous, current in zip(native_ids, native_ids[1:])
        ):
            raise ValueError("native step IDs must strictly increase")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "executed_action_count", executed_action_count)


@dataclass(frozen=True)
class PolicyExecution:
    """The action and observation trace exposed back to a policy after execution."""

    action_plan_sha256: str
    executed_actions: Array
    delivered_observations: Tuple[ObservationRecord, ...]
    terminal_signal: BackendSignal

    def __post_init__(self) -> None:
        observations = tuple(self.delivered_observations)
        if not all(type(item) is ObservationRecord for item in observations):
            raise TypeError(
                "delivered_observations must contain exact ObservationRecord values"
            )
        actions = _freeze_action_array(self.executed_actions, "executed_actions")
        if len(observations) != actions.shape[0]:
            raise ValueError(
                "number of delivered observations must equal executed actions"
            )
        object.__setattr__(
            self,
            "action_plan_sha256",
            _require_sha256(self.action_plan_sha256, "action_plan_sha256"),
        )
        object.__setattr__(self, "executed_actions", actions)
        object.__setattr__(self, "delivered_observations", observations)
        object.__setattr__(
            self, "terminal_signal", _normalize_signal(self.terminal_signal)
        )
