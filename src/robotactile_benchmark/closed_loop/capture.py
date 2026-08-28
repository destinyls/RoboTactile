"""Immutable evaluator-only evidence captured by the closed-loop runner."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    BackendTransition,
)
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.closed_loop.result_hashes import action_trace_sha256
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.contracts import Array, freeze_array, freeze_value

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ActionTraceEntry:
    """Exact executed action prefix and the policy plan identity that produced it."""

    action_plan_sha256: str
    source_step_index: int
    executed_actions: Array

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.action_plan_sha256) is None:
            raise ValueError("action_plan_sha256 must be a lowercase SHA256")
        if isinstance(self.source_step_index, (bool, np.bool_)) or not isinstance(
            self.source_step_index, Integral
        ):
            raise TypeError("source_step_index must be an integer")
        source_step_index = int(self.source_step_index)
        if source_step_index < 0:
            raise ValueError("source_step_index must be non-negative")
        actions = freeze_array(self.executed_actions)
        if actions.dtype != np.float32 or actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError("executed_actions must be canonical float32 [N, 8]")
        if actions.shape[0] < 1 or not np.isfinite(actions).all():
            raise ValueError("executed_actions must be non-empty and finite")
        object.__setattr__(self, "source_step_index", source_step_index)
        object.__setattr__(self, "executed_actions", actions)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ActionTraceEntry:
        """Freeze one runner-owned hash entry without consulting policy internals."""

        if set(value) != {
            "action_plan_sha256",
            "source_step_index",
            "executed_actions",
        }:
            raise ValueError("action trace entry fields mismatch")
        executed_actions = value["executed_actions"]
        if not isinstance(executed_actions, np.ndarray):
            raise TypeError("executed_actions must be an ndarray")
        return cls(
            action_plan_sha256=value["action_plan_sha256"],  # type: ignore[arg-type]
            source_step_index=value["source_step_index"],  # type: ignore[arg-type]
            executed_actions=executed_actions,
        )

    def to_hash_entry(self) -> dict[str, object]:
        """Return the exact Task 3 action-trace hash domain."""

        return {
            "action_plan_sha256": self.action_plan_sha256,
            "source_step_index": self.source_step_index,
            "executed_actions": self.executed_actions,
        }


@dataclass(frozen=True)
class TransitionTraceEntry:
    """One executed backend transition with its JSON-safe diagnostics."""

    benchmark_step_index: int
    native_step_id: int
    signal: BackendSignal
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("benchmark_step_index", "native_step_id"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.signal, BackendSignal):
            object.__setattr__(self, "signal", BackendSignal(self.signal))
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("transition diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))

    @classmethod
    def from_backend_transition(
        cls, transition: BackendTransition
    ) -> TransitionTraceEntry:
        """Drop bulky records while retaining exact backend witnesses."""

        return cls(
            benchmark_step_index=transition.clean_record.observation.step_index,
            native_step_id=transition.native_step_id,
            signal=transition.signal,
            diagnostics=transition.diagnostics,
        )


@dataclass(frozen=True)
class ClosedLoopExecutionEvidence:
    """Defensive snapshot of evaluator-owned evidence after runner teardown."""

    result: ClosedLoopTrialResult
    finalization: Optional[DeliveryFinalization]
    action_entries: Tuple[ActionTraceEntry, ...]
    transition_entries: Tuple[TransitionTraceEntry, ...] = ()
    initial_diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.result, ClosedLoopTrialResult):
            raise TypeError("result must be a ClosedLoopTrialResult")
        if self.finalization is not None and not isinstance(
            self.finalization, DeliveryFinalization
        ):
            raise TypeError("finalization must be a DeliveryFinalization or None")
        entries = tuple(self.action_entries)
        if not all(isinstance(entry, ActionTraceEntry) for entry in entries):
            raise TypeError("action_entries must contain ActionTraceEntry values")
        if (
            action_trace_sha256(tuple(entry.to_hash_entry() for entry in entries))
            != self.result.action_trace_sha256
        ):
            raise ValueError("captured actions do not match the terminal result")
        if self.finalization is not None and (
            self.finalization.clean_trace_sha256 != self.result.clean_trace_sha256
            or self.finalization.delivered_trace_sha256
            != self.result.delivered_trace_sha256
        ):
            raise ValueError("captured delivery does not match the terminal result")
        transitions = tuple(self.transition_entries)
        if not all(isinstance(entry, TransitionTraceEntry) for entry in transitions):
            raise TypeError(
                "transition_entries must contain TransitionTraceEntry values"
            )
        if transitions:
            executed = sum(entry.executed_actions.shape[0] for entry in entries)
            if executed != len(transitions):
                raise ValueError("captured transition/action lengths do not match")
            benchmark_steps = [entry.benchmark_step_index for entry in transitions]
            native_steps = [entry.native_step_id for entry in transitions]
            if any(
                current != previous + 1
                for previous, current in zip(benchmark_steps, benchmark_steps[1:])
            ):
                raise ValueError("captured transition benchmark steps are not dense")
            if any(
                current <= previous
                for previous, current in zip(native_steps, native_steps[1:])
            ):
                raise ValueError("captured transition native steps are not increasing")
        object.__setattr__(self, "action_entries", entries)
        object.__setattr__(self, "transition_entries", transitions)
        if not isinstance(self.initial_diagnostics, Mapping):
            raise TypeError("initial_diagnostics must be a mapping")
        object.__setattr__(
            self, "initial_diagnostics", freeze_value(self.initial_diagnostics)
        )

    @classmethod
    def from_runner_entries(
        cls,
        result: ClosedLoopTrialResult,
        finalization: Optional[DeliveryFinalization],
        entries: Sequence[Mapping[str, object]],
        transitions: Sequence[BackendTransition] = (),
        initial_diagnostics: Mapping[str, object] | None = None,
    ) -> ClosedLoopExecutionEvidence:
        """Copy live runner entries into immutable evaluator evidence."""

        return cls(
            result=result,
            finalization=finalization,
            action_entries=tuple(
                ActionTraceEntry.from_mapping(item) for item in entries
            ),
            transition_entries=tuple(
                TransitionTraceEntry.from_backend_transition(item)
                for item in transitions
            ),
            initial_diagnostics=(
                {} if initial_diagnostics is None else initial_diagnostics
            ),
        )

    @property
    def action_trace_sha256(self) -> str:
        return action_trace_sha256(
            tuple(entry.to_hash_entry() for entry in self.action_entries)
        )
