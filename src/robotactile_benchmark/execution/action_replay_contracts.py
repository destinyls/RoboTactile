"""Immutable contracts for diagnostic live action-trace replay."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Tuple

from robotactile_benchmark.closed_loop.capture import ActionTraceEntry
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.result_hashes import action_trace_sha256
from robotactile_benchmark.contracts import canonical_hash, freeze_value, thaw_value
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.trials import Condition, TerminalStatus, TrialManifest

ACTION_REPLAY_EVIDENCE_LEVEL = "diagnostic_live_action_trace_replay_v1"
ACTION_REPLAY_SEMANTIC_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ActionReplayError(RuntimeError):
    """Stable failure for a replay whose comparison contract is invalid."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _require_nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class ActionReplaySource:
    """Verified Clean action trace used as a simulator-only replay source."""

    trial: TrialManifest
    run_spec: ClosedLoopRunSpec
    artifact_root_sha256: str
    initial_state_sha256: str
    action_trace_sha256: str
    source_terminal_status: TerminalStatus
    source_score_success: bool
    source_control_cycle_count: int
    entries: Tuple[ActionTraceEntry, ...]

    def __post_init__(self) -> None:
        if type(self.trial) is not TrialManifest:
            raise TypeError("trial must be an exact TrialManifest")
        if self.trial.condition is not Condition.CLEAN:
            raise ValueError("action replay source must be Clean")
        if type(self.run_spec) is not ClosedLoopRunSpec:
            raise TypeError("run_spec must be an exact ClosedLoopRunSpec")
        for name in (
            "artifact_root_sha256",
            "initial_state_sha256",
            "action_trace_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        status = TerminalStatus(self.source_terminal_status)
        if status not in {
            TerminalStatus.SUCCESS,
            TerminalStatus.TASK_FAILURE,
            TerminalStatus.EARLY_STOP,
            TerminalStatus.TIMEOUT,
        }:
            raise ValueError("source must be a completed score-eligible episode")
        if type(self.source_score_success) is not bool:
            raise TypeError("source_score_success must be bool")
        if self.source_score_success is not (status is TerminalStatus.SUCCESS):
            raise ValueError("source success disagrees with terminal status")
        cycles = _require_nonnegative(
            self.source_control_cycle_count, "source_control_cycle_count"
        )
        entries = tuple(self.entries)
        if not entries or not all(type(item) is ActionTraceEntry for item in entries):
            raise ValueError("action replay source requires exact action entries")
        expected_step = 0
        for entry in entries:
            if entry.source_step_index != expected_step:
                raise ValueError("source action entries are not densely anchored")
            expected_step += int(entry.executed_actions.shape[0])
        if cycles != len(entries):
            raise ValueError("source control cycles disagree with action entries")
        computed = action_trace_sha256(
            tuple(entry.to_hash_entry() for entry in entries)
        )
        if computed != self.action_trace_sha256:
            raise ValueError("source action entries disagree with action trace SHA256")
        object.__setattr__(self, "source_terminal_status", status)
        object.__setattr__(self, "source_control_cycle_count", cycles)
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_artifact(cls, artifact: LoadedLiveUniVTACArtifact) -> ActionReplaySource:
        """Extract an independently validated Clean trace from a live artifact."""

        if type(artifact) is not LoadedLiveUniVTACArtifact:
            raise TypeError("artifact must be an exact LoadedLiveUniVTACArtifact")
        result = artifact.evidence.result
        if (
            artifact.trial.condition is not Condition.CLEAN
            or not result.score_eligible
            or type(result.score_success) is not bool
            or result.validation_passed is not True
            or result.initial_state_sha256 is None
        ):
            raise ValueError(
                "action replay requires a validated score-eligible Clean artifact"
            )
        return cls(
            trial=artifact.trial,
            run_spec=artifact.run_spec,
            artifact_root_sha256=artifact.external_root_sha256,
            initial_state_sha256=result.initial_state_sha256,
            action_trace_sha256=result.action_trace_sha256,
            source_terminal_status=result.terminal_status,
            source_score_success=result.score_success,
            source_control_cycle_count=result.control_cycle_count,
            entries=artifact.evidence.action_entries,
        )

    @property
    def planned_action_count(self) -> int:
        return sum(int(entry.executed_actions.shape[0]) for entry in self.entries)


@dataclass(frozen=True)
class ActionTraceReplayReceipt:
    """Hashable diagnostic receipt for a fresh live simulator action replay."""

    task: str
    initial_seed: int
    exogenous_seed: int
    source_artifact_root_sha256: str
    source_trial_manifest_sha256: str
    source_action_trace_sha256: str
    source_initial_state_sha256: str
    source_terminal_status: TerminalStatus
    source_score_success: bool
    replay_initial_state_sha256: str
    initial_state_exact_match: bool
    planned_control_cycle_count: int
    replayed_control_cycle_count: int
    planned_action_count: int
    replayed_action_count: int
    replayed_action_trace_sha256: str
    transition_trace_sha256: str
    terminal_status: TerminalStatus
    task_success: bool
    failure_code: Optional[str]
    initial_diagnostics: Mapping[str, object] = field(default_factory=dict)
    final_transition_diagnostics: Mapping[str, object] = field(default_factory=dict)
    evidence_level: str = ACTION_REPLAY_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = True
    policy_inference_executed: bool = False
    paper_claim: bool = False
    semantic_version: str = ACTION_REPLAY_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task must be non-empty")
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(
                self, name, _require_nonnegative(getattr(self, name), name)
            )
        for name in (
            "source_artifact_root_sha256",
            "source_trial_manifest_sha256",
            "source_action_trace_sha256",
            "source_initial_state_sha256",
            "replay_initial_state_sha256",
            "replayed_action_trace_sha256",
            "transition_trace_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        source_status = TerminalStatus(self.source_terminal_status)
        status = TerminalStatus(self.terminal_status)
        for name in (
            "source_score_success",
            "initial_state_exact_match",
            "task_success",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.source_score_success is not (source_status is TerminalStatus.SUCCESS):
            raise ValueError("source success disagrees with source terminal status")
        if not self.initial_state_exact_match or (
            self.source_initial_state_sha256 != self.replay_initial_state_sha256
        ):
            raise ValueError("published replay must have an exact initial-state match")
        if self.task_success is not (status is TerminalStatus.SUCCESS):
            raise ValueError("replay success disagrees with terminal status")
        for name in (
            "planned_control_cycle_count",
            "replayed_control_cycle_count",
            "planned_action_count",
            "replayed_action_count",
        ):
            object.__setattr__(
                self, name, _require_nonnegative(getattr(self, name), name)
            )
        if (
            self.replayed_control_cycle_count > self.planned_control_cycle_count
            or self.replayed_action_count > self.planned_action_count
        ):
            raise ValueError("replay cannot exceed its source action trace")
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str) or not self.failure_code
        ):
            raise ValueError("failure_code must be non-empty or None")
        if status is TerminalStatus.TIMEOUT:
            if self.failure_code != "action_trace_exhausted":
                raise ValueError("replay timeout must identify action trace exhaustion")
        elif self.failure_code is not None:
            raise ValueError(
                "terminal backend signal cannot carry a replay failure code"
            )
        for name in ("initial_diagnostics", "final_transition_diagnostics"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, freeze_value(value))
        if (
            self.evidence_level != ACTION_REPLAY_EVIDENCE_LEVEL
            or self.simulator_execution_claimed is not True
            or self.policy_inference_executed is not False
            or self.paper_claim is not False
            or self.semantic_version != ACTION_REPLAY_SEMANTIC_VERSION
        ):
            raise ValueError("action replay evidence contract mismatch")
        object.__setattr__(self, "source_terminal_status", source_status)
        object.__setattr__(self, "terminal_status", status)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "initial_seed": self.initial_seed,
            "exogenous_seed": self.exogenous_seed,
            "source_artifact_root_sha256": self.source_artifact_root_sha256,
            "source_trial_manifest_sha256": self.source_trial_manifest_sha256,
            "source_action_trace_sha256": self.source_action_trace_sha256,
            "source_initial_state_sha256": self.source_initial_state_sha256,
            "source_terminal_status": self.source_terminal_status.value,
            "source_score_success": self.source_score_success,
            "replay_initial_state_sha256": self.replay_initial_state_sha256,
            "initial_state_exact_match": self.initial_state_exact_match,
            "planned_control_cycle_count": self.planned_control_cycle_count,
            "replayed_control_cycle_count": self.replayed_control_cycle_count,
            "planned_action_count": self.planned_action_count,
            "replayed_action_count": self.replayed_action_count,
            "replayed_action_trace_sha256": self.replayed_action_trace_sha256,
            "transition_trace_sha256": self.transition_trace_sha256,
            "terminal_status": self.terminal_status.value,
            "task_success": self.task_success,
            "failure_code": self.failure_code,
            "initial_diagnostics": thaw_value(self.initial_diagnostics),
            "final_transition_diagnostics": thaw_value(
                self.final_transition_diagnostics
            ),
            "evidence_level": self.evidence_level,
            "simulator_execution_claimed": self.simulator_execution_claimed,
            "policy_inference_executed": self.policy_inference_executed,
            "paper_claim": self.paper_claim,
            "semantic_version": self.semantic_version,
        }


__all__ = [
    "ACTION_REPLAY_EVIDENCE_LEVEL",
    "ACTION_REPLAY_SEMANTIC_VERSION",
    "ActionReplayError",
    "ActionReplaySource",
    "ActionTraceReplayReceipt",
]
