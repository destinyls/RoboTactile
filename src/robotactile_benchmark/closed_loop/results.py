"""Fail-closed public receipt for a completed closed-loop trial."""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.closed_loop.contracts import SEMANTIC_VERSION
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.trials import TerminalStatus

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CRASH_FAILURE_STAGES = frozenset(
    {
        "reset",
        "policy_reset",
        "observe",
        "delivery",
        "infer",
        "execute",
        "commit",
        "validation",
        "close",
    }
)


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _optional_sha256(value: object, name: str) -> Optional[str]:
    return None if value is None else _require_sha256(value, name)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _optional_bool(value: object, name: str) -> Optional[bool]:
    if value is None:
        return None
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool or None")
    return value


def _optional_code(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string or None")
    return value


def _status(value: object, name: str) -> TerminalStatus:
    if type(value) is not TerminalStatus:
        raise TypeError(f"{name} must be an exact TerminalStatus")
    return value


@dataclass(frozen=True)
class ClosedLoopTrialResult:
    """Immutable receipt whose terminal hash verifies all public fields."""

    trial_manifest_sha256: str
    pair_key: str
    run_spec_sha256: str
    initial_state_sha256: Optional[str]
    terminal_status: TerminalStatus
    execution_status: Optional[TerminalStatus]
    score_eligible: bool
    score_success: Optional[bool]
    validation_passed: Optional[bool]
    validation_failure_codes: Tuple[str, ...]
    clean_trace_sha256: str
    delivered_trace_sha256: str
    action_trace_sha256: str
    terminal_trace_sha256: str
    observation_count: int
    control_cycle_count: int
    failure_stage: Optional[str]
    failure_code: Optional[str]
    semantic_version: str = SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "trial_manifest_sha256",
            "pair_key",
            "run_spec_sha256",
            "clean_trace_sha256",
            "delivered_trace_sha256",
            "action_trace_sha256",
            "terminal_trace_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "initial_state_sha256",
            _optional_sha256(self.initial_state_sha256, "initial_state_sha256"),
        )
        terminal_status = _status(self.terminal_status, "terminal_status")
        execution_status = self.execution_status
        if execution_status is not None:
            execution_status = _status(execution_status, "execution_status")
        if type(self.score_eligible) is not bool:
            raise TypeError("score_eligible must be bool")
        score_success = _optional_bool(self.score_success, "score_success")
        validation_passed = _optional_bool(
            self.validation_passed,
            "validation_passed",
        )
        validation_codes = self._validate_codes(self.validation_failure_codes)
        observation_count = _nonnegative_int(
            self.observation_count,
            "observation_count",
        )
        control_cycle_count = _nonnegative_int(
            self.control_cycle_count,
            "control_cycle_count",
        )
        failure_stage = _optional_code(self.failure_stage, "failure_stage")
        failure_code = _optional_code(self.failure_code, "failure_code")
        if self.semantic_version != SEMANTIC_VERSION:
            raise ValueError(f"semantic_version must be {SEMANTIC_VERSION}")
        self._validate_semantics(
            terminal_status,
            execution_status,
            score_success,
            validation_passed,
            validation_codes,
            failure_stage,
            failure_code,
        )
        self._validate_reset_state(terminal_status)
        self._validate_terminal_hash(
            terminal_status,
            execution_status,
            score_success,
            validation_passed,
            validation_codes,
            observation_count,
            control_cycle_count,
            failure_stage,
            failure_code,
        )
        object.__setattr__(self, "execution_status", execution_status)
        object.__setattr__(self, "score_success", score_success)
        object.__setattr__(self, "validation_passed", validation_passed)
        object.__setattr__(self, "validation_failure_codes", validation_codes)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "control_cycle_count", control_cycle_count)
        object.__setattr__(self, "failure_stage", failure_stage)
        object.__setattr__(self, "failure_code", failure_code)

    @staticmethod
    def _validate_codes(value: object) -> Tuple[str, ...]:
        if not isinstance(value, tuple):
            raise TypeError("validation_failure_codes must be a tuple")
        if not all(isinstance(code, str) and code for code in value):
            raise ValueError("validation_failure_codes must contain non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("validation_failure_codes must be unique")
        return value

    def _validate_semantics(
        self,
        terminal_status: TerminalStatus,
        execution_status: Optional[TerminalStatus],
        score_success: Optional[bool],
        validation_passed: Optional[bool],
        validation_codes: Tuple[str, ...],
        failure_stage: Optional[str],
        failure_code: Optional[str],
    ) -> None:
        normal = {
            TerminalStatus.SUCCESS,
            TerminalStatus.TASK_FAILURE,
            TerminalStatus.EARLY_STOP,
            TerminalStatus.TIMEOUT,
        }
        if terminal_status in normal:
            if (
                execution_status is not terminal_status
                or not self.score_eligible
                or score_success is not (terminal_status is TerminalStatus.SUCCESS)
                or validation_passed is not True
                or validation_codes
                or failure_stage is not None
                or failure_code is not None
            ):
                raise ValueError(
                    "normal terminal status has incoherent result semantics"
                )
            return
        if terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT:
            if (
                execution_status is not None
                or self.score_eligible
                or score_success is not None
                or validation_passed is not None
                or validation_codes
                or failure_stage != "preflight"
                or failure_code is None
            ):
                raise ValueError("unsupported result has incoherent semantics")
            return
        if terminal_status is TerminalStatus.VALIDATOR_REJECTED:
            if (
                execution_status is None
                or self.score_eligible
                or score_success is not None
                or validation_passed is not False
                or not validation_codes
                or failure_stage != "validation"
                or failure_code != "validator_rejected"
            ):
                raise ValueError("validator rejection has incoherent semantics")
            return
        if terminal_status is TerminalStatus.CRASH:
            if (
                not self.score_eligible
                or score_success is not False
                or execution_status is None
                or failure_stage is None
                or failure_code is None
                or failure_stage not in _CRASH_FAILURE_STAGES
                or (failure_stage != "reset" and self.initial_state_sha256 is None)
                or (validation_passed is False and not validation_codes)
                or (validation_passed is not False and validation_codes)
            ):
                raise ValueError("crash result has incoherent semantics")
            return
        raise ValueError("unsupported terminal status")

    def _validate_reset_state(self, terminal_status: TerminalStatus) -> None:
        completed = {
            TerminalStatus.SUCCESS,
            TerminalStatus.TASK_FAILURE,
            TerminalStatus.EARLY_STOP,
            TerminalStatus.TIMEOUT,
            TerminalStatus.VALIDATOR_REJECTED,
        }
        if terminal_status in completed and self.initial_state_sha256 is None:
            raise ValueError("completed result requires an initial state hash")
        if (
            terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
            and self.initial_state_sha256 is not None
        ):
            raise ValueError("unsupported result cannot include a reset state hash")

    def _validate_terminal_hash(
        self,
        terminal_status: TerminalStatus,
        execution_status: Optional[TerminalStatus],
        score_success: Optional[bool],
        validation_passed: Optional[bool],
        validation_codes: Tuple[str, ...],
        observation_count: int,
        control_cycle_count: int,
        failure_stage: Optional[str],
        failure_code: Optional[str],
    ) -> None:
        from robotactile_benchmark.closed_loop.result_hashes import (
            terminal_trace_sha256,
        )

        expected = terminal_trace_sha256(
            trial_manifest_sha256=self.trial_manifest_sha256,
            pair_key=self.pair_key,
            run_spec_sha256=self.run_spec_sha256,
            initial_state_sha256=self.initial_state_sha256,
            terminal_status=terminal_status,
            execution_status=execution_status,
            score_eligible=self.score_eligible,
            score_success=score_success,
            validation_passed=validation_passed,
            validation_failure_codes=validation_codes,
            clean_trace_sha256=self.clean_trace_sha256,
            delivered_trace_sha256=self.delivered_trace_sha256,
            action_trace_sha256_value=self.action_trace_sha256,
            observation_count=observation_count,
            control_cycle_count=control_cycle_count,
            failure_stage=failure_stage,
            failure_code=failure_code,
            semantic_version=self.semantic_version,
        )
        if self.terminal_trace_sha256 != expected:
            raise ValueError("terminal_trace_sha256 does not match receipt fields")

    @property
    def sha256(self) -> str:
        """Return the canonical content address of this verified receipt."""

        return canonical_hash(self)
