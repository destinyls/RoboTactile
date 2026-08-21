"""Trace hashing and result construction for closed-loop trials."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple

from robotactile_benchmark.closed_loop.contracts import (
    SEMANTIC_VERSION,
    ClosedLoopRunSpec,
)
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.trials import TerminalStatus, TrialManifest

_ACTION_TRACE_NAMESPACE = "robotactile_benchmark.closed_loop.action_trace.v1"
_EMPTY_RECORD_TRACE_NAMESPACE = "robotactile_benchmark.closed_loop.empty_trace.v1"
_TERMINAL_TRACE_NAMESPACE = "robotactile_benchmark.closed_loop.terminal_trace.v1"


def action_trace_sha256(entries: Sequence[dict[str, object]]) -> str:
    """Hash ordered policy plans and their exact executed action prefixes."""

    return canonical_hash({"namespace": _ACTION_TRACE_NAMESPACE, "entries": entries})


def empty_record_trace_sha256(kind: str) -> str:
    """Return a deterministic trace digest before any observation is delivered."""

    return canonical_hash({"namespace": _EMPTY_RECORD_TRACE_NAMESPACE, "kind": kind})


def terminal_trace_sha256(
    *,
    trial_manifest_sha256: str,
    pair_key: str,
    run_spec_sha256: str,
    initial_state_sha256: Optional[str],
    terminal_status: TerminalStatus,
    execution_status: Optional[TerminalStatus],
    score_eligible: bool,
    score_success: Optional[bool],
    validation_passed: Optional[bool],
    validation_failure_codes: Tuple[str, ...],
    clean_trace_sha256: str,
    delivered_trace_sha256: str,
    action_trace_sha256_value: str,
    observation_count: int,
    control_cycle_count: int,
    failure_stage: Optional[str],
    failure_code: Optional[str],
    semantic_version: str,
) -> str:
    """Hash every result field except this self-verifying terminal digest."""

    return canonical_hash(
        {
            "namespace": _TERMINAL_TRACE_NAMESPACE,
            "trial_manifest_sha256": trial_manifest_sha256,
            "pair_key": pair_key,
            "run_spec_sha256": run_spec_sha256,
            "initial_state_sha256": initial_state_sha256,
            "terminal_status": terminal_status.value,
            "execution_status": (
                None if execution_status is None else execution_status.value
            ),
            "score_eligible": score_eligible,
            "score_success": score_success,
            "validation_passed": validation_passed,
            "validation_failure_codes": validation_failure_codes,
            "clean_trace_sha256": clean_trace_sha256,
            "delivered_trace_sha256": delivered_trace_sha256,
            "action_trace_sha256": action_trace_sha256_value,
            "observation_count": observation_count,
            "control_cycle_count": control_cycle_count,
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "semantic_version": semantic_version,
        }
    )


def build_trial_result(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    *,
    initial_state_sha256: Optional[str],
    terminal_status: TerminalStatus,
    execution_status: Optional[TerminalStatus],
    finalization: Optional[DeliveryFinalization],
    action_entries: Sequence[dict[str, object]],
    observation_count: int,
    control_cycle_count: int,
    failure_stage: Optional[str],
    failure_code: Optional[str],
    validation_override: Optional[tuple[Optional[bool], Tuple[str, ...]]] = None,
) -> ClosedLoopTrialResult:
    """Bind the complete execution receipt into a fail-closed public result."""

    clean_hash, delivered_hash, validation_passed, validation_codes = _trace_fields(
        finalization
    )
    if validation_override is not None:
        validation_passed, validation_codes = validation_override
    score_eligible, score_success = _score(terminal_status)
    action_hash = action_trace_sha256(action_entries)
    terminal_hash = terminal_trace_sha256(
        trial_manifest_sha256=trial.sha256,
        pair_key=trial.pair_key,
        run_spec_sha256=run_spec.sha256,
        initial_state_sha256=initial_state_sha256,
        terminal_status=terminal_status,
        execution_status=execution_status,
        score_eligible=score_eligible,
        score_success=score_success,
        validation_passed=validation_passed,
        validation_failure_codes=validation_codes,
        clean_trace_sha256=clean_hash,
        delivered_trace_sha256=delivered_hash,
        action_trace_sha256_value=action_hash,
        observation_count=observation_count,
        control_cycle_count=control_cycle_count,
        failure_stage=failure_stage,
        failure_code=failure_code,
        semantic_version=SEMANTIC_VERSION,
    )
    return ClosedLoopTrialResult(
        trial_manifest_sha256=trial.sha256,
        pair_key=trial.pair_key,
        run_spec_sha256=run_spec.sha256,
        initial_state_sha256=initial_state_sha256,
        terminal_status=terminal_status,
        execution_status=execution_status,
        score_eligible=score_eligible,
        score_success=score_success,
        validation_passed=validation_passed,
        validation_failure_codes=validation_codes,
        clean_trace_sha256=clean_hash,
        delivered_trace_sha256=delivered_hash,
        action_trace_sha256=action_hash,
        terminal_trace_sha256=terminal_hash,
        observation_count=observation_count,
        control_cycle_count=control_cycle_count,
        failure_stage=failure_stage,
        failure_code=failure_code,
    )


def _score(status: TerminalStatus) -> tuple[bool, Optional[bool]]:
    if status is TerminalStatus.SUCCESS:
        return True, True
    if status in {
        TerminalStatus.TASK_FAILURE,
        TerminalStatus.EARLY_STOP,
        TerminalStatus.TIMEOUT,
        TerminalStatus.CRASH,
    }:
        return True, False
    return False, None


def _trace_fields(
    finalization: Optional[DeliveryFinalization],
) -> tuple[str, str, Optional[bool], Tuple[str, ...]]:
    if finalization is None:
        return (
            empty_record_trace_sha256("clean"),
            empty_record_trace_sha256("delivered"),
            None,
            (),
        )
    if finalization.validation is None:
        return (
            finalization.clean_trace_sha256,
            finalization.delivered_trace_sha256,
            True,
            (),
        )
    return (
        finalization.clean_trace_sha256,
        finalization.delivered_trace_sha256,
        finalization.validation.passed,
        finalization.validation.failure_codes,
    )
