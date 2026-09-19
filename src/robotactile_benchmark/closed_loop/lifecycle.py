"""Evidence capture and resource teardown helpers for closed-loop execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Tuple

from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import BackendTransition
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult


class ClosedLoopResourceCloseError(RuntimeError):
    """Report teardown failures after pre-close evidence publication."""

    def __init__(self, failure_codes: Tuple[str, ...]) -> None:
        if not failure_codes:
            raise ValueError("close failure codes cannot be empty")
        self.failure_codes = tuple(failure_codes)
        super().__init__(
            "closed-loop resource teardown failed after evidence publication: "
            + ",".join(self.failure_codes)
        )


def capture_execution_evidence(
    result: ClosedLoopTrialResult,
    finalization: Optional[DeliveryFinalization],
    action_entries: list[dict[str, object]],
    transition_entries: list[BackendTransition],
    initial_diagnostics: Mapping[str, object],
) -> ClosedLoopExecutionEvidence:
    """Freeze evaluator-owned state into one immutable evidence snapshot."""

    return ClosedLoopExecutionEvidence.from_runner_entries(
        result,
        finalization,
        action_entries,
        transition_entries,
        initial_diagnostics,
    )


def pre_close_evidence(
    capture: Optional[list[ClosedLoopExecutionEvidence]],
    normal_result: Optional[ClosedLoopTrialResult],
    finalization: Optional[DeliveryFinalization],
    action_entries: list[dict[str, object]],
    transition_entries: list[BackendTransition],
    initial_diagnostics: Mapping[str, object],
) -> Optional[ClosedLoopExecutionEvidence]:
    """Build the publishable snapshot, or return none before an accepted reset."""

    if capture:
        if len(capture) != 1:
            raise RuntimeError("pre-close evidence capture is not singular")
        return capture[0]
    if normal_result is None:
        return None
    return capture_execution_evidence(
        normal_result,
        finalization,
        action_entries,
        transition_entries,
        initial_diagnostics,
    )


def close_resources(
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
) -> Tuple[str, ...]:
    """Attempt both closes and return stable failure codes without raising."""

    failures = []
    try:
        backend.close()
    except (Exception, SystemExit):
        failures.append("backend_close_failed")
    try:
        policy.close()
    except (Exception, SystemExit):
        failures.append("policy_close_failed")
    return tuple(failures)
