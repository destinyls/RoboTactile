"""Managed pre-close evidence publication for single live runtimes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Tuple

from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import (
    ClosedLoopRunSpec,
    InitialStatePolicy,
)
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.closed_loop.lifecycle import (
    ClosedLoopResourceCloseError,
)
from robotactile_benchmark.closed_loop.runner import _run_closed_loop_trial
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.trials import TrialManifest


def run_closed_loop_trial_with_evidence_before_close(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
    before_close: Callable[[ClosedLoopExecutionEvidence], None],
    *,
    fault_manifest: Optional[FaultManifest] = None,
    rest_references: Optional[RestReferenceBundle] = None,
    monotonic_clock: Optional[Callable[[], float]] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION,
) -> ClosedLoopExecutionEvidence:
    """Publish evidence before teardown and fail closed on returned close errors."""

    captured: list[ClosedLoopExecutionEvidence] = []
    close_failures: list[Tuple[str, ...]] = []
    _run_closed_loop_trial(
        trial,
        run_spec,
        backend,
        policy,
        fault_manifest=fault_manifest,
        rest_references=rest_references,
        monotonic_clock=monotonic_clock,
        stage_observer=stage_observer,
        initial_state_policy=initial_state_policy,
        capture=captured,
        before_close=before_close,
        close_failures=close_failures,
    )
    if len(close_failures) != 1:
        raise RuntimeError("closed-loop runner did not emit one teardown receipt")
    if close_failures[0]:
        raise ClosedLoopResourceCloseError(close_failures[0])
    if len(captured) != 1:
        raise RuntimeError(
            "closed-loop runner did not emit exactly one evidence snapshot"
        )
    return captured[0]


__all__ = ["run_closed_loop_trial_with_evidence_before_close"]
