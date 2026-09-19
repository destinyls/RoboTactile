"""Deterministic in-memory execution of one audited closed-loop trial."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from time import monotonic
from typing import Optional, Tuple, Union

from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    BackendTransition,
    ClosedLoopRunSpec,
    InitialStatePolicy,
    PolicyExecution,
    WallTimeoutRole,
)
from robotactile_benchmark.closed_loop.crash_diagnostics import emit_crash_marker
from robotactile_benchmark.closed_loop.delivery import (
    DeliveryFinalization,
    IdentityDeliverySession,
    OnlineFaultSession,
)
from robotactile_benchmark.closed_loop.failure_evidence import (
    build_runner_failure_evidence,
)
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.closed_loop.lifecycle import (
    capture_execution_evidence,
    close_resources,
    pre_close_evidence,
)
from robotactile_benchmark.closed_loop.result_hashes import build_trial_result
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.closed_loop.runner_checks import (
    RunnerViolation,
    build_episode_context,
    preflight,
    validate_batch,
    validate_clean_record,
    validate_reset,
)
from robotactile_benchmark.contracts import ObservationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.trials import TerminalStatus, TrialManifest

_STRUCTURAL_ABSENCE_OPERATOR_IDS = frozenset({"A1_stream_absence", "A2_frame_erasure"})
_RESET_VIABLE_PATHS = (
    ("reset_viable",),
    ("task", "reset_viable"),
    ("task", "placement_reset_assessment", "reset_viable"),
)
_INITIAL_EARLY_STOP_PATHS = (
    ("early_stop",),
    ("task", "early_stop_predicate"),
    ("task", "placement_reset_assessment", "current_early_stop"),
)
_INITIAL_SUCCESS_PATHS = (
    ("success_check",),
    ("task", "predicate_success"),
)


def run_closed_loop_trial(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
    *,
    fault_manifest: Optional[FaultManifest] = None,
    rest_references: Optional[RestReferenceBundle] = None,
    monotonic_clock: Optional[Callable[[], float]] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION,
) -> ClosedLoopTrialResult:
    """Execute one requested trial with no file-system side effects."""

    return _run_closed_loop_trial(
        trial,
        run_spec,
        backend,
        policy,
        fault_manifest=fault_manifest,
        rest_references=rest_references,
        monotonic_clock=monotonic_clock,
        stage_observer=stage_observer,
        initial_state_policy=initial_state_policy,
        capture=None,
    )


def _run_closed_loop_trial(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
    *,
    fault_manifest: Optional[FaultManifest] = None,
    rest_references: Optional[RestReferenceBundle] = None,
    monotonic_clock: Optional[Callable[[], float]] = None,
    capture: Optional[list[ClosedLoopExecutionEvidence]] = None,
    before_close: Optional[Callable[[ClosedLoopExecutionEvidence], None]] = None,
    close_failures: Optional[list[Tuple[str, ...]]] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION,
) -> ClosedLoopTrialResult:
    """Execute one requested trial with no file-system side effects."""

    clock = monotonic_clock if monotonic_clock is not None else monotonic
    initial_state_sha256: Optional[str] = None
    finalization: Optional[DeliveryFinalization] = None
    action_entries: list[dict[str, object]] = []
    transition_entries: list[BackendTransition] = []
    initial_diagnostics: Mapping[str, object] = {}
    observation_count = 0
    control_cycle_count = 0
    backend_signal = BackendSignal.RUNNING
    session: Optional[Union[IdentityDeliverySession, OnlineFaultSession]] = None
    reset_attempted = False
    policy_reset_entered = False
    previous_native_step_id: Optional[int] = None
    stage = "preflight"
    primary_crash = False
    normal_result: Optional[ClosedLoopTrialResult] = None
    try:
        _record_lifecycle_stage(stage_observer, "closed_loop_preflight")
        preflight(
            trial,
            run_spec,
            backend,
            policy,
            fault_manifest,
            rest_references,
        )
        if (
            fault_manifest is not None
            and fault_manifest.operator_id in _STRUCTURAL_ABSENCE_OPERATOR_IDS
            and not policy.identity.supports_structural_absence
        ):
            result = build_trial_result(
                trial,
                run_spec,
                initial_state_sha256=None,
                terminal_status=TerminalStatus.UNSUPPORTED_CONTRACT,
                execution_status=None,
                finalization=None,
                action_entries=action_entries,
                observation_count=0,
                control_cycle_count=0,
                failure_stage="preflight",
                failure_code="structural_absence_unsupported",
            )
            _record_evidence(
                capture,
                result,
                None,
                action_entries,
                transition_entries,
                initial_diagnostics,
            )
            return result
        context = build_episode_context(trial, run_spec.prompt)
        session = (
            OnlineFaultSession(fault_manifest, rest_references)
            if fault_manifest is not None
            else IdentityDeliverySession()
        )
        stage = "reset"
        _record_lifecycle_stage(stage_observer, stage)
        reset_attempted = True
        receipt = backend.reset(context)
        initial_diagnostics = receipt.diagnostics
        initial_state_sha256 = receipt.simulator_state_sha256
        validate_reset(receipt, context)
        _validate_initial_state(initial_diagnostics, initial_state_policy)
        stage = "policy_reset"
        _record_lifecycle_stage(stage_observer, stage)
        policy_reset_entered = True
        policy.reset(context)
        started_at = clock()
        stage = "observe"
        _record_lifecycle_stage(stage_observer, stage)
        initial_clean = backend.observe()
        validate_clean_record(initial_clean, context, 0)
        stage = "delivery"
        _record_lifecycle_stage(stage_observer, stage)
        delivered = session.deliver(initial_clean)
        if type(delivered.observation) is not ObservationRecord:
            raise RunnerViolation("delivered_observation_type_mismatch")
        current_observation = delivered.observation
        observation_count = 1
        while True:
            if (
                run_spec.wall_timeout_role is WallTimeoutRole.SCORING_BOUNDARY_V1
                and clock() - started_at >= run_spec.wall_timeout_s
            ):
                backend_signal = BackendSignal.TIMEOUT
                break
            if (
                control_cycle_count >= run_spec.max_control_cycles
                or observation_count >= run_spec.max_observation_steps
            ):
                backend_signal = BackendSignal.TIMEOUT
                break
            stage = "infer"
            _record_lifecycle_stage(stage_observer, stage)
            action_plan = policy.infer(current_observation)
            elapsed_after_infer = (
                clock() - started_at
                if run_spec.wall_timeout_role is WallTimeoutRole.SCORING_BOUNDARY_V1
                else None
            )
            if (
                elapsed_after_infer is not None
                and elapsed_after_infer >= run_spec.wall_timeout_s
            ):
                initial_diagnostics = {
                    **dict(initial_diagnostics),
                    "runner_terminal_timing": {
                        "stage": "after_infer",
                        "elapsed_s": elapsed_after_infer,
                    },
                }
                backend_signal = BackendSignal.TIMEOUT
                break
            if action_plan.action_spec != trial.action_spec:
                raise RunnerViolation("action_plan_action_spec_mismatch")
            if action_plan.source_step_index != current_observation.step_index:
                raise RunnerViolation("action_plan_source_step_mismatch")
            requested_count = min(
                run_spec.execute_action_steps,
                run_spec.max_observation_steps - observation_count,
                action_plan.actions.shape[0],
            )
            if requested_count < 1:
                backend_signal = BackendSignal.TIMEOUT
                break
            requested_actions = action_plan.actions[:requested_count]
            stage = "execute"
            _record_lifecycle_stage(stage_observer, stage)
            batch = backend.execute(requested_actions)
            previous_native_step_id = validate_batch(
                batch,
                requested_count,
                context,
                observation_count,
                previous_native_step_id,
            )
            executed_count = batch.executed_action_count
            transition_entries.extend(batch.transitions)
            action_entries.append(
                {
                    "action_plan_sha256": action_plan.sha256,
                    "source_step_index": action_plan.source_step_index,
                    "executed_actions": requested_actions[:executed_count],
                }
            )
            stage = "delivery"
            _record_lifecycle_stage(stage_observer, stage)
            delivered_next = tuple(
                session.deliver(transition.clean_record)
                for transition in batch.transitions
            )
            control_cycle_count += 1
            backend_signal = batch.transitions[-1].signal
            observation_count += executed_count
            current_observation = delivered_next[-1].observation
            elapsed_before_commit = (
                clock() - started_at
                if run_spec.wall_timeout_role is WallTimeoutRole.SCORING_BOUNDARY_V1
                else None
            )
            runner_budget_exhausted = (
                control_cycle_count >= run_spec.max_control_cycles
                or observation_count >= run_spec.max_observation_steps
            )
            if backend_signal is BackendSignal.RUNNING and (
                runner_budget_exhausted
                or (
                    elapsed_before_commit is not None
                    and elapsed_before_commit >= run_spec.wall_timeout_s
                )
            ):
                backend_signal = BackendSignal.TIMEOUT
            stage = "commit"
            _record_lifecycle_stage(stage_observer, stage)
            policy.commit(
                PolicyExecution(
                    action_plan_sha256=action_plan.sha256,
                    executed_actions=requested_actions[:executed_count],
                    delivered_observations=tuple(
                        record.observation for record in delivered_next
                    ),
                    terminal_signal=backend_signal,
                )
            )
            if backend_signal is not BackendSignal.RUNNING:
                break
        stage = "validation"
        _record_lifecycle_stage(stage_observer, stage)
        finalization = session.finalize()
        execution_status = TerminalStatus(backend_signal.value)
        terminal_status = execution_status
        validation_override = None
        if finalization.validation is not None and not finalization.validation.passed:
            validation_override = (
                False,
                finalization.validation.failure_codes,
            )
        if validation_override is None:
            failure_stage = None
            failure_code = None
        else:
            terminal_status = TerminalStatus.VALIDATOR_REJECTED
            failure_stage = "validation"
            failure_code = "validator_rejected"
        normal_result = build_trial_result(
            trial,
            run_spec,
            initial_state_sha256=initial_state_sha256,
            terminal_status=terminal_status,
            execution_status=execution_status,
            finalization=finalization,
            action_entries=action_entries,
            observation_count=observation_count,
            control_cycle_count=control_cycle_count,
            failure_stage=failure_stage,
            failure_code=failure_code,
            validation_override=validation_override,
        )
    except (Exception, SystemExit) as error:
        if isinstance(error, SystemExit):
            failure_code = f"{stage}_system_exit"
        else:
            failure_code = (
                error.code if isinstance(error, RunnerViolation) else f"{stage}_failed"
            )
        initial_diagnostics = {
            **dict(initial_diagnostics),
            "runner_failure": build_runner_failure_evidence(stage, failure_code, error),
        }
        emit_crash_marker(stage, failure_code, error)
        if not reset_attempted:
            raise
        primary_crash = True
        if policy_reset_entered:
            with suppress(Exception, SystemExit):
                policy.abort(failure_code)
        if session is not None:
            try:
                finalization = session.finalize()
            except (Exception, SystemExit):
                finalization = None
        result = build_trial_result(
            trial,
            run_spec,
            initial_state_sha256=initial_state_sha256,
            terminal_status=TerminalStatus.CRASH,
            execution_status=TerminalStatus.CRASH,
            finalization=finalization,
            action_entries=action_entries,
            observation_count=observation_count,
            control_cycle_count=control_cycle_count,
            failure_stage=stage,
            failure_code=failure_code,
        )
        _record_evidence(
            capture,
            result,
            finalization,
            action_entries,
            transition_entries,
            initial_diagnostics,
        )
        return result
    finally:
        try:
            if before_close is not None:
                publishable = pre_close_evidence(
                    capture,
                    normal_result,
                    finalization,
                    action_entries,
                    transition_entries,
                    initial_diagnostics,
                )
                if publishable is not None:
                    try:
                        stage = "artifact_export"
                        _record_lifecycle_stage(stage_observer, stage)
                        before_close(publishable)
                    except (Exception, SystemExit) as error:
                        emit_crash_marker(
                            "artifact_export",
                            "artifact_export_failed",
                            error,
                        )
                        raise
        finally:
            stage = "close"
            try:
                _record_lifecycle_stage(stage_observer, stage)
            finally:
                close_codes = close_resources(backend, policy)
            if close_failures is not None:
                close_failures.append(close_codes)
        if close_codes and not primary_crash and reset_attempted and normal_result:
            normal_result = build_trial_result(
                trial,
                run_spec,
                initial_state_sha256=normal_result.initial_state_sha256,
                terminal_status=TerminalStatus.CRASH,
                execution_status=normal_result.execution_status,
                finalization=finalization,
                action_entries=action_entries,
                observation_count=observation_count,
                control_cycle_count=control_cycle_count,
                failure_stage="close",
                failure_code=(
                    close_codes[0]
                    if len(close_codes) == 1
                    else "multiple_close_failures"
                ),
                validation_override=(
                    normal_result.validation_passed,
                    normal_result.validation_failure_codes,
                ),
            )
    if normal_result is None:
        raise RuntimeError("runner completed without a result")
    _record_evidence(
        capture,
        normal_result,
        finalization,
        action_entries,
        transition_entries,
        initial_diagnostics,
    )
    return normal_result


def run_closed_loop_trial_with_evidence(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
    *,
    fault_manifest: Optional[FaultManifest] = None,
    rest_references: Optional[RestReferenceBundle] = None,
    monotonic_clock: Optional[Callable[[], float]] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION,
) -> ClosedLoopExecutionEvidence:
    """Execute one trial and return evaluator evidence captured before teardown ends."""

    captured: list[ClosedLoopExecutionEvidence] = []
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
    )
    if len(captured) != 1:
        raise RuntimeError(
            "closed-loop runner did not emit exactly one evidence snapshot"
        )
    return captured[0]


def _validate_initial_state(
    diagnostics: Mapping[str, object], policy: InitialStatePolicy
) -> None:
    selected = (
        policy if isinstance(policy, InitialStatePolicy) else InitialStatePolicy(policy)
    )
    if selected is InitialStatePolicy.DIAGNOSTIC_ALLOW_INVALID_V1:
        return

    if selected is InitialStatePolicy.REPLACE_INITIAL_TERMINAL_V1:
        success_check = diagnostics.get("success_check")
        early_stop = diagnostics.get("early_stop")
        if (
            type(success_check) is not bool
            or type(early_stop) is not bool
            or success_check
            or early_stop
        ):
            raise RunnerViolation("invalid_initial_state")

    reset_viable = _diagnostic_bools(diagnostics, _RESET_VIABLE_PATHS)
    if False in reset_viable:
        raise RunnerViolation("qualification_reset_not_viable")

    early_stop = _diagnostic_bools(diagnostics, _INITIAL_EARLY_STOP_PATHS)
    if True in early_stop:
        raise RunnerViolation("qualification_initial_early_stop")

    initial_success = _diagnostic_bools(diagnostics, _INITIAL_SUCCESS_PATHS)
    if True in initial_success:
        raise RunnerViolation("qualification_initial_success")


def _diagnostic_bools(
    diagnostics: Mapping[str, object], paths: tuple[tuple[str, ...], ...]
) -> tuple[bool, ...]:
    values: list[bool] = []
    for path in paths:
        current: object = diagnostics
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if type(current) is not bool:
                raise RunnerViolation("qualification_initial_diagnostics_invalid")
            values.append(current)
    return tuple(values)


def _record_lifecycle_stage(
    observer: Optional[Callable[[str], None]], stage: str
) -> None:
    if observer is not None:
        observer(stage)


def _record_evidence(
    capture: Optional[list[ClosedLoopExecutionEvidence]],
    result: ClosedLoopTrialResult,
    finalization: Optional[DeliveryFinalization],
    action_entries: list[dict[str, object]],
    transition_entries: list[BackendTransition],
    initial_diagnostics: Mapping[str, object],
) -> None:
    if capture is None:
        return
    capture.append(
        capture_execution_evidence(
            result,
            finalization,
            action_entries,
            transition_entries,
            initial_diagnostics,
        )
    )
