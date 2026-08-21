"""Deterministic in-memory execution of one audited closed-loop trial."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from time import monotonic
from typing import Optional, Tuple, Union

from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    BackendSignal,
    ClosedLoopRunSpec,
    PolicyExecution,
)
from robotactile_benchmark.closed_loop.delivery import (
    DeliveryFinalization,
    IdentityDeliverySession,
    OnlineFaultSession,
)
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
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
from robotactile_benchmark.trials import Condition, TerminalStatus, TrialManifest

_STRUCTURAL_ABSENCE_OPERATOR_IDS = frozenset({"A1_stream_absence", "A2_frame_erasure"})


def run_closed_loop_trial(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
    *,
    fault_manifest: Optional[FaultManifest] = None,
    rest_references: Optional[RestReferenceBundle] = None,
    monotonic_clock: Optional[Callable[[], float]] = None,
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
) -> ClosedLoopTrialResult:
    """Execute one requested trial with no file-system side effects."""

    clock = monotonic_clock if monotonic_clock is not None else monotonic
    initial_state_sha256: Optional[str] = None
    finalization: Optional[DeliveryFinalization] = None
    action_entries: list[dict[str, object]] = []
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
            _record_evidence(capture, result, None, action_entries)
            return result
        context = build_episode_context(trial, run_spec.prompt)
        session = (
            OnlineFaultSession(fault_manifest, rest_references)
            if fault_manifest is not None
            else IdentityDeliverySession()
        )
        stage = "reset"
        reset_attempted = True
        receipt = backend.reset(context)
        initial_state_sha256 = receipt.simulator_state_sha256
        validate_reset(receipt, context)
        stage = "policy_reset"
        policy_reset_entered = True
        policy.reset(context)
        started_at = clock()
        stage = "observe"
        initial_clean = backend.observe()
        validate_clean_record(initial_clean, context, 0)
        stage = "delivery"
        delivered = session.deliver(initial_clean)
        if type(delivered.observation) is not ObservationRecord:
            raise RunnerViolation("delivered_observation_type_mismatch")
        current_observation = delivered.observation
        observation_count = 1
        while True:
            if clock() - started_at >= run_spec.wall_timeout_s:
                backend_signal = BackendSignal.TIMEOUT
                break
            if (
                control_cycle_count >= run_spec.max_control_cycles
                or observation_count >= run_spec.max_observation_steps
            ):
                backend_signal = BackendSignal.TIMEOUT
                break
            stage = "infer"
            action_plan = policy.infer(current_observation)
            if action_plan.action_spec != ACTION_SPEC:
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
            control_cycle_count += 1
            batch = backend.execute(requested_actions)
            previous_native_step_id = validate_batch(
                batch,
                requested_count,
                context,
                observation_count,
                previous_native_step_id,
            )
            executed_count = batch.executed_action_count
            stage = "delivery"
            delivered_next = tuple(
                session.deliver(transition.clean_record)
                for transition in batch.transitions
            )
            action_entries.append(
                {
                    "action_plan_sha256": action_plan.sha256,
                    "source_step_index": action_plan.source_step_index,
                    "executed_actions": requested_actions[:executed_count],
                }
            )
            backend_signal = batch.transitions[-1].signal
            observation_count += executed_count
            current_observation = delivered_next[-1].observation
            stage = "commit"
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
        finalization = session.finalize()
        execution_status = TerminalStatus(backend_signal.value)
        terminal_status = execution_status
        validation_override = _restoration_validation(trial, finalization)
        if finalization.validation is not None and not finalization.validation.passed:
            restoration_codes = (
                () if validation_override is None else validation_override[1]
            )
            validation_override = (
                False,
                tuple(
                    dict.fromkeys(
                        finalization.validation.failure_codes + restoration_codes
                    )
                ),
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
    except Exception as error:
        if not reset_attempted:
            raise
        primary_crash = True
        failure_code = (
            error.code if isinstance(error, RunnerViolation) else f"{stage}_failed"
        )
        if policy_reset_entered:
            with suppress(Exception):
                policy.abort(failure_code)
        if session is not None:
            try:
                finalization = session.finalize()
            except Exception:
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
        _record_evidence(capture, result, finalization, action_entries)
        return result
    finally:
        close_codes = _close_resources(backend, policy)
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
    _record_evidence(capture, normal_result, finalization, action_entries)
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
        capture=captured,
    )
    if len(captured) != 1:
        raise RuntimeError(
            "closed-loop runner did not emit exactly one evidence snapshot"
        )
    return captured[0]


def _record_evidence(
    capture: Optional[list[ClosedLoopExecutionEvidence]],
    result: ClosedLoopTrialResult,
    finalization: Optional[DeliveryFinalization],
    action_entries: list[dict[str, object]],
) -> None:
    if capture is None:
        return
    capture.append(
        ClosedLoopExecutionEvidence.from_runner_entries(
            result,
            finalization,
            action_entries,
        )
    )


def _restoration_validation(
    trial: TrialManifest,
    finalization: DeliveryFinalization,
) -> Optional[tuple[bool, Tuple[str, ...]]]:
    if trial.condition is not Condition.RESTORED:
        return None
    if any(
        record.observation.step_index == trial.restoration_index
        for record in finalization.delivered_records
    ):
        return None
    existing = (
        () if finalization.validation is None else finalization.validation.failure_codes
    )
    return False, tuple(dict.fromkeys(existing + ("RESTORATION_NOT_OBSERVED",)))


def _close_resources(
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
) -> Tuple[str, ...]:
    failures = []
    try:
        backend.close()
    except Exception:
        failures.append("backend_close_failed")
    try:
        policy.close()
    except Exception:
        failures.append("policy_close_failed")
    return tuple(failures)
