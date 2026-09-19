"""Preflight and backend-contract checks for the closed-loop runner."""

from __future__ import annotations

from typing import Optional

from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    ClosedLoopRunSpec,
    ExecutionBatch,
    PolicyEpisodeContext,
    ResetReceipt,
)
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.contracts import (
    EvaluationRecord,
    canonical_hash,
    delivered_hash,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.trials import Condition, TrialManifest


class RunnerViolation(RuntimeError):
    """Stable failure code for a violated closed-loop runtime contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def build_episode_context(
    trial: TrialManifest,
    prompt: str,
) -> PolicyEpisodeContext:
    """Build the deterministic, policy-visible reset context."""

    return PolicyEpisodeContext(
        episode_id=canonical_hash(
            {
                "namespace": "robotactile_benchmark.closed_loop.episode.v1",
                "trial_manifest_sha256": trial.sha256,
            }
        ),
        task=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        instruction=prompt,
        action_spec=trial.action_spec,
    )


def preflight(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    backend: SimulationBackend,
    policy: ClosedLoopPolicy,
    fault_manifest: Optional[FaultManifest],
    rest_references: Optional[RestReferenceBundle],
) -> None:
    """Reject configuration mismatches before reset or policy execution."""

    identity = policy.identity
    if (
        identity.system_id != trial.executed_system_id
        or identity.checkpoint_sha256 != trial.checkpoint_sha256
        or identity.config_sha256 != trial.config_sha256
        or identity.action_spec != trial.action_spec
    ):
        raise ValueError("policy identity does not match the trial manifest")
    if backend.action_spec != trial.action_spec:
        raise ValueError("backend action spec does not match the trial manifest")
    if backend.success_predicate_id != run_spec.success_predicate_id:
        raise ValueError("backend success predicate does not match the run spec")
    faulted = trial.condition is Condition.FAULTED
    if faulted and fault_manifest is None:
        raise ValueError("faulted trial requires a fault manifest")
    if not faulted and fault_manifest is not None:
        raise ValueError("clean/no-touch trial cannot receive a fault manifest")
    if (
        fault_manifest is not None
        and fault_manifest.sha256 != trial.fault_manifest_sha256
    ):
        raise ValueError("fault manifest does not match the trial manifest")
    if rest_references is not None and (
        fault_manifest is None
        or not operator_requires_rest_reference(
            fault_manifest.operator_id,
            severity_registry=fault_manifest.severity_registry,
        )
    ):
        raise ValueError("rest references are only valid for rest-reference operators")
    if trial.condition is Condition.NO_TOUCH:
        if identity.consumes_tactile:
            raise ValueError("no-touch trial requires a non-tactile policy")
    elif not identity.consumes_tactile:
        raise ValueError("tactile trial requires a tactile-consuming base policy")


def validate_reset(
    receipt: ResetReceipt,
    context: PolicyEpisodeContext,
) -> None:
    """Validate reset identity without discarding the already returned receipt."""

    if (
        receipt.episode_id != context.episode_id
        or receipt.initial_seed != context.initial_seed
        or receipt.exogenous_seed != context.exogenous_seed
    ):
        raise RunnerViolation("reset_receipt_mismatch")


def validate_clean_record(
    record: EvaluationRecord,
    context: PolicyEpisodeContext,
    expected_step: int,
) -> None:
    """Require fresh evaluator-only clean data with stable identity."""

    if type(record) is not EvaluationRecord:
        raise RunnerViolation("clean_record_type_mismatch")
    observation = record.observation
    if (
        observation.episode_id != context.episode_id
        or observation.task != context.task
        or observation.seed != context.initial_seed
    ):
        raise RunnerViolation("clean_record_identity_mismatch")
    if observation.step_index != expected_step:
        raise RunnerViolation("clean_record_step_mismatch")
    if any(item.active_fault_ids for item in record.provenance):
        raise RunnerViolation("clean_record_contains_fault")
    content_hash = delivered_hash(observation, record.provenance)
    if (
        record.clean_record_sha256 != content_hash
        or record.delivered_record_sha256 != content_hash
    ):
        raise RunnerViolation("clean_record_hash_mismatch")


def validate_batch(
    batch: ExecutionBatch,
    requested_action_count: int,
    context: PolicyEpisodeContext,
    expected_step: int,
    previous_native_step_id: Optional[int],
) -> int:
    """Validate one action/transition batch and return its final native step."""

    if (
        batch.executed_action_count > requested_action_count
        or len(batch.transitions) != batch.executed_action_count
    ):
        raise RunnerViolation("transition_action_count_mismatch")
    if (
        batch.executed_action_count < requested_action_count
        and batch.transitions[-1].signal is BackendSignal.RUNNING
    ):
        raise RunnerViolation("running_partial_batch")
    terminal_seen = False
    for offset, transition in enumerate(batch.transitions):
        if terminal_seen:
            raise RunnerViolation("terminal_transition_followed")
        if (
            previous_native_step_id is not None
            and transition.native_step_id <= previous_native_step_id
        ):
            raise RunnerViolation("native_step_id_not_monotonic")
        validate_clean_record(
            transition.clean_record,
            context,
            expected_step + offset,
        )
        terminal_seen = transition.signal is not BackendSignal.RUNNING
        previous_native_step_id = transition.native_step_id
    if previous_native_step_id is None:
        raise RunnerViolation("empty_execution_batch")
    return previous_native_step_id
