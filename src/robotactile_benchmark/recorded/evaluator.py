"""N0-only recorded robustness evaluation over canonical UniVTAC records."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    PolicyEpisodeContext,
    PolicyIdentity,
)
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    REST_REFERENCE_OPERATOR_IDS,
)
from robotactile_benchmark.contracts import Array, ObservationRecord, freeze_array
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.n0_official import n0_training_prompt
from robotactile_benchmark.recorded.metrics import ActionErrorMetrics, action_metrics
from robotactile_benchmark.recorded.source import RecordedEpisode
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.severity import severity_value

UNSUPPORTED_NATIVE_N0_OPERATORS = frozenset({"A1_stream_absence", "A2_frame_erasure"})


class RecordedN0Policy(Protocol):
    """Policy surface needed by one-shot recorded evaluation."""

    identity: PolicyIdentity

    def reset(self, context: PolicyEpisodeContext) -> None: ...

    def infer(self, observation: ObservationRecord) -> ActionPlan: ...

    def abort(self, reason_code: str) -> None: ...

    def close(self) -> None: ...


PolicyFactory = Callable[[], RecordedN0Policy]


@dataclass(frozen=True)
class RecordedConditionResult:
    """One clean or atomic-fault N0 prediction and its measurements."""

    condition_id: str
    operator_id: Optional[str]
    severity_level: Optional[int]
    anchor_index: int
    native_anchor_step: int
    status: str
    reason_code: Optional[str]
    prediction: Optional[Array]
    manifest: Optional[FaultManifest]
    delivery_trace_sha256: Optional[str]
    delivery_validation_passed: Optional[bool]
    delivery_failure_codes: Tuple[str, ...]
    expert_anchor: Optional[ActionErrorMetrics]
    expert_full: Optional[ActionErrorMetrics]
    expert_future_only: Optional[ActionErrorMetrics]
    clean_drift_full: Optional[ActionErrorMetrics]
    severity_identifiable_at_anchor: bool

    def __post_init__(self) -> None:
        if self.prediction is not None:
            prediction = freeze_array(self.prediction, np.float32)
            if prediction.shape != (12, 8) or not np.isfinite(prediction).all():
                raise ValueError("recorded N0 prediction must be finite float32 [12,8]")
            object.__setattr__(self, "prediction", prediction)


@dataclass(frozen=True)
class RecordedExperimentResult:
    """Source-bound clean baseline and atomic-fault results."""

    evidence_level: str
    episode: RecordedEpisode
    policy_identity: PolicyIdentity
    exogenous_seed: int
    fault_start_index: int
    rest_references_sha256: str
    release_episode: Optional[RecordedEpisode]
    release_clean_prediction: Optional[Array]
    results: Tuple[RecordedConditionResult, ...]

    def __post_init__(self) -> None:
        if self.release_clean_prediction is not None:
            prediction = freeze_array(self.release_clean_prediction, np.float32)
            if prediction.shape != (12, 8) or not np.isfinite(prediction).all():
                raise ValueError("release clean prediction must be float32 [12,8]")
            object.__setattr__(self, "release_clean_prediction", prediction)

    @property
    def completed_condition_count(self) -> int:
        return sum(item.status == "completed" for item in self.results)

    @property
    def unsupported_condition_count(self) -> int:
        return sum(item.status == "unsupported_contract" for item in self.results)


def _window_for_anchor(
    operator_id: str,
    severity_level: int,
    fault_start_index: int,
    anchor_index: int,
) -> tuple[int, int]:
    if fault_start_index > anchor_index:
        raise ValueError("fault start must not follow the recorded anchor")
    dose = severity_value(operator_id, severity_level)
    if operator_id == "T1_fixed_source_delay":
        return max(fault_start_index, int(dose)), anchor_index + 1
    if operator_id == "T2_held_last_freeze":
        duration = int(dose)
        return max(0, anchor_index - duration + 1), anchor_index + 1
    if operator_id in {
        "A1_stream_absence",
        "A2_frame_erasure",
        "C1_sensor_identity_misrouting",
    }:
        window = 32
        count = max(1, min(window, int(round(window * float(dose)))))
        start = anchor_index - count + 1
        if start < 0:
            raise ValueError("anchor lacks warm-up for availability/context scheduling")
        return start, start + window
    return fault_start_index, anchor_index + 1


def build_anchor_fault_manifest(
    *,
    operator_id: str,
    severity_level: int,
    operator_seed: int,
    fault_start_index: int,
    anchor_index: int,
    rest_references: RestReferenceBundle,
) -> FaultManifest:
    """Schedule one registered operator so the selected anchor receives its dose."""

    if operator_id not in CORE_OPERATOR_IDS:
        raise KeyError(f"unknown core operator: {operator_id}")
    start, stop = _window_for_anchor(
        operator_id, severity_level, fault_start_index, anchor_index
    )
    parameters: dict[str, object] = {}
    if operator_id in REST_REFERENCE_OPERATOR_IDS:
        parameters["rest_reference_sha256"] = rest_references.sha256
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity_level,
        operator_seed=operator_seed,
        start_index=start,
        stop_index=stop,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters=parameters,
    )


def infer_recorded_once(
    factory: PolicyFactory,
    context: PolicyEpisodeContext,
    observation: ObservationRecord,
) -> Array:
    policy = factory()
    try:
        policy.reset(context)
        plan = policy.infer(observation)
        if plan.action_spec != EE8_ACTION_SPEC or plan.actions.shape != (12, 8):
            raise ValueError("official N0 cold prediction must be EE8 [12,8]")
        return freeze_array(plan.actions, np.float32)
    finally:
        policy.abort("recorded_one_shot_complete")
        policy.close()


def _metrics(
    prediction: Array,
    expert: Array,
    clean: Optional[Array],
) -> tuple[
    ActionErrorMetrics,
    ActionErrorMetrics,
    ActionErrorMetrics,
    Optional[ActionErrorMetrics],
]:
    anchor = action_metrics(prediction, expert, horizon_stop=1)
    full = action_metrics(prediction, expert)
    future = action_metrics(prediction, expert, horizon_start=1)
    drift = None if clean is None else action_metrics(prediction, clean)
    return anchor, full, future, drift


def run_recorded_n0_experiment(
    *,
    episode: RecordedEpisode,
    release_episode: Optional[RecordedEpisode],
    rest_references: RestReferenceBundle,
    policy_identity: PolicyIdentity,
    policy_factory: PolicyFactory,
    exogenous_seed: int,
    fault_start_index: int,
    operator_ids: Sequence[str],
    severity_levels: Sequence[int],
) -> RecordedExperimentResult:
    """Run clean plus selected atomic faults without executing predicted actions."""

    if policy_identity.action_spec != EE8_ACTION_SPEC:
        raise ValueError("recorded N0 experiment requires ee8_absolute")
    if exogenous_seed < 0:
        raise ValueError("exogenous seed must be non-negative")
    operators = tuple(operator_ids)
    severities = tuple(severity_levels)
    if len(set(operators)) != len(operators) or any(
        item not in CORE_OPERATOR_IDS for item in operators
    ):
        raise ValueError("operator IDs must be unique registered core operators")
    if (
        not severities
        or len(set(severities)) != len(severities)
        or any(item not in range(1, 6) for item in severities)
    ):
        raise ValueError("severity levels must be unique values in [1,5]")
    context = PolicyEpisodeContext(
        episode_id=episode.episode_id,
        task=episode.task_id,
        initial_seed=episode.initial_seed,
        exogenous_seed=exogenous_seed,
        instruction=n0_training_prompt(episode.task_id),
        action_spec=EE8_ACTION_SPEC,
    )
    clean = infer_recorded_once(
        policy_factory, context, episode.anchor_record.observation
    )
    clean_anchor, clean_full, clean_future, _ = _metrics(
        clean, episode.expert_actions, None
    )
    release_clean: Optional[Array] = None
    if "F6_history_residual_imprint" in operators:
        if release_episode is None:
            raise ValueError("F6 recorded evaluation requires a release episode view")
        if (
            release_episode.source_sha256 != episode.source_sha256
            or release_episode.task_id != episode.task_id
            or release_episode.episode_id != episode.episode_id
            or release_episode.initial_seed != episode.initial_seed
        ):
            raise ValueError("release episode view does not match the contact episode")
        if not all(
            release_episode.anchor_record.provenance_for(slot).phase.value == "release"
            for slot in ("left", "right")
        ):
            raise ValueError("F6 release anchor must be release for both sensors")
        release_clean = infer_recorded_once(
            policy_factory, context, release_episode.anchor_record.observation
        )
    results: list[RecordedConditionResult] = [
        RecordedConditionResult(
            condition_id="clean",
            operator_id=None,
            severity_level=None,
            anchor_index=episode.anchor_index,
            native_anchor_step=episode.native_steps[episode.anchor_index],
            status="completed",
            reason_code=None,
            prediction=clean,
            manifest=None,
            delivery_trace_sha256=None,
            delivery_validation_passed=None,
            delivery_failure_codes=(),
            expert_anchor=clean_anchor,
            expert_full=clean_full,
            expert_future_only=clean_future,
            clean_drift_full=None,
            severity_identifiable_at_anchor=True,
        )
    ]
    for operator_id in operators:
        for severity_level in severities:
            manifest = build_anchor_fault_manifest(
                operator_id=operator_id,
                severity_level=severity_level,
                operator_seed=exogenous_seed,
                fault_start_index=fault_start_index,
                anchor_index=(
                    release_episode.anchor_index
                    if operator_id == "F6_history_residual_imprint"
                    and release_episode is not None
                    else episode.anchor_index
                ),
                rest_references=rest_references,
            )
            condition_id = f"{operator_id}.s{severity_level}"
            if operator_id in UNSUPPORTED_NATIVE_N0_OPERATORS:
                results.append(
                    RecordedConditionResult(
                        condition_id=condition_id,
                        operator_id=operator_id,
                        severity_level=severity_level,
                        anchor_index=episode.anchor_index,
                        native_anchor_step=episode.native_steps[episode.anchor_index],
                        status="unsupported_contract",
                        reason_code="official_n0_requires_both_tactile_streams",
                        prediction=None,
                        manifest=manifest,
                        delivery_trace_sha256=None,
                        delivery_validation_passed=None,
                        delivery_failure_codes=(),
                        expert_anchor=None,
                        expert_full=None,
                        expert_future_only=None,
                        clean_drift_full=None,
                        severity_identifiable_at_anchor=True,
                    )
                )
                continue
            condition_episode = (
                release_episode
                if operator_id == "F6_history_residual_imprint"
                else episode
            )
            if condition_episode is None:
                raise RuntimeError("F6 release episode disappeared")
            replay = apply_fault(condition_episode.records, manifest, rest_references)
            if not replay.validation.passed:
                raise RuntimeError(
                    f"fault delivery failed validation: {condition_id}: "
                    f"{replay.validation.failure_codes}"
                )
            prediction = infer_recorded_once(
                policy_factory,
                context,
                replay.records[condition_episode.anchor_index].observation,
            )
            clean_reference = (
                release_clean if operator_id == "F6_history_residual_imprint" else clean
            )
            if clean_reference is None:
                raise RuntimeError("F6 release clean prediction disappeared")
            expert_anchor, expert_full, expert_future, clean_drift = _metrics(
                prediction, condition_episode.expert_actions, clean_reference
            )
            results.append(
                RecordedConditionResult(
                    condition_id=condition_id,
                    operator_id=operator_id,
                    severity_level=severity_level,
                    anchor_index=condition_episode.anchor_index,
                    native_anchor_step=condition_episode.native_steps[
                        condition_episode.anchor_index
                    ],
                    status="completed",
                    reason_code=None,
                    prediction=prediction,
                    manifest=manifest,
                    delivery_trace_sha256=replay.trace_sha256,
                    delivery_validation_passed=True,
                    delivery_failure_codes=tuple(replay.validation.failure_codes),
                    expert_anchor=expert_anchor,
                    expert_full=expert_full,
                    expert_future_only=expert_future,
                    clean_drift_full=clean_drift,
                    severity_identifiable_at_anchor=(
                        operator_id != "C1_sensor_identity_misrouting"
                    ),
                )
            )
    return RecordedExperimentResult(
        evidence_level="recorded_model_only_n0_v1",
        episode=episode,
        policy_identity=policy_identity,
        exogenous_seed=exogenous_seed,
        fault_start_index=fault_start_index,
        rest_references_sha256=rest_references.sha256,
        release_episode=release_episode,
        release_clean_prediction=release_clean,
        results=tuple(results),
    )


def result_counts(result: RecordedExperimentResult) -> Mapping[str, int]:
    """Return explicit completed/unsupported counts for CLI summaries."""

    return {
        "condition_count": len(result.results),
        "completed_condition_count": result.completed_condition_count,
        "unsupported_condition_count": result.unsupported_condition_count,
    }
