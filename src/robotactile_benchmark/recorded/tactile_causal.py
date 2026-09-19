"""Offline Clean-versus-tactile-absence diagnostics for recorded N0 episodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    PolicyEpisodeContext,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, freeze_array
from robotactile_benchmark.policies.n0_official import n0_training_prompt
from robotactile_benchmark.recorded.evaluator import (
    PolicyFactory,
    infer_recorded_once,
)
from robotactile_benchmark.recorded.metrics import ActionErrorMetrics, action_metrics
from robotactile_benchmark.recorded.selection import RecordedAnchorSelection
from robotactile_benchmark.recorded.source import RecordedEpisode

EVIDENCE_LEVEL = "recorded_hdf5_offline_tactile_causal_diagnostic_v1"
PRIMARY_TASK_IDS = ("grasp_classify", "lift_can")
METRIC_FIELDS = (
    "translation_mean_l2_m",
    "translation_final_l2_m",
    "rotation_mean_geodesic_deg",
    "rotation_final_geodesic_deg",
    "gripper_mean_abs",
    "gripper_final_abs",
)


@dataclass(frozen=True)
class RecordedTactileCausalResult:
    """Paired N0 predictions for one immutable HDF5 contact anchor."""

    episode: RecordedEpisode
    anchor_selection: RecordedAnchorSelection
    clean_identity: PolicyIdentity
    absence_identity: PolicyIdentity
    exogenous_seed: int
    clean_prediction: Array
    absence_prediction: Array
    clean_expert_full: ActionErrorMetrics
    absence_expert_full: ActionErrorMetrics
    clean_expert_future: ActionErrorMetrics
    absence_expert_future: ActionErrorMetrics
    clean_absence_drift_full: ActionErrorMetrics

    def __post_init__(self) -> None:
        for field_name in ("clean_prediction", "absence_prediction"):
            value = freeze_array(getattr(self, field_name), np.float32)
            if value.shape != (12, 8) or not np.isfinite(value).all():
                raise ValueError(f"{field_name} must be finite float32 [12,8]")
            object.__setattr__(self, field_name, value)
        if self.anchor_selection.anchor_index != self.episode.anchor_index:
            raise ValueError("selected anchor and loaded episode disagree")


def _validate_identity_pair(clean: PolicyIdentity, absence: PolicyIdentity) -> None:
    if (
        clean.action_spec != EE8_ACTION_SPEC
        or absence.action_spec != EE8_ACTION_SPEC
        or not clean.consumes_tactile
        or not absence.consumes_tactile
        or clean.supports_structural_absence
        or not absence.supports_structural_absence
        or clean.checkpoint_sha256 != absence.checkpoint_sha256
        or clean.config_sha256 != absence.config_sha256
    ):
        raise ValueError("clean/absence policy identities violate the paired contract")


def run_recorded_tactile_causal_episode(
    *,
    episode: RecordedEpisode,
    anchor_selection: RecordedAnchorSelection,
    clean_identity: PolicyIdentity,
    absence_identity: PolicyIdentity,
    clean_policy_factory: PolicyFactory,
    absence_policy_factory: PolicyFactory,
    exogenous_seed: int,
) -> RecordedTactileCausalResult:
    """Infer one Clean/absence pair without executing either action chunk."""

    _validate_identity_pair(clean_identity, absence_identity)
    if exogenous_seed < 0:
        raise ValueError("exogenous_seed must be non-negative")
    context = PolicyEpisodeContext(
        episode_id=episode.episode_id,
        task=episode.task_id,
        initial_seed=episode.initial_seed,
        exogenous_seed=exogenous_seed,
        instruction=n0_training_prompt(episode.task_id),
        action_spec=EE8_ACTION_SPEC,
    )
    clean_observation = episode.anchor_record.observation
    absence_observation = replace(
        clean_observation,
        tactile=tuple(
            sensor.without_payload(False) for sensor in clean_observation.tactile
        ),
    )
    clean = infer_recorded_once(clean_policy_factory, context, clean_observation)
    absence = infer_recorded_once(absence_policy_factory, context, absence_observation)
    return RecordedTactileCausalResult(
        episode=episode,
        anchor_selection=anchor_selection,
        clean_identity=clean_identity,
        absence_identity=absence_identity,
        exogenous_seed=exogenous_seed,
        clean_prediction=clean,
        absence_prediction=absence,
        clean_expert_full=action_metrics(clean, episode.expert_actions),
        absence_expert_full=action_metrics(absence, episode.expert_actions),
        clean_expert_future=action_metrics(
            clean, episode.expert_actions, horizon_start=1
        ),
        absence_expert_future=action_metrics(
            absence, episode.expert_actions, horizon_start=1
        ),
        clean_absence_drift_full=action_metrics(absence, clean),
    )


def _metric_mean(values: Sequence[ActionErrorMetrics]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate an empty metric collection")
    return {
        field: float(np.mean([float(getattr(value, field)) for value in values]))
        for field in METRIC_FIELDS
    }


def _metric_delta(
    absence: Mapping[str, float], clean: Mapping[str, float]
) -> dict[str, float]:
    return {field: float(absence[field] - clean[field]) for field in METRIC_FIELDS}


def _scope_summary(
    results: Sequence[RecordedTactileCausalResult],
) -> dict[str, object]:
    clean = _metric_mean([item.clean_expert_full for item in results])
    absence = _metric_mean([item.absence_expert_full for item in results])
    worse_fraction = {
        field: float(
            np.mean(
                [
                    float(
                        getattr(item.absence_expert_full, field)
                        > getattr(item.clean_expert_full, field)
                    )
                    for item in results
                ]
            )
        )
        for field in METRIC_FIELDS
    }
    return {
        "episode_count": len(results),
        "clean_expert_error": clean,
        "observed_tactile_absence_expert_error": absence,
        "absence_minus_clean_error": _metric_delta(absence, clean),
        "absence_worse_episode_fraction": worse_fraction,
        "clean_absence_action_drift": _metric_mean(
            [item.clean_absence_drift_full for item in results]
        ),
    }


def aggregate_recorded_tactile_causal(
    results: Sequence[RecordedTactileCausalResult],
    *,
    expected_task_ids: Sequence[str],
    episodes_per_task: int,
) -> dict[str, object]:
    """Aggregate a complete task-balanced HDF5 cohort without an SR claim."""

    items = tuple(results)
    task_ids = tuple(expected_task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("expected task IDs must be non-empty and unique")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    by_task: dict[str, list[RecordedTactileCausalResult]] = defaultdict(list)
    seen_sources: set[str] = set()
    for item in items:
        source = item.episode.source_sha256
        if source in seen_sources:
            raise ValueError("HDF5 source appears more than once in the cohort")
        seen_sources.add(source)
        by_task[item.episode.task_id].append(item)
    expected = set(task_ids)
    if set(by_task) != expected or any(
        len(by_task[task]) != episodes_per_task for task in task_ids
    ):
        raise ValueError("cohort does not match the task-balanced episode contract")
    primary = [item for item in items if item.episode.task_id in PRIMARY_TASK_IDS]
    return {
        "schema_version": EVIDENCE_LEVEL,
        "evidence_level": EVIDENCE_LEVEL,
        "sampling_unit": "recorded_hdf5_episode",
        "success_rate_claimed": False,
        "completed_episode_count": len(items),
        "expected_episode_count": len(task_ids) * episodes_per_task,
        "micro_all_episodes": _scope_summary(items),
        "primary_tasks": {
            "task_ids": list(PRIMARY_TASK_IDS),
            **_scope_summary(primary),
        },
        "per_task": {task: _scope_summary(by_task[task]) for task in task_ids},
        "claim_boundary": (
            "Paired offline model inference on expert HDF5 observations; predicted "
            "actions were not executed, so this artifact does not report task "
            "success rate or closed-loop robustness."
        ),
    }


__all__ = [
    "EVIDENCE_LEVEL",
    "PRIMARY_TASK_IDS",
    "RecordedTactileCausalResult",
    "aggregate_recorded_tactile_causal",
    "run_recorded_tactile_causal_episode",
]
