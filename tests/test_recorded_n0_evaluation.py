from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    PolicyEpisodeContext,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import build_evaluation_record
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.recorded.artifact import (
    load_recorded_experiment_artifact,
    write_recorded_experiment_artifact,
)
from robotactile_benchmark.recorded.evaluator import run_recorded_n0_experiment
from robotactile_benchmark.recorded.metrics import action_metrics
from robotactile_benchmark.recorded.source import (
    RecordedEpisode,
    build_recorded_rest_references,
    retarget_recorded_episode,
)


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="official-n0-recorded-test",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _ee8(index: int) -> np.ndarray:
    return np.asarray(
        (0.1 + index * 0.001, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02),
        dtype=np.float32,
    )


def _episode() -> RecordedEpisode:
    records = []
    for index, source in enumerate(make_synthetic_episode(40)):
        observation = replace(
            source.observation,
            task="lift_bottle",
            seed=90,
            proprio=_ee8(index),
        )
        records.append(build_evaluation_record(observation, source.provenance))
    expert = np.stack([_ee8(index) for index in range(6, 18)])
    return RecordedEpisode(
        source_path=Path("/tmp/lift_bottle_90.hdf5"),
        source_sha256="c" * 64,
        task_id="lift_bottle",
        episode_id="recorded-test-90",
        initial_seed=90,
        anchor_index=6,
        rest_index=0,
        total_record_count=40,
        records=tuple(records),
        native_steps=tuple(range(104, 144)),
        expert_actions=expert,
    )


class FakeRecordedPolicy:
    def __init__(self) -> None:
        self.identity = _identity()
        self.context: Optional[PolicyEpisodeContext] = None
        self.closed = False

    def reset(self, context: PolicyEpisodeContext) -> None:
        self.context = context

    def infer(self, observation: object) -> ActionPlan:
        assert self.context is not None
        left = observation.sensor("left").payload  # type: ignore[attr-defined]
        step = observation.step_index  # type: ignore[attr-defined]
        assert left is not None
        actions = np.stack([_ee8(index) for index in range(step, step + 12)])
        actions[:, 1] = float(left.mean()) * 1e-4
        return ActionPlan(EE8_ACTION_SPEC, step, actions.astype(np.float32))

    def abort(self, reason_code: str) -> None:
        assert reason_code == "recorded_one_shot_complete"
        self.context = None

    def close(self) -> None:
        self.closed = True


def test_action_metrics_separate_anchor_and_future_and_ignore_quaternion_sign() -> None:
    target = np.stack([_ee8(index) for index in range(12)])
    predicted = target.copy()
    predicted[:, 3:7] *= -1.0
    predicted[0, 0] += 0.1

    anchor = action_metrics(predicted, target, horizon_stop=1)
    future = action_metrics(predicted, target, horizon_start=1)

    assert np.isclose(anchor.translation_mean_l2_m, 0.1)
    assert future.translation_mean_l2_m == 0.0
    assert anchor.rotation_mean_geodesic_deg == 0.0


def test_recorded_n0_runner_marks_native_absence_unsupported_and_writes_artifact(
    tmp_path: Path,
) -> None:
    episode = _episode()
    release_episode = retarget_recorded_episode(episode, 8)
    references = build_recorded_rest_references(episode)
    result = run_recorded_n0_experiment(
        episode=episode,
        release_episode=release_episode,
        rest_references=references,
        policy_identity=_identity(),
        policy_factory=FakeRecordedPolicy,
        exogenous_seed=20260823,
        fault_start_index=3,
        operator_ids=(
            "A1_stream_absence",
            "F2_spatial_sensitivity_loss",
            "T1_fixed_source_delay",
            "C1_sensor_identity_misrouting",
            "F6_history_residual_imprint",
        ),
        severity_levels=(1,),
    )

    assert result.completed_condition_count == 5
    assert result.unsupported_condition_count == 1
    absence = result.results[1]
    assert absence.status == "unsupported_contract"
    c1 = next(
        item
        for item in result.results
        if item.operator_id == "C1_sensor_identity_misrouting"
    )
    assert c1.severity_identifiable_at_anchor is False
    assert c1.delivery_validation_passed is True
    f6 = result.results[-1]
    assert f6.operator_id == "F6_history_residual_imprint"
    assert f6.anchor_index == 8
    assert f6.delivery_validation_passed is True

    output = tmp_path / "recorded"
    receipt = write_recorded_experiment_artifact(output, result)
    loaded = load_recorded_experiment_artifact(output)
    assert loaded["root_receipt_sha256"] == receipt
    assert loaded["condition_count"] == 6
    assert loaded["evidence_level"] == "recorded_model_only_n0_v1"
    assert write_recorded_experiment_artifact(output, result) == receipt
