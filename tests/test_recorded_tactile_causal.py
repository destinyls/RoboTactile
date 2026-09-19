from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pytest import MonkeyPatch

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    PolicyEpisodeContext,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import ObservationRecord, build_evaluation_record
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.recorded import selection as selection_module
from robotactile_benchmark.recorded.selection import RecordedAnchorSelection
from robotactile_benchmark.recorded.source import RecordedEpisode
from robotactile_benchmark.recorded.tactile_causal import (
    RecordedTactileCausalResult,
    aggregate_recorded_tactile_causal,
    run_recorded_tactile_causal_episode,
)
from robotactile_benchmark.recorded.tactile_causal_artifact import (
    load_recorded_tactile_causal_task_artifact,
    write_recorded_tactile_causal_artifact,
    write_recorded_tactile_causal_cohort_from_shards,
    write_recorded_tactile_causal_task_artifact,
)

TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)


class FakeHdf5Root(dict[str, np.ndarray]):
    def __enter__(self) -> "FakeHdf5Root":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _identity(*, absence: bool) -> PolicyIdentity:
    return PolicyIdentity(
        system_id=f"n0-test:{'absent' if absence else 'clean'}",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=absence,
    )


def _ee8(index: int) -> np.ndarray:
    return np.asarray(
        (0.1 + index * 0.001, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02),
        dtype=np.float32,
    )


def _episode(task: str, episode_index: int) -> RecordedEpisode:
    records = []
    for index, source in enumerate(make_synthetic_episode(40)):
        observation = replace(
            source.observation,
            episode_id=f"{task}-{episode_index}",
            task=task,
            seed=episode_index,
            proprio=_ee8(index),
        )
        records.append(build_evaluation_record(observation, source.provenance))
    return RecordedEpisode(
        source_path=Path(f"/tmp/{task}-{episode_index}.hdf5"),
        source_sha256=f"{TASKS.index(task) * 5 + episode_index + 1:064x}",
        task_id=task,
        episode_id=f"{task}-{episode_index}",
        initial_seed=episode_index,
        anchor_index=6,
        rest_index=0,
        total_record_count=40,
        records=tuple(records),
        native_steps=tuple(range(40)),
        expert_actions=np.stack([_ee8(index) for index in range(6, 18)]),
    )


class FakeCausalPolicy:
    def __init__(self, *, absence: bool) -> None:
        self.identity = _identity(absence=absence)
        self._absence = absence
        self._context: PolicyEpisodeContext | None = None

    def reset(self, context: PolicyEpisodeContext) -> None:
        self._context = context

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        assert self._context is not None
        is_absent = all(not sensor.payload_present for sensor in observation.tactile)
        assert is_absent is self._absence
        actions = np.stack(
            [
                _ee8(index)
                for index in range(observation.step_index, observation.step_index + 12)
            ]
        )
        if self._absence:
            actions[:, 0] += np.float32(0.05)
        return ActionPlan(EE8_ACTION_SPEC, observation.step_index, actions)

    def abort(self, reason_code: str) -> None:
        assert reason_code == "recorded_one_shot_complete"
        self._context = None

    def close(self) -> None:
        pass


def _run(task: str, episode_index: int) -> RecordedTactileCausalResult:
    episode = _episode(task, episode_index)
    selection = RecordedAnchorSelection(6, 0, 40, "bilateral", 1.1, 1.2)
    return run_recorded_tactile_causal_episode(
        episode=episode,
        anchor_selection=selection,
        clean_identity=_identity(absence=False),
        absence_identity=_identity(absence=True),
        clean_policy_factory=lambda: FakeCausalPolicy(absence=False),
        absence_policy_factory=lambda: FakeCausalPolicy(absence=True),
        exogenous_seed=29,
    )


def test_recorded_pair_uses_structural_absence_and_measures_action_drift() -> None:
    result = _run("grasp_classify", 0)

    assert result.clean_expert_full.translation_mean_l2_m == 0.0
    assert np.isclose(result.absence_expert_full.translation_mean_l2_m, 0.05)
    assert np.isclose(result.clean_absence_drift_full.translation_mean_l2_m, 0.05)


def test_contact_selector_prefers_strongest_bilateral_frame(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = tmp_path / "0.hdf5"
    source.touch()
    left = np.full((20, 240, 320), 34.0, dtype=np.float32)
    right = left.copy()
    left[4, 0, 0] = 32.7
    right[4, 0, 0] = 32.8
    left[6, 0, 0] = 31.9
    right[6, 0, 0] = 32.0
    root = FakeHdf5Root(
        {
            "tactile/left_gsmini/depth": left,
            "tactile/right_gsmini/depth": right,
        }
    )
    monkeypatch.setattr(
        selection_module,
        "_h5py",
        lambda: SimpleNamespace(File=lambda *_args: root),
    )

    selected = selection_module.select_univtac_contact_anchor(
        source, task_id="grasp_classify"
    )

    assert selected.anchor_index == 6
    assert selected.rest_index == 0
    assert selected.contact_mode == "bilateral"
    assert selected.rest_mode == "bilateral_free"


def test_contact_selector_allows_episode_that_begins_in_contact(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = tmp_path / "0.hdf5"
    source.touch()
    left = np.full((20, 240, 320), 32.7, dtype=np.float32)
    right = left.copy()
    left[4, 0, 0] = 31.9
    right[4, 0, 0] = 32.0
    root = FakeHdf5Root(
        {
            "tactile/left_gsmini/depth": left,
            "tactile/right_gsmini/depth": right,
        }
    )
    monkeypatch.setattr(
        selection_module,
        "_h5py",
        lambda: SimpleNamespace(File=lambda *_args: root),
    )

    selected = selection_module.select_univtac_contact_anchor(
        source, task_id="grasp_classify"
    )

    assert selected.anchor_index == 4
    assert selected.rest_index == 0
    assert selected.rest_mode == "earliest_available_fallback"


def test_frozen40_aggregation_reports_all_tasks_and_no_success_rate() -> None:
    results = tuple(_run(task, index) for task in TASKS for index in range(5))

    summary = aggregate_recorded_tactile_causal(
        results,
        expected_task_ids=TASKS,
        episodes_per_task=5,
    )

    assert summary["completed_episode_count"] == 40
    assert summary["expected_episode_count"] == 40
    assert summary["success_rate_claimed"] is False
    primary = summary["primary_tasks"]
    per_task = summary["per_task"]
    assert isinstance(primary, dict)
    assert isinstance(per_task, dict)
    assert primary["episode_count"] == 10
    assert set(per_task) == set(TASKS)


def test_causal_artifact_preserves_predictions_and_claim_boundary(
    tmp_path: Path,
) -> None:
    result = _run("grasp_classify", 0)
    output = tmp_path / "cohort.json"

    digest = write_recorded_tactile_causal_artifact(
        output,
        results=(result,),
        summary={"success_rate_claimed": False},
        source_commit="c" * 40,
        split_manifest_sha256="d" * 64,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert len(digest) == 64
    assert document["summary"]["success_rate_claimed"] is False
    assert len(document["episodes"][0]["clean_prediction"]) == 12
    delta = document["episodes"][0]["absence_minus_clean_error"]
    assert delta["translation_mean_l2_m"] > 0.0


def test_task_shards_merge_into_strict_frozen40_cohort(tmp_path: Path) -> None:
    shards = []
    for task in TASKS:
        shard = tmp_path / f"{task}.json"
        write_recorded_tactile_causal_task_artifact(
            shard,
            task_id=task,
            results=tuple(_run(task, index) for index in range(5)),
            source_commit="c" * 40,
            split_manifest_sha256="d" * 64,
        )
        assert load_recorded_tactile_causal_task_artifact(shard)["task_id"] == task
        shards.append(shard)

    output = tmp_path / "frozen40.json"
    digest, summary = write_recorded_tactile_causal_cohort_from_shards(
        output,
        shard_paths=shards,
        expected_task_ids=TASKS,
        episodes_per_task=5,
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert len(digest) == 64
    assert summary["completed_episode_count"] == 40
    assert summary["success_rate_claimed"] is False
    assert len(document["episodes"]) == 40
    assert document["summary"]["primary_tasks"]["episode_count"] == 10
