"""Frozen task and cohort accounting for the N0 tactile causal diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocols/n0_twam_tactile_causal_diagnostic_v1.json"
TASK_REGISTRY_PATH = ROOT / "configs/univtac/tasks_v1.json"
FROZEN40_PATH = ROOT / "configs/protocols/univtac_frozen40_hdf5_v1.json"


def test_n0_tactile_causal_diagnostic_scope_and_dataset_accounting() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    registry = json.loads(TASK_REGISTRY_PATH.read_text(encoding="utf-8"))

    task_ids = tuple(item["task_id"] for item in registry["tasks"])
    primary_tasks = tuple(protocol["primary_task_ids"])
    cohort = protocol["dataset_evaluation_cohort"]

    assert primary_tasks == ("grasp_classify", "lift_can")
    assert len(task_ids) == len(set(task_ids)) == cohort["registered_task_count"]
    assert set(primary_tasks) <= set(task_ids)
    assert cohort["physical_split"] == "frozen40"
    assert cohort["total_episode_count"] == (
        cohort["registered_task_count"] * cohort["episodes_per_task"]
    )
    assert cohort["evaluated_episode_count"] == cohort["total_episode_count"]
    assert cohort["primary_task_count"] == len(primary_tasks)
    assert cohort["primary_episode_count"] == (
        len(primary_tasks) * cohort["episodes_per_task"]
    )


def test_frozen40_manifest_has_five_unique_hdf5_episodes_per_task() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    entries = json.loads(FROZEN40_PATH.read_text(encoding="utf-8"))
    task_counts: dict[str, int] = {}
    paths: set[str] = set()

    for entry in entries:
        task = entry["task"]
        path = entry["hdf5_path"]
        assert path.startswith(f"{task}/clean/") and path.endswith(".hdf5")
        assert path not in paths
        paths.add(path)
        task_counts[task] = task_counts.get(task, 0) + 1

    cohort = protocol["dataset_evaluation_cohort"]
    assert len(entries) == cohort["evaluated_episode_count"] == 40
    assert set(task_counts.values()) == {cohort["episodes_per_task"]}
    assert len(task_counts) == cohort["registered_task_count"]


def test_dataset_episodes_are_not_relabelled_as_closed_loop_trials() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["evaluation_mode"] == "recorded_hdf5_offline_causal_diagnostic"
    assert protocol["conditions"] == ["clean", "observed_tactile_absence"]
    assert protocol["closed_loop_success_rate_claimed"] is False
