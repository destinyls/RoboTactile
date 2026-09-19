"""Optional group controls preserve native defaults and export selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.manifests import FaultManifest
from scripts.retrained_evaluation import group


def binding(root: Path) -> dict[str, Any]:
    return {
        "model": "n0_vtla",
        "training_step": 160000,
        "deployment_root": str(root),
        "dataset_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "binding_sha256": "c" * 64,
        "normalizer_sha256": "d" * 64,
        "source_commit": "e" * 40,
        "control_hz": 10,
        "tactile_payload": "rgb",
        "tasks": {"lift_can": {"prompt": "trained prompt"}},
    }


def test_selected_registry_start_and_capture(tmp_path: Path) -> None:
    value = binding(tmp_path)
    value["evaluation"] = {
        "severity_registries": ["optical_marker_v1"],
        "capture_profile": "paper_full_v1",
        "fault_start_index": 20,
    }
    plan_path = group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    plan = json.loads(plan_path.read_text())
    assert plan["capture_profile"] == "paper_full_v1"
    assert len(plan["ordered_requests"]) == 6
    assert {item["registry"] for item in plan["excluded"]} == {"optical_marker_v1"}
    assert (
        sum(item["status"] == "unsupported_contract" for item in plan["excluded"]) == 2
    )
    fault = json.loads(next((plan_path.parent / "faults").rglob("*.json")).read_text())
    assert fault["start_index"] == 20
    clean = json.loads((plan_path.parent / "requests/clean.json").read_text())
    assert clean["execute_action_steps"] == 50


@pytest.mark.parametrize("model, expected", [("n0_vtla", 50), ("n0_twam", 20)])
def test_default_start_and_registries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, expected: int
) -> None:
    value = binding(tmp_path)
    clean = group.build_clean(value, "lift_can", tmp_path / "group", 0)
    value["model"] = model
    monkeypatch.setattr(group, "build_clean", lambda *args: clean)
    path = group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    plan = json.loads(path.read_text())
    assert plan["capture_profile"] == "preview_v1"
    assert len(plan["ordered_requests"]) == 11
    for fault_path in (path.parent / "faults").rglob("*.json"):
        assert json.loads(fault_path.read_text())["start_index"] == expected


@pytest.mark.parametrize(
    "options",
    [
        None,
        {"severity_registries": []},
        {"severity_registries": "optical_marker_v1"},
        {"severity_registries": ["unknown"]},
        {"severity_registries": [["optical_marker_v1"]]},
        {"severity_registries": ["optical_marker_v1", "optical_marker_v1"]},
        {"capture_profile": "unknown"},
        {"fault_start_index": True},
        {"fault_start_index": -1},
        {"fault_start_index": 1.5},
        {"fault_start_index": "20"},
        {"fault_start_index": 1000000},
        {"fault_window_mode": "unknown"},
        {"fault_window_mode": True},
        {"fault_window_mode": "full_episode_v1", "fault_start_index": 20},
        {"fault_window_mode": "early_random_onset_v1", "fault_start_index": 2},
        {"fault_window_mode": "early_random_onset_v1", "fault_onset_max_index": 0},
        {"fault_window_mode": "early_random_onset_v1", "fault_onset_max_index": True},
        {"fault_onset_max_index": 8},
    ],
)
def test_invalid_options_leave_no_output(tmp_path: Path, options: Any) -> None:
    value = binding(tmp_path)
    value["evaluation"] = options
    with pytest.raises(ValueError):
        group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    assert not (tmp_path / "group").exists()


def test_full_episode_group_covers_initial_and_terminal_observation(
    tmp_path: Path,
) -> None:
    value = binding(tmp_path)
    value["evaluation"] = {
        "severity_registries": ["optical_marker_extreme_v1"],
        "fault_window_mode": "full_episode_v1",
    }
    path = group.prepare_group(value, "lift_can", tmp_path / "full", 0)
    plan = json.loads(path.read_text())
    clean = json.loads((path.parent / "requests/clean.json").read_text())
    assert plan["fault_window_mode"] == "full_episode_v1"
    assert plan["fault_start_index"] == 0
    assert plan["fault_stop_index"] == clean["max_observation_steps"]
    for fault_path in (path.parent / "faults").rglob("*.json"):
        fault = FaultManifest.from_dict(json.loads(fault_path.read_text()))
        assert fault.start_index == 0
        assert fault.stop_index == clean["max_observation_steps"]
        assert fault.active(0) and fault.active(fault.stop_index - 1)
        assert not fault.active(fault.stop_index)
        if fault.operator_id.startswith("T"):
            assert fault.parameters["temporal_schedule"] == "full_episode_v1"
        if fault.operator_id == "T2_held_last_freeze":
            assert set(fault.parameters["source_index_map"]) == {0}
            assert fault.parameters["hold_duration_frames"] == fault.stop_index


def test_early_random_onset_is_seeded_early_and_shared_across_faults(
    tmp_path: Path,
) -> None:
    value = binding(tmp_path)
    value["evaluation"] = {
        "operators": [
            "F2_spatial_sensitivity_loss",
            "T1_fixed_source_delay",
            "T2_held_last_freeze",
            "T3_inter_sensor_skew",
        ],
        "severity_registries": ["optical_marker_extreme_v1"],
        "fault_window_mode": "early_random_onset_v1",
        "fault_onset_max_index": 8,
    }
    starts = []
    for seed in range(6):
        path = group.prepare_group(value, "lift_can", tmp_path / f"seed-{seed}", seed)
        plan = json.loads(path.read_text())
        clean = json.loads((path.parent / "requests/clean.json").read_text())
        start = plan["fault_start_index"]
        assert 1 <= start <= min(8, (clean["max_observation_steps"] - 1) // 3)
        assert plan["fault_stop_index"] == clean["max_observation_steps"]
        assert plan["fault_onset_derivation"] == "robotactile.early-random-onset.v1"
        manifests = [
            FaultManifest.from_dict(json.loads(item.read_text()))
            for item in (path.parent / "faults").rglob("*.json")
        ]
        assert {item.start_index for item in manifests} == {start}
        assert {item.stop_index for item in manifests} == {
            clean["max_observation_steps"]
        }
        for item in manifests:
            assert not item.active(0) and item.active(start)
            if item.operator_id.startswith("T"):
                assert item.parameters["temporal_schedule"] == "window_to_end_v1"
            if item.operator_id == "T2_held_last_freeze":
                assert (
                    item.parameters["hold_duration_frames"] == item.stop_index - start
                )
        starts.append(start)
    assert len(set(starts)) >= 2
    assert (
        derive_early_random_onset(
            task="lift_can",
            seed=0,
            stop=clean["max_observation_steps"],
            cap=8,
        )
        == starts[0]
    )


@pytest.mark.parametrize("profile", [None, *LiveCaptureProfile])
def test_run_group_uses_plan_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: LiveCaptureProfile | None
) -> None:
    plan: dict[str, Any] = {"ordered_requests": []}
    if profile is not None:
        plan["capture_profile"] = profile.value
    group.write_json(tmp_path / "group.json", plan)
    group.write_json(tmp_path / "binding.json", {})
    captured: dict[str, Any] = {}

    def execute(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(group, "execute_paired_live_univtac_runs", execute)
    group.run_group(tmp_path / "group.json")
    exporter = captured["artifact_exporter"]
    assert exporter.keywords["capture_profile"] is (
        profile or LiveCaptureProfile.PREVIEW
    )
