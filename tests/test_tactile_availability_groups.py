"""Prepare-only availability groups, with no model, process, or simulator load."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.manifests import FaultManifest
from scripts.retrained_evaluation import group

A1 = "A1_stream_absence"
A2 = "A2_frame_erasure"


def binding(root: Path, mode: str = "native_missing_v1") -> dict[str, Any]:
    evaluation: dict[str, Any] = {
        "tactile_availability_mode": mode,
        "operators": [A1, A2],
        "sensor_slots": ["left"],
        "severity_level": 3,
        "severity_registries": ["optical_marker_v1"],
        "fault_window_mode": "full_episode_v1",
    }
    if mode == "zero_fill_v1":
        evaluation["tactile_zero_shape"] = [240, 320, 3]
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
        "evaluation": evaluation,
    }


@pytest.mark.parametrize("mode", ["native_missing_v1", "zero_fill_v1"])
def test_selected_availability_group_three_conditions_and_request_identity(
    tmp_path: Path,
    mode: str,
) -> None:
    path = group.prepare_group(
        binding(tmp_path, mode), "lift_can", tmp_path / "group", 7
    )
    plan = json.loads(path.read_text())
    assert [Path(name).stem for name in plan["ordered_requests"]] == ["clean", A1, A2]
    assert plan["excluded"] == []
    for name in plan["ordered_requests"]:
        request_path = path.parent / name
        document = json.loads(request_path.read_text())
        assert document["tactile_availability_mode"] == mode
        if mode == "zero_fill_v1":
            assert document["tactile_zero_shape"] == [240, 320, 3]
        else:
            assert "tactile_zero_shape" not in document
        request = load_live_univtac_request(request_path)
        loaded = load_live_univtac_run(request)
        assert loaded.policy_identity.supports_structural_absence
        assert request.tactile_availability_mode.value == mode


def test_default_required_keeps_old_exclusions_and_omits_new_wire_fields(
    tmp_path: Path,
) -> None:
    value = binding(tmp_path)
    del value["evaluation"]
    path = group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    plan = json.loads(path.read_text())
    assert len(plan["ordered_requests"]) == 11
    assert plan["ordered_requests"][0] == "requests/clean.json"
    excluded = [
        item for item in plan["excluded"] if item["status"] == "unsupported_contract"
    ]
    assert len(excluded) == 4
    assert {item["operator"] for item in excluded} == {A1, A2}
    for name in plan["ordered_requests"]:
        document = json.loads((path.parent / name).read_text())
        assert "tactile_availability_mode" not in document
        assert "tactile_zero_shape" not in document


def test_a1_contiguous_a2_intermittent_and_same_native_zero_schedule(
    tmp_path: Path,
) -> None:
    schedules = {}
    for mode in ("native_missing_v1", "zero_fill_v1"):
        path = group.prepare_group(
            binding(tmp_path, mode), "lift_can", tmp_path / mode, 9
        )
        faults = {
            fault.operator_id: fault
            for fault in (
                FaultManifest.from_dict(json.loads(item.read_text()))
                for item in (path.parent / "faults").rglob("*.json")
            )
        }
        assert set(faults) == {A1, A2}
        for fault in faults.values():
            assert fault.start_index == 0 and fault.stop_index == 301
            assert fault.sensor_slots == ("left",)
            assert fault.severity_level == 3
        offsets = list(faults[A1].parameters["affected_offsets"])
        assert offsets == list(range(len(offsets)))
        erased = list(faults[A2].parameters["erased_offsets"])
        assert 0 < len(erased) < 301
        assert len(set(erased)) == len(erased)
        assert erased != list(range(len(erased)))
        schedules[mode] = (faults[A2].operator_seed, erased)
    assert schedules["native_missing_v1"] == schedules["zero_fill_v1"]


@pytest.mark.parametrize("model", ["n0_vtla", "n0_twam", "ftp1_policy", "dream_tac"])
def test_zero_fill_group_prepares_for_all_four_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    value = binding(tmp_path, "zero_fill_v1")
    if model == "n0_twam":
        # Avoid the TWAM weight-bundle reader; all request/group validation stays real.
        clean = replace(
            group.build_clean(value, "lift_can", tmp_path / "group", 0),
            policy_kind=LivePolicyKind.N0,
            retrained_prompt=None,
            retrained_control_hz=None,
            retrained_tactile_payload=None,
            execute_action_steps=24,
        )
        monkeypatch.setattr(group, "build_clean", lambda *args: clean)
    value["model"] = model
    path = group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    assert len(json.loads(path.read_text())["ordered_requests"]) == 3


@pytest.mark.parametrize(
    "update",
    [
        {"tactile_availability_mode": "unknown"},
        {"tactile_availability_mode": "zero_fill_v1"},
        {
            "tactile_availability_mode": "zero_fill_v1",
            "tactile_zero_shape": [0, 320, 3],
        },
        {
            "tactile_availability_mode": "zero_fill_v1",
            "tactile_zero_shape": [240, 320, 1],
        },
        {"tactile_availability_mode": "zero_fill_v1", "tactile_zero_shape": [240, 320]},
        {"operators": []},
        {"operators": [A1, A1]},
        {"operators": ["unknown"]},
        {"operators": A1},
        {"sensor_slots": []},
        {"sensor_slots": ["left", "left"]},
        {"sensor_slots": ["unknown"]},
        {"sensor_slots": "left"},
        {"severity_level": 0},
        {"severity_level": 6},
        {"severity_level": True},
        {"severity_registries": ["optical_marker_extreme_v1"], "severity_level": 3},
    ],
)
def test_invalid_configuration_fails_before_output_creation(
    tmp_path: Path,
    update: dict[str, Any],
) -> None:
    value = binding(tmp_path)
    value["evaluation"].update(update)
    with pytest.raises(ValueError):
        group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    assert not (tmp_path / "group").exists()


@pytest.mark.parametrize("model", ["n0_twam", "ftp1_policy", "dream_tac"])
def test_native_other_models_fail_before_artifact_or_output_access(
    tmp_path: Path, model: str
) -> None:
    value = binding(tmp_path)
    value["model"] = model
    with pytest.raises(ValueError):
        group.prepare_group(value, "lift_can", tmp_path / "group", 0)
    assert not (tmp_path / "group").exists()
