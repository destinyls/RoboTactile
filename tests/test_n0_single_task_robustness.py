"""Single-task reporting does not conflate missing/invalid runs with model SR."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np
import pytest
import test_rest_reference_calibration as calibration_fixtures
from test_rest_reference_calibration import _request, _source_artifact
from test_univtac_rest_calibration import FakeCalibrationTask

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_rest_calibration import (
    install_n0_empty_gripper_rest_calibration,
)
from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.closed_loop.delivery import OnlineFaultSession
from robotactile_benchmark.contracts import ContactPhase, build_evaluation_record
from robotactile_benchmark.execution import (
    load_live_univtac_artifact,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from scripts.n0_twam import single_task_robustness as single
from scripts.n0_twam.single_task_robustness import _render_summary, result_row
from scripts.retrained_evaluation.group import write_json


def test_single_task_defaults_to_early_random_onset() -> None:
    current = single.single_task_evaluation()
    assert current["fault_window_mode"] == "early_random_onset_v1"
    assert current["fault_onset_max_index"] == 8
    assert "fault_start_index" not in current
    full = single.single_task_evaluation("full_episode_v1")
    assert full["fault_start_index"] == 0
    legacy = single.single_task_evaluation("delayed_onset_v1")
    assert legacy["fault_start_index"] == 20
    with pytest.raises(ValueError):
        single.single_task_evaluation("typo")


def test_exposure_counts_first_frame_fault_without_inventing_pixel_change() -> None:
    clean = make_synthetic_episode(length=20)
    first_time = clean[0].provenance_for("left").source_time_s
    second_time = clean[1].provenance_for("left").source_time_s
    assert first_time is not None and second_time is not None
    manifest = FaultManifest(
        operator_id="T2_held_last_freeze",
        severity_level=5,
        operator_seed=0,
        start_index=0,
        stop_index=40,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        severity_registry="optical_marker_extreme_v1",
        parameters={
            "sample_period_s": second_time - first_time,
            "temporal_schedule": "full_episode_v1",
        },
    )
    session = OnlineFaultSession(manifest)
    for record in clean:
        session.deliver(record)
    # A reporting-only typed stub: no simulated task result is produced here.
    artifact = cast(
        LoadedLiveUniVTACArtifact,
        SimpleNamespace(
            fault_manifest=manifest,
            evidence=SimpleNamespace(
                finalization=session.finalize(),
                action_entries=tuple(
                    SimpleNamespace(source_step_index=index) for index in (0, 4, 12)
                ),
            ),
        ),
    )
    metrics = single.exposure_metrics(artifact)
    changed = {
        record.observation.step_index
        for record in clean
        if any(
            not np.array_equal(
                record.observation.sensor(slot).payload,
                clean[0].observation.sensor(slot).payload,
            )
            for slot in ("left", "right")
        )
    }
    assert 0 not in changed and 0 < len(changed) < 20
    assert metrics["scheduled_fault_frames"] == 20
    assert metrics["active_fault_frames"] == 20
    assert metrics["changed_payload_frames"] == len(changed)
    assert metrics["active_fault_action_queries"] == 3
    assert metrics["changed_frames_at_action_query"] == sum(
        index in changed for index in (0, 4, 12)
    )
    assert metrics["scheduled_fault_fraction"] == 1.0
    assert metrics["active_fault_fraction"] == 1.0
    assert metrics["active_fault_action_query_fraction"] == 1.0
    assert metrics["full_episode_window_observed"] is True


def test_rest_calibration_keeps_retrained_ten_hz_without_applying_model_actions() -> (
    None
):
    config = build_univtac_backend_config(
        "grasp_classify",
        EE8_ACTION_SPEC,
        n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    )
    task = cast(Callable[[], Any], FakeCalibrationTask)()
    witness = install_n0_empty_gripper_rest_calibration(task, config)
    task.pre_move()
    task.take_action(np.zeros(8), action_type="ee", force=True)
    assert task.step_count * config.decimation == 12
    assert witness["model_actions_applied"] is False
    assert task._robotactile_n0_last_plan["waypoint_count"] == 0
    assert task.check_success() is False


@pytest.mark.parametrize("success", [False, True])
def test_single_trial_score_is_not_a_statistical_claim(success: bool) -> None:
    result = {
        "score_eligible": True,
        "score_success": success,
        "terminal_status": "success" if success else "timeout",
        "failure_stage": None,
        "validation_passed": True,
    }
    row = result_row(result, "clean")
    assert row["valid_episode_count"] == 1
    assert row["success_rate"] == float(success)


@pytest.mark.parametrize(
    "update",
    [
        {"failure_stage": "delivery", "terminal_status": "crash"},
        {"validation_passed": False},
        {"score_eligible": False},
        {"validation_passed": None},
    ],
)
def test_invalid_execution_has_no_model_success_rate(update: dict[str, Any]) -> None:
    row = result_row(
        {
            "score_eligible": True,
            "score_success": False,
            "terminal_status": "timeout",
            "failure_stage": None,
            "validation_passed": True,
            **update,
        },
        "T1",
    )
    assert row["valid_episode_count"] == 0
    assert row["success_rate"] is None


def test_summary_handles_pending_and_unsupported(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _render_summary(
        [
            {"condition": "clean", "status": "not_executed"},
            {"condition": "A1", "status": "unsupported_contract"},
        ],
        "grasp_classify",
        tmp_path,
    )
    assert (tmp_path / "metrics.png").is_file()


def test_summary_panel_labels_the_actual_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draw_module = pytest.importorskip("PIL.ImageDraw")
    texts: list[str] = []
    original = draw_module.ImageDraw.text

    def capture(self: Any, xy: Any, text: str, **kwargs: Any) -> Any:
        texts.append(text)
        return original(self, xy, text, **kwargs)

    monkeypatch.setattr(draw_module.ImageDraw, "text", capture)
    _render_summary([], "lift_can", tmp_path, "full_episode_v1", seed=2)
    assert any("seed 2" in text for text in texts)
    assert all("seed 0" not in text for text in texts)


def test_calibration_publishes_before_nonreturning_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = "pull_out_key"
    original = cast(Callable[..., Any], calibration_fixtures._calibrated_episode)

    def free_episode(task: str) -> Any:
        return tuple(
            build_evaluation_record(
                record.observation,
                tuple(
                    replace(item, phase=ContactPhase.FREE) for item in record.provenance
                ),
            )
            for record in original(task)
        )

    monkeypatch.setattr(calibration_fixtures, "_calibrated_episode", free_episode)
    source = cast(Callable[[Path, str], Path], _source_artifact)(
        tmp_path / "fixture", task
    )
    artifact = load_live_univtac_artifact(source)
    request = cast(Callable[..., Any], _request)(tmp_path, task)
    loaded = load_live_univtac_run(request)
    output = tmp_path / "calibration"
    binding = tmp_path / "binding.json"
    write_json(binding, {"model": "n0_twam"})
    monkeypatch.setattr(
        single,
        "build_clean",
        lambda *args: replace(request, output_dir=output / "artifacts/clean"),
    )

    def execute(request: Any, **kwargs: Any) -> None:
        kwargs["artifact_exporter"](request.output_dir, loaded, artifact.evidence)
        assert (output / "rest/rest_reference.json").is_file()
        assert (output / "calibration_receipt.json").is_file()
        raise SystemExit(0)  # Isaac close: execution never returns to calibrate().

    monkeypatch.setattr(single, "execute_live_univtac_run", execute)
    with pytest.raises(SystemExit) as stopped:
        single.calibrate(binding, task, output)
    assert stopped.value.code == 0
    rest = load_rest_reference_artifact(output / "rest")
    receipt = json.loads((output / "calibration_receipt.json").read_text())
    assert rest.validation.task == task
    assert receipt["rest_reference_sha256"] == rest.references.sha256
    assert receipt["model_actions_applied"] is False
    assert receipt["scored_episode"] is False
    with pytest.raises(FileExistsError):
        single.calibrate(binding, task, output)
    before = (output / "calibration_receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        single._publish_calibration(
            output / "artifacts/clean",
            loaded,
            artifact.evidence,
            task=task,
            output=output,
        )
    assert (output / "calibration_receipt.json").read_bytes() == before
    with pytest.raises(ValueError, match="match the requested task"):
        single._publish_calibration(
            tmp_path / "wrong/live",
            loaded,
            artifact.evidence,
            task="grasp_classify",
            output=tmp_path / "wrong",
        )
    assert not (tmp_path / "wrong").exists()
