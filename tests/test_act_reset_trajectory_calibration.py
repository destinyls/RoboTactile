"""Reset-only ACT trajectory calibration orchestration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import robotactile_benchmark.act_fault_campaign.reset_trajectory_calibration as calibration
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.trials import Condition


def test_capture_executes_exactly_one_reset_without_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial = SimpleNamespace(
        task="grasp_classify",
        condition=Condition.CLEAN,
        initial_seed=938,
        exogenous_seed=1_000_000,
        pair_key="a" * 64,
        dataset_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        config_sha256="d" * 64,
        action_spec="qpos8_next_step",
    )
    request = SimpleNamespace(
        policy_kind=LivePolicyKind.ACT,
        condition=Condition.CLEAN,
        upstream_root=tmp_path / "UniVTAC",
        runtime_dir=tmp_path / "runtime",
        launcher_args=(),
        simulator_device="cuda:0",
    )
    loaded = SimpleNamespace(
        request=request,
        trial=trial,
        run_spec=SimpleNamespace(prompt="classify"),
        backend_config=SimpleNamespace(
            upstream_commit="e" * 40,
            task=SimpleNamespace(task_source_sha256="f" * 64),
        ),
        content_sha256="1" * 64,
    )
    reference = SimpleNamespace(
        task_id=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        dataset_sha256=trial.dataset_sha256,
        checkpoint_sha256=trial.checkpoint_sha256,
        config_sha256=trial.config_sha256,
        source_run_content_sha256=loaded.content_sha256,
        expected_qpos8=(0.0,) * 8,
        expected_native_step=240,
        expected_simulator_state_sha256="2" * 64,
        sha256="3" * 64,
    )
    runtime = SimpleNamespace(task=object(), close_runtime=lambda: None)
    events: list[object] = []
    backend_init: list[tuple[tuple[object, ...], dict[str, object]]] = []
    finish_kwargs: dict[str, object] = {}
    trajectory = object()

    class _Backend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            backend_init.append((args, kwargs))

        def reset(self, context: object) -> object:
            events.append(("reset", context))
            witness = {
                "native_step": 240,
                "qpos8": [0.001] + [0.0] * 7,
                "reference_checks": {"reference_provided": False},
            }
            return SimpleNamespace(
                sha256="4" * 64,
                simulator_state_sha256="5" * 64,
                diagnostics={
                    "plan_success": True,
                    "task": {"reset_witness": witness},
                },
            )

        def close(self) -> None:
            events.append("close")

    def _finish(**kwargs: object) -> object:
        finish_kwargs.update(kwargs)
        return trajectory

    capture = SimpleNamespace(finish=_finish)
    monkeypatch.setattr(calibration, "load_live_univtac_request", lambda _path: request)
    monkeypatch.setattr(calibration, "load_live_univtac_run", lambda _request: loaded)
    monkeypatch.setattr(
        calibration, "load_act_reset_reference", lambda _path: reference
    )
    monkeypatch.setattr(
        calibration, "launch_univtac_runtime", lambda *_args, **_kwargs: runtime
    )
    monkeypatch.setattr(
        calibration,
        "install_reference_guided_pre_move_ik",
        lambda _task, _qpos, **_kwargs: True,
    )
    monkeypatch.setattr(
        calibration,
        "install_pre_move_trajectory_capture",
        lambda _task, *, expected_native_step: (
            capture if expected_native_step == reference.expected_native_step else None
        ),
    )
    monkeypatch.setattr(calibration, "UniVTACIsaacBackend", _Backend)
    monkeypatch.setattr(calibration, "build_episode_context", lambda *_args: "ctx")

    def _write(path: Path, value: object) -> bool:
        assert path == tmp_path / "trajectory.json"
        assert value is trajectory
        events.append("write")
        return True

    monkeypatch.setattr(calibration, "write_act_reset_trajectory", _write)

    status, result, reset_sha = calibration.capture_act_reset_trajectory(
        clean_request_path=tmp_path / "clean.json",
        reset_reference_path=tmp_path / "reference.json",
        output_path=tmp_path / "trajectory.json",
    )

    assert status == "created"
    assert result is trajectory
    assert reset_sha == "4" * 64
    assert events == [("reset", "ctx"), "write", "close"]
    assert backend_init == [((loaded.backend_config, runtime), {})]
    assert finish_kwargs["capture_simulator_state_sha256"] == "5" * 64
    assert finish_kwargs["capture_native_step"] == 240
    assert finish_kwargs["capture_qpos8"] == (0.001,) + (0.0,) * 7
    assert finish_kwargs["source_qpos8_max_abs_error"] == pytest.approx(0.001)


def test_source_proximity_rejects_wrong_native_step() -> None:
    reference = SimpleNamespace(expected_native_step=240, expected_qpos8=(0.0,) * 8)

    with pytest.raises(
        calibration.ACTFaultCampaignError,
        match="actual=239, expected=240",
    ):
        calibration._require_source_proximate_reset(
            {
                "plan_success": True,
                "task": {"reset_witness": {"native_step": 239, "qpos8": [0.0] * 8}},
            },
            reference,
        )


def test_source_proximity_rejects_qpos_outside_threshold() -> None:
    reference = SimpleNamespace(expected_native_step=240, expected_qpos8=(0.0,) * 8)

    with pytest.raises(
        calibration.ACTFaultCampaignError,
        match=r"max_abs_error=0.0021.*threshold=0.002",
    ):
        calibration._require_source_proximate_reset(
            {
                "plan_success": True,
                "task": {
                    "reset_witness": {
                        "native_step": 240,
                        "qpos8": [0.0021] + [0.0] * 7,
                    }
                },
            },
            reference,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "0.0"])
def test_source_proximity_rejects_nonfinite_or_nonnumeric_qpos(
    bad_value: object,
) -> None:
    reference = SimpleNamespace(expected_native_step=240, expected_qpos8=(0.0,) * 8)

    with pytest.raises(calibration.ACTFaultCampaignError, match="qpos8 is"):
        calibration._require_source_proximate_reset(
            {
                "plan_success": True,
                "task": {
                    "reset_witness": {
                        "native_step": 240,
                        "qpos8": [bad_value] + [0.0] * 7,
                    }
                },
            },
            reference,
        )


def test_failure_is_emitted_before_native_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        calibration.sys.stderr,
        "write",
        lambda _value: events.append("write") or 1,
    )
    monkeypatch.setattr(
        calibration.traceback,
        "print_exception",
        lambda *_args, **_kwargs: events.append("traceback"),
    )
    monkeypatch.setattr(
        calibration.sys.stderr,
        "flush",
        lambda: events.append("flush"),
    )

    calibration._emit_calibration_failure(RuntimeError("reset mismatch"))

    assert events == ["write", "traceback", "flush"]
