"""Owned offline launch and declared-source gates, without GPU execution."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark.n0_fault_campaign import stress_replay_evidence as evidence
from robotactile_benchmark.trials import Condition
from scripts.n0_twam import run_noise_stress_replay as runner


def source_fixture(monkeypatch):
    plan = {
        "task": "lift_bottle",
        "excluded_seeds": [5],
        "binding": {"model_sha256": "a" * 64, "dataset_sha256": "b" * 64},
    }
    manifest = {"group_sha256": "c" * 64, "selected": [{"condition": "clean"}]}
    source = SimpleNamespace(
        trial=SimpleNamespace(
            condition=Condition.CLEAN,
            task="lift_bottle",
            initial_seed=5,
            exogenous_seed=5,
            checkpoint_sha256="a" * 64,
            dataset_sha256="b" * 64,
        ),
        evidence=SimpleNamespace(
            finalization=SimpleNamespace(clean_records=tuple(range(61))),
            result=SimpleNamespace(terminal_status=SimpleNamespace(value="timeout")),
        ),
        external_root_sha256="d" * 64,
    )
    monkeypatch.setattr(
        evidence, "load_stress_group", lambda path: (plan, manifest, [None])
    )
    monkeypatch.setattr(evidence, "load_live_univtac_artifact", lambda path: source)
    monkeypatch.setattr(
        evidence,
        "result_to_dict",
        lambda value: {
            "terminal_status": "timeout",
            "score_eligible": True,
            "validation_passed": True,
            "observation_count": 61,
            "score_success": False,
        },
    )
    return source


def test_declared_task_failure_source_is_retained(tmp_path, monkeypatch):
    source_fixture(monkeypatch)
    _, provenance, records, _ = evidence.load_replay_inputs(tmp_path, tmp_path, 60)
    assert provenance["source_status"] == "timeout" and len(records) == 61


@pytest.mark.parametrize("mode", ["seed", "model", "dataset", "preview", "short"])
def test_replay_rejects_test_data_other_model_and_sparse_prefix(
    tmp_path, monkeypatch, mode
):
    source = source_fixture(monkeypatch)
    if mode == "seed":
        source.trial.initial_seed = 100000
    elif mode == "model":
        source.trial.checkpoint_sha256 = "9" * 64
    elif mode == "dataset":
        source.trial.dataset_sha256 = "9" * 64
    elif mode == "preview":
        source.evidence.finalization = None
    else:
        source.evidence.finalization.clean_records = tuple(range(60))
    with pytest.raises(ValueError, match="development|full captured"):
        evidence.load_replay_inputs(tmp_path, tmp_path, 60)


def arguments(tmp_path, monkeypatch):
    import robotactile_benchmark

    values = {}
    for name in (
        "group",
        "clean_artifact",
        "model_root",
        "code",
        "config",
        "n0_source",
        "digest_cache",
    ):
        path = tmp_path / name
        path.mkdir()
        values[name] = path
    python = tmp_path / "python"
    python.write_text("test launcher")
    args = argparse.Namespace(
        **values,
        package=Path(robotactile_benchmark.__file__).resolve().parent.parent,
        output=tmp_path / "offline",
        n0_python=python,
        gpu_lock=tmp_path / "gpu.lock",
        gpu_id=6,
        port=39000,
        timeout_s=10,
        code_sha256="e" * 64,
        target_steps=[12, 36, 60],
    )
    monkeypatch.setattr(runner, "verify_runtime_code", lambda *a: None)
    monkeypatch.setattr(runner, "validate_runtime_binding", lambda *a: {})
    monkeypatch.setattr(runner, "file_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        runner,
        "load_replay_inputs",
        lambda *a: (
            {"binding": {}, "protocol_sha256": "b" * 64},
            {"model_seed": 5, "source_sha256": "c" * 64},
            [],
            {"fast_f1-s5": []},
        ),
    )
    monkeypatch.setattr(
        runner, "official_server_environment", lambda **kw: kw["inherited"]
    )
    monkeypatch.setattr(runner, "check_port", lambda port: None)
    monkeypatch.setattr(runner, "wait_server", lambda *a: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "0")
    monkeypatch.setattr(
        runner,
        "summarize_replay_probes",
        lambda *a, **kw: {"rng_after_reset_equal_and_complete": True},
    )
    return args


@pytest.mark.parametrize("failure", [False, True])
def test_only_own_server_is_started_stopped_and_no_rerun(
    tmp_path, monkeypatch, failure
):
    args = arguments(tmp_path, monkeypatch)
    launched, stopped = [], []

    class Server:
        pid = 456
        returncode = None

        def __init__(self, command, **kwargs):
            assert "--diagnostic-input-capture-arrays" in command
            assert kwargs["start_new_session"]
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"
            launched.append(self)

    def replay(**kwargs):
        if failure:
            raise RuntimeError("inference failed")
        return {"comparisons": []}

    monkeypatch.setattr(runner.subprocess, "Popen", Server)
    monkeypatch.setattr(runner, "stop_owned", lambda server: stopped.append(server))
    monkeypatch.setattr(runner, "run_fixed_input_replay", replay)
    if failure:
        with pytest.raises(RuntimeError, match="inference failed"):
            runner.run(args)
        assert (args.output / "failure.json").exists()
    else:
        runner.run(args)
        assert (args.output / "report.json").exists()
    assert launched == stopped and len(launched) == 1
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert len(launched) == 1


def test_occupied_gpu_prevents_any_model_start(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "40000")
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not start")
    )
    with pytest.raises(RuntimeError, match="occupied"):
        runner.run(args)
    assert not args.output.exists()
