"""CPU-only capture ownership, frozen inputs and source acceptance checks."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import robotactile_benchmark
from scripts.n0_twam import capture_noise_replay_source as capture


@pytest.fixture
def setup_capture(tmp_path, monkeypatch):
    paths = {}
    for name in ("model_root", "code", "n0_source", "digest_cache"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    for name in ("protocol", "config", "isaac_python", "n0_python"):
        paths[name] = tmp_path / name
        paths[name].write_text("{}")
    args = argparse.Namespace(
        **paths,
        package=Path(robotactile_benchmark.__file__).resolve().parent.parent,
        output=tmp_path / "capture",
        gpu_lock=tmp_path / "gpu.lock",
        gpu_id=0,
        port=29000,
        seed=5,
        timeout_s=10,
        code_sha256="d" * 64,
        isaac_python_sha256="d" * 64,
    )
    plan = {
        "task": "lift_bottle",
        "seeds": [50],
        "excluded_seeds": [5],
        "protocol_sha256": "a" * 64,
        "binding": {"model_sha256": "b" * 64, "dataset_sha256": "c" * 64},
    }
    monkeypatch.setattr(capture, "validate_protocol", lambda raw: plan)
    monkeypatch.setattr(capture, "validate_runtime_binding", lambda *a: {})
    monkeypatch.setattr(capture, "file_sha256", lambda path: "d" * 64)
    monkeypatch.setattr(capture, "check_port", lambda port: None)
    monkeypatch.setattr(capture, "wait_server", lambda *a: None)
    monkeypatch.setattr(
        capture, "official_server_environment", lambda **kw: dict(kw["inherited"])
    )
    monkeypatch.setattr(
        capture,
        "subprocess",
        NS(
            check_output=lambda *a, **kw: "0",
            DEVNULL=-3,
            STDOUT=-2,
        ),
    )
    monkeypatch.setattr(
        capture, "live_univtac_request_to_dict", lambda request: request
    )

    def request(args, seed, output):
        assert args.root == args.model_root
        return {"seed": seed, "upstream_root": str(args.root / "sources/UniVTAC")}

    monkeypatch.setattr(capture, "build_request", request)
    launched, stopped = [], []

    class Process:
        pid = 1234
        returncode = None

        def wait(self, timeout):
            self.returncode = 0
            return 0

    def launch(argv, **kwargs):
        assert (args.output / "plan.json").is_file()
        launched.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(capture.subprocess, "Popen", launch, raising=False)
    monkeypatch.setattr(capture, "stop_owned", lambda p: stopped.append(p))
    artifact = NS(
        trial=NS(
            condition=capture.Condition.CLEAN,
            task="lift_bottle",
            initial_seed=5,
            exogenous_seed=5,
            checkpoint_sha256="b" * 64,
            dataset_sha256="c" * 64,
        ),
        capture_profile=capture.LiveCaptureProfile.PAPER_FULL,
        external_root_sha256="e" * 64,
        evidence=NS(
            result={
                "terminal_status": "task_failure",
                "score_eligible": True,
                "validation_passed": True,
                "score_success": False,
                "observation_count": 61,
            },
            finalization=NS(
                clean_records=[NS(observation=NS(step_index=i)) for i in range(61)]
            ),
        ),
    )
    monkeypatch.setattr(capture, "load_live_univtac_artifact", lambda path: artifact)
    monkeypatch.setattr(capture, "result_to_dict", lambda result: result)
    return args, plan, artifact, launched, stopped


def test_failed_task_is_retained_as_valid_full_development_source(setup_capture):
    args, plan, artifact, launched, stopped = setup_capture
    receipt = capture.run(args)
    assert receipt["valid_for_replay"] and receipt["not_scored_in_campaign"]
    assert receipt["terminal_status"] == "task_failure"
    assert len(launched) == len(stopped) == 2
    argv, env = launched[1]
    assert argv[0] == str(args.isaac_python)
    assert argv[argv.index("--capture-profile") + 1] == "paper_full_v1"
    assert argv[argv.index("--root") + 1] == str(args.model_root)
    frozen = json.loads((args.output / "plan.json").read_text())
    assert frozen["source_protocol_sha256"] == plan["protocol_sha256"]
    assert frozen["max_attempts"] == 1 and not frozen["success_rate_claimed"]


@pytest.mark.parametrize(
    "defect",
    ["dataset", "model", "seed", "condition", "short", "sparse", "metrics", "crash"],
)
def test_strict_source_rejection(setup_capture, defect):
    args, plan, artifact, launched, stopped = setup_capture
    if defect in {"dataset", "model", "seed", "condition"}:
        name, value = {
            "dataset": ("dataset_sha256", "z"),
            "model": ("checkpoint_sha256", "z"),
            "seed": ("initial_seed", 50),
            "condition": ("condition", None),
        }[defect]
        setattr(artifact.trial, name, value)
    elif defect == "short":
        artifact.evidence.finalization.clean_records.pop()
    elif defect == "sparse":
        artifact.evidence.finalization.clean_records[20].observation.step_index = 21
    elif defect == "metrics":
        artifact.capture_profile = capture.LiveCaptureProfile.METRICS_ONLY
    else:
        artifact.evidence.result["terminal_status"] = "crash"
    with pytest.raises(ValueError, match="eligibility"):
        capture.run(args)
    assert len(launched) == len(stopped) == 2
    assert not json.loads((args.output / "receipt.json").read_text())[
        "valid_for_replay"
    ]
    assert (args.output / "failure.json").is_file()


@pytest.mark.parametrize(
    "defect", ["cohort", "undeclared", "occupied", "port", "hash", "existing"]
)
def test_refuses_before_launch(setup_capture, monkeypatch, defect):
    args, plan, artifact, launched, stopped = setup_capture
    if defect == "cohort":
        plan["seeds"].append(5)
    elif defect == "undeclared":
        args.seed = 6
    elif defect == "occupied":
        monkeypatch.setattr(capture.subprocess, "check_output", lambda *a, **kw: "1024")
    elif defect == "port":
        monkeypatch.setattr(
            capture,
            "check_port",
            lambda port: (_ for _ in ()).throw(OSError("occupied")),
        )
    elif defect == "hash":
        args.isaac_python_sha256 = "wrong"
    else:
        args.output.mkdir()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        capture.run(args)
    assert not launched
    assert not (args.output / "plan.json").exists()


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timeout"), InterruptedError("signal"), RuntimeError("startup")],
)
def test_owned_server_cleanup_on_startup_failure(setup_capture, monkeypatch, error):
    args, plan, artifact, launched, stopped = setup_capture
    monkeypatch.setattr(capture, "wait_server", lambda *a: (_ for _ in ()).throw(error))
    with pytest.raises(type(error)):
        capture.run(args)
    assert len(launched) == 1
    assert len(stopped) == 2 and stopped[0] is None and stopped[1] is not None
    assert (args.output / "exit.json").is_file()
    assert (args.output / "failure.json").is_file()


def test_preserves_venv_launcher_symlinks(setup_capture):
    args, plan, artifact, launched, stopped = setup_capture
    launcher = args.isaac_python.with_name("venv-python")
    launcher.symlink_to(args.isaac_python)
    args.isaac_python = launcher
    capture.run(args)
    assert launched[1][0][0] == str(launcher)


def test_corrupt_artifact_retained_without_retry(setup_capture, monkeypatch):
    args, plan, artifact, launched, stopped = setup_capture
    monkeypatch.setattr(
        capture,
        "load_live_univtac_artifact",
        lambda path: (_ for _ in ()).throw(ValueError("root hash mismatch")),
    )
    with pytest.raises(ValueError, match="root hash"):
        capture.run(args)
    assert len(launched) == len(stopped) == 2
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert not receipt["valid_for_replay"] and receipt["source_sha256"] is None
