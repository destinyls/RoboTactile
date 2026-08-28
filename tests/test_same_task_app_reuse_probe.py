from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from scripts.live_univtac import probe_same_task_app_reuse as module


class _FakeRuntime:
    def __init__(self, seed: int, observer: Optional[object]) -> None:
        self.seed = seed
        self.observer = observer
        self.close_count = 0

    def close_runtime(self) -> None:
        self.close_count += 1
        if callable(self.observer):
            self.observer("task_closed")


class _FakeHost:
    def __init__(self) -> None:
        self.runtimes: list[_FakeRuntime] = []
        self.close_count = 0
        self.close_error: Optional[BaseException] = None
        self.close_observer: Optional[Callable[[], None]] = None

    @property
    def runtime_count(self) -> int:
        return len(self.runtimes)

    def create_runtime(
        self,
        *,
        runtime_dir: Path,
        initial_seed: int,
        stage_observer: Optional[object] = None,
    ) -> _FakeRuntime:
        assert not runtime_dir.exists()
        if callable(stage_observer):
            stage_observer("runtime_ready")
        runtime = _FakeRuntime(initial_seed, stage_observer)
        self.runtimes.append(runtime)
        return runtime

    def close(self) -> None:
        self.close_count += 1
        if self.close_observer is not None:
            self.close_observer()
        if self.close_error is not None:
            raise self.close_error


class _FakeBackend:
    fail_seed: Optional[int] = None

    def __init__(self, _config: object, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.latest_state_sha256: Optional[str] = None
        self.initial_clean_record_sha256: Optional[str] = None

    def reset(self, context: PolicyEpisodeContext) -> object:
        seed = context.initial_seed
        if seed == self.fail_seed:
            raise RuntimeError("mock reset timeout")
        assert seed == self.runtime.seed
        self.latest_state_sha256 = f"{seed % 10}" * 64
        self.initial_clean_record_sha256 = f"{(seed + 1) % 10}" * 64
        return SimpleNamespace(
            simulator_state_sha256=self.latest_state_sha256,
            sha256=f"{(seed + 2) % 10}" * 64,
        )

    def observe(self) -> object:
        return SimpleNamespace(clean_record_sha256=self.initial_clean_record_sha256)

    def close(self) -> None:
        self.runtime.close_runtime()


def _arguments(tmp_path: Path, seeds: object = None) -> argparse.Namespace:
    upstream_root = tmp_path / "upstream"
    runtime_root = tmp_path / "runtime"
    upstream_root.mkdir(exist_ok=True)
    runtime_root.mkdir(exist_ok=True)
    return argparse.Namespace(
        antialiasing_mode="TAA",
        device="cuda:0",
        initial_seeds=seeds,
        output=tmp_path / "probe.json",
        preclose_output=None,
        rendering_mode="balanced",
        runtime_root=runtime_root,
        task="insert_tube",
        upstream_root=upstream_root,
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeHost, list[dict[str, object]]]:
    host = _FakeHost()
    launches: list[dict[str, object]] = []
    config = SimpleNamespace(
        action_spec="ee8",
        task=SimpleNamespace(task_id="insert_tube", prompt="insert the tube"),
    )

    def launch(_config: object, **kwargs: object) -> _FakeHost:
        launches.append(dict(kwargs))
        observer = kwargs["stage_observer"]
        assert callable(observer)
        observer("simulation_app_ready")
        return host

    monkeypatch.setattr(
        module, "build_univtac_backend_config", lambda *_args, **_kwargs: config
    )
    monkeypatch.setattr(module, "launch_univtac_app_host", launch)
    monkeypatch.setattr(module, "UniVTACIsaacBackend", _FakeBackend)
    return host, launches


def test_probe_defaults_to_two_distinct_seeds(tmp_path: Path) -> None:
    args = _arguments(tmp_path)

    assert module._initial_seeds(args) == (4_000_000, 4_000_001)

    args.initial_seeds = [4_000_000]
    with pytest.raises(ValueError, match="at least twice"):
        module._initial_seeds(args)


def test_probe_validates_preclose_no_clobber_before_app_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host, launches = _install_fakes(monkeypatch)
    args = _arguments(tmp_path)
    preclose = tmp_path / "existing-preclose.json"
    preclose.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must not already exist"):
        module.main(
            [
                "--upstream-root",
                str(args.upstream_root),
                "--runtime-root",
                str(args.runtime_root),
                "--output",
                str(args.output),
                "--preclose-output",
                str(preclose),
                "--task",
                args.task,
            ]
        )

    assert launches == []
    assert not args.output.exists()


def test_probe_reuses_one_app_with_two_fresh_runtimes_and_no_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeBackend.fail_seed = None
    host, launches = _install_fakes(monkeypatch)
    (tmp_path / "upstream").mkdir()
    (tmp_path / "runtime").mkdir()
    output = tmp_path / "probe.json"

    exit_code = module.main(
        [
            "--upstream-root",
            str(tmp_path / "upstream"),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--output",
            str(output),
            "--task",
            "insert_tube",
        ]
    )

    assert exit_code == 0
    assert len(launches) == 1
    assert launches[0]["initial_seed"] == 4_000_000
    assert launches[0]["launcher_args"] == {
        "enable_cameras": True,
        "headless": True,
        "kit_args": "--/app/hangDetector/enabled=false",
        "rendering_mode": "balanced",
    }
    assert host.close_count == 1
    assert len(host.runtimes) == 2
    assert host.runtimes[0] is not host.runtimes[1]
    assert [runtime.close_count for runtime in host.runtimes] == [1, 1]

    document = json.loads(output.read_text(encoding="utf-8"))
    preclose = json.loads(
        (tmp_path / "probe.preclose.json").read_text(encoding="utf-8")
    )
    assert document["status"] == "passed"
    assert document["receipt_stage"] == "final"
    assert document["policy_loaded"] is False
    assert document["runtime_count"] == 2
    assert document["completed_runtime_count"] == 2
    assert document["same_process_confirmed"] is True
    assert document["app_process_id"] == os.getpid()
    assert [item["process_id"] for item in document["runs"]] == [
        os.getpid(),
        os.getpid(),
    ]
    assert [item["runtime_count"] for item in document["runs"]] == [1, 2]
    assert all(item["initial_state_sha256"] for item in document["runs"])
    assert all(item["initial_clean_record_sha256"] for item in document["runs"])
    assert all(item["construction_duration_s"] is not None for item in document["runs"])
    assert all(item["reset_duration_s"] is not None for item in document["runs"])
    assert all(item["close_duration_s"] is not None for item in document["runs"])
    assert "task_closed" in document["runs"][0]["lifecycle_stages"]
    assert len(document["content_sha256"]) == 64
    assert preclose["status"] == "runtimes_passed_app_close_pending"
    assert preclose["receipt_stage"] == "preclose"
    assert preclose["app_close_status"] == "pending"
    assert preclose["exit_code"] is None
    assert preclose["runtime_count"] == 2
    assert preclose["completed_runtime_count"] == 2
    assert all(item["initial_state_sha256"] for item in preclose["runs"])
    assert all(item["initial_clean_record_sha256"] for item in preclose["runs"])
    assert len(preclose["content_sha256"]) == 64

    with pytest.raises(FileExistsError, match="must not already exist"):
        module.main(
            [
                "--upstream-root",
                str(tmp_path / "upstream"),
                "--runtime-root",
                str(tmp_path / "runtime"),
                "--output",
                str(output),
                "--task",
                "insert_tube",
            ]
        )
    assert len(launches) == 1


def test_probe_publishes_explicit_failure_and_closes_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeBackend.fail_seed = 4_000_001
    host, launches = _install_fakes(monkeypatch)
    args = _arguments(tmp_path)

    exit_code = module.main(
        [
            "--upstream-root",
            str(args.upstream_root),
            "--runtime-root",
            str(args.runtime_root),
            "--output",
            str(args.output),
            "--task",
            args.task,
        ]
    )
    document = json.loads(args.output.read_text(encoding="utf-8"))
    preclose = json.loads(
        (tmp_path / "probe.preclose.json").read_text(encoding="utf-8")
    )

    assert exit_code == 1
    assert len(launches) == 1
    assert host.close_count == 1
    assert [runtime.close_count for runtime in host.runtimes] == [1, 1]
    assert document["status"] == "failed"
    assert document["same_process_confirmed"] is False
    assert document["completed_runtime_count"] == 1
    assert document["runtime_count"] == 2
    assert document["failure"] == {
        "error_message": "mock reset timeout",
        "error_type": "RuntimeError",
        "stage": "runtime[1].reset",
    }
    assert document["runs"][1]["status"] == "failed"  # type: ignore[index]
    assert document["runs"][1]["failure"] == document["failure"]  # type: ignore[index]
    assert preclose["status"] == "failed"
    assert preclose["app_close_status"] == "pending"
    assert preclose["failure"] == document["failure"]


def test_probe_publishes_passed_receipt_after_app_close_system_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeBackend.fail_seed = None
    host, _launches = _install_fakes(monkeypatch)
    host.close_error = SystemExit(0)
    args = _arguments(tmp_path)
    preclose_output = tmp_path / "explicit-preclose.json"

    def observe_close() -> None:
        assert preclose_output.is_file()
        assert not args.output.exists()

    host.close_observer = observe_close

    exit_code = module.main(
        [
            "--upstream-root",
            str(args.upstream_root),
            "--runtime-root",
            str(args.runtime_root),
            "--output",
            str(args.output),
            "--preclose-output",
            str(preclose_output),
            "--task",
            args.task,
        ]
    )
    document = json.loads(args.output.read_text(encoding="utf-8"))
    preclose = json.loads(preclose_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert host.close_count == 1
    assert document["status"] == "passed"
    assert document["exit_code"] == 0
    assert document["app_close_status"] == "system_exit_zero"
    assert "app_close_system_exit_zero" in document["app_lifecycle_stages"]
    assert document["failure"] is None
    assert preclose["status"] == "runtimes_passed_app_close_pending"
    assert preclose["app_close_status"] == "pending"
    assert preclose["receipt_stage"] == "preclose"


@pytest.mark.parametrize(
    ("close_error", "expected_exit_code", "expected_close_status"),
    [
        (RuntimeError("native close failed"), 1, "failed"),
        (SystemExit(7), 7, "system_exit_nonzero"),
        (KeyboardInterrupt(), 130, "interrupted"),
    ],
)
def test_probe_publishes_failed_receipt_for_abnormal_app_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_error: BaseException,
    expected_exit_code: int,
    expected_close_status: str,
) -> None:
    _FakeBackend.fail_seed = None
    host, _launches = _install_fakes(monkeypatch)
    host.close_error = close_error
    args = _arguments(tmp_path)

    exit_code = module.main(
        [
            "--upstream-root",
            str(args.upstream_root),
            "--runtime-root",
            str(args.runtime_root),
            "--output",
            str(args.output),
            "--task",
            args.task,
        ]
    )
    document = json.loads(args.output.read_text(encoding="utf-8"))
    preclose = json.loads(
        (tmp_path / "probe.preclose.json").read_text(encoding="utf-8")
    )

    assert exit_code == expected_exit_code
    assert host.close_count == 1
    assert document["status"] == "failed"
    assert document["exit_code"] == expected_exit_code
    assert document["app_close_status"] == expected_close_status
    assert document["failure"]["stage"] == "app_close"
    assert document["completed_runtime_count"] == 2
    assert document["runtime_count"] == 2
    assert preclose["status"] == "runtimes_passed_app_close_pending"
    assert preclose["app_close_status"] == "pending"
