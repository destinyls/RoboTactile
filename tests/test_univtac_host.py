from __future__ import annotations

import types
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from robotactile_benchmark.backends.qualification_fakes import make_fake_runtime
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    _UniVTACApplicationResources,
    launch_univtac_app_host,
)
from robotactile_benchmark.backends.univtac_host import (
    StageObserver,
    UniVTACSimulationAppHost,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACTaskRuntime


def test_host_reuses_app_but_never_reuses_task(tmp_path: Path) -> None:
    config = build_univtac_backend_config("pull_out_key")
    app_close_count = 0
    tasks = []
    factory_calls: list[tuple[Path, int]] = []

    def factory(
        runtime_dir: Path,
        initial_seed: int,
        stage_observer: Optional[StageObserver],
    ) -> UniVTACTaskRuntime:
        factory_calls.append((runtime_dir, initial_seed))
        if stage_observer is not None:
            stage_observer("runtime_ready")
        runtime, task = make_fake_runtime(
            config,
            construction_seed=initial_seed,
        )
        tasks.append(task)
        return runtime

    def close_application() -> None:
        nonlocal app_close_count
        app_close_count += 1

    host = UniVTACSimulationAppHost(
        task_runtime_factory=factory,
        close_application=close_application,
    )
    stages: list[str] = []

    first = host.create_runtime(
        runtime_dir=tmp_path / "first",
        initial_seed=11,
        stage_observer=stages.append,
    )
    assert host.active
    assert first.construction_seed == 11
    with pytest.raises(UniVTACContractError, match="active task runtime"):
        host.create_runtime(runtime_dir=tmp_path / "overlap", initial_seed=12)
    first.close_runtime()
    first.close_runtime()
    assert not host.active
    assert tasks[0].close_count == 1
    assert app_close_count == 0

    second = host.create_runtime(
        runtime_dir=tmp_path / "second",
        initial_seed=29,
        stage_observer=stages.append,
    )
    assert second.task is not first.task
    assert second.construction_seed == 29
    second.close_runtime()

    assert factory_calls == [
        ((tmp_path / "first").resolve(), 11),
        ((tmp_path / "second").resolve(), 29),
    ]
    assert stages == [
        "runtime_ready",
        "task_closed",
        "runtime_ready",
        "task_closed",
    ]
    assert [task.close_count for task in tasks] == [1, 1]
    assert host.runtime_count == 2
    host.close()
    host.close()
    assert app_close_count == 1


def test_host_close_reclaims_active_task_before_app() -> None:
    config = build_univtac_backend_config("pull_out_key")
    events: list[str] = []
    runtime, task = make_fake_runtime(config)
    task_close = runtime.close_runtime

    def close_task() -> None:
        task_close()
        events.append("task")

    runtime = UniVTACTaskRuntime(
        task=runtime.task,
        handshake=runtime.handshake,
        construction_seed=runtime.construction_seed,
        encode_action=runtime.encode_action,
        prepare_reset=runtime.prepare_reset,
        close_runtime=close_task,
        capture_state=runtime.capture_state,
        restore_state=runtime.restore_state,
        snapshot_state_sha256=runtime.snapshot_state_sha256,
    )
    host = UniVTACSimulationAppHost(
        task_runtime_factory=lambda _dir, _seed, _observer: runtime,
        close_application=lambda: events.append("app"),
    )

    leased = host.create_runtime(runtime_dir=Path("runtime"), initial_seed=11)
    host.close()
    leased.close_runtime()

    assert events == ["task", "app"]
    assert task.close_count == 1
    assert host.closed
    assert not host.active


def test_task_construction_failure_poison_closes_host(tmp_path: Path) -> None:
    app_close_count = 0

    def fail_factory(
        _runtime_dir: Path,
        _initial_seed: int,
        _stage_observer: Optional[StageObserver],
    ) -> UniVTACTaskRuntime:
        raise RuntimeError("partial task construction")

    def close_application() -> None:
        nonlocal app_close_count
        app_close_count += 1

    host = UniVTACSimulationAppHost(
        task_runtime_factory=fail_factory,
        close_application=close_application,
    )

    with pytest.raises(RuntimeError, match="partial task construction"):
        host.create_runtime(runtime_dir=tmp_path, initial_seed=11)
    assert host.closed
    assert app_close_count == 1
    with pytest.raises(UniVTACContractError, match="closed"):
        host.create_runtime(runtime_dir=tmp_path, initial_seed=12)


def test_host_runs_teardown_and_reconstruction_between_tasks(
    tmp_path: Path,
) -> None:
    config = build_univtac_backend_config("pull_out_key")
    events: list[str] = []

    def factory(
        _runtime_dir: Path,
        initial_seed: int,
        _stage_observer: Optional[StageObserver],
    ) -> UniVTACTaskRuntime:
        events.append(f"construct:{initial_seed}")
        runtime, _ = make_fake_runtime(config, construction_seed=initial_seed)
        return runtime

    def finalize(
        runtime: UniVTACTaskRuntime,
        _stage_observer: Optional[StageObserver],
    ) -> None:
        events.append(f"finalize:{runtime.construction_seed}")

    host = UniVTACSimulationAppHost(
        task_runtime_factory=factory,
        task_runtime_finalizer=finalize,
        task_reconstruction_barrier=lambda _observer: events.append("barrier"),
        close_application=lambda: events.append("app:close"),
    )
    stages: list[str] = []

    first = host.create_runtime(
        runtime_dir=tmp_path / "first",
        initial_seed=11,
        stage_observer=stages.append,
    )
    first.close_runtime()
    second = host.create_runtime(
        runtime_dir=tmp_path / "second",
        initial_seed=29,
        stage_observer=stages.append,
    )
    second.close_runtime()
    host.close()

    assert events == [
        "construct:11",
        "finalize:11",
        "barrier",
        "construct:29",
        "finalize:29",
        "app:close",
    ]
    assert stages == ["task_closed", "task_closed"]


def test_task_teardown_failure_poison_closes_host(tmp_path: Path) -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtime, task = make_fake_runtime(config)
    app_close_count = 0

    def close_application() -> None:
        nonlocal app_close_count
        app_close_count += 1

    def fail_finalizer(
        _runtime: UniVTACTaskRuntime,
        _stage_observer: Optional[StageObserver],
    ) -> None:
        raise RuntimeError("native teardown failed")

    host = UniVTACSimulationAppHost(
        task_runtime_factory=lambda _dir, _seed, _observer: runtime,
        task_runtime_finalizer=fail_finalizer,
        close_application=close_application,
    )
    leased = host.create_runtime(runtime_dir=tmp_path, initial_seed=11)

    with pytest.raises(RuntimeError, match="native teardown failed"):
        leased.close_runtime()

    assert task.close_count == 1
    assert host.closed
    assert not host.active
    assert app_close_count == 1
    with pytest.raises(UniVTACContractError, match="closed"):
        host.create_runtime(runtime_dir=tmp_path / "next", initial_seed=12)


def test_reconstruction_failure_poison_closes_host(tmp_path: Path) -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtimes: list[UniVTACTaskRuntime] = []
    app_close_count = 0

    def factory(
        _runtime_dir: Path,
        initial_seed: int,
        _stage_observer: Optional[StageObserver],
    ) -> UniVTACTaskRuntime:
        runtime, _ = make_fake_runtime(config, construction_seed=initial_seed)
        runtimes.append(runtime)
        return runtime

    def close_application() -> None:
        nonlocal app_close_count
        app_close_count += 1

    def fail_reconstruction(_observer: Optional[StageObserver]) -> None:
        raise RuntimeError("stage recreation failed")

    host = UniVTACSimulationAppHost(
        task_runtime_factory=factory,
        task_reconstruction_barrier=fail_reconstruction,
        close_application=close_application,
    )
    first = host.create_runtime(runtime_dir=tmp_path / "first", initial_seed=11)
    first.close_runtime()

    with pytest.raises(RuntimeError, match="stage recreation failed"):
        host.create_runtime(runtime_dir=tmp_path / "second", initial_seed=29)

    assert len(runtimes) == 1
    assert host.closed
    assert app_close_count == 1


def test_public_factory_host_constructs_two_task_local_runtimes(
    tmp_path: Path,
) -> None:
    config = build_univtac_backend_config("pull_out_key")
    app_close_count = 0
    constructed: list[tuple[Path, int]] = []
    tasks = []

    def close_app() -> None:
        nonlocal app_close_count
        app_close_count += 1

    resources = _UniVTACApplicationResources(
        simulation_app=types.SimpleNamespace(close=close_app),
        torch=types.SimpleNamespace(),
        task_cfg_type=lambda: None,
        task_type=lambda *_args, **_kwargs: None,
    )

    def construct_runtime(
        _config: object,
        _resources: object,
        *,
        runtime_dir: Path,
        initial_seed: int,
        device: Optional[str],
        antialiasing_mode: Optional[str],
        n0_action_execution_contract: Optional[str],
        stage_observer: Optional[StageObserver],
    ) -> UniVTACTaskRuntime:
        assert device == "cuda:7"
        assert antialiasing_mode is None
        assert n0_action_execution_contract is None
        assert stage_observer is None
        constructed.append((runtime_dir, initial_seed))
        runtime, task = make_fake_runtime(
            config,
            construction_seed=initial_seed,
        )
        tasks.append(task)
        return runtime

    with (
        patch("robotactile_benchmark.backends.univtac_factory._verify_checkout"),
        patch(
            "robotactile_benchmark.backends.univtac_factory."
            "_prepare_process_determinism"
        ),
        patch(
            "robotactile_benchmark.backends.univtac_factory."
            "_launch_univtac_application",
            return_value=resources,
        ) as launch_app,
        patch(
            "robotactile_benchmark.backends.univtac_factory."
            "_construct_univtac_task_runtime",
            side_effect=construct_runtime,
        ),
        patch(
            "robotactile_benchmark.backends.univtac_factory."
            "prepare_univtac_task_teardown"
        ) as teardown,
        patch(
            "robotactile_benchmark.backends.univtac_factory.reconstruct_univtac_stage"
        ) as reconstruct,
    ):
        host = launch_univtac_app_host(
            config,
            upstream_root=tmp_path,
            initial_seed=11,
            device="cuda:7",
        )
        first = host.create_runtime(runtime_dir=tmp_path / "seed-11", initial_seed=11)
        first.close_runtime()
        second = host.create_runtime(runtime_dir=tmp_path / "seed-29", initial_seed=29)
        second.close_runtime()
        host.close()

    launch_app.assert_called_once()
    assert constructed == [
        ((tmp_path / "seed-11").resolve(), 11),
        ((tmp_path / "seed-29").resolve(), 29),
    ]
    assert tasks[0] is not tasks[1]
    assert [task.close_count for task in tasks] == [1, 1]
    assert app_close_count == 1
    assert teardown.call_count == 2
    reconstruct.assert_called_once_with(
        resources.simulation_app,
        stage_observer=None,
    )
