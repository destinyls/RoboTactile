"""No-GPU contracts for the official seed-bound robustness supervisor."""

import json
import socket
from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.artifact_io import strict_json_bytes
from scripts.n0_twam.run_official_early_fault_seed import (
    _check_server_port_available,
    _clean_template,
    _runtime_environment,
    official_server_command,
    official_server_environment,
    prepare,
    run,
    write_json,
)


def test_port_check_rejects_live_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with pytest.raises(OSError):
            _check_server_port_available(listener.getsockname()[1])


def test_port_check_sets_reuse_before_bind(monkeypatch):
    events = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            events.append(("setsockopt", args))

        def bind(self, *args):
            events.append(("bind", args))

    monkeypatch.setattr(socket, "socket", Probe)
    _check_server_port_available(29695)
    assert events == [
        ("setsockopt", (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)),
        ("bind", (("127.0.0.1", 29695),)),
    ]


def test_supervisor_writes_strict_canonical_json_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plan.json"
    write_json(path, {"seed": 5, "tasks": ["lift_can"]})
    assert strict_json_bytes(path.read_bytes(), "plan") == {
        "seed": 5,
        "tasks": ["lift_can"],
    }
    with pytest.raises(FileExistsError):
        write_json(path, {"seed": 6})


def test_clean_template_requires_unique_live_cell(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    manifest = reference / "fault_campaign/campaign_manifest.json"
    write_json(
        manifest,
        {
            "cells": [
                {
                    "condition": "clean",
                    "disposition": "live_request",
                    "request_relpath": "requests/task/pair/clean.json",
                }
            ]
        },
    )
    assert _clean_template(reference) == (
        reference / "fault_campaign/requests/task/pair/clean.json"
    )


def test_prepare_rejects_duplicate_tasks_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "new-campaign"
    with pytest.raises(ValueError, match="unique"):
        prepare(
            repo=tmp_path,
            output=output,
            reference_a=tmp_path / "reference-a",
            reference_b=tmp_path / "reference-b",
            tasks=("lift_can", "lift_can"),
            seed=5,
            integration_label="source-example",
            port=29693,
            capture_profile="paper_full_v1",
            isaac_python=tmp_path / "isaac-python.sh",
        )
    assert not output.exists()


def test_prepare_rejects_missing_explicit_n0_runtime_before_writing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "new-campaign"
    isaac_python = tmp_path / "isaac-python.sh"
    isaac_python.touch()
    with pytest.raises(FileNotFoundError, match="custom-n0-python"):
        prepare(
            repo=tmp_path,
            output=output,
            reference_a=tmp_path / "reference-a",
            reference_b=tmp_path / "reference-b",
            tasks=("lift_can",),
            seed=5,
            integration_label="source-example",
            port=29693,
            capture_profile="paper_full_v1",
            isaac_python=isaac_python,
            n0_python=tmp_path / "custom-n0-python",
            n0_source=tmp_path / "custom-n0-source",
            gpu_id=3,
        )
    assert not output.exists()


def test_run_reads_custom_n0_runtime_and_gpu_from_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "campaign"
    package_root = tmp_path / "package"
    (package_root / "robotactile_benchmark").mkdir(parents=True)
    isaac_python = tmp_path / "isaac-python.sh"
    n0_python = tmp_path / "custom-n0-python"
    n0_source = tmp_path / "custom-n0-source"
    isaac_python.touch()
    n0_python.touch()
    n0_source.mkdir()
    write_json(
        output / "plan.json",
        {
            "repo": str(tmp_path),
            "tasks": [],
            "seed": 5,
            "port": 29693,
            "isaac_python": str(isaac_python),
            "n0_python": str(n0_python),
            "n0_source": str(n0_source),
            "gpu_id": 3,
        },
    )
    captured = {}

    def capture_write(path: Path, value: object) -> None:
        captured["path"] = path
        captured["value"] = value

    monkeypatch.setattr(
        "scripts.n0_twam.run_official_early_fault_seed.write_json",
        capture_write,
    )
    run(output=output, package_root=package_root)
    assert captured["path"] == output / "finished.json"
    assert captured["value"]["tasks"] == []
    assert json.loads((output / "plan.json").read_text())["gpu_id"] == 3


def test_runtime_environment_uses_planned_gpu(tmp_path: Path) -> None:
    env = _runtime_environment(
        deployment_root=tmp_path / "deployment",
        package_root=tmp_path / "package",
        n0_source=tmp_path / "custom-n0-source",
        gpu_id=3,
        inherited={"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "inherited"},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONPATH"].split(":") == [
        str(tmp_path / "package"),
        str(tmp_path / "custom-n0-source"),
        "inherited",
    ]
    assert Path(env["ROBOTACTILE_N0_DIGEST_CACHE_DIR"]).parent == (
        tmp_path / "deployment/runtime/artifact-digest-cache/n0-twam"
    )


def test_official_server_uses_single_rank_torchrun(tmp_path: Path) -> None:
    command = official_server_command(
        n0_python=tmp_path / "n0-python",
        repo=tmp_path,
        port=29693,
        save_root=tmp_path / "server-dumps",
    )
    assert command[:5] == [
        str(tmp_path / "n0-python"),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
    ]


def test_official_server_binds_task_pool_and_local_bundle(tmp_path: Path) -> None:
    model_root = tmp_path / "artifacts/models/n0_twam"
    pool = model_root / "serve-pools/grasp_classify"
    write_json(
        pool / "serve_bundle_manifest.json",
        {
            "task_id": "grasp_classify",
            "action_mode": "delta",
            "serve_pool_root": str(pool),
            "serve_task_id": "univtac_grasp_classify_hdf5_current",
        },
    )
    vae = model_root / "serve-bundle/vae"
    vae.mkdir(parents=True)
    (vae / "config.json").write_text("{}", encoding="utf-8")
    env = official_server_environment(
        root=tmp_path,
        task="grasp_classify",
        save_root=tmp_path / "server-dumps",
        inherited={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert env["TWAM_SERVE_POOL"] == str(pool)
    assert env["TWAM_SERVE_TASK"] == "univtac_grasp_classify_hdf5_current"
    assert env["TWAM_SERVE_ACTION_MODE"] == "delta"
    assert env["TWAM_SERVE_BUNDLE"] == str(model_root / "serve-bundle")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
