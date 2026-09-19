"""No GPU/server execution: queue argv, ownership, locks and failure semantics."""

import argparse
import json
from types import SimpleNamespace

import pytest

from scripts.n0_twam import noise_stress_queue as queue


def arguments(tmp_path):
    paths = {}
    for name in ("root", "model_root", "code", "package", "n0_source", "digest_cache"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    for name in ("protocol", "config", "isaac_python", "n0_python"):
        path = tmp_path / name
        path.write_text("{}")
        paths[name] = path
    return argparse.Namespace(
        **paths,
        campaign=tmp_path / "campaign",
        gpu_lock=tmp_path / "gpu-0.lock",
        gpu_id=0,
        port=29000,
        max_attempts=1,
        episode_timeout_s=10,
        startup_timeout_s=10,
        rest_reference=None,
        spatial_calibration=None,
    )


def test_argv_preserves_paths_and_probe_stage(tmp_path):
    args = arguments(tmp_path)
    args.code = tmp_path / "code with spaces"
    group = tmp_path / "group with spaces"
    command = queue.worker_command(args, group)
    assert command[command.index("--group") + 1] == str(group)
    assert command.count("--port") == 1
    assert "--capture-profile" not in command
    assert "--diagnostic-input-trace-root" in queue.server_command(
        args, {"stage": "screening"}
    )
    assert "--diagnostic-input-trace-root" not in queue.server_command(
        args, {"stage": "confirmation"}
    )


def test_gpu_lock_is_exclusive_and_inode_preserved(tmp_path):
    path = tmp_path / "gpu.lock"
    with queue.gpu_lock(path), pytest.raises(BlockingIOError), queue.gpu_lock(path):
        pass
    assert path.exists()
    with queue.gpu_lock(path):
        pass


@pytest.mark.parametrize("worker_exit,accepted", [(0, True), (7, True), (0, False)])
def test_owned_process_cleanup_and_no_outcome_retries(
    tmp_path, monkeypatch, worker_exit, accepted
):
    args = arguments(tmp_path)
    plan = {
        "stage": "screening",
        "protocol_sha256": "a" * 64,
        "binding": {"dataset_sha256": "b" * 64},
        "seeds": [50, 51],
        "variants": [{"label": "fast_f1-s5"}],
    }
    monkeypatch.setattr(queue, "validate_protocol", lambda raw: plan)
    monkeypatch.setattr(queue, "validate_runtime_binding", lambda *a: {})
    monkeypatch.setattr(queue, "file_sha256", lambda path: "d" * 64)
    monkeypatch.setattr(queue, "check_port", lambda port: None)
    monkeypatch.setattr(
        queue, "official_server_environment", lambda **kw: dict(kw["inherited"])
    )
    monkeypatch.setattr(queue, "wait_server", lambda *a: None)
    monkeypatch.setattr(
        queue, "build_request", lambda args, seed, source: {"seed": seed}
    )
    monkeypatch.setattr(queue, "live_univtac_request_to_dict", lambda obj: obj)

    def prepare(group, **kwargs):
        group.mkdir()
        (group / "group_result.json").write_text("{}")

    monkeypatch.setattr(queue, "prepare_stress_group", prepare)
    monkeypatch.setattr(
        queue, "read_stress_rows", lambda root: [{"group_accepted": accepted}] * 2
    )
    monkeypatch.setattr(
        queue, "summarize_stress", lambda plan, rows: {"verified_rows": len(rows)}
    )
    processes, stopped = [], []

    class Process:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.pid = 100 + len(processes)
            self.returncode = None
            assert kwargs["start_new_session"] is True
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = worker_exit
            return worker_exit

    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setattr(queue, "stop_owned", lambda process: stopped.append(process))
    if worker_exit or not accepted:
        with pytest.raises(RuntimeError, match="stopped queue"):
            queue.run(args)
        assert len(processes) == 2
        assert (args.campaign / "failure.json").is_file()
    else:
        queue.run(args)
        assert len(processes) == 3
    assert stopped[-1] is processes[0]
    assert stopped[-2] is processes[-1]
    assert all(process in processes for process in stopped)
    summary = json.loads((args.campaign / "summary.json").read_text())
    assert summary["abnormal_worker_exit_count"] == int(worker_exit != 0)
    if worker_exit and accepted:
        assert summary["verified_rows"] == 2
        assert summary["worker_exits"][0]["pre_close_group_result_present"] is True
    original = (args.campaign / "launch.json").read_bytes()
    with pytest.raises(FileExistsError):
        queue.run(args)
    assert (args.campaign / "launch.json").read_bytes() == original


def test_rejects_retry_override_before_launch(tmp_path):
    args = arguments(tmp_path)
    args.max_attempts = 2
    with pytest.raises(ValueError, match="one frozen attempt"):
        queue.run(args)
    assert not args.campaign.exists()


def test_shard_filter_preserves_protocol_order_and_rejects_invalid_members():
    plan = {"seeds": [103, 101, 107, 109]}
    assert queue.shard_seeds(plan, [109, 103]) == [103, 109]
    assert queue.shard_seeds(plan, None) == plan["seeds"]
    for selected in ([103, 103], [999], [], [True]):
        with pytest.raises(ValueError, match="unique members"):
            queue.shard_seeds(plan, selected)
    assert plan == {"seeds": [103, 101, 107, 109]}


def test_serving_paths_are_bound_to_verified_artifact_config(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    base = args.model_root / "artifacts/models/n0_twam"
    manifest = SimpleNamespace(
        task_id="lift_bottle",
        checkpoint_sha256="a" * 64,
        serve_pool_root=base / "serve-pools/lift_bottle",
        serve_bundle_root=base / "serve-bundle",
        serve_bundle_manifest_path=base
        / "serve-pools/lift_bottle/serve_bundle_manifest.json",
    )
    runtime = SimpleNamespace(manifest=manifest, manifest_path=args.config)
    plan = {
        "binding": {
            "integration_config_sha256": "b" * 64,
            "model_sha256": "a" * 64,
            "code_sha256": "c" * 64,
        }
    }
    monkeypatch.setattr(queue, "resolve_n0_runtime_artifacts", lambda path: runtime)
    monkeypatch.setattr(queue, "verify_runtime_code", lambda *args: None)
    monkeypatch.setattr(queue, "file_sha256", lambda path: "b" * 64)
    assert queue.validate_runtime_binding(args, plan)["checkpoint_sha256"] == "a" * 64
    for field in ("serve_pool_root", "serve_bundle_root", "serve_bundle_manifest_path"):
        original = getattr(manifest, field)
        setattr(manifest, field, tmp_path / "unrelated-model")
        with pytest.raises(ValueError, match=field):
            queue.validate_runtime_binding(args, plan)
        setattr(manifest, field, original)


def test_launcher_symlinks_hash_real_files_without_changing_argv(tmp_path):
    args = arguments(tmp_path)
    target = tmp_path / "real-python"
    target.write_bytes(b"real interpreter fixture")
    for name in ("n0_python", "isaac_python"):
        link = tmp_path / (name + "-venv-link")
        link.symlink_to(target)
        setattr(args, name, link)
    worker = args.code / "scripts/n0_twam/run_noise_stress.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# worker fixture")
    server = args.n0_source / "n0_twam/n0_twam_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# server fixture")
    value = queue.launch_file_provenance(args)
    assert value["n0_python_resolved"] == str(target)
    assert value["isaac_python_resolved"] == str(target)
    assert value["n0_launcher_sha256"] == queue.file_sha256(target)
    assert value["isaac_launcher_sha256"] == queue.file_sha256(target)
    assert queue.server_command(args, {"stage": "screening"})[0] == str(args.n0_python)
    assert queue.worker_command(args, tmp_path / "group")[0] == str(args.isaac_python)
    assert args.n0_python.is_symlink() and args.isaac_python.is_symlink()


def test_launch_hash_failure_does_not_create_campaign(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    plan = {"seeds": [101], "binding": {"dataset_sha256": "a" * 64}}
    monkeypatch.setattr(queue, "validate_protocol", lambda raw: plan)
    monkeypatch.setattr(queue, "validate_runtime_binding", lambda *a: {})
    # Real launch hashing fails on the absent worker script after hashing both
    # valid launcher files. This must happen before mkdir or Popen.
    with pytest.raises((FileNotFoundError, ValueError)):
        queue.run(args)
    assert not args.campaign.exists()
    assert not args.gpu_lock.exists()
