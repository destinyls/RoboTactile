"""Unit contracts for the owned N0 Clean task-shard launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
    SameTaskWorkerReadyIdentity,
)

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/n0_twam/run_clean_task_shard.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("n0_clean_task_shard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shard_receipt_is_no_clobber_and_idempotent(tmp_path: Path) -> None:
    write_once = _module()._write_once
    path = tmp_path / "outputs/shard.json"
    document = {"semantic_version": "1.0", "task_id": "pull_out_key"}

    write_once(path, document)
    write_once(path, document)

    assert path.read_bytes() == canonical_json_bytes(document)
    with pytest.raises(FileExistsError):
        write_once(path, {"semantic_version": "1.0", "task_id": "lift_can"})


def test_shard_parser_requires_explicit_task_and_gpu_inventory() -> None:
    args = (
        _module()
        ._parser()
        .parse_args(
            [
                "--manifest",
                "campaign_manifest.json",
                "--task",
                "pull_out_key",
                "--gpus",
                "0",
            ]
        )
    )

    assert args.task == "pull_out_key"
    assert args.gpus == "0"
    assert args.integration_config is None
    assert args.qualification is None
    assert args.qualification_sha256 is None
    assert args.capture_profile == "paper_full_v1"
    assert args.worker_contract == "fresh_process_v1"
    assert args.reset_equivalence_receipt is None
    assert args.action_execution_contract == "robotactile_n0_training_60hz_ee_v1"


def test_runtime_launcher_allows_only_internal_symlink(tmp_path: Path) -> None:
    require_launcher = _module()._require_runtime_launcher
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    executable = runtime / "python3.11"
    executable.write_text("launcher", encoding="utf-8")
    launcher = runtime / "python"
    launcher.symlink_to(executable.name)

    assert require_launcher(launcher, runtime, "test") == launcher.absolute()

    external = tmp_path / "external-python"
    external.write_text("launcher", encoding="utf-8")
    escaped = runtime / "escaped-python"
    escaped.symlink_to(external)
    with pytest.raises(ValueError, match="escapes its runtime root"):
        require_launcher(escaped, runtime, "test")


def test_shard_passes_validated_integration_config_to_campaign_runner(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    config = root / "artifacts/models/n0_twam/configs/lift_bottle/v11.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    selected = module._optional_integration_config(root, config)
    command = module._campaign_runner_command(
        repository_root=ROOT,
        deployment_root=root,
        manifest_path=root / "requests/clean/campaign_manifest.json",
        task="lift_bottle",
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        n0_source_root=root / "src/N0-TWAM",
        n0_port=29601,
        integration_config=selected,
        action_execution_contract="robotactile_fixed_endpoint_v1",
    )

    index = command.index("--config")
    assert command[index + 1] == str(config.absolute())
    execution_index = command.index("--action-execution-contract")
    assert command[execution_index + 1] == "robotactile_fixed_endpoint_v1"
    assert "--config" not in module._campaign_runner_command(
        repository_root=ROOT,
        deployment_root=root,
        manifest_path=root / "requests/clean/campaign_manifest.json",
        task="lift_bottle",
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        n0_source_root=root / "src/N0-TWAM",
        n0_port=29601,
        integration_config=None,
    )
    default_command = module._campaign_runner_command(
        repository_root=ROOT,
        deployment_root=root,
        manifest_path=root / "requests/clean/campaign_manifest.json",
        task="lift_bottle",
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        n0_source_root=root / "src/N0-TWAM",
        n0_port=29601,
        integration_config=None,
    )
    assert (
        default_command[default_command.index("--action-execution-contract") + 1]
        == "robotactile_n0_training_60hz_ee_v1"
    )
    assert default_command[default_command.index("--capture-profile") + 1] == (
        "paper_full_v1"
    )
    preview_command = module._campaign_runner_command(
        repository_root=ROOT,
        deployment_root=root,
        manifest_path=root / "requests/clean/campaign_manifest.json",
        task="lift_bottle",
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        n0_source_root=root / "src/N0-TWAM",
        n0_port=29601,
        integration_config=None,
        capture_profile=module.LiveCaptureProfile.PREVIEW,
    )
    assert preview_command[preview_command.index("--capture-profile") + 1] == (
        "preview_v1"
    )


def test_shard_rejects_config_outside_root_and_symlink(tmp_path: Path) -> None:
    optional_config = _module()._optional_integration_config
    root = tmp_path / "deployment"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="below"):
        optional_config(root, outside)

    config = root / "config.json"
    config.write_text("{}", encoding="utf-8")
    symlink = root / "config-link.json"
    symlink.symlink_to(config.name)
    with pytest.raises(ValueError, match="non-symlink"):
        optional_config(root, symlink)


def test_shard_binds_source_qualification_and_server_attestation_to_runner(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    command = module._campaign_runner_command(
        repository_root=ROOT,
        deployment_root=root,
        manifest_path=root / "requests/clean/campaign_manifest.json",
        task="lift_bottle",
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        n0_source_root=root / "sources/N0-TWAM",
        n0_port=29601,
        integration_config=(
            root
            / "artifacts/models/n0_twam/configs/lift_bottle/integration_config.json"
        ),
        qualification_path=root / "artifacts/deployment/qualification-v3.json",
        attestation_path=(
            root / "outputs/n0-twam/lift_bottle/runtime-attestations/session.json"
        ),
        attestation_sha256="a" * 64,
    )

    assert command[command.index("--qualification") + 1].endswith(
        "qualification-v3.json"
    )
    assert command[command.index("--n0-server-attestation") + 1].endswith(
        "session.json"
    )
    assert command[command.index("--n0-server-attestation-sha256") + 1] == ("a" * 64)
    with pytest.raises(ValueError, match="incomplete"):
        module._campaign_runner_command(
            repository_root=ROOT,
            deployment_root=root,
            manifest_path=root / "requests/clean/campaign_manifest.json",
            task="lift_bottle",
            isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
            n0_source_root=root / "sources/N0-TWAM",
            n0_port=29601,
            integration_config=None,
            qualification_path=root / "artifacts/deployment/qualification-v3.json",
        )


def test_source_bound_shard_rejects_selected_executor_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    qualification_path = root / "artifacts/deployment/qualification-v3.json"
    qualification_path.parent.mkdir(parents=True)
    qualification_path.write_text("{}", encoding="utf-8")
    production = "robotactile_n0_training_60hz_ee_v1"
    source = SimpleNamespace(
        action_execution_contract=production,
        native_step_contract="fixed_physics_steps_per_action_v1",
        verify_against_current_runtime=lambda: None,
    )
    qualification = SimpleNamespace(
        semantic_version="3.0",
        sha256="a" * 64,
        tasks=("insert_hole",),
        task_source_bindings=(source,),
    )
    monkeypatch.setattr(
        module,
        "verify_all_task_qualification",
        lambda _root, _path: qualification,
    )

    assert module._optional_source_bound_qualification(
        root,
        qualification_path,
        "insert_hole",
        "a" * 64,
        production,
    ) == (qualification_path.absolute(), "a" * 64)
    with pytest.raises(ValueError, match="action execution contract mismatch"):
        module._optional_source_bound_qualification(
            root,
            qualification_path,
            "insert_hole",
            "a" * 64,
            "univtac_stock_ee_v1",
        )


def test_shard_preserves_runner_replacement_and_budget_state() -> None:
    module = _module()
    parsed = module._parse_runner_summary(
        b'{"remaining_unattempted_candidates":2,'
        b'"replacement_requires_fresh_server":true,'
        b'"stopped_for_budget":false,"target_complete":false}\n'
    )

    subset = module._runner_summary_subset(parsed)

    assert subset["remaining_unattempted_candidates"] == 2
    assert subset["replacement_requires_fresh_server"] is True
    assert subset["stopped_for_budget"] is False
    assert subset["target_complete"] is False
    with pytest.raises(RuntimeError, match="one JSON object"):
        module._parse_runner_summary(b"not-json")


def test_persistent_worker_parser_and_command_contract(tmp_path: Path) -> None:
    module = _module()
    args = module._parser().parse_args(
        [
            "--manifest",
            "campaign.json",
            "--task",
            "insert_hole",
            "--gpus",
            "0",
            "--worker-contract",
            "same_task_worker_v1",
        ]
    )
    assert args.worker_contract == "same_task_worker_v1"

    root = tmp_path / "deployment"
    reset = root / "artifacts/deployment/reset-equivalence.json"
    reset.parent.mkdir(parents=True)
    reset.write_text("{}", encoding="utf-8")
    reset_sha = module._sha256_file(reset)
    command = module._worker_command(
        deployment_root=root,
        manifest_path=root / "requests/campaign.json",
        task_id="insert_hole",
        socket_path=root / "runtime/ipc/worker.sock",
        ready_receipt=root / "outputs/workers/ready.json",
        session_receipt=root / "outputs/workers/session.json",
        integration_config=root / "artifacts/models/config.json",
        n0_source_root=root / "sources/N0-TWAM",
        n0_port=29601,
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
        restart_generation=0,
        qualification_path=root / "artifacts/deployment/qualification.json",
        attestation_path=root / "outputs/n0/server.json",
        attestation_sha256="a" * 64,
        reset_proof=(reset, reset_sha),
    )

    assert command[:2] == [
        "-m",
        "robotactile_benchmark.execution.same_task_worker_server",
    ]
    assert command[command.index("--socket") + 1].endswith("worker.sock")
    assert command[command.index("--session-receipt") + 1].endswith("session.json")
    assert command[command.index("--reset-equivalence-receipt-sha256") + 1] == (
        reset_sha
    )


def test_reset_equivalence_gate_is_formal_only_and_hash_bound(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()

    assert (
        module._reset_equivalence_proof(
            root, requested=None, expected_sha256=None, required=False
        )
        is None
    )
    with pytest.raises(ValueError, match="pilot/paper"):
        module._reset_equivalence_proof(
            root, requested=None, expected_sha256=None, required=True
        )

    receipt = root / "artifacts/deployment/reset.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module._reset_equivalence_proof(
            root,
            requested=receipt,
            expected_sha256="0" * 64,
            required=True,
        )


def test_wait_for_worker_ready_requires_bound_socket_and_matching_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    socket_path = tmp_path / "worker.sock"
    socket_path.touch()
    monkeypatch.setattr(module.stat, "S_ISSOCK", lambda _mode: True)
    ready_path = tmp_path / "ready.json"
    process = SimpleNamespace(pid=4321, poll=lambda: None)
    process_groups = {4321: 4321, 4322: 4321, 4323: 9999, 4324: 4321}
    sessions = {4321: 4321, 4322: 4321, 4323: 9999, 4324: 9999}
    monkeypatch.setattr(module.os, "getpgid", process_groups.__getitem__)
    monkeypatch.setattr(module.os, "getsid", sessions.__getitem__)
    ready = SameTaskWorkerReadyIdentity(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        session_id="session-1",
        task_id="lift_can",
        process_id=4322,
        campaign_manifest_sha256="a" * 64,
        source_binding_sha256="b" * 64,
        integration_config_sha256="c" * 64,
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
    )
    ready_path.write_bytes(canonical_json_bytes(ready.to_document()))

    loaded = module._wait_for_worker_ready(
        process=process,
        socket_path=socket_path,
        ready_receipt=ready_path,
        expected_task_id="lift_can",
        timeout_s=0.5,
    )
    assert loaded == ready

    wrong_group = SameTaskWorkerReadyIdentity(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        session_id="session-2",
        task_id="lift_can",
        process_id=4323,
        campaign_manifest_sha256="a" * 64,
        source_binding_sha256="b" * 64,
        integration_config_sha256="c" * 64,
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
    )
    ready_path.write_bytes(canonical_json_bytes(wrong_group.to_document()))
    with pytest.raises(ValueError, match="process-group mismatch"):
        module._wait_for_worker_ready(
            process=process,
            socket_path=socket_path,
            ready_receipt=ready_path,
            expected_task_id="lift_can",
            timeout_s=0.5,
        )

    wrong_session = SameTaskWorkerReadyIdentity(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        session_id="session-3",
        task_id="lift_can",
        process_id=4324,
        campaign_manifest_sha256="a" * 64,
        source_binding_sha256="b" * 64,
        integration_config_sha256="c" * 64,
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
    )
    ready_path.write_bytes(canonical_json_bytes(wrong_session.to_document()))
    with pytest.raises(ValueError, match="ready session mismatch"):
        module._wait_for_worker_ready(
            process=process,
            socket_path=socket_path,
            ready_receipt=ready_path,
            expected_task_id="lift_can",
            timeout_s=0.5,
        )

    sessions[4321] = 9998
    ready_path.write_bytes(canonical_json_bytes(ready.to_document()))
    with pytest.raises(ValueError, match="launcher session mismatch"):
        module._wait_for_worker_ready(
            process=process,
            socket_path=socket_path,
            ready_receipt=ready_path,
            expected_task_id="lift_can",
            timeout_s=0.5,
        )
    sessions[4321] = 4321

    ready_path.write_bytes(canonical_json_bytes(ready.to_document()))

    with pytest.raises(ValueError, match="task mismatch"):
        module._wait_for_worker_ready(
            process=process,
            socket_path=socket_path,
            ready_receipt=ready_path,
            expected_task_id="insert_hole",
            timeout_s=0.5,
        )


def test_verified_worker_session_distinguishes_python_pid_from_group_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    reset = root / "artifacts/deployment/reset.json"
    reset.parent.mkdir(parents=True)
    reset.write_text("{}", encoding="utf-8")
    receipt = SimpleNamespace(
        task_id="insert_tube",
        campaign_id="campaign-1",
        worker_session_id="worker-session-1",
        worker_pid=4322,
        worker_process_group_id=4321,
        worker_posix_session_id=4321,
        restart_generation=0,
        episode_dispatches=(object(), object()),
        reset_equivalence_receipt_relpath=(reset.relative_to(root).as_posix()),
        reset_equivalence_receipt_sha256="a" * 64,
    )
    monkeypatch.setattr(
        module,
        "load_persistent_worker_session_receipt",
        lambda _path: receipt,
    )

    loaded = module._verified_worker_session(
        receipt_path=root / "session.json",
        task_id="insert_tube",
        campaign_id="campaign-1",
        worker_session_id="worker-session-1",
        worker_pid=4322,
        worker_process_group_id=4321,
        worker_posix_session_id=4321,
        expected_dispatch_count=2,
        reset_proof=(reset, "a" * 64),
        deployment_root=root,
    )

    assert loaded is receipt

    mismatches = {
        "worker_session_id": "worker-session-2",
        "worker_pid": 4999,
        "worker_process_group_id": 4998,
        "worker_posix_session_id": 4997,
    }
    for field, value in mismatches.items():
        invalid = SimpleNamespace(**vars(receipt))
        setattr(invalid, field, value)
        monkeypatch.setattr(
            module,
            "load_persistent_worker_session_receipt",
            lambda _path, selected=invalid: selected,
        )
        with pytest.raises(ValueError, match="identity mismatch"):
            module._verified_worker_session(
                receipt_path=root / "session.json",
                task_id="insert_tube",
                campaign_id="campaign-1",
                worker_session_id="worker-session-1",
                worker_pid=4322,
                worker_process_group_id=4321,
                worker_posix_session_id=4321,
                expected_dispatch_count=2,
                reset_proof=(reset, "a" * 64),
                deployment_root=root,
            )


def test_persistent_worker_shutdown_precedes_n0_shutdown() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    finally_block = source[source.index("        finally:") :]
    assert finally_block.index("if worker is not None") < finally_block.index(
        "if server is not None"
    )
