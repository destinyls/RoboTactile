"""Same-task worker dispatch coverage for the Clean campaign supervisor."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robotactile_benchmark.clean_baseline.campaign_command import (
    build_live_command,
    require_worker_socket,
)
from robotactile_benchmark.clean_baseline.contracts import CleanCampaignTrialSpec
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleIdentity
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
)

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/live_univtac/run_clean_campaign.py"


def _module() -> ModuleType:
    name = "clean_campaign_worker_dispatch"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _entry() -> CleanCampaignTrialSpec:
    digest = "a" * 64
    return CleanCampaignTrialSpec(
        ordinal=0,
        task="lift_bottle",
        initial_seed=17,
        exogenous_seed=29,
        base_system_id="official-n0-test",
        dataset_sha256=digest,
        base_system_manifest_sha256=digest,
        checkpoint_sha256=digest,
        config_sha256=digest,
        trial_manifest_sha256=digest,
        pair_key=digest,
        run_spec_sha256=digest,
        run_content_sha256=digest,
        request_file_sha256=digest,
        request_relpath="requests/clean/0000/request.json",
        artifact_relpath="artifacts/live/0000",
        max_control_cycles=1,
        max_observation_steps=2,
        execute_action_steps=24,
    )


def test_fresh_process_command_is_byte_for_byte_compatible(tmp_path: Path) -> None:
    command = build_live_command(
        isaac_python=tmp_path / "python.sh",
        deployment_root=tmp_path,
        request_path=tmp_path / "request.json",
        n0_source_root=tmp_path / "N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29601,
        integration_config=None,
    )

    assert command == (
        str(tmp_path / "python.sh"),
        "-m",
        "robotactile_benchmark.cli",
        "live-univtac-run",
        "--root",
        str(tmp_path),
        "--request",
        str(tmp_path / "request.json"),
        "--n0-source-root",
        str(tmp_path / "N0-TWAM"),
        "--n0-host",
        "127.0.0.1",
        "--n0-port",
        "29601",
        "--capture-profile",
        "paper_full_v1",
    )


def test_worker_socket_must_be_real_non_symlink_unix_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    socket_path = root / "worker.sock"
    socket_path.touch()
    original_lstat = Path.lstat

    def fake_lstat(path: Path) -> object:
        if path == socket_path:
            return SimpleNamespace(st_mode=stat.S_IFSOCK | 0o600)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    assert require_worker_socket(root, socket_path) == socket_path.resolve()

    symlink = root / "worker-link.sock"
    symlink.symlink_to(socket_path.name)
    with pytest.raises(ValueError, match="non-symlink Unix socket"):
        require_worker_socket(root, symlink)

    regular = root / "regular.file"
    regular.write_text("not a socket", encoding="utf-8")
    with pytest.raises(ValueError, match="non-symlink Unix socket"):
        require_worker_socket(root, regular)
    with pytest.raises(ValueError, match="below deployment root"):
        require_worker_socket(root, root.parent / "missing.sock")


def test_worker_command_uses_strict_identity_and_lightweight_client(
    tmp_path: Path,
) -> None:
    module = _module()
    entry = _entry()
    campaign_digest = "b" * 64
    identity = LifecycleIdentity(
        campaign_manifest_sha256=campaign_digest,
        request_file_sha256=entry.request_file_sha256,
        trial_manifest_sha256=entry.trial_manifest_sha256,
        task_id=entry.task,
        ordinal=entry.ordinal,
        attempt_id="attempt-1",
    )
    worker_request = module._same_task_worker_request(entry, identity)
    context = SimpleNamespace(
        campaign_manifest_sha256=campaign_digest,
        task_id=entry.task,
    )
    socket_path = tmp_path / "worker.sock"
    request_path = tmp_path / "request.json"
    journal_path = tmp_path / "attempt.journal"

    command = build_live_command(
        isaac_python=tmp_path / "unused-isaac-python",
        deployment_root=tmp_path,
        request_path=request_path,
        n0_source_root=tmp_path / "unused-N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29601,
        integration_config=None,
        lifecycle_journal=journal_path,
        source_bound_context=context,
        worker_socket=socket_path,
        worker_request=worker_request,
    )

    assert worker_request.worker_contract == SAME_TASK_WORKER_CONTRACT
    assert worker_request.semantic_version == SAME_TASK_WORKER_SEMANTIC_VERSION
    assert command == (
        sys.executable,
        "-m",
        "robotactile_benchmark.execution.same_task_worker_client",
        "--socket",
        str(socket_path),
        "--request",
        str(request_path),
        "--lifecycle-journal",
        str(journal_path),
        "--request-id",
        worker_request.request_id,
        "--campaign-manifest-sha256",
        campaign_digest,
        "--task",
        entry.task,
        "--ordinal",
        "0",
        "--attempt-id",
        "attempt-1",
        "--request-file-sha256",
        entry.request_file_sha256,
        "--trial-manifest-sha256",
        entry.trial_manifest_sha256,
    )
    assert "live-univtac-run" not in command
    assert "--n0-source-root" not in command

    with pytest.raises(ValueError, match="source-bound context"):
        build_live_command(
            isaac_python=tmp_path / "python",
            deployment_root=tmp_path,
            request_path=request_path,
            n0_source_root=tmp_path / "N0-TWAM",
            n0_host="127.0.0.1",
            n0_port=29601,
            integration_config=None,
            lifecycle_journal=journal_path,
            worker_socket=socket_path,
            worker_request=worker_request,
        )
    with pytest.raises(ValueError, match="do not support same_task_worker"):
        build_live_command(
            isaac_python=tmp_path / "python",
            deployment_root=tmp_path,
            request_path=request_path,
            n0_source_root=tmp_path / "N0-TWAM",
            n0_host="127.0.0.1",
            n0_port=29601,
            integration_config=None,
            lifecycle_journal=journal_path,
            source_bound_context=context,
            worker_socket=socket_path,
            worker_request=worker_request,
            capture_profile=LiveCaptureProfile.PREVIEW,
        )


def test_worker_execute_entry_preserves_attempt_command_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    layout = DeploymentLayout(root)
    entry = _entry()
    request_path = root / entry.request_relpath
    request_path.parent.mkdir(parents=True)
    request_path.write_bytes(canonical_json_bytes({}))
    campaign_digest = "b" * 64
    worker_socket = root / "worker.sock"
    captured: dict[str, tuple[str, ...]] = {}

    def fake_popen(command: tuple[str, ...], **_kwargs: object) -> object:
        captured["command"] = command
        return SimpleNamespace(pid=9876)

    artifact = {
        "artifact_root_sha256": "c" * 64,
        "execution_status": "completed",
        "score_eligible": True,
        "score_success": False,
        "terminal_status": "task_failure",
    }
    monkeypatch.setattr(
        module,
        "load_live_univtac_request",
        lambda _path: SimpleNamespace(wall_timeout_s=1.0),
    )
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        module,
        "wait_with_lifecycle_watchdog",
        lambda *_args, **_kwargs: module.HardLifecycleOutcome(0, False, False, False),
    )
    monkeypatch.setattr(
        module,
        "_inspect_artifact",
        lambda *_args: module._ArtifactInspection(artifact, None, None),
    )
    context = SimpleNamespace(
        deployment_root=root,
        campaign_id="campaign",
        campaign_manifest_sha256=campaign_digest,
        task_id=entry.task,
    )

    disposition, receipt = module._execute_entry(
        layout=layout,
        campaign_id="campaign",
        campaign_manifest_sha256=campaign_digest,
        entry=entry,
        request_path=request_path,
        artifact_path=root / entry.artifact_relpath,
        isaac_python=root / "unused-python",
        n0_source_root=root / "unused-N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29601,
        integration_config=None,
        hard_lifecycle_timeout_s=2.0,
        watchdog_term_grace_s=0.1,
        semantic_version="2.0",
        source_bound_context=context,
        worker_socket=worker_socket,
    )

    assert disposition == module._VALID_OUTCOME
    assert receipt["command_kind"] == "robotactile_live_univtac_run_v1"
    command = captured["command"]
    assert command[:3] == (
        sys.executable,
        "-m",
        "robotactile_benchmark.execution.same_task_worker_client",
    )
    assert command[command.index("--request-file-sha256") + 1] == (
        entry.request_file_sha256
    )
    assert command[command.index("--trial-manifest-sha256") + 1] == (
        entry.trial_manifest_sha256
    )
