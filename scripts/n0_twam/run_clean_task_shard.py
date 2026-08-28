#!/usr/bin/env python3
"""Own one N0 server and execute a resumable Clean campaign task shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_EE_ACTION_EXECUTION_CONTRACTS,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    validate_n0_ee_action_execution_contract,
)
from robotactile_benchmark.clean_baseline.contracts import CleanCampaignProtocol
from robotactile_benchmark.clean_baseline.io import load_clean_campaign_manifest
from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_V3_SEMANTIC_VERSION,
    verify_all_task_qualification,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.persistent_worker_receipt import (
    PersistentWorkerSessionReceipt,
    load_persistent_worker_session_receipt,
)
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SameTaskWorkerReadyIdentity,
)
from robotactile_benchmark.runtime_attestation import (
    load_n0_server_runtime_attestation,
    n0_server_attestation_sha256,
    verify_live_n0_server_attestation,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_N0_ISOLATED_ENVIRONMENT_KEYS = frozenset(
    {
        "LD_LIBRARY_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
    }
)
_RUNNER_SUMMARY_FIELDS = (
    "candidate_pool_exhausted",
    "action_execution_contract",
    "capture_profile",
    "completed_for_task",
    "fatal_error",
    "new_trials_attempted",
    "recorded_failures_for_task",
    "remaining_unattempted_candidates",
    "replacement_requires_fresh_server",
    "stopped_for_budget",
    "target_complete",
    "target_valid_trials_for_task",
)
_FRESH_PROCESS_WORKER_CONTRACT = "fresh_process_v1"
_WORKER_CONTRACTS = (_FRESH_PROCESS_WORKER_CONTRACT, SAME_TASK_WORKER_CONTRACT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--n0-port", type=int, default=29601)
    parser.add_argument("--master-port", type=int, default=29988)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--integration-config", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--qualification-sha256")
    parser.add_argument("--max-new-trials", type=int)
    parser.add_argument("--continue-on-infrastructure-failure", action="store_true")
    parser.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        default=LiveCaptureProfile.PAPER_FULL.value,
    )
    parser.add_argument(
        "--worker-contract",
        choices=_WORKER_CONTRACTS,
        default=_FRESH_PROCESS_WORKER_CONTRACT,
    )
    parser.add_argument("--reset-equivalence-receipt", type=Path)
    parser.add_argument("--reset-equivalence-receipt-sha256")
    parser.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        default=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, document: object) -> None:
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("shard receipt cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("shard receipt already exists with different bytes")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _port_is_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _n0_subprocess_environment() -> dict[str, str]:
    """Remove parent-runtime paths before entering the isolated N0 runtime."""

    return {
        name: value
        for name, value in os.environ.items()
        if name not in _N0_ISOLATED_ENVIRONMENT_KEYS
    }


def _require_runtime_launcher(path: Path, runtime_root: Path, label: str) -> Path:
    """Allow an internal venv symlink while rejecting runtime path escapes."""

    try:
        resolved_root = runtime_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(f"{label} launcher escapes its runtime root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} launcher is unavailable")
    return path.absolute()


def _optional_integration_config(root: Path, requested: Path | None) -> Path | None:
    if requested is None:
        return None
    root_resolved = root.resolve(strict=True)
    selected = requested.absolute()
    try:
        resolved = selected.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(
            "integration config must remain below the deployment root"
        ) from error
    if selected.is_symlink() or not resolved.is_file():
        raise ValueError("integration config must be a regular non-symlink file")
    return selected


def _optional_source_bound_qualification(
    root: Path,
    requested: Path | None,
    task_id: str,
    expected_sha256: str | None = None,
    action_execution_contract: str = N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
) -> tuple[Path, str] | None:
    if requested is None:
        if expected_sha256 is not None:
            raise ValueError("qualification SHA256 requires --qualification")
        return None
    selected = requested.absolute()
    try:
        selected.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(
            "qualification must remain below the deployment root"
        ) from error
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("qualification must be a regular non-symlink file")
    qualification = verify_all_task_qualification(root, selected)
    if expected_sha256 is not None and qualification.sha256 != expected_sha256:
        raise ValueError("qualification SHA256 mismatch")
    if qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION:
        raise ValueError("source-bound task shard requires qualification v3")
    try:
        task_index = qualification.tasks.index(task_id)
    except ValueError as error:
        raise ValueError("qualification does not cover the task shard") from error
    source = qualification.task_source_bindings[task_index]
    source.verify_against_current_runtime()
    expected_native = (
        N0_STOCK_EE_NATIVE_STEP_CONTRACT
        if action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT
        else FIXED_NATIVE_STEP_CONTRACT
    )
    if (
        source.action_execution_contract != action_execution_contract
        or source.native_step_contract != expected_native
    ):
        raise ValueError("qualification action execution contract mismatch")
    return selected, qualification.sha256


def _campaign_runner_command(
    *,
    repository_root: Path,
    deployment_root: Path,
    manifest_path: Path,
    task: str,
    isaac_python: Path,
    n0_source_root: Path,
    n0_port: int,
    integration_config: Path | None,
    action_execution_contract: str = N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    qualification_path: Path | None = None,
    attestation_path: Path | None = None,
    attestation_sha256: str | None = None,
    worker_socket: Path | None = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> list[str]:
    command = [
        sys.executable,
        str(repository_root / "scripts/live_univtac/run_clean_campaign.py"),
        "--root",
        str(deployment_root),
        "--manifest",
        str(manifest_path),
        "--task",
        task,
        "--isaac-python",
        str(isaac_python),
        "--n0-source-root",
        str(n0_source_root),
        "--n0-port",
        str(n0_port),
        "--capture-profile",
        capture_profile.value,
    ]
    if integration_config is not None:
        command.extend(("--config", str(integration_config)))
    command.extend(("--action-execution-contract", action_execution_contract))
    source_bound_values = (
        qualification_path,
        attestation_path,
        attestation_sha256,
    )
    if any(value is not None for value in source_bound_values):
        if not all(value is not None for value in source_bound_values):
            raise ValueError("source-bound campaign runner arguments are incomplete")
        command.extend(
            (
                "--qualification",
                str(qualification_path),
                "--n0-server-attestation",
                str(attestation_path),
                "--n0-server-attestation-sha256",
                cast(str, attestation_sha256),
            )
        )
    if worker_socket is not None:
        command.extend(("--worker-socket", str(worker_socket)))
    return command


def _reset_equivalence_proof(
    root: Path,
    *,
    requested: Path | None,
    expected_sha256: str | None,
    required: bool,
) -> tuple[Path, str] | None:
    if requested is None:
        if expected_sha256 is not None:
            raise ValueError(
                "reset equivalence SHA256 requires --reset-equivalence-receipt"
            )
        if required:
            raise ValueError(
                "pilot/paper persistent worker requires reset-equivalence proof"
            )
        return None
    selected = requested.absolute()
    try:
        selected.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(
            "reset-equivalence receipt must remain below the deployment root"
        ) from error
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("reset-equivalence receipt must be a regular non-symlink file")
    digest = _sha256_file(selected)
    if expected_sha256 is None or digest != expected_sha256:
        raise ValueError("reset-equivalence receipt SHA256 mismatch")
    return selected, digest


def _worker_paths(
    layout: DeploymentLayout, *, task_id: str, run_id: str
) -> tuple[Path, Path, Path]:
    token = hashlib.sha256(f"{task_id}:{run_id}".encode("utf-8")).hexdigest()[:16]
    socket_path = layout.runtime / "ipc" / f"stw-{token}.sock"
    receipt_root = layout.outputs / "persistent-workers" / task_id
    return (
        socket_path,
        receipt_root / f"{run_id}.ready.json",
        receipt_root / f"{run_id}.session.json",
    )


def _wait_for_worker_ready(
    *,
    process: subprocess.Popen[bytes],
    socket_path: Path,
    ready_receipt: Path,
    expected_task_id: str,
    timeout_s: float,
    expected_campaign_manifest_sha256: str | None = None,
    expected_source_binding_sha256: str | None = None,
    expected_integration_config_sha256: str | None = None,
    expected_action_execution_contract: str | None = None,
) -> SameTaskWorkerReadyIdentity:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"same-task worker exited before readiness: {return_code}"
            )
        if socket_path.exists() and ready_receipt.is_file():
            if socket_path.is_symlink() or ready_receipt.is_symlink():
                raise ValueError("same-task worker readiness paths cannot be symlinks")
            if not stat.S_ISSOCK(socket_path.stat().st_mode):
                raise ValueError("same-task worker did not publish a Unix socket")
            document = json.loads(ready_receipt.read_text(encoding="utf-8"))
            if not isinstance(document, Mapping):
                raise TypeError("same-task worker ready receipt must be an object")
            identity = SameTaskWorkerReadyIdentity.from_document(document)
            if identity.task_id != expected_task_id:
                raise ValueError("same-task worker ready task mismatch")
            try:
                launcher_process_group_id = os.getpgid(process.pid)
                launcher_session_id = os.getsid(process.pid)
                ready_process_group_id = os.getpgid(identity.process_id)
                ready_session_id = os.getsid(identity.process_id)
            except ProcessLookupError as error:
                raise RuntimeError(
                    "same-task worker disappeared during readiness verification"
                ) from error
            if launcher_process_group_id != process.pid:
                raise ValueError("same-task worker launcher process-group mismatch")
            if launcher_session_id != process.pid:
                raise ValueError("same-task worker launcher session mismatch")
            if ready_process_group_id != launcher_process_group_id:
                raise ValueError("same-task worker ready process-group mismatch")
            if ready_session_id != launcher_session_id:
                raise ValueError("same-task worker ready session mismatch")
            expected_values = (
                (
                    identity.campaign_manifest_sha256,
                    expected_campaign_manifest_sha256,
                ),
                (identity.source_binding_sha256, expected_source_binding_sha256),
                (
                    identity.integration_config_sha256,
                    expected_integration_config_sha256,
                ),
                (
                    identity.action_execution_contract,
                    expected_action_execution_contract,
                ),
            )
            if any(
                expected is not None and actual != expected
                for actual, expected in expected_values
            ):
                raise ValueError("same-task worker ready source identity mismatch")
            return identity
        time.sleep(0.2)
    raise TimeoutError("same-task worker readiness timed out")


def _worker_command(
    *,
    deployment_root: Path,
    manifest_path: Path,
    task_id: str,
    socket_path: Path,
    ready_receipt: Path,
    session_receipt: Path,
    integration_config: Path | None,
    n0_source_root: Path,
    n0_port: int,
    action_execution_contract: str,
    restart_generation: int,
    qualification_path: Path | None,
    attestation_path: Path | None,
    attestation_sha256: str | None,
    reset_proof: tuple[Path, str] | None,
) -> list[str]:
    command = [
        "-m",
        "robotactile_benchmark.execution.same_task_worker_server",
        "--root",
        str(deployment_root),
        "--campaign-manifest",
        str(manifest_path),
        "--task",
        task_id,
        "--socket",
        str(socket_path),
        "--ready-receipt",
        str(ready_receipt),
        "--session-receipt",
        str(session_receipt),
        "--n0-source-root",
        str(n0_source_root),
        "--n0-host",
        "127.0.0.1",
        "--n0-port",
        str(n0_port),
        "--action-execution-contract",
        action_execution_contract,
        "--restart-generation",
        str(restart_generation),
    ]
    if integration_config is not None:
        command.extend(("--config", str(integration_config)))
    source_values = (qualification_path, attestation_path, attestation_sha256)
    if any(value is not None for value in source_values):
        if not all(value is not None for value in source_values):
            raise ValueError("same-task worker source arguments are incomplete")
        command.extend(
            (
                "--qualification",
                str(qualification_path),
                "--n0-server-attestation",
                str(attestation_path),
                "--n0-server-attestation-sha256",
                cast(str, attestation_sha256),
            )
        )
    if reset_proof is not None:
        reset_path, reset_sha256 = reset_proof
        command.extend(
            (
                "--reset-equivalence-receipt",
                str(reset_path),
                "--reset-equivalence-receipt-sha256",
                reset_sha256,
            )
        )
    return command


def _verified_worker_session(
    *,
    receipt_path: Path,
    task_id: str,
    campaign_id: str,
    worker_session_id: str,
    worker_pid: int,
    worker_process_group_id: int,
    worker_posix_session_id: int,
    expected_dispatch_count: int,
    reset_proof: tuple[Path, str],
    deployment_root: Path,
) -> PersistentWorkerSessionReceipt:
    """Verify formal worker evidence after its process group has exited."""

    if expected_dispatch_count < 1:
        raise ValueError("formal worker session requires at least one dispatch")
    receipt = load_persistent_worker_session_receipt(receipt_path)
    reset_path, reset_sha256 = reset_proof
    if (
        receipt.task_id != task_id
        or receipt.campaign_id != campaign_id
        or receipt.worker_session_id != worker_session_id
        or receipt.worker_pid != worker_pid
        or receipt.worker_process_group_id != worker_process_group_id
        or receipt.worker_posix_session_id != worker_posix_session_id
        or receipt.restart_generation != 0
        or len(receipt.episode_dispatches) != expected_dispatch_count
        or receipt.reset_equivalence_receipt_relpath
        != reset_path.relative_to(deployment_root).as_posix()
        or receipt.reset_equivalence_receipt_sha256 != reset_sha256
    ):
        raise ValueError("persistent worker session receipt identity mismatch")
    return receipt


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.1)
    return not _process_group_exists(process_group_id)


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> str:
    leader_exited = process.poll() is not None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    if not leader_exited:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30.0)
    if _wait_for_process_group_exit(process.pid, 30.0):
        return "sigterm_group_after_leader_exit" if leader_exited else "sigterm"
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "sigterm"
    if process.poll() is None:
        process.wait(timeout=10.0)
    if not _wait_for_process_group_exit(process.pid, 10.0):
        raise RuntimeError("owned N0 process group survived SIGKILL")
    return "sigkill_after_timeout"


def _parse_runner_summary(payload: bytes) -> dict[str, object]:
    output = payload.decode("utf-8", errors="replace").strip()
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"campaign runner did not return one JSON object: {output[-2000:]}"
        ) from error
    if not isinstance(document, Mapping):
        raise TypeError("campaign runner JSON must be an object")
    return dict(document)


def _runner_summary_subset(summary: Mapping[str, object] | None) -> dict[str, object]:
    if summary is None:
        return {name: None for name in _RUNNER_SUMMARY_FIELDS}
    return {name: summary.get(name) for name in _RUNNER_SUMMARY_FIELDS}


def _wait_for_server(
    *,
    process: subprocess.Popen[bytes],
    check_command: tuple[str, ...],
    environment: dict[str, str],
    timeout_s: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_error = "server probe has not run"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"N0 server exited before readiness: {return_code}")
        try:
            completed = subprocess.run(
                check_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=30.0,
            )
        except subprocess.TimeoutExpired:
            last_error = "server metadata probe timed out"
        else:
            output = completed.stdout.decode("utf-8", errors="replace").strip()
            if completed.returncode == 0:
                document = json.loads(output)
                if not isinstance(document, dict):
                    raise TypeError("server readiness payload must be an object")
                return document
            last_error = output[-1000:]
        time.sleep(5.0)
    raise TimeoutError(f"N0 server readiness timed out: {last_error}")


def run_task_shard(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    if _IDENTIFIER.fullmatch(args.task) is None:
        raise ValueError("task contains unsupported characters")
    if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpus) is None:
        raise ValueError("gpus must be a comma-separated integer list")
    if not 1 <= args.n0_port <= 65535 or not 1 <= args.master_port <= 65535:
        raise ValueError("ports must be in [1,65535]")
    if args.server_ready_timeout_s <= 0:
        raise ValueError("server_ready_timeout_s must be positive")
    if args.max_new_trials is not None and args.max_new_trials < 0:
        raise ValueError("max_new_trials must be non-negative")
    action_execution_contract = validate_n0_ee_action_execution_contract(
        args.action_execution_contract
    )
    capture_profile = LiveCaptureProfile(args.capture_profile)
    manifest_path = args.manifest.absolute()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("campaign manifest must be a regular non-symlink file")
    manifest_file_sha256 = _sha256_file(manifest_path)
    manifest = load_clean_campaign_manifest(manifest_path)
    if _port_is_open("127.0.0.1", args.n0_port):
        raise RuntimeError("N0 port is already occupied; refusing an ambiguous server")

    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    persistent_worker = args.worker_contract == SAME_TASK_WORKER_CONTRACT
    if persistent_worker and capture_profile is not LiveCaptureProfile.PAPER_FULL:
        raise ValueError("light capture profiles do not support same_task_worker_v1")
    formal_protocol = manifest.protocol_id in {
        CleanCampaignProtocol.PILOT,
        CleanCampaignProtocol.PAPER,
    }
    if formal_protocol and capture_profile is not LiveCaptureProfile.PAPER_FULL:
        raise ValueError("pilot/paper campaigns require paper_full_v1 capture")
    reset_proof = _reset_equivalence_proof(
        layout.root,
        requested=args.reset_equivalence_receipt,
        expected_sha256=args.reset_equivalence_receipt_sha256,
        required=persistent_worker and formal_protocol,
    )
    if not persistent_worker and reset_proof is not None:
        raise ValueError("reset-equivalence proof requires same_task_worker_v1")
    qualification_info = _optional_source_bound_qualification(
        layout.root,
        getattr(args, "qualification", None),
        args.task,
        getattr(args, "qualification_sha256", None),
        action_execution_contract,
    )
    if persistent_worker and qualification_info is None:
        raise ValueError("same-task worker requires source-bound qualification v3")
    requested_config = args.integration_config
    if qualification_info is not None and requested_config is None:
        requested_config = (
            layout.model_artifacts
            / "n0_twam/configs"
            / args.task
            / "integration_config.json"
        )
    integration_config = _optional_integration_config(layout.root, requested_config)
    if qualification_info is not None and integration_config is None:
        raise ValueError("source-bound task shard requires an integration config")
    repository_root = Path(__file__).resolve().parents[2]
    n0_python = layout.runtime / "n0-twam/bin/python"
    n0_source_root = layout.sources / "N0-TWAM"
    n0_python = _require_runtime_launcher(
        n0_python,
        layout.runtime / "n0-twam",
        "N0 Python",
    )
    isaac_python = _require_runtime_launcher(
        args.isaac_python or layout.runtime / "isaac-sim-4.5.0/python.sh",
        layout.runtime / "isaac-sim-4.5.0",
        "Isaac Python",
    )
    if n0_source_root.is_symlink() or not n0_source_root.is_dir():
        raise ValueError("N0 source root is unavailable")

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = layout.logs / "clean-campaign-shards" / args.task / f"{run_id}.log"
    receipt_path = (
        layout.outputs / "clean-campaign-shards" / args.task / f"{run_id}.json"
    )
    attestation_path = (
        layout.outputs
        / "n0-twam"
        / args.task
        / "runtime-attestations"
        / f"{run_id}.json"
    )
    if qualification_info is not None and (
        attestation_path.exists() or attestation_path.is_symlink()
    ):
        raise FileExistsError("N0 server attestation output already exists")
    worker_socket, worker_ready_receipt, worker_session_receipt = _worker_paths(
        layout, task_id=args.task, run_id=run_id
    )
    if persistent_worker:
        for path in (worker_socket, worker_ready_receipt, worker_session_receipt):
            if path.exists() or path.is_symlink():
                raise FileExistsError("same-task worker output path already exists")
        worker_socket.parent.mkdir(parents=True, exist_ok=True)
        worker_ready_receipt.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server_command_list = [
        "bash",
        str(repository_root / "scripts/n0_twam/serve_univtac.sh"),
        "--root",
        str(layout.root),
        "--task",
        args.task,
        "--gpus",
        args.gpus,
        "--port",
        str(args.n0_port),
        "--master-port",
        str(args.master_port),
    ]
    if qualification_info is not None:
        qualification_path, _ = qualification_info
        server_command_list.extend(
            (
                "--session-id",
                run_id,
                "--attestation",
                str(attestation_path),
                "--qualification",
                str(qualification_path),
                "--integration-config",
                str(integration_config),
            )
        )
    server_command = tuple(server_command_list)
    runner_command = _campaign_runner_command(
        repository_root=repository_root,
        deployment_root=layout.root,
        manifest_path=manifest_path,
        task=args.task,
        isaac_python=isaac_python,
        n0_source_root=n0_source_root,
        n0_port=args.n0_port,
        integration_config=integration_config,
        action_execution_contract=action_execution_contract,
        capture_profile=capture_profile,
    )
    if args.max_new_trials is not None:
        runner_command.extend(("--max-new-trials", str(args.max_new_trials)))
    if args.continue_on_infrastructure_failure:
        runner_command.append("--continue-on-infrastructure-failure")

    server: subprocess.Popen[bytes] | None = None
    worker: subprocess.Popen[bytes] | None = None
    readiness: dict[str, object] | None = None
    worker_readiness: dict[str, object] | None = None
    runner_return_code: int | None = None
    runner_summary: dict[str, object] | None = None
    error_type: str | None = None
    error_message: str | None = None
    shutdown = "not_started"
    attestation_sha256: str | None = None
    server_process_group_id: int | None = None
    worker_session_id: str | None = None
    worker_process_id: int | None = None
    worker_process_group_id: int | None = None
    worker_posix_session_id: int | None = None
    worker_shutdown = "not_started"
    worker_source_binding_sha256: str | None = None
    n0_environment = _n0_subprocess_environment()
    with log_path.open("xb") as log_stream:
        try:
            server = subprocess.Popen(
                server_command,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=n0_environment,
            )
            server_process_group_id = os.getpgid(server.pid)
            if server_process_group_id != server.pid:
                raise RuntimeError("owned N0 server did not become its group leader")
            check_command = (
                str(n0_python),
                str(repository_root / "scripts/n0_twam/check_server.py"),
                "--source-root",
                str(n0_source_root),
                "--port",
                str(args.n0_port),
            )
            readiness = _wait_for_server(
                process=server,
                check_command=check_command,
                environment=n0_environment,
                timeout_s=args.server_ready_timeout_s,
            )
            if qualification_info is not None:
                qualification_path, _ = qualification_info
                attestation_sha256 = n0_server_attestation_sha256(attestation_path)
                attestation = load_n0_server_runtime_attestation(
                    layout.root,
                    attestation_path,
                    expected_sha256=attestation_sha256,
                )
                verify_live_n0_server_attestation(
                    attestation,
                    expected_task_id=args.task,
                    expected_session_id=run_id,
                    expected_process_group_id=server.pid,
                )
                worker_source_binding_sha256 = canonical_hash(
                    attestation.task_source_binding.to_dict()
                )
                runner_command = _campaign_runner_command(
                    repository_root=repository_root,
                    deployment_root=layout.root,
                    manifest_path=manifest_path,
                    task=args.task,
                    isaac_python=isaac_python,
                    n0_source_root=n0_source_root,
                    n0_port=args.n0_port,
                    integration_config=integration_config,
                    action_execution_contract=action_execution_contract,
                    qualification_path=qualification_path,
                    attestation_path=attestation_path,
                    attestation_sha256=attestation_sha256,
                    capture_profile=capture_profile,
                )
                if args.max_new_trials is not None:
                    runner_command.extend(
                        ("--max-new-trials", str(args.max_new_trials))
                    )
                if args.continue_on_infrastructure_failure:
                    runner_command.append("--continue-on-infrastructure-failure")
            if persistent_worker:
                worker_qualification_path = (
                    None if qualification_info is None else qualification_info[0]
                )
                worker_command = _worker_command(
                    deployment_root=layout.root,
                    manifest_path=manifest_path,
                    task_id=args.task,
                    socket_path=worker_socket,
                    ready_receipt=worker_ready_receipt,
                    session_receipt=worker_session_receipt,
                    integration_config=integration_config,
                    n0_source_root=n0_source_root,
                    n0_port=args.n0_port,
                    action_execution_contract=action_execution_contract,
                    restart_generation=0,
                    qualification_path=worker_qualification_path,
                    attestation_path=(
                        attestation_path if qualification_info is not None else None
                    ),
                    attestation_sha256=attestation_sha256,
                    reset_proof=reset_proof,
                )
                worker = subprocess.Popen(
                    [str(isaac_python), *worker_command],
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=dict(os.environ),
                )
                worker_process_group_id = os.getpgid(worker.pid)
                worker_posix_session_id = os.getsid(worker.pid)
                if worker_process_group_id != worker.pid:
                    raise RuntimeError(
                        "owned same-task worker did not become its group leader"
                    )
                if worker_posix_session_id != worker.pid:
                    raise RuntimeError(
                        "owned same-task worker did not become its session leader"
                    )
                ready = _wait_for_worker_ready(
                    process=worker,
                    socket_path=worker_socket,
                    ready_receipt=worker_ready_receipt,
                    expected_task_id=args.task,
                    timeout_s=args.server_ready_timeout_s,
                    expected_campaign_manifest_sha256=manifest.sha256,
                    expected_source_binding_sha256=(worker_source_binding_sha256),
                    expected_integration_config_sha256=_sha256_file(
                        cast(Path, integration_config)
                    ),
                    expected_action_execution_contract=(action_execution_contract),
                )
                worker_session_id = ready.session_id
                worker_process_id = ready.process_id
                worker_readiness = ready.to_document()
                runner_command.extend(("--worker-socket", str(worker_socket)))
            completed = subprocess.run(
                runner_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            runner_return_code = completed.returncode
            log_stream.write(b"\nROBOTACTILE_CAMPAIGN_RUNNER_OUTPUT\n")
            log_stream.write(completed.stdout)
            if completed.stdout and not completed.stdout.endswith(b"\n"):
                log_stream.write(b"\n")
            log_stream.flush()
            runner_summary = _parse_runner_summary(completed.stdout)
            if runner_summary.get("capture_profile") != capture_profile.value:
                raise ValueError("campaign runner capture profile mismatch")
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
            error_type = type(error).__name__
            error_message = str(error)
            log_stream.write(
                f"\nROBOTACTILE_SHARD_ERROR {error_type}: {error_message}\n".encode(
                    "utf-8", errors="replace"
                )
            )
            log_stream.flush()
        finally:
            if worker is not None:
                try:
                    worker_shutdown = _terminate_owned_process(worker)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    worker_shutdown = "shutdown_failed"
                    if error_type is None:
                        error_type = type(error).__name__
                        error_message = str(error)
            if server is not None:
                try:
                    shutdown = _terminate_owned_process(server)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    shutdown = "shutdown_failed"
                    if error_type is None:
                        error_type = type(error).__name__
                        error_message = str(error)

    new_worker_dispatches = (
        None if runner_summary is None else runner_summary.get("new_trials_attempted")
    )
    if (
        persistent_worker
        and formal_protocol
        and type(new_worker_dispatches) is int
        and new_worker_dispatches > 0
    ):
        try:
            if (
                worker_session_id is None
                or worker_process_id is None
                or worker_process_group_id is None
                or worker_posix_session_id is None
                or reset_proof is None
            ):
                raise ValueError("formal persistent worker identity is incomplete")
            session = _verified_worker_session(
                receipt_path=worker_session_receipt,
                task_id=args.task,
                campaign_id=manifest.campaign_id,
                worker_session_id=worker_session_id,
                worker_pid=worker_process_id,
                worker_process_group_id=worker_process_group_id,
                worker_posix_session_id=worker_posix_session_id,
                expected_dispatch_count=new_worker_dispatches,
                reset_proof=reset_proof,
                deployment_root=layout.root,
            )
            if runner_return_code in {0, 4} and session.shutdown_status != "clean":
                raise ValueError("successful worker session did not shut down cleanly")
        except (OSError, TypeError, ValueError) as error:
            if error_type is None:
                error_type = type(error).__name__
                error_message = str(error)

    finished = datetime.now(timezone.utc)
    document = {
        "action_execution_contract": action_execution_contract,
        "capture_profile": capture_profile.value,
        "error_type": error_type,
        "error_message": error_message,
        "finished_at_utc": finished.isoformat(),
        "gpus": args.gpus,
        "campaign_manifest_file_sha256": manifest_file_sha256,
        "log_relpath": log_path.relative_to(layout.root).as_posix(),
        "log_sha256": _sha256_file(log_path),
        "n0_port": args.n0_port,
        "native_step_contract": (
            N0_STOCK_EE_NATIVE_STEP_CONTRACT
            if action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT
            else FIXED_NATIVE_STEP_CONTRACT
        ),
        "readiness": readiness,
        "replacement_requires_fresh_server": runner_return_code == 3,
        "runner_return_code": runner_return_code,
        "runner_summary": _runner_summary_subset(runner_summary),
        "semantic_version": "2.0",
        "server_shutdown": shutdown,
        "started_at_utc": started.isoformat(),
        "task_id": args.task,
        "worker_contract": args.worker_contract,
        "worker_process_group_id": worker_process_group_id,
        "worker_readiness": worker_readiness,
        "worker_shutdown": worker_shutdown,
        "worker_ready_receipt_relpath": (
            worker_ready_receipt.relative_to(layout.root).as_posix()
            if worker_ready_receipt.is_file()
            else None
        ),
        "worker_ready_receipt_sha256": (
            _sha256_file(worker_ready_receipt)
            if worker_ready_receipt.is_file()
            else None
        ),
        "worker_session_receipt_relpath": (
            worker_session_receipt.relative_to(layout.root).as_posix()
            if worker_session_receipt.is_file()
            else None
        ),
        "worker_session_receipt_sha256": (
            _sha256_file(worker_session_receipt)
            if worker_session_receipt.is_file()
            else None
        ),
        "reset_equivalence_receipt_relpath": (
            reset_proof[0].relative_to(layout.root).as_posix()
            if reset_proof is not None
            else None
        ),
        "reset_equivalence_receipt_sha256": (
            reset_proof[1] if reset_proof is not None else None
        ),
    }
    if qualification_info is not None:
        qualification_path, qualification_sha256 = qualification_info
        document.update(
            {
                "integration_config_relpath": (
                    cast(Path, integration_config).relative_to(layout.root).as_posix()
                ),
                "integration_config_sha256": _sha256_file(
                    cast(Path, integration_config)
                ),
                "n0_server_attestation_relpath": attestation_path.relative_to(
                    layout.root
                ).as_posix(),
                "n0_server_attestation_sha256": attestation_sha256,
                "qualification_relpath": qualification_path.relative_to(
                    layout.root
                ).as_posix(),
                "qualification_sha256": qualification_sha256,
                "semantic_version": "3.0",
                "server_process_group_id": server_process_group_id,
                "session_id": run_id,
                "source_bound": True,
            }
        )
    _write_once(receipt_path, document)
    summary = {**document, "receipt": str(receipt_path)}
    if error_type is None and runner_return_code == 0:
        return summary, 0
    if error_type is None and runner_return_code == 3:
        return summary, 3
    if error_type is None and runner_return_code == 4:
        return summary, 4
    return summary, 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, exit_code = run_task_shard(args)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
