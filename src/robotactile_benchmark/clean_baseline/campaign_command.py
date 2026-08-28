"""Small command/path helpers for the Clean campaign process supervisor."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

from robotactile_benchmark.clean_baseline.source_bound_attempts import (
    SourceBoundCampaignContext,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SameTaskWorkerRequest,
)


def optional_integration_config(root: Path, requested: Path | None) -> Path | None:
    """Resolve one optional non-symlink integration config below deployment root."""

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


def require_worker_socket(root: Path, requested: Path | None) -> Path | None:
    """Resolve one existing non-symlink Unix socket below deployment root."""

    if requested is None:
        return None
    root_resolved = root.resolve(strict=True)
    selected = requested.absolute()
    try:
        lexical_relative = selected.relative_to(root_resolved)
        resolved = selected.resolve(strict=True)
        resolved.relative_to(root_resolved)
        mode = selected.lstat().st_mode
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("worker socket must remain below deployment root") from error
    current = root_resolved
    traverses_symlink = False
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            traverses_symlink = True
            break
    if traverses_symlink or not stat.S_ISSOCK(mode):
        raise ValueError("worker socket must be a non-symlink Unix socket")
    return resolved


def build_live_command(
    *,
    isaac_python: Path,
    deployment_root: Path,
    request_path: Path,
    n0_source_root: Path,
    n0_host: str,
    n0_port: int,
    integration_config: Path | None,
    action_execution_contract: str | None = None,
    lifecycle_journal: Path | None = None,
    source_bound_context: SourceBoundCampaignContext | None = None,
    isaac_attestation_output: Path | None = None,
    worker_socket: Path | None = None,
    worker_request: SameTaskWorkerRequest | None = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> tuple[str, ...]:
    """Build the exact argv executed by the Isaac child process."""

    if (worker_socket is None) != (worker_request is None):
        raise ValueError("worker socket and worker request must be provided together")
    if worker_socket is not None:
        if capture_profile is not LiveCaptureProfile.PAPER_FULL:
            raise ValueError(
                "light capture profiles do not support same_task_worker_v1"
            )
        assert worker_request is not None
        if lifecycle_journal is None:
            raise ValueError("same-task worker command requires a lifecycle journal")
        if source_bound_context is None:
            raise ValueError("same-task worker command requires source-bound context")
        if (
            worker_request.campaign_manifest_sha256
            != source_bound_context.campaign_manifest_sha256
            or worker_request.task_id != source_bound_context.task_id
        ):
            raise ValueError("same-task worker request source identity mismatch")
        return (
            sys.executable,
            "-m",
            "robotactile_benchmark.execution.same_task_worker_client",
            "--socket",
            str(worker_socket),
            "--request",
            str(request_path),
            "--lifecycle-journal",
            str(lifecycle_journal),
            "--request-id",
            worker_request.request_id,
            "--campaign-manifest-sha256",
            worker_request.campaign_manifest_sha256,
            "--task",
            worker_request.task_id,
            "--ordinal",
            str(worker_request.ordinal),
            "--attempt-id",
            worker_request.attempt_id,
            "--request-file-sha256",
            worker_request.request_file_sha256,
            "--trial-manifest-sha256",
            worker_request.trial_manifest_sha256,
        )

    command = [
        str(isaac_python),
        "-m",
        "robotactile_benchmark.cli",
        "live-univtac-run",
        "--root",
        str(deployment_root),
        "--request",
        str(request_path),
        "--n0-source-root",
        str(n0_source_root),
        "--n0-host",
        n0_host,
        "--n0-port",
        str(n0_port),
        "--capture-profile",
        capture_profile.value,
    ]
    if integration_config is not None:
        command.extend(("--config", str(integration_config)))
    if action_execution_contract is not None:
        command.extend(("--action-execution-contract", action_execution_contract))
    if lifecycle_journal is not None:
        command.extend(("--lifecycle-journal", str(lifecycle_journal)))
    if source_bound_context is not None:
        if isaac_attestation_output is None:
            raise ValueError("source-bound command lacks Isaac attestation output")
        command.extend(
            (
                "--campaign-id",
                source_bound_context.campaign_id,
                "--qualification",
                str(source_bound_context.qualification_path),
                "--n0-server-attestation",
                str(source_bound_context.n0_server_attestation_path),
                "--n0-server-attestation-sha256",
                source_bound_context.n0_server_attestation_sha256,
                "--isaac-attestation-output",
                str(isaac_attestation_output),
            )
        )
    return tuple(command)


__all__ = [
    "build_live_command",
    "optional_integration_config",
    "require_worker_socket",
]
