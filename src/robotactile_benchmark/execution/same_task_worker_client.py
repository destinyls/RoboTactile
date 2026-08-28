"""Lightweight per-episode client for a persistent same-task Isaac worker."""

from __future__ import annotations

import argparse
import json
import socket
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Optional, cast

from robotactile_benchmark.closed_loop.crash_diagnostics import emit_crash_marker
from robotactile_benchmark.execution.lifecycle_watchdog import (
    LifecycleStageJournal,
    sha256_file,
)
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
)
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.official_act import official_act_live_summary
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
    SameTaskWorkerRequest,
    SameTaskWorkerResponse,
    read_frame,
    write_frame,
)


class SameTaskWorkerRemoteError(RuntimeError):
    """One identity-bound worker request ended without a valid artifact."""

    def __init__(self, failure_code: str) -> None:
        super().__init__("same-task worker request failed")
        self.failure_code = failure_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--lifecycle-journal", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--campaign-manifest-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--request-file-sha256", required=True)
    parser.add_argument("--trial-manifest-sha256", required=True)
    return parser


def _worker_request(args: argparse.Namespace) -> SameTaskWorkerRequest:
    return SameTaskWorkerRequest(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        request_id=args.request_id,
        campaign_manifest_sha256=args.campaign_manifest_sha256,
        task_id=args.task,
        ordinal=args.ordinal,
        attempt_id=args.attempt_id,
        request_file_sha256=args.request_file_sha256,
        trial_manifest_sha256=args.trial_manifest_sha256,
    )


def _validated_journal(
    path: Path, request: SameTaskWorkerRequest
) -> LifecycleStageJournal:
    journal = LifecycleStageJournal.open(path)
    identity = journal.identity
    if (
        identity.campaign_manifest_sha256 != request.campaign_manifest_sha256
        or identity.request_file_sha256 != request.request_file_sha256
        or identity.trial_manifest_sha256 != request.trial_manifest_sha256
        or identity.task_id != request.task_id
        or identity.ordinal != request.ordinal
        or identity.attempt_id != request.attempt_id
    ):
        raise ValueError("worker client lifecycle identity mismatch")
    return journal


def _validated_socket(path: Path) -> Path:
    selected = Path(path).absolute()
    if selected.is_symlink():
        raise ValueError("worker socket cannot be a symlink")
    try:
        mode = selected.stat().st_mode
    except OSError as error:
        raise ValueError("worker socket is unavailable") from error
    if not stat.S_ISSOCK(mode):
        raise ValueError("worker socket must be an AF_UNIX socket")
    return selected


def execute_client(args: argparse.Namespace) -> dict[str, object]:
    """Dispatch one frozen request and verify the returned live artifact."""

    request_identity = _worker_request(args)
    _validated_journal(args.lifecycle_journal, request_identity)
    request_path = Path(args.request).absolute()
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("worker request must be a regular file")
    if sha256_file(request_path) != request_identity.request_file_sha256:
        raise ValueError("worker client request SHA256 mismatch")
    live_request = load_live_univtac_request(request_path)
    if live_request.task_id != request_identity.task_id:
        raise ValueError("worker client request task mismatch")
    socket_path = _validated_socket(args.socket)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        with client.makefile("rwb") as stream:
            binary_stream = cast(BinaryIO, stream)
            write_frame(binary_stream, request_identity.to_document())
            response = SameTaskWorkerResponse.from_document(read_frame(binary_stream))
    response.validate_against(request_identity)
    if response.status != "completed":
        raise SameTaskWorkerRemoteError(
            response.exception_code or "same_task_worker_episode_failed"
        )
    if live_request.output_dir is None:
        raise ValueError("worker request has no output directory")
    artifact = load_live_univtac_artifact(live_request.output_dir)
    if artifact.external_root_sha256 != response.artifact_root_sha256:
        raise ValueError("worker response artifact SHA256 mismatch")
    return official_act_live_summary(artifact)


def _last_stage(journal_path: Path) -> str:
    try:
        record = LifecycleStageJournal.open(journal_path).last_record()
    except (OSError, ValueError):
        return "process_spawn"
    return "process_spawn" if record is None else record.stage


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = execute_client(args)
    except BaseException as error:
        failure_code = (
            error.failure_code
            if isinstance(error, SameTaskWorkerRemoteError)
            else "same_task_worker_client_failed"
        )
        emit_crash_marker(_last_stage(args.lifecycle_journal), failure_code, error)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SameTaskWorkerRemoteError", "execute_client", "main"]
