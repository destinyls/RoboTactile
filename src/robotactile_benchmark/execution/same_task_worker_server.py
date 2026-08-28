"""Persistent task-local Isaac worker for sequential Clean episodes."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Optional, cast

from robotactile_benchmark.backends.univtac_contracts import (
    N0_EE_ACTION_EXECUTION_CONTRACTS,
)
from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignProtocol,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.io import (
    load_clean_campaign_manifest,
    write_canonical_no_clobber,
)
from robotactile_benchmark.clean_baseline.source_bound_attempts import (
    isaac_attestation_output_path,
    load_source_bound_campaign_context,
)
from robotactile_benchmark.closed_loop.crash_diagnostics import emit_crash_marker
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacAttestationRequest,
)
from robotactile_benchmark.execution.lifecycle_watchdog import (
    LifecycleStageJournal,
    campaign_watchdog_paths,
    sha256_file,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.persistent_worker_receipt import (
    EpisodeDispatch,
    PersistentWorkerSessionReceipt,
    write_persistent_worker_session_receipt,
)
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
    SameTaskWorkerReadyIdentity,
    SameTaskWorkerRequest,
    SameTaskWorkerResponse,
    read_frame,
    write_frame,
)
from robotactile_benchmark.execution.same_task_worker_runtime import (
    SameTaskEpisodeExecutor,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--session-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n0-source-root", type=Path, required=True)
    parser.add_argument("--n0-host", default="127.0.0.1")
    parser.add_argument("--n0-port", type=int, required=True)
    parser.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        required=True,
    )
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--n0-server-attestation", type=Path, required=True)
    parser.add_argument("--n0-server-attestation-sha256", required=True)
    parser.add_argument("--restart-generation", type=int, default=0)
    parser.add_argument("--reset-equivalence-receipt", type=Path)
    parser.add_argument("--reset-equivalence-receipt-sha256")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_member(root: Path, relative: str, name: str) -> Path:
    candidate = (root / relative).absolute()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain below deployment root") from error
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    return candidate


def _relative(root: Path, path: Path, name: str) -> str:
    target = Path(path).absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain below deployment root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} path is unsafe")
    return relative.as_posix()


def _validate_reset_proof(
    root: Path,
    manifest_protocol: CleanCampaignProtocol,
    path: Optional[Path],
    expected_sha256: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if (path is None) != (expected_sha256 is None):
        raise ValueError("reset-equivalence receipt arguments must be complete")
    if path is None:
        if manifest_protocol is not CleanCampaignProtocol.DIAGNOSTIC:
            raise ValueError(
                "persistent pilot/paper execution requires reset-equivalence evidence"
            )
        return None, None
    selected = Path(path).absolute()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("reset-equivalence receipt must be a regular file")
    digest = sha256_file(selected)
    if digest != expected_sha256:
        raise ValueError("reset-equivalence receipt SHA256 mismatch")
    return _relative(root, selected, "reset-equivalence receipt"), digest


class SameTaskWorkerServer:
    """Bind one app host to a frozen task/campaign/source identity."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._root = Path(args.root).resolve(strict=True)
        self._manifest_path = Path(args.campaign_manifest).absolute()
        self._manifest = load_clean_campaign_manifest(self._manifest_path)
        self._task_id = args.task
        self._entries = {
            item.ordinal: item
            for item in self._manifest.trials
            if item.task == self._task_id
        }
        if not self._entries:
            raise ValueError("campaign does not contain the selected worker task")
        # The loader already proves exact canonical on-disk bytes.  The manifest
        # identity intentionally hashes canonical JSON without its terminating
        # newline, whereas the file digest includes that newline; comparing the
        # two would reject every valid manifest.
        self._context = load_source_bound_campaign_context(
            deployment_root=self._root,
            campaign_id=self._manifest.campaign_id,
            campaign_manifest_sha256=self._manifest.sha256,
            task_id=self._task_id,
            qualification_path=args.qualification,
            n0_server_attestation_path=args.n0_server_attestation,
            n0_server_attestation_sha256=args.n0_server_attestation_sha256,
        )
        self._action_execution_contract = args.action_execution_contract
        if self._context.runtime_source_binding.action_execution_contract != (
            self._action_execution_contract
        ):
            raise ValueError("worker action contract differs from source binding")
        self._config = Path(args.config).absolute()
        if self._config.is_symlink() or not self._config.is_file():
            raise ValueError("worker integration config must be a regular file")
        runtime = resolve_n0_runtime_artifacts(self._config)
        if runtime.manifest.task_id != self._task_id:
            raise ValueError("worker integration config task mismatch")
        source = self._context.runtime_source_binding
        if (
            runtime.manifest.external_commit != source.n0_source_commit
            or runtime.manifest.checkpoint_sha256 != source.checkpoint_sha256
            or runtime.manifest.config_sha256 != source.config_sha256
            or runtime.manifest.normalizer_sha256 != source.normalizer_sha256
            or runtime.manifest.serve_bundle_sha256 != source.serve_bundle_sha256
            or runtime.manifest.prompt_manifest_sha256 != source.prompt_manifest_sha256
        ):
            raise ValueError("worker integration config differs from source binding")
        bootstrap_entry = self._entries[min(self._entries)]
        bootstrap_path = _safe_member(
            self._root, bootstrap_entry.request_relpath, "bootstrap request"
        )
        if sha256_file(bootstrap_path) != bootstrap_entry.request_file_sha256:
            raise ValueError("bootstrap request SHA256 mismatch")
        bootstrap = load_live_univtac_run(load_live_univtac_request(bootstrap_path))
        if bootstrap.trial.sha256 != bootstrap_entry.trial_manifest_sha256:
            raise ValueError("bootstrap trial manifest SHA256 mismatch")
        self._executor = SameTaskEpisodeExecutor(
            bootstrap=bootstrap,
            manifest=runtime.manifest,
            n0_source_root=args.n0_source_root,
            n0_host=args.n0_host,
            n0_port=args.n0_port,
            action_execution_contract=self._action_execution_contract,
        )
        self._socket_path = Path(args.socket).absolute()
        self._ready_receipt = Path(args.ready_receipt).absolute()
        self._session_receipt = Path(args.session_receipt).absolute()
        for path, name in (
            (self._ready_receipt, "ready receipt"),
            (self._session_receipt, "session receipt"),
        ):
            _relative(self._root, path, name)
        self._reset_relpath, self._reset_sha256 = _validate_reset_proof(
            self._root,
            self._manifest.protocol_id,
            args.reset_equivalence_receipt,
            args.reset_equivalence_receipt_sha256,
        )
        if type(args.restart_generation) is not int or args.restart_generation < 0:
            raise ValueError("restart generation must be non-negative")
        self._restart_generation = args.restart_generation
        self._session_id = f"same-task-{uuid.uuid4().hex}"
        self._started_at = _utc_now()
        self._dispatches: list[EpisodeDispatch] = []
        self._stop_requested = False
        self._crashed = False

    def request_stop(self, _signum: int, _frame: Optional[FrameType]) -> None:
        self._stop_requested = True

    def serve(self) -> int:
        listener = self._bind_listener()
        try:
            self._write_ready_receipt()
            while not self._stop_requested and not self._crashed:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except InterruptedError:
                    continue
                with connection:
                    self._serve_connection(connection)
        finally:
            listener.close()
            self._socket_path.unlink(missing_ok=True)
            try:
                self._executor.close()
            except BaseException as error:
                self._crashed = True
                emit_crash_marker("close", "same_task_worker_close_failed", error)
            self._write_session_receipt()
        return 2 if self._crashed else 0

    def _bind_listener(self) -> socket.socket:
        if len(os.fsencode(self._socket_path)) >= 100:
            raise ValueError("worker socket path is too long")
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._socket_path.parent.is_symlink() or self._socket_path.exists():
            raise FileExistsError("worker socket path must not already exist")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self._socket_path))
        os.chmod(self._socket_path, 0o600)
        listener.listen(1)
        listener.settimeout(1.0)
        return listener

    def _write_ready_receipt(self) -> None:
        ready = SameTaskWorkerReadyIdentity(
            worker_contract=SAME_TASK_WORKER_CONTRACT,
            semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
            session_id=self._session_id,
            task_id=self._task_id,
            process_id=os.getpid(),
            campaign_manifest_sha256=self._manifest.sha256,
            source_binding_sha256=canonical_hash(
                self._context.runtime_source_binding.to_dict()
            ),
            integration_config_sha256=sha256_file(self._config),
            action_execution_contract=self._action_execution_contract,
        )
        write_canonical_no_clobber(self._ready_receipt, ready.to_document())

    def _serve_connection(self, connection: socket.socket) -> None:
        dispatched_at = _utc_now()
        request: Optional[SameTaskWorkerRequest] = None
        entry: Optional[CleanCampaignTrialSpec] = None
        dispatch_recorded = False
        try:
            with connection.makefile("rwb") as stream:
                binary_stream = cast(BinaryIO, stream)
                request = SameTaskWorkerRequest.from_document(read_frame(binary_stream))
                entry, request_path, journal = self._validate_request(request)
                artifact = self._executor.execute(
                    load_live_univtac_request(request_path),
                    lifecycle_journal=journal,
                    isaac_attestation_request=IsaacAttestationRequest(
                        campaign_id=self._manifest.campaign_id,
                        deployment_root=self._root,
                        output_path=isaac_attestation_output_path(
                            self._context,
                            ordinal=request.ordinal,
                            attempt_id=request.attempt_id,
                        ),
                        qualification_path=self._context.qualification_path,
                        n0_server_attestation_path=(
                            self._context.n0_server_attestation_path
                        ),
                        n0_server_attestation_sha256=(
                            self._context.n0_server_attestation_sha256
                        ),
                    ),
                )
                response = SameTaskWorkerResponse(
                    worker_contract=SAME_TASK_WORKER_CONTRACT,
                    semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
                    request_id=request.request_id,
                    task_id=request.task_id,
                    ordinal=request.ordinal,
                    attempt_id=request.attempt_id,
                    status="completed",
                    return_code=0,
                    artifact_root_sha256=artifact.external_root_sha256,
                    exception_code=None,
                )
                self._record_dispatch(entry, request, dispatched_at)
                dispatch_recorded = True
                write_frame(binary_stream, response.to_document())
        except BaseException as error:
            self._crashed = True
            stage = "process_spawn"
            if request is not None:
                stage = self._last_stage(request)
                if entry is not None and not dispatch_recorded:
                    self._record_dispatch(entry, request, dispatched_at)
                self._write_failure_response(connection, request)
            emit_crash_marker(stage, "same_task_worker_episode_failed", error)

    def _record_dispatch(
        self,
        entry: CleanCampaignTrialSpec,
        request: SameTaskWorkerRequest,
        dispatched_at: str,
    ) -> None:
        self._dispatches.append(
            EpisodeDispatch(
                task_id=self._task_id,
                campaign_id=self._manifest.campaign_id,
                sequence=len(self._dispatches),
                ordinal=entry.ordinal,
                attempt_id=request.attempt_id,
                initial_seed=entry.initial_seed,
                exogenous_seed=entry.exogenous_seed,
                request_file_sha256=entry.request_file_sha256,
                trial_manifest_sha256=entry.trial_manifest_sha256,
                dispatched_at_utc=dispatched_at,
                completed_at_utc=_utc_now(),
            )
        )

    def _validate_request(
        self, request: SameTaskWorkerRequest
    ) -> tuple[CleanCampaignTrialSpec, Path, LifecycleStageJournal]:
        if (
            request.campaign_manifest_sha256 != self._manifest.sha256
            or request.task_id != self._task_id
        ):
            raise ValueError("worker request campaign/task mismatch")
        try:
            entry = self._entries[request.ordinal]
        except KeyError as error:
            raise ValueError(
                "worker request ordinal is outside the task shard"
            ) from error
        if (
            request.request_file_sha256 != entry.request_file_sha256
            or request.trial_manifest_sha256 != entry.trial_manifest_sha256
        ):
            raise ValueError("worker request content identity mismatch")
        request_path = _safe_member(self._root, entry.request_relpath, "request")
        if sha256_file(request_path) != entry.request_file_sha256:
            raise ValueError("worker request file changed after campaign freeze")
        lifecycle_root = (
            self._root
            / "outputs"
            / "clean-campaigns"
            / self._manifest.campaign_id
            / "lifecycle"
            / self._task_id
        )
        journal_path, _ = campaign_watchdog_paths(
            lifecycle_root, request.ordinal, request.attempt_id
        )
        journal = LifecycleStageJournal.open(journal_path)
        if journal.identity.to_dict() != {
            "attempt_id": request.attempt_id,
            "campaign_manifest_sha256": request.campaign_manifest_sha256,
            "ordinal": request.ordinal,
            "request_file_sha256": request.request_file_sha256,
            "semantic_version": "1.0",
            "task_id": request.task_id,
            "trial_manifest_sha256": request.trial_manifest_sha256,
        }:
            raise ValueError("worker lifecycle identity mismatch")
        return entry, request_path, journal

    def _last_stage(self, request: SameTaskWorkerRequest) -> str:
        lifecycle_root = (
            self._root
            / "outputs"
            / "clean-campaigns"
            / self._manifest.campaign_id
            / "lifecycle"
            / self._task_id
        )
        journal_path, _ = campaign_watchdog_paths(
            lifecycle_root, request.ordinal, request.attempt_id
        )
        try:
            record = LifecycleStageJournal.open(journal_path).last_record()
        except (OSError, ValueError):
            return "process_spawn"
        return "process_spawn" if record is None else record.stage

    @staticmethod
    def _write_failure_response(
        connection: socket.socket, request: SameTaskWorkerRequest
    ) -> None:
        response = SameTaskWorkerResponse(
            worker_contract=SAME_TASK_WORKER_CONTRACT,
            semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
            request_id=request.request_id,
            task_id=request.task_id,
            ordinal=request.ordinal,
            attempt_id=request.attempt_id,
            status="failed",
            return_code=2,
            artifact_root_sha256=None,
            exception_code="same_task_worker_episode_failed",
        )
        try:
            with connection.makefile("rwb") as stream:
                write_frame(cast(BinaryIO, stream), response.to_document())
        except (BrokenPipeError, OSError, ValueError):
            return

    def _write_session_receipt(self) -> None:
        if (
            self._reset_relpath is None
            or self._reset_sha256 is None
            or not self._dispatches
        ):
            return
        receipt = PersistentWorkerSessionReceipt.build(
            task_id=self._task_id,
            campaign_id=self._manifest.campaign_id,
            worker_session_id=self._session_id,
            worker_pid=os.getpid(),
            worker_process_group_id=os.getpgrp(),
            worker_posix_session_id=os.getsid(0),
            restart_generation=self._restart_generation,
            runtime_source_binding=self._context.runtime_source_binding,
            qualification_relpath=self._context.qualification_relpath,
            qualification_sha256=self._context.qualification_sha256,
            n0_server_attestation_relpath=(self._context.n0_server_attestation_relpath),
            n0_server_attestation_sha256=(self._context.n0_server_attestation_sha256),
            reset_equivalence_receipt_relpath=self._reset_relpath,
            reset_equivalence_receipt_sha256=self._reset_sha256,
            episode_dispatches=tuple(self._dispatches),
            started_at_utc=self._started_at,
            finished_at_utc=_utc_now(),
            shutdown_status="crashed" if self._crashed else "clean",
        )
        write_persistent_worker_session_receipt(self._session_receipt, receipt)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    server = SameTaskWorkerServer(args)
    signal.signal(signal.SIGTERM, server.request_stop)
    signal.signal(signal.SIGINT, server.request_stop)
    return server.serve()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SameTaskWorkerServer", "main"]
