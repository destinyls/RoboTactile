#!/usr/bin/env python3
"""Execute and resume one frozen task shard of an N0 Clean campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path, PurePosixPath

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_EE_ACTION_EXECUTION_CONTRACTS,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    load_univtac_task_registry,
    validate_n0_ee_action_execution_contract,
)
from robotactile_benchmark.clean_baseline import (
    CleanCampaignManifest,
    CleanCampaignProtocol,
    CleanCampaignTrialSpec,
    load_clean_campaign_manifest,
)
from robotactile_benchmark.clean_baseline import runner_recovery as _runner_recovery
from robotactile_benchmark.clean_baseline.campaign_attempt_io import (
    load_prior_attempts as _load_prior_attempts,
)
from robotactile_benchmark.clean_baseline.campaign_command import (
    build_live_command as _live_command,
)
from robotactile_benchmark.clean_baseline.campaign_command import (
    optional_integration_config as _optional_config,
)
from robotactile_benchmark.clean_baseline.campaign_command import (
    require_worker_socket as _require_worker_socket,
)
from robotactile_benchmark.clean_baseline.source_bound_attempts import (
    SourceBoundCampaignContext,
    isaac_attestation_output_path,
    load_source_bound_campaign_context,
    source_bound_arguments_requested,
    source_bound_attempt_fields,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution import load_live_univtac_request
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.lifecycle_watchdog import (
    HardLifecycleOutcome,
    LifecycleIdentity,
    LifecycleStageJournal,
    campaign_watchdog_paths,
    require_positive_finite,
    resolve_hard_lifecycle_timeout,
    sha256_file,
    wait_with_lifecycle_watchdog,
    watchdog_proves_fatal_unattempted,
    watchdog_receipt_document,
    write_canonical_no_clobber,
)
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
    SameTaskWorkerRequest,
    compute_request_id,
)
from robotactile_benchmark.integrations.n0_twam.official_protocol import (
    validate_official_n0_clean_claim_request,
)
from robotactile_benchmark.trials import Condition

_ArtifactInspection = _runner_recovery.ArtifactInspection
_EXCEPTION_REPLACED = _runner_recovery.EXCEPTION_REPLACED
_FATAL_UNATTEMPTED = _runner_recovery.FATAL_UNATTEMPTED
_OfficialPreflight = _runner_recovery.OfficialPreflight
_RunnerStageEvidence = _runner_recovery.RunnerStageEvidence
_VALID_OUTCOME = _runner_recovery.VALID_OUTCOME
_adoption_receipt = _runner_recovery.adoption_receipt
_candidate_disposition = _runner_recovery.candidate_disposition
_close_timeout_proof = _runner_recovery._close_timeout_proof
_candidate_logs = _runner_recovery._candidate_logs
_existing_artifact = _runner_recovery.existing_artifact
_has_successful_attempt = _runner_recovery.has_successful_attempt
_inspect_artifact = _runner_recovery.inspect_artifact
_official_attempt_disposition = _runner_recovery.official_attempt_disposition
_preflight_official_candidates_impl = _runner_recovery.preflight_official_candidates
_replacement_requires_fresh_n0_server = (
    _runner_recovery.replacement_requires_fresh_n0_server
)
_runner_stage_evidence = _runner_recovery.runner_stage_evidence
_trusted_log_evidence = _runner_recovery._trusted_log_evidence
_write_receipt = write_canonical_no_clobber

_DEFAULT_WATCHDOG_TERM_GRACE_S = 30.0


def _required_action_execution_contract(
    protocol: CleanCampaignProtocol,
    requested: str | None = None,
) -> str:
    """Require training-aligned cadence for formal campaign protocols."""

    if protocol in {CleanCampaignProtocol.PILOT, CleanCampaignProtocol.PAPER}:
        if requested not in {
            None,
            N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        }:
            raise ValueError(
                f"{protocol.value} requires "
                f"{N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT}"
            )
        return N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    return validate_n0_ee_action_execution_contract(
        requested or N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )


def _native_step_contract(action_execution_contract: str) -> str:
    """Return the native-step contract implied by one N0 EE surface."""

    if action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT:
        return N0_STOCK_EE_NATIVE_STEP_CONTRACT
    return FIXED_NATIVE_STEP_CONTRACT


def _require_source_bound_execution_contract(
    context: SourceBoundCampaignContext,
    action_execution_contract: str,
) -> None:
    """Reject source evidence created for a different executor cadence."""

    source = context.runtime_source_binding
    if (
        source.action_execution_contract != action_execution_contract
        or source.native_step_contract
        != _native_step_contract(action_execution_contract)
    ):
        raise ValueError("source-bound action execution contract mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--n0-source-root", type=Path)
    parser.add_argument("--n0-host", default="127.0.0.1")
    parser.add_argument("--n0-port", type=int, default=29601)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--n0-server-attestation", type=Path)
    parser.add_argument("--n0-server-attestation-sha256")
    parser.add_argument(
        "--worker-socket",
        type=Path,
        help="dispatch episodes to an already-qualified same-task worker",
    )
    parser.add_argument("--max-new-trials", type=int)
    parser.add_argument("--continue-on-infrastructure-failure", action="store_true")
    parser.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        default=LiveCaptureProfile.PAPER_FULL.value,
    )
    parser.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        help=(
            "defaults to the training-aligned 60 Hz N0 EE contract; legacy "
            "fixed-endpoint and stock EE contracts are explicit diagnostic "
            "or reference surfaces only"
        ),
    )
    parser.add_argument(
        "--hard-lifecycle-timeout-s",
        type=float,
        default=None,
        help="hard per-episode deadline; default is request wall timeout plus 600s",
    )
    parser.add_argument(
        "--watchdog-term-grace-s",
        type=float,
        default=_DEFAULT_WATCHDOG_TERM_GRACE_S,
        help="TERM grace before process-group KILL",
    )
    return parser


def _safe_path(root: Path, relative: object, name: str) -> Path:
    if type(relative) is not str:
        raise TypeError(f"{name} must be a POSIX path string")
    member = PurePosixPath(relative)
    if (
        member.is_absolute()
        or not member.parts
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise ValueError(f"{name} is not a safe relative path")
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / Path(*member.parts)).resolve(strict=False)
    if root_resolved not in candidate.parents:
        raise ValueError(f"{name} escapes the deployment root")
    return candidate


def _validate_entry(root: Path, entry: CleanCampaignTrialSpec) -> tuple[Path, Path]:
    request_path = _safe_path(root, entry.request_relpath, "request_relpath")
    artifact_path = _safe_path(root, entry.artifact_relpath, "artifact_relpath")
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("frozen request is not a regular file")
    if sha256_file(request_path) != entry.request_file_sha256:
        raise ValueError("frozen request file hash mismatch")
    request = load_live_univtac_request(request_path)
    if (
        request.condition is not Condition.CLEAN
        or request.task_id != entry.task
        or request.initial_seed != entry.initial_seed
        or request.exogenous_seed != entry.exogenous_seed
        or request.output_dir != artifact_path
    ):
        raise ValueError("frozen request semantic identity mismatch")
    return request_path, artifact_path


def _same_task_worker_request(
    entry: CleanCampaignTrialSpec,
    identity: LifecycleIdentity,
) -> SameTaskWorkerRequest:
    """Bind one worker dispatch to the already-validated campaign identity."""

    if (
        identity.task_id != entry.task
        or identity.ordinal != entry.ordinal
        or identity.request_file_sha256 != entry.request_file_sha256
        or identity.trial_manifest_sha256 != entry.trial_manifest_sha256
    ):
        raise ValueError("same-task worker lifecycle identity mismatch")
    request_id = compute_request_id(
        campaign_manifest_sha256=identity.campaign_manifest_sha256,
        task_id=identity.task_id,
        ordinal=identity.ordinal,
        attempt_id=identity.attempt_id,
        request_file_sha256=identity.request_file_sha256,
    )
    return SameTaskWorkerRequest(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        request_id=request_id,
        campaign_manifest_sha256=identity.campaign_manifest_sha256,
        task_id=identity.task_id,
        ordinal=identity.ordinal,
        attempt_id=identity.attempt_id,
        request_file_sha256=identity.request_file_sha256,
        trial_manifest_sha256=identity.trial_manifest_sha256,
    )


_prior_attempts = _load_prior_attempts


def _watchdog_filtered_candidate_logs(
    layout: DeploymentLayout,
    campaign_id: str,
    entry: CleanCampaignTrialSpec,
    *,
    campaign_manifest_sha256: str,
) -> tuple[Path, ...]:
    kept: list[Path] = []
    lifecycle_root = (
        layout.outputs / "clean-campaigns" / campaign_id / "lifecycle" / entry.task
    )
    for log in _candidate_logs(layout, campaign_id, entry):
        attempt_id = log.stem.removeprefix(f"{entry.ordinal:04d}-")
        identity = LifecycleIdentity(
            campaign_manifest_sha256,
            entry.request_file_sha256,
            entry.trial_manifest_sha256,
            entry.task,
            entry.ordinal,
            attempt_id,
        )
        journal, receipt = campaign_watchdog_paths(
            lifecycle_root, entry.ordinal, attempt_id
        )
        if not receipt.exists() or not watchdog_proves_fatal_unattempted(
            receipt_path=receipt, journal_path=journal, log_path=log, identity=identity
        ):
            kept.append(log)
    return tuple(kept)


def _preflight_official_candidates(
    *,
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    selected: Sequence[CleanCampaignTrialSpec],
    target: int,
    attempt_semantic_version: str = "2.0",
) -> _OfficialPreflight:
    """Bind script-local validators to the reusable recovery state machine."""

    return _preflight_official_candidates_impl(
        layout=layout,
        campaign_id=campaign_id,
        campaign_manifest_sha256=campaign_manifest_sha256,
        selected=selected,
        target=target,
        validate_entry=_validate_entry,
        prior_attempts=partial(
            _prior_attempts,
            required_semantic_version=attempt_semantic_version,
        ),
        inspect_artifact_fn=_inspect_artifact,
        candidate_logs_fn=partial(
            _watchdog_filtered_candidate_logs,
            campaign_manifest_sha256=campaign_manifest_sha256,
        ),
        trusted_log_evidence_fn=_trusted_log_evidence,
    )


def _execute_entry(
    *,
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    entry: CleanCampaignTrialSpec,
    request_path: Path,
    artifact_path: Path,
    isaac_python: Path,
    n0_source_root: Path,
    n0_host: str,
    n0_port: int,
    integration_config: Path | None,
    hard_lifecycle_timeout_s: float | None,
    watchdog_term_grace_s: float,
    action_execution_contract: str | None = None,
    semantic_version: str = "1.0",
    source_bound_context: SourceBoundCampaignContext | None = None,
    worker_socket: Path | None = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> tuple[str, dict[str, object]]:
    request = load_live_univtac_request(request_path)
    hard_timeout_s, hard_timeout_source, startup_teardown_grace_s = (
        resolve_hard_lifecycle_timeout(request.wall_timeout_s, hard_lifecycle_timeout_s)
    )
    term_grace_s = require_positive_finite(
        watchdog_term_grace_s, "watchdog_term_grace_s"
    )
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    attempt_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = (
        layout.logs
        / "clean-campaigns"
        / campaign_id
        / entry.task
        / f"{entry.ordinal:04d}-{attempt_id}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_root = (
        layout.outputs / "clean-campaigns" / campaign_id / "lifecycle" / entry.task
    )
    lifecycle_journal_path, watchdog_receipt_path = campaign_watchdog_paths(
        lifecycle_root, entry.ordinal, attempt_id
    )
    identity = LifecycleIdentity(
        campaign_manifest_sha256=campaign_manifest_sha256,
        request_file_sha256=entry.request_file_sha256,
        trial_manifest_sha256=entry.trial_manifest_sha256,
        task_id=entry.task,
        ordinal=entry.ordinal,
        attempt_id=attempt_id,
    )
    worker_request = None
    if worker_socket is not None:
        if source_bound_context is None:
            raise ValueError("same-task worker execution requires source-bound context")
        worker_request = _same_task_worker_request(entry, identity)
    lifecycle_journal = LifecycleStageJournal.create(lifecycle_journal_path, identity)
    lifecycle_journal.record("process_spawn")
    isaac_attestation_path = (
        None
        if source_bound_context is None
        else isaac_attestation_output_path(
            source_bound_context,
            ordinal=entry.ordinal,
            attempt_id=attempt_id,
        )
    )
    command = _live_command(
        isaac_python=isaac_python,
        deployment_root=layout.root,
        request_path=request_path,
        n0_source_root=n0_source_root,
        n0_host=n0_host,
        n0_port=n0_port,
        integration_config=integration_config,
        action_execution_contract=action_execution_contract,
        lifecycle_journal=lifecycle_journal_path,
        source_bound_context=source_bound_context,
        isaac_attestation_output=isaac_attestation_path,
        worker_socket=worker_socket,
        worker_request=worker_request,
        capture_profile=capture_profile,
    )
    launch_error_type: str | None = None
    watchdog_error_type: str | None = None
    process_id: int | None = None
    outcome = HardLifecycleOutcome(126, False, False, False)
    with log_path.open("xb") as stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
                start_new_session=True,
            )
            process_id = process.pid
            try:
                outcome = wait_with_lifecycle_watchdog(
                    process,
                    hard_timeout_s=hard_timeout_s,
                    term_grace_s=term_grace_s,
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                watchdog_error_type = type(error).__name__
                outcome = HardLifecycleOutcome(125, False, False, False)
                stream.write(
                    (
                        "\nROBOTACTILE_LIFECYCLE_WATCHDOG_ERROR "
                        f"exception_type={watchdog_error_type}\n"
                    ).encode("ascii")
                )
        except OSError as error:
            launch_error_type = type(error).__name__
            stream.write(
                (
                    "\nROBOTACTILE_CAMPAIGN_COMMAND_ERROR "
                    f"exception_type={launch_error_type}\n"
                ).encode("ascii")
            )
        if outcome.timed_out:
            last_during_timeout = lifecycle_journal.last_record()
            last_stage = (
                "none" if last_during_timeout is None else last_during_timeout.stage
            )
            stream.write(
                (
                    "\nROBOTACTILE_HARD_LIFECYCLE_TIMEOUT "
                    f"last_stage={last_stage} timeout_s={hard_timeout_s}\n"
                ).encode("ascii")
            )
        stream.flush()
    finished = datetime.now(timezone.utc)
    duration_s = time.monotonic() - started_monotonic
    last_stage_receipt = lifecycle_journal.last_record()
    watchdog_document = watchdog_receipt_document(
        identity=identity,
        last_stage_receipt=last_stage_receipt,
        journal_relpath=lifecycle_journal_path.relative_to(layout.root).as_posix(),
        log_relpath=log_path.relative_to(layout.root).as_posix(),
        log_sha256=sha256_file(log_path),
        process_id=process_id,
        outcome=outcome,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_s=duration_s,
        hard_lifecycle_timeout_s=hard_timeout_s,
        soft_wall_timeout_s=request.wall_timeout_s,
        hard_timeout_source=hard_timeout_source,
        startup_teardown_grace_s=startup_teardown_grace_s,
        term_grace_s=term_grace_s,
        watchdog_error_type=watchdog_error_type,
    )
    write_canonical_no_clobber(watchdog_receipt_path, watchdog_document)
    inspection = _inspect_artifact(entry, artifact_path)
    artifact = inspection.artifact
    artifact_error = inspection.error_type
    if (
        artifact is not None
        and inspection.capture_profile is not None
        and inspection.capture_profile is not capture_profile
    ):
        artifact = None
        artifact_error = "CaptureProfileMismatch"
    log_evidence = _runner_stage_evidence(log_path)
    journal_evidence = (
        None
        if last_stage_receipt is None
        else _RunnerStageEvidence(
            stage=last_stage_receipt.stage,
            failure_code=(
                "hard_lifecycle_timeout" if outcome.timed_out else "abrupt_process_exit"
            ),
            exception_type=("TimeoutExpired" if outcome.timed_out else "ProcessExit"),
        )
    )
    stage_evidence = inspection.stage_evidence or log_evidence or journal_evidence
    close_after_export_timeout = False
    if (
        outcome.timed_out
        and artifact is not None
        and artifact.get("terminal_status") != "crash"
    ):
        try:
            proof = _close_timeout_proof(
                layout,
                campaign_id,
                campaign_manifest_sha256,
                entry,
                log_path.relative_to(layout.root).as_posix(),
            )
            close_after_export_timeout = proof.return_code == outcome.return_code
        except (OSError, TypeError, ValueError):
            close_after_export_timeout = False
    if semantic_version in {"2.0", "3.0"}:
        try:
            disposition, exception_code = _candidate_disposition(
                return_code=outcome.return_code,
                artifact=artifact,
                artifact_error=artifact_error,
                stage_evidence=stage_evidence,
                close_after_export_timeout=close_after_export_timeout,
            )
        except ValueError as error:
            disposition = _FATAL_UNATTEMPTED
            exception_code = f"candidate_classification_{type(error).__name__}"
        if launch_error_type is not None:
            exception_code = f"launcher_{launch_error_type}"
        elif watchdog_error_type is not None:
            exception_code = f"watchdog_{watchdog_error_type}"
    else:
        disposition = (
            _VALID_OUTCOME
            if outcome.return_code == 0 and artifact is not None
            else _EXCEPTION_REPLACED
        )
        exception_code = None
    receipt: dict[str, object] = {
        "artifact": artifact,
        "artifact_validation_error_type": artifact_error,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "command_kind": "robotactile_live_univtac_run_v1",
        "duration_s": duration_s,
        "finished_at_utc": finished.isoformat(),
        "log_relpath": log_path.relative_to(layout.root).as_posix(),
        "log_sha256": sha256_file(log_path),
        "ordinal": entry.ordinal,
        "request_file_sha256": entry.request_file_sha256,
        "return_code": outcome.return_code,
        "semantic_version": semantic_version,
        "started_at_utc": started.isoformat(),
        "task_id": entry.task,
        "trial_manifest_sha256": entry.trial_manifest_sha256,
    }
    if semantic_version in {"2.0", "3.0"}:
        receipt.update(
            {
                "candidate_disposition": disposition,
                "exception_code": exception_code,
            }
        )
    if semantic_version == "3.0" and disposition != _FATAL_UNATTEMPTED:
        if source_bound_context is None or isaac_attestation_path is None:
            disposition = _FATAL_UNATTEMPTED
            receipt["exception_code"] = "source_bound_context_missing"
        else:
            try:
                fields, _ = source_bound_attempt_fields(
                    source_bound_context,
                    identity=identity,
                    isaac_attestation_path=isaac_attestation_path,
                )
                receipt.update(fields)
            except (OSError, TypeError, ValueError) as error:
                disposition = _FATAL_UNATTEMPTED
                receipt["exception_code"] = f"isaac_attestation_{type(error).__name__}"
    if disposition != _FATAL_UNATTEMPTED:
        receipt_path = (
            layout.outputs
            / "clean-campaigns"
            / campaign_id
            / "attempts"
            / entry.task
            / f"{entry.ordinal:04d}-{attempt_id}.json"
        )
        write_canonical_no_clobber(receipt_path, receipt)
    return disposition, receipt


def _run_official_task(
    *,
    args: argparse.Namespace,
    layout: DeploymentLayout,
    manifest: CleanCampaignManifest,
    selected: Sequence[CleanCampaignTrialSpec],
    isaac_python: Path,
    n0_source_root: Path,
    integration_config: Path | None,
    action_execution_contract: str | None,
    source_bound_context: SourceBoundCampaignContext | None,
    worker_socket: Path | None,
    capture_profile: LiveCaptureProfile,
) -> tuple[dict[str, object], int]:
    """Run task-local candidates until the first target valid outcomes exist."""

    sampling = manifest.sampling
    if sampling is None:
        raise ValueError("official campaign lacks a valid sampling target")
    target = sampling.target_valid_trials_per_task
    if args.continue_on_infrastructure_failure:
        raise ValueError(
            "official replacement semantics always require a fresh server after an exception"
        )
    campaign_id = manifest.campaign_id
    manifest_sha256 = manifest.sha256
    valid_count = 0
    replacement_count = 0
    unused_reserve_count = 0
    new_count = 0
    remaining_unattempted_count = 0
    replacement_requires_fresh_server = False
    last_receipt: dict[str, object] | None = None
    stopped_for_budget = False
    fatal_error: dict[str, object] | None = None
    while True:
        try:
            preflight = _preflight_official_candidates(
                layout=layout,
                campaign_id=campaign_id,
                campaign_manifest_sha256=manifest_sha256,
                selected=selected,
                target=target,
                attempt_semantic_version=(
                    "3.0" if source_bound_context is not None else "2.0"
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            fatal_error = {
                "error_type": type(error).__name__,
                "message": str(error),
                "stage": "candidate_preflight",
            }
            break
        if any(
            state.artifact is not None
            and state.capture_profile is not None
            and state.capture_profile is not capture_profile
            for state in preflight.states
        ):
            fatal_error = {
                "error_type": "CaptureProfileMismatch",
                "message": "existing artifact uses another capture profile",
                "stage": "candidate_preflight",
            }
            break
        valid_count = preflight.valid_count
        replacement_count = preflight.replacement_count
        unused_reserve_count = preflight.unused_reserve_count
        if preflight.adoptions:
            try:
                for state in preflight.adoptions:
                    receipt_path, receipt = _adoption_receipt(
                        layout=layout,
                        campaign_id=campaign_id,
                        campaign_manifest_sha256=manifest_sha256,
                        state=state,
                    )
                    if source_bound_context is not None:
                        if state.adoption_log is None:
                            raise ValueError("source-bound adoption lacks log identity")
                        attempt_id = state.adoption_log.stem.removeprefix(
                            f"{state.entry.ordinal:04d}-"
                        )
                        identity = LifecycleIdentity(
                            campaign_manifest_sha256=manifest_sha256,
                            request_file_sha256=state.entry.request_file_sha256,
                            trial_manifest_sha256=state.entry.trial_manifest_sha256,
                            task_id=state.entry.task,
                            ordinal=state.entry.ordinal,
                            attempt_id=attempt_id,
                        )
                        fields, _ = source_bound_attempt_fields(
                            source_bound_context,
                            identity=identity,
                            isaac_attestation_path=isaac_attestation_output_path(
                                source_bound_context,
                                ordinal=state.entry.ordinal,
                                attempt_id=attempt_id,
                            ),
                        )
                        receipt["semantic_version"] = "3.0"
                        receipt.update(fields)
                    write_canonical_no_clobber(receipt_path, receipt)
                    last_receipt = receipt
            except (OSError, TypeError, ValueError) as error:
                fatal_error = {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "stage": "artifact_adoption",
                }
                break
            continue
        if valid_count >= target:
            break
        if preflight.next_index is None:
            remaining_unattempted_count = 0
            break
        next_index = preflight.next_index
        remaining_unattempted_count = len(selected) - next_index
        if args.max_new_trials is not None and new_count >= args.max_new_trials:
            stopped_for_budget = True
            break
        state = preflight.states[next_index]
        disposition, last_receipt = _execute_entry(
            layout=layout,
            campaign_id=campaign_id,
            campaign_manifest_sha256=manifest_sha256,
            entry=state.entry,
            request_path=state.request_path,
            artifact_path=state.artifact_path,
            isaac_python=isaac_python,
            n0_source_root=n0_source_root,
            n0_host=args.n0_host,
            n0_port=args.n0_port,
            integration_config=integration_config,
            hard_lifecycle_timeout_s=args.hard_lifecycle_timeout_s,
            watchdog_term_grace_s=args.watchdog_term_grace_s,
            action_execution_contract=action_execution_contract,
            semantic_version=("3.0" if source_bound_context is not None else "2.0"),
            source_bound_context=source_bound_context,
            worker_socket=worker_socket,
            capture_profile=capture_profile,
        )
        if disposition == _FATAL_UNATTEMPTED:
            fatal_error = {
                "candidate_ordinal": state.entry.ordinal,
                "error_type": "UntrustedCandidateFailure",
                "exception_code": last_receipt.get("exception_code"),
                "log_relpath": last_receipt.get("log_relpath"),
                "message": "live command failed without trusted episode evidence",
                "stage": "live_command",
            }
            break
        new_count += 1
        if disposition == _EXCEPTION_REPLACED:
            replacement_count += 1
            requires_fresh_server = worker_socket is not None or (
                _replacement_requires_fresh_n0_server(
                    last_receipt.get("exception_code")
                )
            )
            if not requires_fresh_server:
                remaining_unattempted_count = len(selected) - next_index - 1
                continue
            replacement_requires_fresh_server = True
            remaining_unattempted_count = len(selected) - next_index - 1
            break
    target_complete = valid_count == target
    candidate_pool_exhausted = (
        not target_complete
        and not stopped_for_budget
        and not replacement_requires_fresh_server
        and fatal_error is None
        and remaining_unattempted_count == 0
    )
    summary = {
        "action_execution_contract": action_execution_contract,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": manifest_sha256,
        "capture_profile": capture_profile.value,
        "candidate_pool_exhausted": candidate_pool_exhausted,
        "completed_for_task": valid_count,
        "fatal_error": fatal_error,
        "last_attempt": last_receipt,
        "new_trials_attempted": new_count,
        "recorded_failures_for_task": replacement_count,
        "remaining_unattempted_candidates": remaining_unattempted_count,
        "replacement_requires_fresh_server": replacement_requires_fresh_server,
        "selected_task": args.task,
        "stopped_for_budget": stopped_for_budget,
        "stopped_on_infrastructure_failure": fatal_error is not None,
        "target_complete": target_complete,
        "target_valid_trials_for_task": target,
        "total_planned_for_task": len(selected),
        "unused_reserve_for_task": unused_reserve_count,
    }
    if fatal_error is not None:
        return summary, 2
    if replacement_requires_fresh_server:
        return summary, 3
    if candidate_pool_exhausted:
        return summary, 2
    if stopped_for_budget:
        return summary, 4
    return summary, 0


def run_campaign(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    manifest_path = args.manifest.absolute()
    try:
        manifest_relative = manifest_path.relative_to(layout.root)
    except ValueError as error:
        raise ValueError(
            "campaign manifest must remain below deployment root"
        ) from error
    if not manifest_relative.parts or manifest_relative.parts[0] != "requests":
        raise ValueError("campaign manifest must remain below requests/")
    manifest = load_clean_campaign_manifest(manifest_path)
    if manifest.policy_kind is not LivePolicyKind.N0:
        raise ValueError("N0 campaign runner requires an N0 policy manifest")
    integration_config = _optional_config(layout.root, args.config)
    worker_socket = _require_worker_socket(layout.root, args.worker_socket)
    capture_profile = LiveCaptureProfile(args.capture_profile)
    if (
        manifest.protocol_id
        in {CleanCampaignProtocol.PILOT, CleanCampaignProtocol.PAPER}
        and capture_profile is not LiveCaptureProfile.PAPER_FULL
    ):
        raise ValueError("pilot/paper campaigns require paper_full_v1 capture")
    if (
        worker_socket is not None
        and capture_profile is not LiveCaptureProfile.PAPER_FULL
    ):
        raise ValueError("light capture profiles do not support same_task_worker_v1")
    action_execution_contract = _required_action_execution_contract(
        manifest.protocol_id,
        args.action_execution_contract,
    )
    campaign_id = manifest.campaign_id
    entries = manifest.trials
    selected = tuple(item for item in entries if item.task == args.task)
    if not selected:
        raise ValueError("selected task is absent from the frozen campaign")
    if manifest.protocol_id in {
        CleanCampaignProtocol.PILOT,
        CleanCampaignProtocol.PAPER,
    }:
        action_horizon = load_univtac_task_registry().task(args.task).action_horizon
        for entry in selected:
            request_path, _ = _validate_entry(layout.root, entry)
            request = load_live_univtac_request(request_path)
            validate_official_n0_clean_claim_request(
                request,
                action_horizon=action_horizon,
            )
    if args.max_new_trials is not None and args.max_new_trials < 0:
        raise ValueError("max_new_trials must be non-negative")
    isaac_python = (
        args.isaac_python or layout.runtime / "isaac-sim-4.5.0/python.sh"
    ).absolute()
    n0_source_root = (args.n0_source_root or layout.sources / "N0-TWAM").absolute()
    if isaac_python.is_symlink() or not isaac_python.is_file():
        raise ValueError("Isaac Python launcher is unavailable")
    if n0_source_root.is_symlink() or not n0_source_root.is_dir():
        raise ValueError("N0 source root is unavailable")
    if not 1 <= args.n0_port <= 65535:
        raise ValueError("n0_port is out of range")
    if args.hard_lifecycle_timeout_s is not None:
        require_positive_finite(
            args.hard_lifecycle_timeout_s, "hard_lifecycle_timeout_s"
        )
    require_positive_finite(args.watchdog_term_grace_s, "watchdog_term_grace_s")

    source_bound_context = None
    if source_bound_arguments_requested(
        args.qualification,
        args.n0_server_attestation,
        args.n0_server_attestation_sha256,
    ):
        if manifest.semantic_version != "2.0":
            raise ValueError("source-bound attempts require an official v2 campaign")
        assert args.qualification is not None
        assert args.n0_server_attestation is not None
        assert args.n0_server_attestation_sha256 is not None
        source_bound_context = load_source_bound_campaign_context(
            deployment_root=layout.root,
            campaign_id=campaign_id,
            campaign_manifest_sha256=manifest.sha256,
            task_id=args.task,
            qualification_path=args.qualification,
            n0_server_attestation_path=args.n0_server_attestation,
            n0_server_attestation_sha256=args.n0_server_attestation_sha256,
        )
        _require_source_bound_execution_contract(
            source_bound_context,
            action_execution_contract,
        )
    if worker_socket is not None and source_bound_context is None:
        raise ValueError("same-task worker mode requires source-bound campaign inputs")

    if manifest.semantic_version == "2.0":
        return _run_official_task(
            args=args,
            layout=layout,
            manifest=manifest,
            selected=selected,
            isaac_python=isaac_python,
            n0_source_root=n0_source_root,
            integration_config=integration_config,
            action_execution_contract=action_execution_contract,
            source_bound_context=source_bound_context,
            worker_socket=worker_socket,
            capture_profile=capture_profile,
        )

    completed_count = 0
    recorded_failure_count = 0
    new_count = 0
    stopped = False
    last_receipt: dict[str, object] | None = None
    for entry in selected:
        request_path, artifact_path = _validate_entry(layout.root, entry)
        inspection = _inspect_artifact(entry, artifact_path)
        artifact = inspection.artifact
        if (
            artifact is not None
            and inspection.capture_profile is not None
            and inspection.capture_profile is not capture_profile
        ):
            raise ValueError("existing artifact uses another capture profile")
        attempts = _prior_attempts(
            layout=layout,
            campaign_id=campaign_id,
            campaign_manifest_sha256=manifest.sha256,
            entry=entry,
        )
        if artifact is not None:
            if not _has_successful_attempt(attempts, artifact):
                raise ValueError(
                    "live artifact has no matching successful campaign attempt"
                )
            completed_count += 1
            continue
        if attempts:
            recorded_failure_count += 1
            if not args.continue_on_infrastructure_failure:
                stopped = True
                break
            continue
        if args.max_new_trials is not None and new_count >= args.max_new_trials:
            break
        disposition, last_receipt = _execute_entry(
            layout=layout,
            campaign_id=campaign_id,
            campaign_manifest_sha256=manifest.sha256,
            entry=entry,
            request_path=request_path,
            artifact_path=artifact_path,
            isaac_python=isaac_python,
            n0_source_root=n0_source_root,
            n0_host=args.n0_host,
            n0_port=args.n0_port,
            integration_config=integration_config,
            hard_lifecycle_timeout_s=args.hard_lifecycle_timeout_s,
            watchdog_term_grace_s=args.watchdog_term_grace_s,
            action_execution_contract=action_execution_contract,
            capture_profile=capture_profile,
        )
        new_count += 1
        if disposition == _VALID_OUTCOME:
            completed_count += 1
        elif not args.continue_on_infrastructure_failure:
            stopped = True
            break
    summary = {
        "action_execution_contract": action_execution_contract,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": manifest.sha256,
        "capture_profile": capture_profile.value,
        "completed_for_task": completed_count,
        "last_attempt": last_receipt,
        "new_trials_attempted": new_count,
        "recorded_failures_for_task": recorded_failure_count,
        "selected_task": args.task,
        "stopped_on_infrastructure_failure": stopped,
        "total_planned_for_task": len(selected),
    }
    return summary, 2 if stopped else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, exit_code = run_campaign(args)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
