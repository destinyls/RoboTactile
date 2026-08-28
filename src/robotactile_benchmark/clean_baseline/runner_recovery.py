"""Crash-safe state recovery for the official Clean campaign runner."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from robotactile_benchmark.clean_baseline.contracts import CleanCampaignTrialSpec
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution import load_live_univtac_artifact
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.lifecycle_watchdog import (
    LifecycleIdentity,
    VerifiedCloseAfterExportTimeout,
    verify_close_after_export_timeout,
)
from robotactile_benchmark.trials import Condition

VALID_OUTCOME = "valid_outcome"
EXCEPTION_REPLACED = "exception_replaced"
FATAL_UNATTEMPTED = "fatal_unattempted"
INITIAL_STATE_REJECTED = "initial_state_rejected"
CLOSE_AFTER_EXPORT_HARD_TIMEOUT = "close_after_export_hard_timeout"
_TRUSTED_EPISODE_STAGES = frozenset(
    {
        "reset",
        "policy_reset",
        "observe",
        "delivery",
        "infer",
        "execute",
        "commit",
        "validation",
        "artifact_export",
        "close",
    }
)
_CRASH_MARKER = re.compile(
    rb"^ROBOTACTILE_CLOSED_LOOP_CRASH "
    rb"stage=(?P<stage>[a-z_]+) "
    rb"exception_type=(?P<exception_type>[A-Za-z0-9_.]+) "
    rb"failure_code=(?P<failure_code>[a-z0-9_]+) "
    rb"system_exit_code=[^\r\n]+$",
    re.MULTILINE,
)


class EntryValidator(Protocol):
    def __call__(
        self, root: Path, entry: CleanCampaignTrialSpec
    ) -> tuple[Path, Path]: ...


class PriorAttemptsLoader(Protocol):
    def __call__(
        self,
        *,
        layout: DeploymentLayout,
        campaign_id: str,
        campaign_manifest_sha256: str,
        entry: CleanCampaignTrialSpec,
        semantic_version: str = "1.0",
    ) -> tuple[Mapping[str, object], ...]: ...


@dataclass(frozen=True)
class RunnerStageEvidence:
    stage: str
    failure_code: str
    exception_type: str


@dataclass(frozen=True)
class ArtifactInspection:
    artifact: Mapping[str, object] | None
    error_type: str | None
    stage_evidence: RunnerStageEvidence | None
    capture_profile: LiveCaptureProfile | None = None


@dataclass(frozen=True)
class CandidateState:
    entry: CleanCampaignTrialSpec
    request_path: Path
    artifact_path: Path
    artifact: Mapping[str, object] | None
    artifact_error: str | None
    stage_evidence: RunnerStageEvidence | None
    attempts: tuple[Mapping[str, object], ...]
    disposition: str | None
    adoption_log: Path | None
    candidate_logs: tuple[Path, ...]
    capture_profile: LiveCaptureProfile | None = None


@dataclass(frozen=True)
class OfficialPreflight:
    states: tuple[CandidateState, ...]
    valid_count: int
    replacement_count: int
    unused_reserve_count: int
    next_index: int | None
    adoptions: tuple[CandidateState, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_artifact(
    entry: CleanCampaignTrialSpec, path: Path
) -> tuple[
    dict[str, object] | None,
    RunnerStageEvidence | None,
    LiveCaptureProfile | None,
]:
    if not path.exists():
        return None, None, None
    loaded = load_live_univtac_artifact(path)
    if (
        loaded.trial.task != entry.task
        or loaded.trial.initial_seed != entry.initial_seed
        or loaded.trial.exogenous_seed != entry.exogenous_seed
        or loaded.trial.sha256 != entry.trial_manifest_sha256
        or loaded.trial.condition is not Condition.CLEAN
    ):
        raise ValueError("existing live artifact does not match the frozen trial")
    result = loaded.evidence.result
    artifact: dict[str, object] = {
        "artifact_root_sha256": loaded.external_root_sha256,
        "execution_status": (
            None if result.execution_status is None else result.execution_status.value
        ),
        "score_eligible": result.score_eligible,
        "score_success": result.score_success,
        "terminal_status": result.terminal_status.value,
    }
    stage_evidence: RunnerStageEvidence | None = None
    if result.terminal_status.value == "crash":
        failure = loaded.evidence.initial_diagnostics.get("runner_failure")
        exception_type = "artifact_contract"
        if isinstance(failure, Mapping):
            raw_exception_type = failure.get("exception_type")
            if isinstance(raw_exception_type, str) and raw_exception_type:
                exception_type = raw_exception_type
        if result.failure_stage is None or result.failure_code is None:
            raise ValueError("crash artifact lacks runner stage evidence")
        stage_evidence = RunnerStageEvidence(
            stage=result.failure_stage,
            failure_code=result.failure_code,
            exception_type=exception_type,
        )
    return artifact, stage_evidence, loaded.capture_profile


def existing_artifact(
    entry: CleanCampaignTrialSpec, path: Path
) -> dict[str, object] | None:
    artifact, _, _ = _load_existing_artifact(entry, path)
    return artifact


def inspect_artifact(entry: CleanCampaignTrialSpec, path: Path) -> ArtifactInspection:
    try:
        artifact, stage_evidence, capture_profile = _load_existing_artifact(entry, path)
    except (OSError, TypeError, ValueError) as error:
        return ArtifactInspection(None, type(error).__name__, None)
    return ArtifactInspection(artifact, None, stage_evidence, capture_profile)


def runner_stage_evidence(log_path: Path) -> RunnerStageEvidence | None:
    if log_path.is_symlink() or not log_path.is_file():
        raise ValueError("candidate log must be a regular non-symlink file")
    last_match: re.Match[bytes] | None = None
    with log_path.open("rb") as stream:
        for line in stream:
            match = _CRASH_MARKER.match(line.rstrip(b"\r\n"))
            if match is not None:
                last_match = match
    if last_match is None:
        return None
    return RunnerStageEvidence(
        stage=last_match.group("stage").decode("ascii"),
        failure_code=last_match.group("failure_code").decode("ascii"),
        exception_type=last_match.group("exception_type").decode("ascii"),
    )


def candidate_disposition(
    *,
    return_code: int,
    artifact: Mapping[str, object] | None,
    artifact_error: str | None,
    stage_evidence: RunnerStageEvidence | None = None,
    close_after_export_timeout: bool = False,
) -> tuple[str, str | None]:
    """Classify only evidence known to be inside the official episode try."""

    if (
        return_code == 0
        and artifact is not None
        and artifact_error is None
        and artifact.get("terminal_status") != "crash"
        and artifact.get("score_eligible") is True
    ):
        return VALID_OUTCOME, None
    if (
        close_after_export_timeout
        and return_code != 0
        and artifact is not None
        and artifact_error is None
        and artifact.get("terminal_status") != "crash"
    ):
        return EXCEPTION_REPLACED, CLOSE_AFTER_EXPORT_HARD_TIMEOUT
    if artifact is not None and artifact.get("terminal_status") != "crash":
        raise ValueError("non-crash ineligible artifact is not a replaceable exception")
    if artifact is not None:
        if (
            stage_evidence is not None
            and stage_evidence.failure_code == "invalid_initial_state"
        ):
            return EXCEPTION_REPLACED, INITIAL_STATE_REJECTED
        return EXCEPTION_REPLACED, "live_artifact_crash"
    if stage_evidence is not None and stage_evidence.stage in _TRUSTED_EPISODE_STAGES:
        return (
            EXCEPTION_REPLACED,
            f"runner_stage_{stage_evidence.stage}_{stage_evidence.failure_code}",
        )
    if artifact_error is not None:
        return FATAL_UNATTEMPTED, "invalid_live_artifact"
    if return_code != 0:
        return FATAL_UNATTEMPTED, f"live_command_return_code_{return_code}"
    return FATAL_UNATTEMPTED, "missing_live_artifact"


def has_successful_attempt(
    attempts: Sequence[Mapping[str, object]], artifact: Mapping[str, object]
) -> bool:
    """Return whether legacy/v2 receipts bind one successful artifact."""

    return any(
        attempt["return_code"] == 0
        and attempt["artifact_validation_error_type"] is None
        and attempt["artifact"] == artifact
        and (
            attempt["semantic_version"] == "1.0"
            or attempt.get("candidate_disposition") == VALID_OUTCOME
        )
        for attempt in attempts
    )


def replacement_requires_fresh_n0_server(exception_code: object) -> bool:
    """Keep one untouched N0 server only for reset-time seed rejection."""

    return exception_code != INITIAL_STATE_REJECTED


def official_attempt_disposition(
    attempts: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object] | None,
    artifact_error: str | None = None,
    *,
    layout: DeploymentLayout | None = None,
    campaign_id: str | None = None,
    campaign_manifest_sha256: str | None = None,
    entry: CleanCampaignTrialSpec | None = None,
) -> str | None:
    if not attempts:
        return None
    if len(attempts) != 1:
        raise ValueError("official candidate must have at most one attempt receipt")
    attempt = attempts[0]
    disposition = attempt.get("candidate_disposition")
    if disposition not in {VALID_OUTCOME, EXCEPTION_REPLACED}:
        raise ValueError("official candidate disposition is invalid")
    if artifact != attempt["artifact"]:
        raise ValueError("official attempt artifact binding mismatch")
    bound_error = attempt["artifact_validation_error_type"]
    if artifact_error != bound_error:
        raise ValueError("official attempt artifact error binding mismatch")
    return_code = attempt["return_code"]
    exception_code = attempt.get("exception_code")
    if disposition == VALID_OUTCOME:
        if (
            return_code != 0
            or bound_error is not None
            or artifact is None
            or artifact.get("terminal_status") == "crash"
            or artifact.get("score_eligible") is not True
            or exception_code is not None
        ):
            raise ValueError("valid outcome receipt is internally inconsistent")
        return VALID_OUTCOME
    if not isinstance(exception_code, str) or not exception_code:
        raise ValueError("replacement receipt requires an exception code")
    if artifact is not None and artifact.get("terminal_status") != "crash":
        if exception_code != CLOSE_AFTER_EXPORT_HARD_TIMEOUT:
            raise ValueError("non-crash artifact cannot be replaced as an exception")
        if (
            layout is None
            or campaign_id is None
            or campaign_manifest_sha256 is None
            or entry is None
        ):
            raise ValueError("close-timeout replacement lacks campaign identity")
        proof = _close_timeout_proof(
            layout, campaign_id, campaign_manifest_sha256, entry, attempt["log_relpath"]
        )
        if (
            proof.return_code != return_code
            or proof.log_sha256 != attempt["log_sha256"]
            or proof.started_at_utc != attempt["started_at_utc"]
            or proof.finished_at_utc != attempt["finished_at_utc"]
            or proof.duration_s != attempt["duration_s"]
        ):
            raise ValueError("close-timeout attempt/watchdog binding mismatch")
    if return_code == 0 and artifact is None and bound_error is None:
        raise ValueError("replacement receipt lacks exception evidence")
    return EXCEPTION_REPLACED


def _candidate_logs(
    layout: DeploymentLayout, campaign_id: str, entry: CleanCampaignTrialSpec
) -> tuple[Path, ...]:
    directory = layout.logs / "clean-campaigns" / campaign_id / entry.task
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("candidate log directory must be a real directory")
    paths = tuple(sorted(directory.glob(f"{entry.ordinal:04d}-*.log")))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("candidate logs must be regular non-symlink files")
    return paths


def _log_matches_artifact(
    log_path: Path,
    entry: CleanCampaignTrialSpec,
    artifact: Mapping[str, object],
) -> bool:
    matched = False
    with log_path.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.lstrip().startswith(b"{"):
                continue
            try:
                document = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(document, Mapping):
                continue
            current = (
                document.get("artifact_root_sha256")
                == artifact.get("artifact_root_sha256")
                and document.get("condition") == "clean"
                and document.get("task_id") == entry.task
                and document.get("terminal_status") == artifact.get("terminal_status")
            )
            if current and matched:
                raise ValueError("candidate log repeats its artifact summary")
            matched = matched or current
    return matched


def _adoption_log(
    logs: Sequence[Path],
    entry: CleanCampaignTrialSpec,
    artifact: Mapping[str, object],
) -> Path | None:
    matches = tuple(
        path for path in logs if _log_matches_artifact(path, entry, artifact)
    )
    if len(matches) > 1:
        raise ValueError("multiple candidate logs claim the same live artifact")
    return None if not matches else matches[0]


def _close_timeout_proof(
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    entry: CleanCampaignTrialSpec,
    log_relpath: object,
) -> VerifiedCloseAfterExportTimeout:
    if not isinstance(log_relpath, str):
        raise ValueError("close-timeout log path must be a string")
    log_path = layout.root / log_relpath
    prefix = f"{entry.ordinal:04d}-"
    if not log_path.stem.startswith(prefix):
        raise ValueError("close-timeout log name does not match candidate ordinal")
    identity = LifecycleIdentity(
        campaign_manifest_sha256=campaign_manifest_sha256,
        request_file_sha256=entry.request_file_sha256,
        trial_manifest_sha256=entry.trial_manifest_sha256,
        task_id=entry.task,
        ordinal=entry.ordinal,
        attempt_id=log_path.stem[len(prefix) :],
    )
    return verify_close_after_export_timeout(
        root=layout.root,
        campaign_id=campaign_id,
        identity=identity,
        log_path=log_path,
    )


def _close_timeout_adoption_log(
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    entry: CleanCampaignTrialSpec,
    logs: Sequence[Path],
) -> Path | None:
    matches: list[Path] = []
    for log in logs:
        try:
            _close_timeout_proof(
                layout,
                campaign_id,
                campaign_manifest_sha256,
                entry,
                log.relative_to(layout.root).as_posix(),
            )
        except (OSError, TypeError, ValueError):
            continue
        matches.append(log)
    if len(matches) > 1:
        raise ValueError("multiple close-timeout logs claim one candidate")
    return None if not matches else matches[0]


def _trusted_log_evidence(
    logs: Sequence[Path],
) -> tuple[Path, RunnerStageEvidence] | None:
    matches = tuple(
        (path, evidence)
        for path in logs
        if (evidence := runner_stage_evidence(path)) is not None
        and evidence.stage in _TRUSTED_EPISODE_STAGES
    )
    if len(matches) > 1:
        raise ValueError(
            "multiple logs contain episode-stage evidence for one candidate"
        )
    return None if not matches else matches[0]


def preflight_official_candidates(
    *,
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    selected: Sequence[CleanCampaignTrialSpec],
    target: int,
    validate_entry: EntryValidator,
    prior_attempts: PriorAttemptsLoader,
    inspect_artifact_fn: Callable[
        [CleanCampaignTrialSpec, Path], ArtifactInspection
    ] = inspect_artifact,
    candidate_logs_fn: Callable[
        [DeploymentLayout, str, CleanCampaignTrialSpec], tuple[Path, ...]
    ] = _candidate_logs,
    trusted_log_evidence_fn: Callable[
        [Sequence[Path]], tuple[Path, RunnerStageEvidence] | None
    ] = _trusted_log_evidence,
) -> OfficialPreflight:
    states: list[CandidateState] = []
    adoptions: list[CandidateState] = []
    for entry in selected:
        request_path, artifact_path = validate_entry(layout.root, entry)
        inspection = inspect_artifact_fn(entry, artifact_path)
        attempts = prior_attempts(
            layout=layout,
            campaign_id=campaign_id,
            campaign_manifest_sha256=campaign_manifest_sha256,
            entry=entry,
            semantic_version="2.0",
        )
        logs = candidate_logs_fn(layout, campaign_id, entry)
        trusted_log = trusted_log_evidence_fn(logs)
        stage_evidence = inspection.stage_evidence
        if stage_evidence is None and trusted_log is not None:
            stage_evidence = trusted_log[1]
        disposition = official_attempt_disposition(
            attempts,
            inspection.artifact,
            inspection.error_type,
            layout=layout,
            campaign_id=campaign_id,
            campaign_manifest_sha256=campaign_manifest_sha256,
            entry=entry,
        )
        adoption_log = None
        if not attempts and inspection.artifact is not None:
            adoption_log = _adoption_log(logs, entry, inspection.artifact)
            if adoption_log is None:
                adoption_log = _close_timeout_adoption_log(
                    layout,
                    campaign_id,
                    campaign_manifest_sha256,
                    entry,
                    logs,
                )
            if adoption_log is not None:
                if _log_matches_artifact(adoption_log, entry, inspection.artifact):
                    disposition, _ = candidate_disposition(
                        return_code=0,
                        artifact=inspection.artifact,
                        artifact_error=None,
                        stage_evidence=inspection.stage_evidence,
                    )
                else:
                    disposition = EXCEPTION_REPLACED
        state = CandidateState(
            entry=entry,
            request_path=request_path,
            artifact_path=artifact_path,
            artifact=inspection.artifact,
            artifact_error=inspection.error_type,
            stage_evidence=stage_evidence,
            attempts=attempts,
            disposition=disposition,
            adoption_log=adoption_log,
            candidate_logs=logs,
            capture_profile=inspection.capture_profile,
        )
        states.append(state)
        if adoption_log is not None:
            adoptions.append(state)
    return _validate_sequence(tuple(states), target, tuple(adoptions))


def _validate_sequence(
    states: tuple[CandidateState, ...],
    target: int,
    adoptions: tuple[CandidateState, ...],
) -> OfficialPreflight:
    valid_count = replacement_count = unused_reserve_count = 0
    next_index: int | None = None
    for index, state in enumerate(states):
        orphan_stage = (
            not state.attempts
            and state.disposition is None
            and state.stage_evidence is not None
            and state.stage_evidence.stage in _TRUSTED_EPISODE_STAGES
        )
        unbound_artifact = (
            not state.attempts
            and state.disposition is None
            and (state.artifact is not None or state.artifact_error is not None)
        )
        orphan_command_log = (
            not state.attempts
            and state.disposition is None
            and bool(state.candidate_logs)
        )
        if valid_count >= target:
            if (
                state.disposition is not None
                or orphan_stage
                or unbound_artifact
                or orphan_command_log
            ):
                raise ValueError("official campaign attempted a candidate after target")
            unused_reserve_count += 1
        elif state.disposition is not None:
            if next_index is not None:
                raise ValueError(
                    "official campaign has an attempted candidate after a gap"
                )
            if state.disposition == VALID_OUTCOME:
                valid_count += 1
            elif state.disposition == EXCEPTION_REPLACED:
                replacement_count += 1
            else:
                raise ValueError("fatal candidate state cannot consume a seed")
        elif unbound_artifact:
            raise ValueError(
                "live artifact cannot be adopted without matching log evidence"
            )
        elif orphan_stage:
            raise ValueError(
                "episode-stage log evidence lacks an immutable attempt receipt"
            )
        elif orphan_command_log:
            raise ValueError(
                "candidate command log lacks immutable outcome evidence; refusing rerun"
            )
        elif next_index is None:
            next_index = index
    return OfficialPreflight(
        states,
        valid_count,
        replacement_count,
        unused_reserve_count,
        next_index,
        adoptions,
    )


def adoption_receipt(
    *,
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    state: CandidateState,
) -> tuple[Path, dict[str, object]]:
    if state.artifact is None or state.adoption_log is None:
        raise ValueError("adoption requires a strict artifact and matching log")
    normal_adoption = _log_matches_artifact(
        state.adoption_log, state.entry, state.artifact
    )
    if normal_adoption:
        disposition, exception_code = candidate_disposition(
            return_code=0,
            artifact=state.artifact,
            artifact_error=None,
            stage_evidence=state.stage_evidence,
        )
        return_code = 0
        started_at = finished_at = _timestamp_from_attempt_id(state.adoption_log)
        duration_s = 0.0
        log_sha256 = _sha256_file(state.adoption_log)
    else:
        proof = _close_timeout_proof(
            layout,
            campaign_id,
            campaign_manifest_sha256,
            state.entry,
            state.adoption_log.relative_to(layout.root).as_posix(),
        )
        disposition = EXCEPTION_REPLACED
        exception_code = CLOSE_AFTER_EXPORT_HARD_TIMEOUT
        return_code = proof.return_code
        started_at = proof.started_at_utc
        finished_at = proof.finished_at_utc
        duration_s = proof.duration_s
        log_sha256 = proof.log_sha256
    artifact_sha256 = state.artifact.get("artifact_root_sha256")
    if not isinstance(artifact_sha256, str):
        raise ValueError("adopted artifact lacks its canonical root hash")
    prefix = f"{state.entry.ordinal:04d}-"
    if not state.adoption_log.stem.startswith(prefix):
        raise ValueError("adoption log name does not match candidate ordinal")
    document: dict[str, object] = {
        "artifact": state.artifact,
        "artifact_validation_error_type": None,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "candidate_disposition": disposition,
        "command_kind": "robotactile_live_univtac_run_v1",
        "duration_s": duration_s,
        "exception_code": exception_code,
        "finished_at_utc": finished_at,
        "log_relpath": state.adoption_log.relative_to(layout.root).as_posix(),
        "log_sha256": log_sha256,
        "ordinal": state.entry.ordinal,
        "request_file_sha256": state.entry.request_file_sha256,
        "return_code": return_code,
        "semantic_version": "2.0",
        "started_at_utc": started_at,
        "task_id": state.entry.task,
        "trial_manifest_sha256": state.entry.trial_manifest_sha256,
    }
    path = (
        layout.outputs
        / "clean-campaigns"
        / campaign_id
        / "attempts"
        / state.entry.task
        / f"{state.entry.ordinal:04d}-adopted-{artifact_sha256[:16]}.json"
    )
    return path, document


def _timestamp_from_attempt_id(log_path: Path) -> str:
    token = log_path.stem.split("-", 1)[1]
    try:
        return (
            datetime.strptime(token, "%Y%m%dT%H%M%S.%fZ")
            .replace(tzinfo=timezone.utc)
            .isoformat()
        )
    except ValueError as error:
        raise ValueError("adoption log lacks a canonical attempt timestamp") from error


__all__ = [
    "EXCEPTION_REPLACED",
    "CLOSE_AFTER_EXPORT_HARD_TIMEOUT",
    "FATAL_UNATTEMPTED",
    "INITIAL_STATE_REJECTED",
    "VALID_OUTCOME",
    "CandidateState",
    "RunnerStageEvidence",
    "adoption_receipt",
    "candidate_disposition",
    "has_successful_attempt",
    "existing_artifact",
    "inspect_artifact",
    "official_attempt_disposition",
    "preflight_official_candidates",
    "replacement_requires_fresh_n0_server",
    "runner_stage_evidence",
]
