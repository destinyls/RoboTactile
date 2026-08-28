"""Crash-safe lifecycle stage journals and hard-watchdog receipts."""

from __future__ import annotations

import hashlib
import math
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)

LIFECYCLE_STAGES = frozenset(
    {
        "process_spawn",
        "request_loading",
        "source_binding",
        "runtime_attestation",
        "capability_preflight",
        "run_loading",
        "backend_construction",
        "dependency_preflight",
        "app_launcher_import",
        "app_launcher",
        "task_reconstruction_start",
        "gc_collected",
        "usd_stage_recreated",
        "task_reconstruction_ready",
        "runtime_preparation",
        "tactile_constructor_hook",
        "univtac_task_construction",
        "tactile_attachment_validation",
        "runtime_compatibility_installation",
        "runtime_ready",
        "backend_ready",
        "policy_construction",
        "startup_close",
        "closed_loop_preflight",
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
        "task_teardown_start",
        "renderer_released",
        "timeline_stopped",
        "task_closed",
        "execution_completed",
        "artifact_verified",
        "completed",
    }
)
TRUSTED_EPISODE_STAGES = frozenset(
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
        "task_teardown_start",
        "renderer_released",
        "timeline_stopped",
        "task_closed",
        "execution_completed",
        "artifact_verified",
        "completed",
    }
)
DEFAULT_STARTUP_TEARDOWN_GRACE_S = 600.0
_HEX = frozenset("0123456789abcdef")
_IDENTITY_FIELDS = frozenset(
    {
        "attempt_id",
        "campaign_manifest_sha256",
        "ordinal",
        "request_file_sha256",
        "semantic_version",
        "task_id",
        "trial_manifest_sha256",
    }
)
_STAGE_FIELDS = frozenset(
    {
        "identity",
        "previous_record_sha256",
        "process_group_id",
        "process_id",
        "recorded_at_utc",
        "semantic_version",
        "sequence",
        "stage",
    }
)
_WATCHDOG_FIELDS = frozenset(
    {
        "duration_s",
        "finished_at_utc",
        "hard_lifecycle_timeout_s",
        "hard_timeout_source",
        "identity",
        "journal_last_record_sha256",
        "journal_last_sequence",
        "journal_last_stage",
        "journal_relpath",
        "kill_signal_sent",
        "log_relpath",
        "log_sha256",
        "process_group_id",
        "process_id",
        "return_code",
        "semantic_version",
        "soft_wall_timeout_s",
        "started_at_utc",
        "startup_teardown_grace_s",
        "term_grace_s",
        "termination_signal_sent",
        "timed_out",
        "watchdog_error_type",
    }
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _require_token(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a non-empty filesystem-safe token")
    return value


@dataclass(frozen=True)
class LifecycleIdentity:
    """Frozen campaign/candidate identity shared by parent and child."""

    campaign_manifest_sha256: str
    request_file_sha256: str
    trial_manifest_sha256: str
    task_id: str
    ordinal: int
    attempt_id: str
    semantic_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.semantic_version != "1.0":
            raise ValueError("lifecycle identity semantic version mismatch")
        for name in (
            "campaign_manifest_sha256",
            "request_file_sha256",
            "trial_manifest_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        object.__setattr__(self, "task_id", _require_token(self.task_id, "task_id"))
        object.__setattr__(
            self, "attempt_id", _require_token(self.attempt_id, "attempt_id")
        )
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("ordinal must be an integer")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "ordinal": self.ordinal,
            "request_file_sha256": self.request_file_sha256,
            "semantic_version": self.semantic_version,
            "task_id": self.task_id,
            "trial_manifest_sha256": self.trial_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> LifecycleIdentity:
        if not isinstance(value, dict) or set(value) != _IDENTITY_FIELDS:
            raise ValueError("lifecycle identity fields mismatch")
        return cls(
            campaign_manifest_sha256=value["campaign_manifest_sha256"],
            request_file_sha256=value["request_file_sha256"],
            trial_manifest_sha256=value["trial_manifest_sha256"],
            task_id=value["task_id"],
            ordinal=value["ordinal"],
            attempt_id=value["attempt_id"],
            semantic_version=value["semantic_version"],
        )


@dataclass(frozen=True)
class LifecycleStageReceipt:
    """One immutable hash-chained stage entry."""

    identity: LifecycleIdentity
    sequence: int
    stage: str
    previous_record_sha256: Optional[str]
    recorded_at_utc: str
    process_id: int
    process_group_id: int
    semantic_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.semantic_version != "1.0":
            raise ValueError("lifecycle stage semantic version mismatch")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("lifecycle sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("lifecycle sequence must be non-negative")
        if self.stage not in LIFECYCLE_STAGES:
            raise ValueError("unknown lifecycle stage")
        if self.sequence == 0:
            if self.previous_record_sha256 is not None:
                raise ValueError("first lifecycle stage cannot have a previous hash")
        else:
            _require_sha256(self.previous_record_sha256, "previous_record_sha256")
        for value, name in (
            (self.process_id, "process_id"),
            (self.process_group_id, "process_group_id"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            parsed = datetime.fromisoformat(self.recorded_at_utc)
        except ValueError as error:
            raise ValueError("recorded_at_utc must be ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError("recorded_at_utc must include a timezone")

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "previous_record_sha256": self.previous_record_sha256,
            "process_group_id": self.process_group_id,
            "process_id": self.process_id,
            "recorded_at_utc": self.recorded_at_utc,
            "semantic_version": self.semantic_version,
            "sequence": self.sequence,
            "stage": self.stage,
        }

    @property
    def sha256(self) -> str:
        return _sha256(canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> LifecycleStageReceipt:
        if not isinstance(value, dict) or set(value) != _STAGE_FIELDS:
            raise ValueError("lifecycle stage fields mismatch")
        return cls(
            identity=LifecycleIdentity.from_dict(value["identity"]),
            sequence=value["sequence"],
            stage=value["stage"],
            previous_record_sha256=value["previous_record_sha256"],
            recorded_at_utc=value["recorded_at_utc"],
            process_id=value["process_id"],
            process_group_id=value["process_group_id"],
            semantic_version=value["semantic_version"],
        )


@dataclass(frozen=True)
class LifecycleStageJournal:
    """Directory-backed no-clobber journal safe across abrupt child exits."""

    path: Path
    identity: LifecycleIdentity

    @classmethod
    def create(cls, path: Path, identity: LifecycleIdentity) -> LifecycleStageJournal:
        selected = Path(path).absolute()
        selected.parent.mkdir(parents=True, exist_ok=True)
        if selected.parent.is_symlink() or not selected.parent.is_dir():
            raise ValueError("lifecycle journal parent must be a real directory")
        selected.mkdir()
        return cls(selected, identity)

    @classmethod
    def open(cls, path: Path) -> LifecycleStageJournal:
        selected = Path(path).absolute()
        records = _load_records(selected)
        if not records:
            raise ValueError("lifecycle journal is empty")
        return cls(selected, records[0].identity)

    def records(self) -> tuple[LifecycleStageReceipt, ...]:
        records = _load_records(self.path)
        if records and records[0].identity != self.identity:
            raise ValueError("lifecycle journal identity changed")
        return records

    def last_record(self) -> Optional[LifecycleStageReceipt]:
        records = self.records()
        return None if not records else records[-1]

    def record(self, stage: str) -> LifecycleStageReceipt:
        records = self.records()
        previous = None if not records else records[-1].sha256
        receipt = LifecycleStageReceipt(
            identity=self.identity,
            sequence=len(records),
            stage=stage,
            previous_record_sha256=previous,
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            process_id=os.getpid(),
            process_group_id=os.getpgrp(),
        )
        destination = self.path / f"{receipt.sequence:06d}-{receipt.stage}.json"
        write_canonical_no_clobber(destination, receipt.to_dict())
        return receipt

    def observe(self, stage: str) -> None:
        """Record a stage through callback interfaces that return ``None``."""

        self.record(stage)


@dataclass(frozen=True)
class HardLifecycleOutcome:
    """Result of waiting for and, when necessary, killing one owned process group."""

    return_code: int
    timed_out: bool
    termination_signal_sent: bool
    kill_signal_sent: bool


@dataclass(frozen=True)
class VerifiedCloseAfterExportTimeout:
    """Durable watchdog facts for a process killed only during teardown."""

    return_code: int
    started_at_utc: str
    finished_at_utc: str
    duration_s: float
    log_sha256: str


def campaign_watchdog_paths(
    root: Path, ordinal: int, attempt_id: str
) -> tuple[Path, Path]:
    """Return deterministic sibling paths for one journal and watchdog receipt."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    token = _require_token(attempt_id, "attempt_id")
    prefix = f"{ordinal:04d}-{token}"
    return root / f"{prefix}.journal", root / f"{prefix}.watchdog.json"


def require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def resolve_hard_lifecycle_timeout(
    soft_wall_timeout_s: object, requested_hard_timeout_s: object
) -> tuple[float, str, Optional[float]]:
    """Resolve a hard budget without collapsing it into the evaluator budget."""

    soft = require_positive_finite(soft_wall_timeout_s, "soft_wall_timeout_s")
    if requested_hard_timeout_s is None:
        return (
            soft + DEFAULT_STARTUP_TEARDOWN_GRACE_S,
            "soft_plus_startup_teardown_grace",
            DEFAULT_STARTUP_TEARDOWN_GRACE_S,
        )
    hard = require_positive_finite(requested_hard_timeout_s, "hard_lifecycle_timeout_s")
    if hard <= soft:
        raise ValueError("hard_lifecycle_timeout_s must exceed soft_wall_timeout_s")
    return hard, "explicit", None


def wait_with_lifecycle_watchdog(
    process: subprocess.Popen[bytes],
    *,
    hard_timeout_s: float,
    term_grace_s: float,
) -> HardLifecycleOutcome:
    """Wait one process group, escalating TERM to KILL at the hard deadline."""

    hard_timeout_s = require_positive_finite(hard_timeout_s, "hard_timeout_s")
    term_grace_s = require_positive_finite(term_grace_s, "term_grace_s")
    process_group_id = process.pid
    try:
        return_code = process.wait(timeout=hard_timeout_s)
    except subprocess.TimeoutExpired:
        term_sent, kill_sent = _terminate_process_group(
            process, process_group_id, term_grace_s
        )
        if process.returncode is None:
            raise RuntimeError("watchdog did not reap the lifecycle process") from None
        return HardLifecycleOutcome(process.returncode, True, term_sent, kill_sent)
    if not _process_group_exists(process_group_id):
        return HardLifecycleOutcome(return_code, False, False, False)
    term_sent, kill_sent = _terminate_process_group(
        process, process_group_id, term_grace_s
    )
    return HardLifecycleOutcome(return_code, False, term_sent, kill_sent)


def watchdog_receipt_document(
    *,
    identity: LifecycleIdentity,
    last_stage_receipt: Optional[LifecycleStageReceipt],
    journal_relpath: str,
    log_relpath: str,
    log_sha256: str,
    process_id: Optional[int],
    outcome: HardLifecycleOutcome,
    started_at_utc: str,
    finished_at_utc: str,
    duration_s: float,
    hard_lifecycle_timeout_s: float,
    soft_wall_timeout_s: float,
    hard_timeout_source: str,
    startup_teardown_grace_s: Optional[float],
    term_grace_s: float,
    watchdog_error_type: Optional[str],
) -> dict[str, object]:
    """Build an identity-bound watchdog receipt without claiming an outcome."""

    return {
        "duration_s": duration_s,
        "finished_at_utc": finished_at_utc,
        "hard_lifecycle_timeout_s": hard_lifecycle_timeout_s,
        "hard_timeout_source": hard_timeout_source,
        "identity": identity.to_dict(),
        "journal_last_record_sha256": (
            None if last_stage_receipt is None else last_stage_receipt.sha256
        ),
        "journal_last_sequence": (
            None if last_stage_receipt is None else last_stage_receipt.sequence
        ),
        "journal_last_stage": (
            None if last_stage_receipt is None else last_stage_receipt.stage
        ),
        "journal_relpath": journal_relpath,
        "kill_signal_sent": outcome.kill_signal_sent,
        "log_relpath": log_relpath,
        "log_sha256": _require_sha256(log_sha256, "log_sha256"),
        "process_group_id": process_id,
        "process_id": process_id,
        "return_code": outcome.return_code,
        "semantic_version": "1.0",
        "started_at_utc": started_at_utc,
        "startup_teardown_grace_s": startup_teardown_grace_s,
        "soft_wall_timeout_s": soft_wall_timeout_s,
        "term_grace_s": term_grace_s,
        "termination_signal_sent": outcome.termination_signal_sent,
        "timed_out": outcome.timed_out,
        "watchdog_error_type": watchdog_error_type,
    }


def watchdog_proves_fatal_unattempted(
    *,
    receipt_path: Path,
    journal_path: Path,
    log_path: Path,
    identity: LifecycleIdentity,
) -> bool:
    """Validate durable evidence that a failed process never entered reset."""

    for path, name in (
        (receipt_path, "watchdog receipt"),
        (log_path, "watchdog log"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} must be a regular non-symlink file")
    value = strict_json_bytes(receipt_path.read_bytes(), "lifecycle watchdog receipt")
    if not isinstance(value, Mapping) or set(value) != _WATCHDOG_FIELDS:
        raise ValueError("lifecycle watchdog receipt fields mismatch")
    if value["identity"] != identity.to_dict():
        raise ValueError("lifecycle watchdog receipt identity mismatch")
    if value["log_sha256"] != sha256_file(log_path):
        raise ValueError("lifecycle watchdog log hash mismatch")
    return_code = value["return_code"]
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise ValueError("lifecycle watchdog return code is invalid")
    journal = LifecycleStageJournal.open(journal_path)
    if journal.identity != identity:
        raise ValueError("lifecycle watchdog journal identity mismatch")
    last = journal.last_record()
    if last is None:
        raise ValueError("lifecycle watchdog journal has no stage")
    if (
        value["journal_last_stage"] != last.stage
        or value["journal_last_sequence"] != last.sequence
        or value["journal_last_record_sha256"] != last.sha256
    ):
        raise ValueError("lifecycle watchdog last-stage binding mismatch")
    return return_code != 0 and last.stage not in TRUSTED_EPISODE_STAGES


def verify_close_after_export_timeout(
    *,
    root: Path,
    campaign_id: str,
    identity: LifecycleIdentity,
    log_path: Path,
) -> VerifiedCloseAfterExportTimeout:
    """Verify the sole non-crash-artifact exception replacement whitelist."""

    deployment_root = Path(root).absolute()
    campaign = _require_token(campaign_id, "campaign_id")
    expected_log = (
        deployment_root
        / "logs"
        / "clean-campaigns"
        / campaign
        / identity.task_id
        / f"{identity.ordinal:04d}-{identity.attempt_id}.log"
    )
    selected_log = Path(log_path).absolute()
    if selected_log != expected_log:
        raise ValueError("close-timeout log path does not match lifecycle identity")
    lifecycle_root = (
        deployment_root
        / "outputs"
        / "clean-campaigns"
        / campaign
        / "lifecycle"
        / identity.task_id
    )
    journal_path, receipt_path = campaign_watchdog_paths(
        lifecycle_root, identity.ordinal, identity.attempt_id
    )
    for path, name, directory in (
        (selected_log, "close-timeout log", False),
        (receipt_path, "close-timeout watchdog receipt", False),
        (journal_path, "close-timeout lifecycle journal", True),
    ):
        if _traverses_symlink(deployment_root, path):
            raise ValueError(f"{name} traverses a symlink")
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            raise ValueError(f"{name} has the wrong filesystem type")
    value = strict_json_bytes(receipt_path.read_bytes(), "lifecycle watchdog receipt")
    if not isinstance(value, Mapping) or set(value) != _WATCHDOG_FIELDS:
        raise ValueError("lifecycle watchdog receipt fields mismatch")
    expected_journal_relpath = journal_path.relative_to(deployment_root).as_posix()
    expected_log_relpath = selected_log.relative_to(deployment_root).as_posix()
    if (
        value["identity"] != identity.to_dict()
        or value["journal_relpath"] != expected_journal_relpath
        or value["log_relpath"] != expected_log_relpath
    ):
        raise ValueError("close-timeout watchdog identity/path binding mismatch")
    log_sha256 = sha256_file(selected_log)
    if value["log_sha256"] != log_sha256:
        raise ValueError("close-timeout watchdog log hash mismatch")
    return_code = value["return_code"]
    if (
        isinstance(return_code, bool)
        or not isinstance(return_code, int)
        or return_code == 0
        or value["timed_out"] is not True
        or value["watchdog_error_type"] is not None
    ):
        raise ValueError("close-timeout watchdog does not prove a hard timeout")
    journal = LifecycleStageJournal.open(journal_path)
    records = journal.records()
    if journal.identity != identity or len(records) < 2:
        raise ValueError("close-timeout lifecycle journal identity is invalid")
    if tuple(record.stage for record in records[-2:]) != (
        "artifact_export",
        "close",
    ):
        raise ValueError("close-timeout did not follow artifact export")
    last = records[-1]
    if (
        value["journal_last_stage"] != last.stage
        or value["journal_last_sequence"] != last.sequence
        or value["journal_last_record_sha256"] != last.sha256
    ):
        raise ValueError("close-timeout watchdog last-stage binding mismatch")
    duration = value["duration_s"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        raise ValueError("close-timeout watchdog duration is invalid")
    for field_name in ("started_at_utc", "finished_at_utc"):
        timestamp = value[field_name]
        if not isinstance(timestamp, str):
            raise ValueError("close-timeout watchdog timestamp is invalid")
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            raise ValueError("close-timeout watchdog timestamp lacks timezone")
    return VerifiedCloseAfterExportTimeout(
        return_code=return_code,
        started_at_utc=value["started_at_utc"],
        finished_at_utc=value["finished_at_utc"],
        duration_s=float(duration),
        log_sha256=log_sha256,
    )


def _traverses_symlink(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_group_signal(process_group_id: int, selected: signal.Signals) -> bool:
    if process_group_id == os.getpgrp():
        raise RuntimeError("refusing to signal the watchdog owner process group")
    try:
        os.killpg(process_group_id, selected)
    except ProcessLookupError:
        return False
    return True


def _wait_for_group_exit(
    process: subprocess.Popen[bytes], process_group_id: int, timeout_s: float
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    process.poll()
    return not _process_group_exists(process_group_id)


def _terminate_process_group(
    process: subprocess.Popen[bytes], process_group_id: int, grace_s: float
) -> tuple[bool, bool]:
    term_sent = _send_group_signal(process_group_id, signal.SIGTERM)
    kill_sent = False
    if term_sent and not _wait_for_group_exit(process, process_group_id, grace_s):
        kill_sent = _send_group_signal(process_group_id, signal.SIGKILL)
        if kill_sent and not _wait_for_group_exit(process, process_group_id, grace_s):
            raise RuntimeError("owned lifecycle process group survived SIGKILL")
    if process.poll() is None:
        process.wait(timeout=grace_s)
    return term_sent, kill_sent


def _load_records(path: Path) -> tuple[LifecycleStageReceipt, ...]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("lifecycle journal must be a real directory")
    records: list[LifecycleStageReceipt] = []
    previous_sha256: Optional[str] = None
    identity: Optional[LifecycleIdentity] = None
    for expected_sequence, member in enumerate(sorted(path.iterdir())):
        if member.is_symlink() or not member.is_file() or member.suffix != ".json":
            raise ValueError("lifecycle journal contains an invalid member")
        document = strict_json_bytes(member.read_bytes(), "lifecycle stage receipt")
        receipt = LifecycleStageReceipt.from_dict(document)
        expected_name = f"{expected_sequence:06d}-{receipt.stage}.json"
        if member.name != expected_name or receipt.sequence != expected_sequence:
            raise ValueError("lifecycle journal sequence is not contiguous")
        if receipt.previous_record_sha256 != previous_sha256:
            raise ValueError("lifecycle journal hash chain mismatch")
        if identity is None:
            identity = receipt.identity
        elif receipt.identity != identity:
            raise ValueError("lifecycle journal contains mixed identities")
        records.append(receipt)
        previous_sha256 = receipt.sha256
    return tuple(records)


def write_canonical_no_clobber(path: Path, document: object) -> str:
    """Publish one canonical JSON receipt without replacing existing bytes."""

    payload = canonical_json_bytes(document)
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError("receipt parent must be a real directory")
    if destination.is_symlink():
        raise ValueError("receipt cannot be a symlink")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError("receipt already exists with different bytes")
        return _sha256(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != payload:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(payload)


__all__ = [
    "LIFECYCLE_STAGES",
    "TRUSTED_EPISODE_STAGES",
    "DEFAULT_STARTUP_TEARDOWN_GRACE_S",
    "HardLifecycleOutcome",
    "LifecycleIdentity",
    "LifecycleStageJournal",
    "LifecycleStageReceipt",
    "VerifiedCloseAfterExportTimeout",
    "campaign_watchdog_paths",
    "require_positive_finite",
    "resolve_hard_lifecycle_timeout",
    "sha256_file",
    "wait_with_lifecycle_watchdog",
    "watchdog_receipt_document",
    "watchdog_proves_fatal_unattempted",
    "verify_close_after_export_timeout",
    "write_canonical_no_clobber",
]
