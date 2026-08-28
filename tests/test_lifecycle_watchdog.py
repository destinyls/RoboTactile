from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from robotactile_benchmark.execution.lifecycle_watchdog import (
    LIFECYCLE_STAGES,
    TRUSTED_EPISODE_STAGES,
    HardLifecycleOutcome,
    LifecycleIdentity,
    LifecycleStageJournal,
    resolve_hard_lifecycle_timeout,
    sha256_file,
    verify_close_after_export_timeout,
    wait_with_lifecycle_watchdog,
    watchdog_proves_fatal_unattempted,
    watchdog_receipt_document,
    write_canonical_no_clobber,
)

_REUSE_LIFECYCLE_STAGES = frozenset(
    {
        "task_teardown_start",
        "renderer_released",
        "timeline_stopped",
        "task_closed",
        "task_reconstruction_start",
        "gc_collected",
        "usd_stage_recreated",
        "task_reconstruction_ready",
    }
)
_REUSE_TRUSTED_EPISODE_STAGES = frozenset(
    {
        "task_teardown_start",
        "renderer_released",
        "timeline_stopped",
        "task_closed",
    }
)


def _identity() -> LifecycleIdentity:
    return LifecycleIdentity(
        campaign_manifest_sha256="a" * 64,
        request_file_sha256="b" * 64,
        trial_manifest_sha256="c" * 64,
        task_id="lift_bottle",
        ordinal=3,
        attempt_id="20260825T120000.000000Z",
    )


def test_stage_journal_is_identity_bound_contiguous_and_hash_chained(
    tmp_path: Path,
) -> None:
    journal = LifecycleStageJournal.create(tmp_path / "attempt.journal", _identity())
    first = journal.record("process_spawn")
    second = journal.record("reset")

    reopened = LifecycleStageJournal.open(journal.path)
    assert reopened.identity == _identity()
    assert reopened.records() == (first, second)
    assert second.sequence == 1
    assert second.previous_record_sha256 == first.sha256
    assert reopened.last_record() == second

    member = journal.path / "000001-reset.json"
    member.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        reopened.records()


@pytest.mark.parametrize("stage", sorted(_REUSE_LIFECYCLE_STAGES))
def test_stage_journal_accepts_univtac_reuse_stages(
    tmp_path: Path,
    stage: str,
) -> None:
    journal = LifecycleStageJournal.create(tmp_path / f"{stage}.journal", _identity())

    receipt = journal.record(stage)

    assert receipt.stage == stage
    assert journal.last_record() == receipt


def test_only_post_episode_univtac_reuse_stages_are_trusted() -> None:
    assert _REUSE_LIFECYCLE_STAGES <= LIFECYCLE_STAGES
    assert (
        _REUSE_LIFECYCLE_STAGES & TRUSTED_EPISODE_STAGES
        == _REUSE_TRUSTED_EPISODE_STAGES
    )


def test_canonical_receipt_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    digest = write_canonical_no_clobber(path, {"status": "first"})
    assert write_canonical_no_clobber(path, {"status": "first"}) == digest
    with pytest.raises(FileExistsError):
        write_canonical_no_clobber(path, {"status": "different"})


def test_hard_watchdog_terminates_an_owned_process_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    outcome = wait_with_lifecycle_watchdog(
        process, hard_timeout_s=0.05, term_grace_s=0.5
    )

    assert outcome.timed_out is True
    assert outcome.termination_signal_sent is True
    assert outcome.return_code == -signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_hard_watchdog_escalates_to_kill_when_term_is_ignored() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "print('ready',flush=True);time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == b"ready\n"

    outcome = wait_with_lifecycle_watchdog(
        process, hard_timeout_s=0.05, term_grace_s=0.05
    )

    assert outcome.timed_out is True
    assert outcome.termination_signal_sent is True
    assert outcome.kill_signal_sent is True
    assert outcome.return_code == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_watchdog_receipt_binds_last_stage_and_separate_hard_budget(
    tmp_path: Path,
) -> None:
    journal = LifecycleStageJournal.create(tmp_path / "attempt.journal", _identity())
    journal.record("process_spawn")
    last = journal.record("infer")
    document = watchdog_receipt_document(
        identity=_identity(),
        last_stage_receipt=last,
        journal_relpath="outputs/attempt.journal",
        log_relpath="logs/attempt.log",
        log_sha256="d" * 64,
        process_id=123,
        outcome=HardLifecycleOutcome(-9, True, True, True),
        started_at_utc="2026-08-25T12:00:00+00:00",
        finished_at_utc="2026-08-25T12:30:00+00:00",
        duration_s=1800.0,
        hard_lifecycle_timeout_s=2400.0,
        soft_wall_timeout_s=1800.0,
        hard_timeout_source="soft_plus_startup_teardown_grace",
        startup_teardown_grace_s=600.0,
        term_grace_s=30.0,
        watchdog_error_type=None,
    )

    assert document["identity"] == _identity().to_dict()
    assert document["journal_last_stage"] == "infer"
    assert document["journal_last_record_sha256"] == last.sha256
    assert document["hard_lifecycle_timeout_s"] == 2400.0
    assert document["soft_wall_timeout_s"] == 1800.0
    assert document["startup_teardown_grace_s"] == 600.0


def test_default_hard_budget_adds_startup_teardown_grace() -> None:
    resolved = resolve_hard_lifecycle_timeout(1800.0, None)
    assert resolved == (2400.0, "soft_plus_startup_teardown_grace", 600.0)
    assert resolve_hard_lifecycle_timeout(1800.0, 2500.0) == (
        2500.0,
        "explicit",
        None,
    )
    with pytest.raises(ValueError, match="must exceed"):
        resolve_hard_lifecycle_timeout(1800.0, 1800.0)


def test_startup_watchdog_receipt_allows_same_candidate_recovery(
    tmp_path: Path,
) -> None:
    journal = LifecycleStageJournal.create(tmp_path / "attempt.journal", _identity())
    journal.record("process_spawn")
    last = journal.record("app_launcher")
    log = tmp_path / "attempt.log"
    log.write_text("startup timed out\n", encoding="utf-8")
    document = watchdog_receipt_document(
        identity=_identity(),
        last_stage_receipt=last,
        journal_relpath="outputs/attempt.journal",
        log_relpath="logs/attempt.log",
        log_sha256=sha256_file(log),
        process_id=123,
        outcome=HardLifecycleOutcome(-9, True, True, True),
        started_at_utc="2026-08-25T12:00:00+00:00",
        finished_at_utc="2026-08-25T12:40:00+00:00",
        duration_s=2400.0,
        hard_lifecycle_timeout_s=2400.0,
        soft_wall_timeout_s=1800.0,
        hard_timeout_source="soft_plus_startup_teardown_grace",
        startup_teardown_grace_s=600.0,
        term_grace_s=30.0,
        watchdog_error_type=None,
    )
    receipt = tmp_path / "attempt.watchdog.json"
    write_canonical_no_clobber(receipt, document)

    assert watchdog_proves_fatal_unattempted(
        receipt_path=receipt,
        journal_path=journal.path,
        log_path=log,
        identity=_identity(),
    )


def test_close_timeout_requires_artifact_export_as_penultimate_stage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    identity = _identity()
    lifecycle_root = (
        root / "outputs/clean-campaigns/campaign/lifecycle" / identity.task_id
    )
    journal_path = lifecycle_root / (
        f"{identity.ordinal:04d}-{identity.attempt_id}.journal"
    )
    receipt_path = lifecycle_root / (
        f"{identity.ordinal:04d}-{identity.attempt_id}.watchdog.json"
    )
    log_path = (
        root
        / "logs/clean-campaigns/campaign"
        / identity.task_id
        / f"{identity.ordinal:04d}-{identity.attempt_id}.log"
    )
    log_path.parent.mkdir(parents=True)
    log_path.write_text("hard timeout during close\n", encoding="utf-8")
    journal = LifecycleStageJournal.create(journal_path, identity)
    journal.record("artifact_export")
    last = journal.record("close")
    document = watchdog_receipt_document(
        identity=identity,
        last_stage_receipt=last,
        journal_relpath=journal_path.relative_to(root).as_posix(),
        log_relpath=log_path.relative_to(root).as_posix(),
        log_sha256=sha256_file(log_path),
        process_id=123,
        outcome=HardLifecycleOutcome(-15, True, True, False),
        started_at_utc="2026-08-25T12:00:00+00:00",
        finished_at_utc="2026-08-25T12:40:00+00:00",
        duration_s=2400.0,
        hard_lifecycle_timeout_s=2400.0,
        soft_wall_timeout_s=1800.0,
        hard_timeout_source="soft_plus_startup_teardown_grace",
        startup_teardown_grace_s=600.0,
        term_grace_s=30.0,
        watchdog_error_type=None,
    )
    write_canonical_no_clobber(receipt_path, document)

    proof = verify_close_after_export_timeout(
        root=root,
        campaign_id="campaign",
        identity=identity,
        log_path=log_path,
    )

    assert proof.return_code == -15
    assert proof.log_sha256 == sha256_file(log_path)
