"""Strict clean artifact inventory, attempt gate, and summary tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_clean_campaign_support import (
    digest,
    make_layout,
    write_artifact_and_attempt,
    write_clean_request,
    write_exception_attempt_without_artifact,
)

from robotactile_benchmark.clean_baseline import (
    CLEAN_UNIVTAC_SEED_DERIVATION,
    CleanBaselineSummary,
    CleanCampaignError,
    CleanCampaignProtocol,
    CleanCampaignSamplingSpec,
    IncompleteCleanCampaignError,
    aggregate_clean_campaign,
    build_clean_campaign_manifest,
    load_clean_artifact_inventory,
    load_clean_baseline_summary,
    write_clean_baseline_summary,
)
from robotactile_benchmark.clean_baseline.seeds import univtac_task_seed_start
from robotactile_benchmark.clean_baseline.summary import (
    CleanCandidateClassification,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.contracts import BackendSignal
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.lifecycle_watchdog import (
    HardLifecycleOutcome,
    LifecycleIdentity,
    LifecycleStageJournal,
    sha256_file,
    watchdog_receipt_document,
    write_canonical_no_clobber,
)


def _campaign(tmp_path: Path):
    layout = make_layout(tmp_path)
    request_path, request = write_clean_request(layout)
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=(request_path,),
        campaign_id="campaign-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=20260823,
        bootstrap_resamples=200,
        bootstrap_seed=13,
    )
    return layout, request, manifest


def _v2_campaign(tmp_path: Path, candidate_count: int = 3):
    layout = make_layout(tmp_path)
    seed_start = univtac_task_seed_start(0)
    requests = tuple(
        write_clean_request(
            layout,
            ordinal=ordinal,
            initial_seed=seed_start + ordinal,
            exogenous_seed=seed_start + ordinal,
        )
        for ordinal in range(candidate_count)
    )
    sampling = CleanCampaignSamplingSpec(
        seed_protocol=CLEAN_UNIVTAC_SEED_DERIVATION,
        exception_handling="replace_exception_until_target_valid_v1",
        target_valid_trials_per_task=1,
        candidate_trials_per_task=candidate_count,
        official_eval_seed=0,
        task_seed_start=seed_start,
        policy_seed_mode="same_as_task_seed_v1",
    )
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=tuple(item[0] for item in requests),
        campaign_id="campaign-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=0,
        bootstrap_resamples=200,
        bootstrap_seed=13,
        sampling=sampling,
    )
    return layout, tuple(item[1] for item in requests), manifest


def test_verified_artifact_and_success_attempt_produce_strict_statistics(
    tmp_path: Path,
) -> None:
    layout, request, manifest = _campaign(tmp_path)
    spec = manifest.trials[0]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=spec.ordinal,
        request_file_sha256=spec.request_file_sha256,
    )

    summary = aggregate_clean_campaign(layout.root, manifest)

    assert summary.loaded_artifact_count == 1
    assert summary.missing_artifact_count == 0
    assert summary.protocol_invalid_count == 0
    assert summary.eligible_trial_count == 1
    assert summary.success_count + summary.failure_count == 1
    assert summary.pooled_wilson is not None
    assert summary.task_stratified_bootstrap is not None
    assert summary.statistically_complete is True
    assert summary.paper_claim_eligible is False
    assert summary.paper_claim_blockers == (
        "protocol_not_paper",
        "simulator_not_qualified",
    )
    assert len(summary.artifact_root_sha256s) == 1
    assert len(summary.attempt_receipt_sha256s) == 1
    assert summary.semantic_version == "1.0"
    assert not {
        "candidate_provenance",
        "candidate_trial_count",
        "exception_attempt_receipt_sha256s",
        "target_valid_trial_count",
    } & set(summary.to_dict())

    output = layout.outputs / "clean-campaigns/campaign-test/summary.json"
    assert write_clean_baseline_summary(output, summary) is True
    assert write_clean_baseline_summary(output, summary) is False
    assert load_clean_baseline_summary(output) == summary

    legacy_paper = summary.to_dict()
    legacy_paper["protocol_id"] = "paper_v1"
    legacy_paper["protocol_allows_paper_claim"] = False
    loaded_legacy_paper = CleanBaselineSummary.from_dict(legacy_paper)
    assert loaded_legacy_paper.protocol_allows_paper_claim is False
    assert "protocol_not_paper" in loaded_legacy_paper.paper_claim_blockers
    polluted_v1 = summary.to_dict()
    polluted_v1["candidate_trial_count"] = None
    with pytest.raises(CleanCampaignError, match="summary fields mismatch"):
        CleanBaselineSummary.from_dict(polluted_v1)


@pytest.mark.parametrize(
    "capture_profile",
    (LiveCaptureProfile.METRICS_ONLY, LiveCaptureProfile.PREVIEW),
)
def test_compact_capture_requires_explicit_diagnostic_aggregation(
    tmp_path: Path,
    capture_profile: LiveCaptureProfile,
) -> None:
    layout, request, manifest = _campaign(tmp_path)
    spec = manifest.trials[0]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=spec.ordinal,
        request_file_sha256=spec.request_file_sha256,
        capture_profile=capture_profile,
    )

    rejected = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)
    accepted = aggregate_clean_campaign(
        layout.root,
        manifest,
        allow_compact=True,
    )

    assert rejected.loaded_artifact_count == 0
    assert rejected.protocol_invalid_count == 1
    assert rejected.protocol_invalid_trials[0].reason_code == (
        "artifact_validation_failed"
    )
    assert accepted.loaded_artifact_count == 1
    assert accepted.protocol_invalid_count == 0
    assert accepted.eligible_trial_count == 1
    assert accepted.success_count + accepted.failure_count == 1
    assert accepted.statistically_complete is True


def test_compact_aggregation_flag_is_bool(tmp_path: Path) -> None:
    layout, _, manifest = _campaign(tmp_path)

    with pytest.raises(TypeError, match="allow_compact must be bool"):
        aggregate_clean_campaign(
            layout.root,
            manifest,
            allow_compact=1,  # type: ignore[arg-type]
        )


def test_compact_opt_in_rejects_non_diagnostic_protocol(tmp_path: Path) -> None:
    layout, _, manifest = _campaign(tmp_path)
    object.__setattr__(manifest, "protocol_id", CleanCampaignProtocol.PILOT)

    with pytest.raises(
        CleanCampaignError,
        match="compact capture aggregation requires a diagnostic_v1 campaign",
    ):
        load_clean_artifact_inventory(
            layout.root,
            manifest,
            allow_compact=True,
        )


def test_missing_artifact_fails_closed_unless_partial_is_explicit(
    tmp_path: Path,
) -> None:
    layout, _, manifest = _campaign(tmp_path)

    with pytest.raises(IncompleteCleanCampaignError, match="allow_partial"):
        aggregate_clean_campaign(layout.root, manifest)
    summary = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)

    assert summary.loaded_artifact_count == 0
    assert summary.missing_artifact_count == 1
    assert summary.protocol_invalid_count == 0
    assert summary.eligible_success_rate is None
    assert summary.statistically_complete is False
    assert "missing_artifacts" in summary.paper_claim_blockers


@pytest.mark.parametrize(
    ("include_attempt", "return_code", "reason"),
    (
        (False, 0, "attempt_receipt_missing"),
        (True, 7, "attempt_return_code_nonzero"),
    ),
)
def test_artifact_without_proven_successful_command_is_protocol_invalid(
    tmp_path: Path,
    include_attempt: bool,
    return_code: int,
    reason: str,
) -> None:
    layout, request, manifest = _campaign(tmp_path)
    spec = manifest.trials[0]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=spec.ordinal,
        request_file_sha256=spec.request_file_sha256,
        include_attempt=include_attempt,
        return_code=return_code,
    )

    summary = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)

    assert summary.loaded_artifact_count == 0
    assert summary.protocol_invalid_count == 1
    assert summary.protocol_invalid_trials[0].reason_code == reason
    assert summary.statistically_complete is False
    assert "protocol_invalid_trials" in summary.paper_claim_blockers


def test_tampered_artifact_is_reported_as_protocol_invalid(tmp_path: Path) -> None:
    layout, request, manifest = _campaign(tmp_path)
    spec = manifest.trials[0]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=spec.ordinal,
        request_file_sha256=spec.request_file_sha256,
    )
    assert request.output_dir is not None
    (request.output_dir / "unknown.bin").write_bytes(b"tamper")

    summary = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)

    assert summary.protocol_invalid_trials[0].reason_code == (
        "artifact_validation_failed"
    )


def test_v2_task_failure_is_valid_denominator_and_reserve_is_unused(
    tmp_path: Path,
) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    spec = manifest.trials[0]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[0],
        manifest_ordinal=spec.ordinal,
        request_file_sha256=spec.request_file_sha256,
        semantic_version="2.0",
        candidate_disposition="valid_outcome",
        terminal_signal=BackendSignal.TASK_FAILURE,
    )

    summary = aggregate_clean_campaign(layout.root, manifest)

    assert summary.semantic_version == "2.0"
    assert summary.target_valid_trial_count == 1
    assert summary.candidate_trial_count == 3
    assert summary.attempted_candidate_count == 1
    assert summary.valid_outcome_count == 1
    assert summary.exception_replacement_count == 0
    assert summary.unused_reserve_count == 2
    assert summary.missing_required_candidate_count == 0
    assert summary.eligible_trial_count == 1
    assert summary.failure_count == 1
    assert summary.success_count == 0
    assert summary.terminal_status_counts == {"task_failure": 1}
    assert summary.statistically_complete is True
    assert tuple(item.classification for item in summary.candidate_provenance) == (
        CleanCandidateClassification.VALID_OUTCOME,
        CleanCandidateClassification.UNUSED_RESERVE,
        CleanCandidateClassification.UNUSED_RESERVE,
    )
    assert summary.task_summaries[0].valid_outcome_count == 1
    assert summary.task_summaries[0].unused_reserve_count == 2

    output = layout.outputs / "clean-campaigns/campaign-test/summary-v2.json"
    assert write_clean_baseline_summary(output, summary) is True
    assert load_clean_baseline_summary(output) == summary
    mixed_task_version = summary.to_dict()
    del mixed_task_version["task_summaries"][0]["candidate_trial_count"]
    with pytest.raises(CleanCampaignError, match="task summary fields mismatch"):
        CleanBaselineSummary.from_dict(mixed_task_version)


def test_v2_crash_artifact_is_replaced_and_next_candidate_is_valid(
    tmp_path: Path,
) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    first, second = manifest.trials[:2]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[0],
        manifest_ordinal=first.ordinal,
        request_file_sha256=first.request_file_sha256,
        semantic_version="2.0",
        candidate_disposition="exception_replaced",
        exception_code="policy_inference_crash",
        crash=True,
    )
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[1],
        manifest_ordinal=second.ordinal,
        request_file_sha256=second.request_file_sha256,
        semantic_version="2.0",
        candidate_disposition="valid_outcome",
        terminal_signal=BackendSignal.TASK_FAILURE,
    )

    summary = aggregate_clean_campaign(layout.root, manifest)

    assert summary.loaded_artifact_count == 2
    assert summary.eligible_trial_count == 1
    assert summary.failure_count == 1
    assert summary.ineligible_count == 1
    assert summary.attempted_candidate_count == 2
    assert summary.exception_replacement_count == 1
    assert summary.valid_outcome_count == 1
    assert summary.unused_reserve_count == 1
    assert summary.terminal_status_counts == {"crash": 1, "task_failure": 1}
    assert len(summary.exception_attempt_receipt_sha256s) == 1
    assert "ineligible_outcomes" not in summary.paper_claim_blockers
    assert summary.statistically_complete is True


def test_v2_verified_close_timeout_is_replaced_without_scoring_artifact(
    tmp_path: Path,
) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    first, second = manifest.trials[:2]
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[0],
        manifest_ordinal=first.ordinal,
        request_file_sha256=first.request_file_sha256,
        semantic_version="2.0",
        candidate_disposition="exception_replaced",
        exception_code="close_after_export_hard_timeout",
        return_code=-15,
        terminal_signal=BackendSignal.TASK_FAILURE,
    )
    attempt_id = "attempt"
    identity = LifecycleIdentity(
        campaign_manifest_sha256=manifest.sha256,
        request_file_sha256=first.request_file_sha256,
        trial_manifest_sha256=first.trial_manifest_sha256,
        task_id=first.task,
        ordinal=first.ordinal,
        attempt_id=attempt_id,
    )
    lifecycle_root = (
        layout.outputs
        / "clean-campaigns"
        / manifest.campaign_id
        / "lifecycle"
        / first.task
    )
    journal_path = lifecycle_root / f"{first.ordinal:04d}-{attempt_id}.journal"
    receipt_path = lifecycle_root / f"{first.ordinal:04d}-{attempt_id}.watchdog.json"
    log_path = (
        layout.logs
        / "clean-campaigns"
        / manifest.campaign_id
        / first.task
        / f"{first.ordinal:04d}-{attempt_id}.log"
    )
    journal = LifecycleStageJournal.create(journal_path, identity)
    journal.record("artifact_export")
    last = journal.record("close")
    started = datetime(2026, 8, 23, tzinfo=timezone.utc)
    write_canonical_no_clobber(
        receipt_path,
        watchdog_receipt_document(
            identity=identity,
            last_stage_receipt=last,
            journal_relpath=journal_path.relative_to(layout.root).as_posix(),
            log_relpath=log_path.relative_to(layout.root).as_posix(),
            log_sha256=sha256_file(log_path),
            process_id=123,
            outcome=HardLifecycleOutcome(-15, True, True, False),
            started_at_utc=started.isoformat(),
            finished_at_utc=(started + timedelta(seconds=1)).isoformat(),
            duration_s=1.0,
            hard_lifecycle_timeout_s=6.0,
            soft_wall_timeout_s=5.0,
            hard_timeout_source="explicit",
            startup_teardown_grace_s=None,
            term_grace_s=1.0,
            watchdog_error_type=None,
        ),
    )
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[1],
        manifest_ordinal=second.ordinal,
        request_file_sha256=second.request_file_sha256,
        semantic_version="2.0",
        candidate_disposition="valid_outcome",
        terminal_signal=BackendSignal.TASK_FAILURE,
    )

    summary = aggregate_clean_campaign(layout.root, manifest)

    assert summary.exception_replacement_count == 1
    assert summary.valid_outcome_count == 1
    assert summary.eligible_trial_count == 1
    assert summary.terminal_status_counts == {"task_failure": 2}


def test_v2_command_exception_without_artifact_uses_next_candidate(
    tmp_path: Path,
) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    first, second = manifest.trials[:2]
    write_exception_attempt_without_artifact(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[0],
        manifest_ordinal=first.ordinal,
        request_file_sha256=first.request_file_sha256,
    )
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[1],
        manifest_ordinal=second.ordinal,
        request_file_sha256=second.request_file_sha256,
        semantic_version="2.0",
        candidate_disposition="valid_outcome",
        terminal_signal=BackendSignal.TASK_FAILURE,
    )

    summary = aggregate_clean_campaign(layout.root, manifest)

    assert summary.loaded_artifact_count == 1
    assert summary.attempted_candidate_count == 2
    assert summary.exception_replacement_count == 1
    assert summary.valid_outcome_count == 1
    assert summary.unused_reserve_count == 1
    assert len(summary.artifact_root_sha256s) == 1
    assert len(summary.attempt_receipt_sha256s) == 1
    assert len(summary.exception_attempt_receipt_sha256s) == 1
    assert summary.candidate_provenance[0].artifact_root_sha256 is None
    assert summary.statistically_complete is True


def test_v2_gap_marks_remaining_required_candidates_missing_not_unused(
    tmp_path: Path,
) -> None:
    layout, _, manifest = _v2_campaign(tmp_path)

    with pytest.raises(IncompleteCleanCampaignError, match="allow_partial"):
        aggregate_clean_campaign(layout.root, manifest)
    summary = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)

    assert summary.missing_artifact_count == 1
    assert summary.missing_required_candidate_count == 3
    assert summary.unused_reserve_count == 0
    assert summary.valid_outcome_count == 0
    assert all(
        item.classification is CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE
        for item in summary.candidate_provenance
    )
    assert summary.statistically_complete is False
    assert "replacement_pool_exhausted" in summary.paper_claim_blockers


def test_v2_rejects_candidate_overrun_after_first_target_valid(
    tmp_path: Path,
) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    for index, spec in enumerate(manifest.trials[:2]):
        write_artifact_and_attempt(
            layout,
            manifest.campaign_id,
            manifest.sha256,
            requests[index],
            manifest_ordinal=spec.ordinal,
            request_file_sha256=spec.request_file_sha256,
            semantic_version="2.0",
            candidate_disposition="valid_outcome",
            terminal_signal=BackendSignal.TASK_FAILURE,
        )

    with pytest.raises(CleanCampaignError, match="overrun.*target-valid"):
        aggregate_clean_campaign(layout.root, manifest)


def test_v2_exhausted_replacement_pool_fails_closed(tmp_path: Path) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    for request, spec in zip(requests, manifest.trials):
        write_exception_attempt_without_artifact(
            layout,
            manifest.campaign_id,
            manifest.sha256,
            request,
            manifest_ordinal=spec.ordinal,
            request_file_sha256=spec.request_file_sha256,
        )

    with pytest.raises(IncompleteCleanCampaignError, match="allow_partial"):
        aggregate_clean_campaign(layout.root, manifest)
    summary = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)

    assert summary.attempted_candidate_count == 3
    assert summary.exception_replacement_count == 3
    assert summary.valid_outcome_count == 0
    assert summary.unused_reserve_count == 0
    assert summary.missing_required_candidate_count == 0
    assert len(summary.exception_attempt_receipt_sha256s) == 3
    assert summary.statistically_complete is False
    assert "replacement_pool_exhausted" in summary.paper_claim_blockers


def test_v2_receipt_identity_error_is_protocol_invalid_not_replacement(
    tmp_path: Path,
) -> None:
    layout, requests, manifest = _v2_campaign(tmp_path)
    first = manifest.trials[0]
    write_exception_attempt_without_artifact(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        requests[0],
        manifest_ordinal=first.ordinal,
        request_file_sha256=first.request_file_sha256,
    )
    receipt_path = (
        layout.outputs
        / "clean-campaigns"
        / manifest.campaign_id
        / "attempts"
        / first.task
        / f"{first.ordinal:04d}-attempt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["campaign_manifest_sha256"] = digest("wrong-manifest")
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    summary = aggregate_clean_campaign(layout.root, manifest, allow_partial=True)

    assert summary.protocol_invalid_count == 1
    assert summary.protocol_invalid_trials[0].reason_code == "attempt_receipt_invalid"
    assert summary.exception_replacement_count == 0
    assert summary.missing_required_candidate_count == 3
    assert summary.eligible_trial_count == 0
