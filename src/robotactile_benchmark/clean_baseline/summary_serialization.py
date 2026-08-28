"""Canonical dictionary conversion for clean campaign summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignProtocol,
)
from robotactile_benchmark.clean_baseline.summary_values import (
    CleanCandidateProvenance,
    CleanTaskSummary,
    ProtocolInvalidTrial,
    bootstrap_from_dict,
    bootstrap_to_dict,
    wilson_from_dict,
    wilson_to_dict,
)
from robotactile_benchmark.execution.contracts import LivePolicyKind

if TYPE_CHECKING:
    from robotactile_benchmark.clean_baseline.summary import CleanBaselineSummary


_V2_SUMMARY_FIELDS = frozenset(
    {
        "target_valid_trial_count",
        "candidate_trial_count",
        "attempted_candidate_count",
        "exception_replacement_count",
        "unused_reserve_count",
        "valid_outcome_count",
        "missing_required_candidate_count",
        "candidate_provenance",
        "exception_attempt_receipt_sha256s",
    }
)
_V2_TASK_FIELDS = frozenset(
    {
        "target_valid_trial_count",
        "candidate_trial_count",
        "attempted_candidate_count",
        "exception_replacement_count",
        "unused_reserve_count",
        "valid_outcome_count",
        "missing_required_candidate_count",
    }
)


def clean_summary_to_dict(summary: CleanBaselineSummary) -> dict[str, object]:
    document: dict[str, object] = {
        "campaign_id": summary.campaign_id,
        "campaign_manifest_sha256": summary.campaign_manifest_sha256,
        "protocol_id": summary.protocol_id.value,
        "protocol_allows_paper_claim": summary.protocol_allows_paper_claim,
        "paper_claim_eligible": summary.paper_claim_eligible,
        "paper_claim_blockers": list(summary.paper_claim_blockers),
        "evidence_level": summary.evidence_level,
        "simulator_qualification_claimed": summary.simulator_qualification_claimed,
        "policy_kind": summary.policy_kind.value,
        "task_registry_sha256": summary.task_registry_sha256,
        "planned_trial_count": summary.planned_trial_count,
        "loaded_artifact_count": summary.loaded_artifact_count,
        "missing_artifact_count": summary.missing_artifact_count,
        "protocol_invalid_count": summary.protocol_invalid_count,
        "eligible_trial_count": summary.eligible_trial_count,
        "success_count": summary.success_count,
        "failure_count": summary.failure_count,
        "ineligible_count": summary.ineligible_count,
        "artifact_completion_rate": summary.artifact_completion_rate,
        "eligible_success_rate": summary.eligible_success_rate,
        "macro_task_success_rate": summary.macro_task_success_rate,
        "pooled_wilson": wilson_to_dict(summary.pooled_wilson),
        "task_stratified_bootstrap": bootstrap_to_dict(
            summary.task_stratified_bootstrap
        ),
        "task_summaries": [item.to_dict() for item in summary.task_summaries],
        "terminal_status_counts": dict(summary.terminal_status_counts),
        "missing_trial_manifest_sha256s": list(summary.missing_trial_manifest_sha256s),
        "protocol_invalid_trials": [
            item.to_dict() for item in summary.protocol_invalid_trials
        ],
        "artifact_root_sha256s": list(summary.artifact_root_sha256s),
        "attempt_receipt_sha256s": list(summary.attempt_receipt_sha256s),
        "statistically_complete": summary.statistically_complete,
        "confidence_level": summary.confidence_level,
        "bootstrap_resamples": summary.bootstrap_resamples,
        "bootstrap_seed": summary.bootstrap_seed,
        "semantic_version": summary.semantic_version,
    }
    if summary.semantic_version == "2.0":
        document.update(
            {
                "target_valid_trial_count": summary.target_valid_trial_count,
                "candidate_trial_count": summary.candidate_trial_count,
                "attempted_candidate_count": summary.attempted_candidate_count,
                "exception_replacement_count": summary.exception_replacement_count,
                "unused_reserve_count": summary.unused_reserve_count,
                "valid_outcome_count": summary.valid_outcome_count,
                "missing_required_candidate_count": (
                    summary.missing_required_candidate_count
                ),
                "candidate_provenance": [
                    item.to_dict() for item in summary.candidate_provenance
                ],
                "exception_attempt_receipt_sha256s": list(
                    summary.exception_attempt_receipt_sha256s
                ),
            }
        )
    return document


def clean_summary_from_dict(value: object) -> CleanBaselineSummary:
    from robotactile_benchmark.clean_baseline.summary import CleanBaselineSummary

    if not isinstance(value, Mapping):
        raise CleanCampaignError("clean baseline summary must be an object")
    all_fields = frozenset(CleanBaselineSummary.__dataclass_fields__)
    document_fields = frozenset(value)
    semantic_version = value.get("semantic_version")
    expected_fields = (
        all_fields if semantic_version == "2.0" else all_fields - _V2_SUMMARY_FIELDS
    )
    if semantic_version not in {"1.0", "2.0"} or document_fields != expected_fields:
        raise CleanCampaignError("clean baseline summary fields mismatch")
    sequence_names: tuple[str, ...] = (
        "task_summaries",
        "paper_claim_blockers",
        "missing_trial_manifest_sha256s",
        "protocol_invalid_trials",
        "artifact_root_sha256s",
        "attempt_receipt_sha256s",
    )
    if semantic_version == "2.0":
        sequence_names += (
            "candidate_provenance",
            "exception_attempt_receipt_sha256s",
        )
    if any(not isinstance(value[name], list) for name in sequence_names):
        raise CleanCampaignError("clean baseline sequence fields must be lists")
    document = cast(Mapping[str, Any], value)
    task_fields = frozenset(CleanTaskSummary.__dataclass_fields__)
    expected_task_fields = (
        task_fields if semantic_version == "2.0" else task_fields - _V2_TASK_FIELDS
    )
    if any(
        not isinstance(item, Mapping) or frozenset(item) != expected_task_fields
        for item in document["task_summaries"]
    ):
        raise CleanCampaignError("clean task summary fields mismatch")
    kwargs = dict(document)
    kwargs["protocol_id"] = CleanCampaignProtocol(document["protocol_id"])
    kwargs["policy_kind"] = LivePolicyKind(document["policy_kind"])
    kwargs["paper_claim_blockers"] = tuple(document["paper_claim_blockers"])
    kwargs["pooled_wilson"] = wilson_from_dict(document["pooled_wilson"])
    kwargs["task_stratified_bootstrap"] = bootstrap_from_dict(
        document["task_stratified_bootstrap"]
    )
    kwargs["task_summaries"] = tuple(
        CleanTaskSummary.from_dict(item) for item in document["task_summaries"]
    )
    kwargs["missing_trial_manifest_sha256s"] = tuple(
        document["missing_trial_manifest_sha256s"]
    )
    kwargs["protocol_invalid_trials"] = tuple(
        ProtocolInvalidTrial.from_dict(item)
        for item in document["protocol_invalid_trials"]
    )
    kwargs["artifact_root_sha256s"] = tuple(document["artifact_root_sha256s"])
    kwargs["attempt_receipt_sha256s"] = tuple(document["attempt_receipt_sha256s"])
    if semantic_version == "2.0":
        kwargs["candidate_provenance"] = tuple(
            CleanCandidateProvenance.from_dict(item)
            for item in document["candidate_provenance"]
        )
        kwargs["exception_attempt_receipt_sha256s"] = tuple(
            document["exception_attempt_receipt_sha256s"]
        )
    return CleanBaselineSummary(**kwargs)


__all__ = ["clean_summary_from_dict", "clean_summary_to_dict"]
