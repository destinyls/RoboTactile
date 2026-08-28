"""Typed campaign-level aggregate for one strict clean-only campaign."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Tuple, cast

from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_BASELINE_EVIDENCE_LEVEL,
    CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION,
    CleanCampaignError,
    CleanCampaignProtocol,
    require_integer,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.clean_baseline.summary_serialization import (
    clean_summary_from_dict,
    clean_summary_to_dict,
)
from robotactile_benchmark.clean_baseline.summary_values import (
    PROTOCOL_INVALID_REASON_CODES,
    CleanCandidateClassification,
    CleanCandidateProvenance,
    CleanTaskSummary,
    ProtocolInvalidTrial,
    bootstrap_from_dict,
    bootstrap_to_dict,
    require_bool,
    require_rate,
    require_status_counts,
    wilson_from_dict,
    wilson_to_dict,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.reporting.statistics import (
    BinomialInterval,
    IntervalEstimate,
)


@dataclass(frozen=True)
class CleanBaselineSummary:
    campaign_id: str
    campaign_manifest_sha256: str
    protocol_id: CleanCampaignProtocol
    protocol_allows_paper_claim: bool
    paper_claim_eligible: bool
    paper_claim_blockers: Tuple[str, ...]
    evidence_level: str
    simulator_qualification_claimed: bool
    policy_kind: LivePolicyKind
    task_registry_sha256: str
    planned_trial_count: int
    loaded_artifact_count: int
    missing_artifact_count: int
    protocol_invalid_count: int
    eligible_trial_count: int
    success_count: int
    failure_count: int
    ineligible_count: int
    artifact_completion_rate: float
    eligible_success_rate: Optional[float]
    macro_task_success_rate: Optional[float]
    pooled_wilson: Optional[BinomialInterval]
    task_stratified_bootstrap: Optional[IntervalEstimate]
    task_summaries: Tuple[CleanTaskSummary, ...]
    terminal_status_counts: Mapping[str, int]
    missing_trial_manifest_sha256s: Tuple[str, ...]
    protocol_invalid_trials: Tuple[ProtocolInvalidTrial, ...]
    artifact_root_sha256s: Tuple[str, ...]
    attempt_receipt_sha256s: Tuple[str, ...]
    statistically_complete: bool
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    semantic_version: str = CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION
    target_valid_trial_count: Optional[int] = None
    candidate_trial_count: Optional[int] = None
    attempted_candidate_count: Optional[int] = None
    exception_replacement_count: Optional[int] = None
    unused_reserve_count: Optional[int] = None
    valid_outcome_count: Optional[int] = None
    missing_required_candidate_count: Optional[int] = None
    candidate_provenance: Tuple[CleanCandidateProvenance, ...] = ()
    exception_attempt_receipt_sha256s: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._normalize_identity()
        self._validate_inventory()
        self._validate_statistics()

    def _normalize_identity(self) -> None:
        object.__setattr__(
            self, "campaign_id", require_nonempty(self.campaign_id, "campaign_id")
        )
        object.__setattr__(
            self,
            "campaign_manifest_sha256",
            require_sha256(self.campaign_manifest_sha256, "campaign_manifest_sha256"),
        )
        protocol = (
            self.protocol_id
            if isinstance(self.protocol_id, CleanCampaignProtocol)
            else CleanCampaignProtocol(self.protocol_id)
        )
        policy = (
            self.policy_kind
            if isinstance(self.policy_kind, LivePolicyKind)
            else LivePolicyKind(self.policy_kind)
        )
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "policy_kind", policy)
        object.__setattr__(
            self,
            "task_registry_sha256",
            require_sha256(self.task_registry_sha256, "task_registry_sha256"),
        )
        if self.evidence_level != CLEAN_BASELINE_EVIDENCE_LEVEL:
            raise CleanCampaignError("clean baseline evidence level mismatch")
        if require_bool(
            self.simulator_qualification_claimed,
            "simulator_qualification_claimed",
        ):
            raise CleanCampaignError(
                "clean baseline summary cannot claim qualification"
            )
        if require_bool(self.paper_claim_eligible, "paper_claim_eligible"):
            raise CleanCampaignError(
                "unqualified clean artifacts cannot support a paper claim"
            )
        allows_claim = require_bool(
            self.protocol_allows_paper_claim, "protocol_allows_paper_claim"
        )
        expected_allows = protocol.allows_paper_claim and self.semantic_version == "2.0"
        if allows_claim != expected_allows:
            raise CleanCampaignError("protocol paper-claim flag mismatch")
        if self.semantic_version not in {
            CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION,
            "2.0",
        }:
            raise CleanCampaignError("unsupported clean baseline summary version")

    def _validate_inventory(self) -> None:
        names = (
            "planned_trial_count",
            "loaded_artifact_count",
            "missing_artifact_count",
            "protocol_invalid_count",
            "eligible_trial_count",
            "success_count",
            "failure_count",
            "ineligible_count",
            "bootstrap_resamples",
            "bootstrap_seed",
        )
        for name in names:
            minimum = 1 if name in {"planned_trial_count", "bootstrap_resamples"} else 0
            object.__setattr__(
                self, name, require_integer(getattr(self, name), name, minimum)
            )
        self._normalize_v2_inventory()
        if (
            (
                self.semantic_version == CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION
                and self.planned_trial_count
                != self.loaded_artifact_count
                + self.missing_artifact_count
                + self.protocol_invalid_count
            )
            or self.loaded_artifact_count
            != self.eligible_trial_count + self.ineligible_count
            or self.eligible_trial_count != self.success_count + self.failure_count
        ):
            raise CleanCampaignError("clean baseline inventory counts are inconsistent")
        tasks = tuple(self.task_summaries)
        if not tasks or any(type(item) is not CleanTaskSummary for item in tasks):
            raise CleanCampaignError("task_summaries must contain typed task summaries")
        task_names = tuple(item.task for item in tasks)
        if task_names != tuple(sorted(task_names)) or len(set(task_names)) != len(
            tasks
        ):
            raise CleanCampaignError("task_summaries must be uniquely task-sorted")
        aggregate = tuple(
            sum(getattr(item, field) for item in tasks)
            for field in (
                "planned",
                "loaded",
                "missing",
                "protocol_invalid",
                "eligible",
                "successes",
                "failures",
                "ineligible",
            )
        )
        expected = (
            self.planned_trial_count,
            self.loaded_artifact_count,
            self.missing_artifact_count,
            self.protocol_invalid_count,
            self.eligible_trial_count,
            self.success_count,
            self.failure_count,
            self.ineligible_count,
        )
        if aggregate != expected:
            raise CleanCampaignError(
                "task summaries do not aggregate to campaign counts"
            )
        statuses = require_status_counts(self.terminal_status_counts)
        task_statuses: Counter[str] = Counter()
        for item in tasks:
            task_statuses.update(item.terminal_status_counts)
        if dict(task_statuses) != dict(statuses):
            raise CleanCampaignError("campaign terminal status counts are inconsistent")
        missing = tuple(
            require_sha256(item, "missing trial manifest")
            for item in self.missing_trial_manifest_sha256s
        )
        invalid = tuple(self.protocol_invalid_trials)
        roots = tuple(
            require_sha256(item, "artifact root") for item in self.artifact_root_sha256s
        )
        attempts = tuple(
            require_sha256(item, "attempt receipt")
            for item in self.attempt_receipt_sha256s
        )
        self._validate_root_inventories(missing, invalid, roots, attempts)
        complete = require_bool(self.statistically_complete, "statistically_complete")
        expected_complete = self._expected_statistical_completeness()
        if complete != expected_complete:
            raise CleanCampaignError("statistical completeness flag mismatch")
        object.__setattr__(self, "task_summaries", tasks)
        object.__setattr__(self, "terminal_status_counts", statuses)
        object.__setattr__(self, "missing_trial_manifest_sha256s", missing)
        object.__setattr__(self, "protocol_invalid_trials", invalid)
        object.__setattr__(self, "artifact_root_sha256s", roots)
        object.__setattr__(self, "attempt_receipt_sha256s", attempts)

    def _normalize_v2_inventory(self) -> None:
        names = (
            "target_valid_trial_count",
            "candidate_trial_count",
            "attempted_candidate_count",
            "exception_replacement_count",
            "unused_reserve_count",
            "valid_outcome_count",
            "missing_required_candidate_count",
        )
        raw = tuple(getattr(self, name) for name in names)
        if self.semantic_version == CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION:
            if any(item is not None for item in raw):
                raise CleanCampaignError(
                    "v1 summary cannot contain v2 candidate counts"
                )
            if self.candidate_provenance or self.exception_attempt_receipt_sha256s:
                raise CleanCampaignError("v1 summary cannot contain v2 provenance")
            return
        if any(item is None for item in raw):
            raise CleanCampaignError("v2 candidate counts must be present")
        values = tuple(
            require_integer(cast(int, item), name) for name, item in zip(names, raw)
        )
        target, candidates, attempted, replaced, unused, valid, missing = values
        if (
            candidates != self.planned_trial_count
            or attempted != valid + replaced
            or candidates != attempted + unused + missing
            or valid != self.eligible_trial_count
            or valid > target
        ):
            raise CleanCampaignError("v2 campaign candidate counts are inconsistent")
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)
        self._validate_v2_task_counts(values)
        self._validate_v2_provenance(candidates, replaced)

    def _validate_v2_task_counts(self, values: tuple[int, ...]) -> None:
        fields = (
            "target_valid_trial_count",
            "candidate_trial_count",
            "attempted_candidate_count",
            "exception_replacement_count",
            "unused_reserve_count",
            "valid_outcome_count",
            "missing_required_candidate_count",
        )
        aggregate = tuple(
            sum(cast(int, getattr(task, name)) for task in self.task_summaries)
            for name in fields
        )
        if aggregate != values:
            raise CleanCampaignError("v2 task candidate counts do not aggregate")

    def _validate_v2_provenance(self, candidates: int, replaced: int) -> None:
        provenance = tuple(self.candidate_provenance)
        if (
            len(provenance) != candidates
            or any(type(item) is not CleanCandidateProvenance for item in provenance)
            or tuple(item.ordinal for item in provenance) != tuple(range(candidates))
            or len({item.trial_manifest_sha256 for item in provenance}) != candidates
        ):
            raise CleanCampaignError("v2 candidate provenance inventory mismatch")
        counts = Counter(item.classification for item in provenance)
        expected = {
            CleanCandidateClassification.VALID_OUTCOME: self.valid_outcome_count,
            CleanCandidateClassification.EXCEPTION_REPLACED: replaced,
            CleanCandidateClassification.UNUSED_RESERVE: self.unused_reserve_count,
            CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE: (
                self.missing_required_candidate_count
            ),
        }
        if any(counts[key] != value for key, value in expected.items()):
            raise CleanCampaignError("v2 candidate classifications mismatch counts")
        provenance_roots = tuple(
            cast(str, item.artifact_root_sha256)
            for item in provenance
            if item.artifact_root_sha256 is not None
        )
        provenance_artifact_attempts = tuple(
            cast(str, item.attempt_receipt_sha256)
            for item in provenance
            if item.artifact_root_sha256 is not None
        )
        if provenance_roots != tuple(
            self.artifact_root_sha256s
        ) or provenance_artifact_attempts != tuple(self.attempt_receipt_sha256s):
            raise CleanCampaignError("v2 provenance artifact inventories mismatch")
        exception_attempts = tuple(
            require_sha256(item, "exception attempt receipt")
            for item in self.exception_attempt_receipt_sha256s
        )
        expected_attempts = tuple(
            cast(str, item.attempt_receipt_sha256)
            for item in provenance
            if item.classification is CleanCandidateClassification.EXCEPTION_REPLACED
        )
        if (
            exception_attempts != expected_attempts
            or len(exception_attempts) != replaced
        ):
            raise CleanCampaignError("v2 exception attempt receipts mismatch")
        object.__setattr__(self, "candidate_provenance", provenance)
        object.__setattr__(
            self, "exception_attempt_receipt_sha256s", exception_attempts
        )

    def _expected_statistical_completeness(self) -> bool:
        if self.semantic_version == "2.0":
            return (
                self.valid_outcome_count == self.target_valid_trial_count
                and self.missing_required_candidate_count == 0
                and self.protocol_invalid_count == 0
            )
        return (
            self.missing_artifact_count == 0
            and self.protocol_invalid_count == 0
            and self.ineligible_count == 0
        )

    def _validate_root_inventories(
        self,
        missing: Tuple[str, ...],
        invalid: Tuple[ProtocolInvalidTrial, ...],
        roots: Tuple[str, ...],
        attempts: Tuple[str, ...],
    ) -> None:
        if len(missing) != self.missing_artifact_count or len(set(missing)) != len(
            missing
        ):
            raise CleanCampaignError("missing trial manifest inventory mismatch")
        invalid_hashes = tuple(item.trial_manifest_sha256 for item in invalid)
        if (
            len(invalid) != self.protocol_invalid_count
            or any(type(item) is not ProtocolInvalidTrial for item in invalid)
            or len(set(invalid_hashes)) != len(invalid)
            or set(missing) & set(invalid_hashes)
        ):
            raise CleanCampaignError("protocol-invalid trial inventory mismatch")
        if len(roots) != self.loaded_artifact_count or len(set(roots)) != len(roots):
            raise CleanCampaignError("artifact root inventory mismatch")
        if len(attempts) != self.loaded_artifact_count or len(set(attempts)) != len(
            attempts
        ):
            raise CleanCampaignError("attempt receipt inventory mismatch")

    def _validate_statistics(self) -> None:
        confidence = require_rate(self.confidence_level, "confidence_level")
        if confidence is None or confidence in {0.0, 1.0}:
            raise CleanCampaignError("confidence_level must lie in (0, 1)")
        completion = require_rate(
            self.artifact_completion_rate, "artifact_completion_rate"
        )
        expected_completion = self.loaded_artifact_count / self.planned_trial_count
        if completion is None or not math.isclose(
            completion, expected_completion, abs_tol=1e-15
        ):
            raise CleanCampaignError("artifact completion rate mismatch")
        eligible_rate = require_rate(
            self.eligible_success_rate, "eligible_success_rate", optional=True
        )
        macro_rate = require_rate(
            self.macro_task_success_rate,
            "macro_task_success_rate",
            optional=True,
        )
        eligible_tasks = [item for item in self.task_summaries if item.eligible > 0]
        if self.eligible_trial_count == 0:
            if any(
                value is not None
                for value in (
                    eligible_rate,
                    macro_rate,
                    self.pooled_wilson,
                    self.task_stratified_bootstrap,
                )
            ):
                raise CleanCampaignError(
                    "empty eligible inventory cannot have statistics"
                )
        else:
            self._validate_nonempty_statistics(
                eligible_rate, macro_rate, eligible_tasks
            )
        self._validate_interval_confidence(confidence)
        blockers = tuple(
            require_nonempty(item, "paper claim blocker")
            for item in self.paper_claim_blockers
        )
        if blockers != _paper_claim_blockers(self):
            raise CleanCampaignError("paper claim blockers mismatch")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "artifact_completion_rate", completion)
        object.__setattr__(self, "eligible_success_rate", eligible_rate)
        object.__setattr__(self, "macro_task_success_rate", macro_rate)
        object.__setattr__(self, "paper_claim_blockers", blockers)

    def _validate_nonempty_statistics(
        self,
        eligible_rate: Optional[float],
        macro_rate: Optional[float],
        eligible_tasks: list[CleanTaskSummary],
    ) -> None:
        expected_rate = self.success_count / self.eligible_trial_count
        expected_macro = sum(
            cast(float, item.eligible_success_rate) for item in eligible_tasks
        ) / len(eligible_tasks)
        if eligible_rate is None or not math.isclose(
            eligible_rate, expected_rate, abs_tol=1e-15
        ):
            raise CleanCampaignError("eligible success rate mismatch")
        if macro_rate is None or not math.isclose(
            macro_rate, expected_macro, abs_tol=1e-15
        ):
            raise CleanCampaignError("macro task success rate mismatch")
        if (
            self.pooled_wilson is None
            or self.pooled_wilson.success_count != self.success_count
            or self.pooled_wilson.trial_count != self.eligible_trial_count
        ):
            raise CleanCampaignError("pooled Wilson interval mismatch")
        bootstrap = self.task_stratified_bootstrap
        if (
            bootstrap is None
            or bootstrap.task_count != len(eligible_tasks)
            or bootstrap.pair_count != self.eligible_trial_count
            or bootstrap.valid_resamples != self.bootstrap_resamples
        ):
            raise CleanCampaignError("task-stratified bootstrap inventory mismatch")

    def _validate_interval_confidence(self, confidence: float) -> None:
        intervals = [
            item.wilson for item in self.task_summaries if item.wilson is not None
        ]
        if self.pooled_wilson is not None:
            intervals.append(self.pooled_wilson)
        if any(
            not math.isclose(item.confidence_level, confidence, abs_tol=1e-15)
            for item in intervals
        ):
            raise CleanCampaignError("Wilson confidence level mismatch")
        bootstrap = self.task_stratified_bootstrap
        if bootstrap is not None and not math.isclose(
            bootstrap.confidence_level, confidence, abs_tol=1e-15
        ):
            raise CleanCampaignError("bootstrap confidence level mismatch")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return clean_summary_to_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> CleanBaselineSummary:
        if cls is not CleanBaselineSummary:
            raise TypeError("clean baseline summary does not support subclass loading")
        return clean_summary_from_dict(value)


def _paper_claim_blockers(summary: CleanBaselineSummary) -> Tuple[str, ...]:
    blockers = []
    if not summary.protocol_allows_paper_claim:
        blockers.append("protocol_not_paper")
    if summary.missing_artifact_count:
        blockers.append("missing_artifacts")
    if summary.ineligible_count and summary.semantic_version == "1.0":
        blockers.append("ineligible_outcomes")
    if summary.protocol_invalid_count:
        blockers.append("protocol_invalid_trials")
    if (
        summary.semantic_version == "2.0"
        and summary.valid_outcome_count != summary.target_valid_trial_count
    ):
        blockers.append("replacement_pool_exhausted")
    blockers.append("simulator_not_qualified")
    return tuple(blockers)


__all__ = [
    "PROTOCOL_INVALID_REASON_CODES",
    "CleanBaselineSummary",
    "CleanCandidateClassification",
    "CleanCandidateProvenance",
    "CleanTaskSummary",
    "ProtocolInvalidTrial",
    "bootstrap_from_dict",
    "bootstrap_to_dict",
    "wilson_from_dict",
    "wilson_to_dict",
]
