"""Reusable interval and per-task values for clean baseline summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, cast

from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    require_integer,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.reporting.statistics import (
    BinomialInterval,
    IntervalEstimate,
)
from robotactile_benchmark.trials import TerminalStatus

PROTOCOL_INVALID_REASON_CODES = frozenset(
    {
        "artifact_validation_failed",
        "attempt_receipt_missing",
        "attempt_receipt_invalid",
        "attempt_return_code_nonzero",
        "attempt_artifact_binding_mismatch",
        "multiple_attempt_receipts",
    }
)


class CleanCandidateClassification(str, Enum):
    """Exhaustive official-sampling disposition for one frozen candidate."""

    VALID_OUTCOME = "valid_outcome"
    EXCEPTION_REPLACED = "exception_replaced"
    UNUSED_RESERVE = "unused_reserve"
    MISSING_REQUIRED_CANDIDATE = "missing_required_candidate"


@dataclass(frozen=True)
class CleanCandidateProvenance:
    """Content-addressed audit row for one official v2 candidate."""

    task: str
    ordinal: int
    trial_manifest_sha256: str
    classification: CleanCandidateClassification
    attempt_receipt_sha256: Optional[str]
    artifact_root_sha256: Optional[str]
    exception_code: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", require_nonempty(self.task, "task"))
        object.__setattr__(self, "ordinal", require_integer(self.ordinal, "ordinal"))
        object.__setattr__(
            self,
            "trial_manifest_sha256",
            require_sha256(self.trial_manifest_sha256, "trial_manifest_sha256"),
        )
        classification = (
            self.classification
            if isinstance(self.classification, CleanCandidateClassification)
            else CleanCandidateClassification(self.classification)
        )
        attempt = _optional_sha256(
            self.attempt_receipt_sha256, "attempt_receipt_sha256"
        )
        artifact = _optional_sha256(self.artifact_root_sha256, "artifact_root_sha256")
        exception = self.exception_code
        if exception is not None:
            exception = require_nonempty(exception, "exception_code")
        if classification is CleanCandidateClassification.VALID_OUTCOME:
            if attempt is None or artifact is None or exception is not None:
                raise CleanCampaignError("valid candidate provenance is incoherent")
        elif classification is CleanCandidateClassification.EXCEPTION_REPLACED:
            if attempt is None or exception is None:
                raise CleanCampaignError(
                    "replacement candidate provenance is incomplete"
                )
        elif any(item is not None for item in (attempt, artifact, exception)):
            raise CleanCampaignError("unattempted candidate cannot bind evidence")
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "attempt_receipt_sha256", attempt)
        object.__setattr__(self, "artifact_root_sha256", artifact)
        object.__setattr__(self, "exception_code", exception)

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "ordinal": self.ordinal,
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "classification": self.classification.value,
            "attempt_receipt_sha256": self.attempt_receipt_sha256,
            "artifact_root_sha256": self.artifact_root_sha256,
            "exception_code": self.exception_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> CleanCandidateProvenance:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise CleanCampaignError("candidate provenance fields mismatch")
        return cls(**cast(Any, dict(value)))


def _optional_sha256(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    return require_sha256(value, name)


def require_rate(value: object, name: str, optional: bool = False) -> Optional[float]:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CleanCampaignError(f"{name} must be a real rate")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CleanCampaignError(f"{name} must lie in [0, 1]")
    return result


def require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise CleanCampaignError(f"{name} must be bool")
    return value


def require_status_counts(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise CleanCampaignError("terminal_status_counts must be an object")
    known = {status.value for status in TerminalStatus}
    counts: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in known:
            raise CleanCampaignError("terminal_status_counts has an unknown status")
        counts[key] = require_integer(item, f"terminal_status_counts[{key}]")
    return cast(Mapping[str, int], freeze_value(counts))


def wilson_to_dict(value: Optional[BinomialInterval]) -> Optional[dict[str, object]]:
    if value is None:
        return None
    return {
        "estimate": value.estimate,
        "lower": value.lower,
        "upper": value.upper,
        "confidence_level": value.confidence_level,
        "success_count": value.success_count,
        "trial_count": value.trial_count,
    }


def wilson_from_dict(value: object) -> Optional[BinomialInterval]:
    if value is None:
        return None
    fields = set(BinomialInterval.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CleanCampaignError("Wilson interval fields mismatch")
    return BinomialInterval(**cast(Any, dict(value)))


def bootstrap_to_dict(value: Optional[IntervalEstimate]) -> Optional[dict[str, object]]:
    if value is None:
        return None
    return {
        "estimate": value.estimate,
        "lower": value.lower,
        "upper": value.upper,
        "confidence_level": value.confidence_level,
        "valid_resamples": value.valid_resamples,
        "task_count": value.task_count,
        "pair_count": value.pair_count,
    }


def bootstrap_from_dict(value: object) -> Optional[IntervalEstimate]:
    if value is None:
        return None
    fields = set(IntervalEstimate.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CleanCampaignError("bootstrap interval fields mismatch")
    return IntervalEstimate(**cast(Any, dict(value)))


@dataclass(frozen=True)
class CleanTaskSummary:
    task: str
    planned: int
    loaded: int
    missing: int
    protocol_invalid: int
    eligible: int
    successes: int
    failures: int
    ineligible: int
    eligible_success_rate: Optional[float]
    wilson: Optional[BinomialInterval]
    unique_initial_state_count: int
    terminal_status_counts: Mapping[str, int]
    target_valid_trial_count: Optional[int] = None
    candidate_trial_count: Optional[int] = None
    attempted_candidate_count: Optional[int] = None
    exception_replacement_count: Optional[int] = None
    unused_reserve_count: Optional[int] = None
    valid_outcome_count: Optional[int] = None
    missing_required_candidate_count: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", require_nonempty(self.task, "task"))
        for name in (
            "planned",
            "loaded",
            "missing",
            "protocol_invalid",
            "eligible",
            "successes",
            "failures",
            "ineligible",
            "unique_initial_state_count",
        ):
            object.__setattr__(self, name, require_integer(getattr(self, name), name))
        v2_values = self._normalize_v2_counts()
        if (
            (
                v2_values is None
                and self.planned != self.loaded + self.missing + self.protocol_invalid
            )
            or self.loaded != self.eligible + self.ineligible
            or self.eligible != self.successes + self.failures
            or self.unique_initial_state_count > self.loaded
        ):
            raise CleanCampaignError("clean task inventory counts are inconsistent")
        rate = require_rate(
            self.eligible_success_rate, "eligible_success_rate", optional=True
        )
        if self.eligible == 0:
            if rate is not None or self.wilson is not None:
                raise CleanCampaignError("empty task outcomes cannot have an interval")
        else:
            expected = self.successes / self.eligible
            if rate is None or not math.isclose(rate, expected, abs_tol=1e-15):
                raise CleanCampaignError("task success rate does not match counts")
            if (
                self.wilson is None
                or self.wilson.success_count != self.successes
                or self.wilson.trial_count != self.eligible
            ):
                raise CleanCampaignError("task Wilson interval does not match counts")
        statuses = require_status_counts(self.terminal_status_counts)
        if sum(statuses.values()) != self.loaded:
            raise CleanCampaignError("task terminal status counts do not match loaded")
        object.__setattr__(self, "eligible_success_rate", rate)
        object.__setattr__(self, "terminal_status_counts", statuses)

    def _normalize_v2_counts(self) -> Optional[tuple[int, ...]]:
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
        if all(item is None for item in raw):
            return None
        if any(item is None for item in raw):
            raise CleanCampaignError("v2 clean task counts must be all present")
        values = tuple(
            require_integer(cast(int, item), name) for name, item in zip(names, raw)
        )
        target, candidates, attempted, replaced, unused, valid, missing = values
        if (
            candidates != self.planned
            or attempted != valid + replaced
            or candidates != attempted + unused + missing
            or valid != self.eligible
            or valid > target
        ):
            raise CleanCampaignError("v2 clean task candidate counts are inconsistent")
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)
        return values

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "task": self.task,
            "planned": self.planned,
            "loaded": self.loaded,
            "missing": self.missing,
            "protocol_invalid": self.protocol_invalid,
            "eligible": self.eligible,
            "successes": self.successes,
            "failures": self.failures,
            "ineligible": self.ineligible,
            "eligible_success_rate": self.eligible_success_rate,
            "wilson": wilson_to_dict(self.wilson),
            "unique_initial_state_count": self.unique_initial_state_count,
            "terminal_status_counts": dict(self.terminal_status_counts),
        }
        if self.candidate_trial_count is not None:
            document.update(
                {
                    "target_valid_trial_count": self.target_valid_trial_count,
                    "candidate_trial_count": self.candidate_trial_count,
                    "attempted_candidate_count": self.attempted_candidate_count,
                    "exception_replacement_count": self.exception_replacement_count,
                    "unused_reserve_count": self.unused_reserve_count,
                    "valid_outcome_count": self.valid_outcome_count,
                    "missing_required_candidate_count": (
                        self.missing_required_candidate_count
                    ),
                }
            )
        return document

    @classmethod
    def from_dict(cls, value: object) -> CleanTaskSummary:
        all_fields = set(cls.__dataclass_fields__)
        v2_fields = {
            "target_valid_trial_count",
            "candidate_trial_count",
            "attempted_candidate_count",
            "exception_replacement_count",
            "unused_reserve_count",
            "valid_outcome_count",
            "missing_required_candidate_count",
        }
        if not isinstance(value, Mapping) or frozenset(value) not in {
            frozenset(all_fields),
            frozenset(all_fields - v2_fields),
        }:
            raise CleanCampaignError("clean task summary fields mismatch")
        kwargs = dict(value)
        kwargs["wilson"] = wilson_from_dict(value["wilson"])
        return cls(**cast(Any, kwargs))


@dataclass(frozen=True)
class ProtocolInvalidTrial:
    trial_manifest_sha256: str
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trial_manifest_sha256",
            require_sha256(self.trial_manifest_sha256, "trial_manifest_sha256"),
        )
        reason = require_nonempty(self.reason_code, "reason_code")
        if reason not in PROTOCOL_INVALID_REASON_CODES:
            raise CleanCampaignError("unknown protocol-invalid reason code")
        object.__setattr__(self, "reason_code", reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProtocolInvalidTrial:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise CleanCampaignError("protocol-invalid trial fields mismatch")
        return cls(**cast(Any, dict(value)))


__all__ = [
    "PROTOCOL_INVALID_REASON_CODES",
    "CleanCandidateClassification",
    "CleanCandidateProvenance",
    "CleanTaskSummary",
    "ProtocolInvalidTrial",
    "bootstrap_from_dict",
    "bootstrap_to_dict",
    "require_bool",
    "require_rate",
    "require_status_counts",
    "wilson_from_dict",
    "wilson_to_dict",
]
