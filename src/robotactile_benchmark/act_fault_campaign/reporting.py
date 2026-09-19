"""Clean/Faulted-only reporting for official ACT robustness campaigns."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, cast

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.reporting.contracts import require_sha256
from robotactile_benchmark.trials import Condition

from .reporting_inputs import CellOutcome, load_campaign_inputs

ACT_FAULT_REPORT_SEMANTIC_VERSION = "1.0"


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_metric(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ACTFaultDispositionBreakdown:
    """Mutually exclusive dispositions; only model outcomes enter SR."""

    requested_count: int
    model_success_count: int
    model_failure_count: int
    unsupported_contract_count: int
    infrastructure_failure_count: int
    validator_failure_count: int
    missing_artifact_count: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("ACT disposition counts must be non-negative integers")
        if self.requested_count != sum(values[1:]):
            raise ValueError("ACT dispositions do not cover every requested cell")

    @property
    def score_denominator(self) -> int:
        return self.model_success_count + self.model_failure_count

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "score_denominator": self.score_denominator}


@dataclass(frozen=True)
class ACTFaultOperatorCellReport:
    operator_id: str
    severity_level: int
    planned_count: int
    eligible_pair_count: int
    clean_success_count: int
    clean_score_denominator: int
    faulted_success_count: int
    faulted_score_denominator: int
    clean_success_rate: Optional[float]
    faulted_success_rate: Optional[float]
    delta_success_rate: Optional[float]
    retention: Optional[float]
    dispositions: ACTFaultDispositionBreakdown

    def __post_init__(self) -> None:
        if self.operator_id not in CORE_OPERATOR_IDS:
            raise ValueError("unknown ACT fault operator")
        if not 1 <= self.severity_level <= 5:
            raise ValueError("ACT severity level must lie in [1, 5]")
        for name in (
            "planned_count",
            "eligible_pair_count",
            "clean_success_count",
            "clean_score_denominator",
            "faulted_success_count",
            "faulted_score_denominator",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.clean_success_count > self.clean_score_denominator:
            raise ValueError("clean successes exceed the cell denominator")
        if self.faulted_success_count > self.faulted_score_denominator:
            raise ValueError("faulted successes exceed the cell denominator")
        if self.dispositions.requested_count != self.planned_count:
            raise ValueError("cell disposition inventory mismatch")
        for name in (
            "clean_success_rate",
            "faulted_success_rate",
            "delta_success_rate",
            "retention",
        ):
            object.__setattr__(self, name, _optional_metric(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "dispositions": self.dispositions.to_dict()}


@dataclass(frozen=True)
class ACTFaultCampaignReport:
    campaign_id: str
    campaign_manifest_sha256: str
    clean_success_count: int
    clean_score_denominator: int
    faulted_success_count: int
    faulted_score_denominator: int
    clean_success_rate: Optional[float]
    faulted_success_rate: Optional[float]
    delta_success_rate: Optional[float]
    retention: Optional[float]
    dispositions: ACTFaultDispositionBreakdown
    operator_cells: Tuple[ACTFaultOperatorCellReport, ...]
    complete: bool
    completeness_blockers: Tuple[str, ...]
    semantic_version: str = ACT_FAULT_REPORT_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        _nonempty_text(self.campaign_id, "campaign_id")
        require_sha256(self.campaign_manifest_sha256, "campaign_manifest_sha256")
        for name in (
            "clean_success_count",
            "clean_score_denominator",
            "faulted_success_count",
            "faulted_score_denominator",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.clean_success_count > self.clean_score_denominator:
            raise ValueError("clean successes exceed the campaign denominator")
        if self.faulted_success_count > self.faulted_score_denominator:
            raise ValueError("faulted successes exceed the campaign denominator")
        for name in (
            "clean_success_rate",
            "faulted_success_rate",
            "delta_success_rate",
            "retention",
        ):
            object.__setattr__(self, name, _optional_metric(getattr(self, name), name))
        blockers = tuple(sorted(set(self.completeness_blockers)))
        if self.complete != (not blockers):
            raise ValueError("ACT report completeness disagrees with blockers")
        if self.semantic_version != ACT_FAULT_REPORT_SEMANTIC_VERSION:
            raise ValueError("unsupported ACT fault report semantic version")
        object.__setattr__(self, "operator_cells", tuple(self.operator_cells))
        object.__setattr__(self, "completeness_blockers", blockers)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "clean_success_count": self.clean_success_count,
            "clean_score_denominator": self.clean_score_denominator,
            "faulted_success_count": self.faulted_success_count,
            "faulted_score_denominator": self.faulted_score_denominator,
            "clean_success_rate": self.clean_success_rate,
            "faulted_success_rate": self.faulted_success_rate,
            "delta_success_rate": self.delta_success_rate,
            "retention": self.retention,
            "dispositions": self.dispositions.to_dict(),
            "operator_cells": [item.to_dict() for item in self.operator_cells],
            "complete": self.complete,
            "completeness_blockers": list(self.completeness_blockers),
            "semantic_version": self.semantic_version,
        }


def _breakdown(outcomes: Sequence[CellOutcome]) -> ACTFaultDispositionBreakdown:
    counts = Counter(item.category for item in outcomes)
    return ACTFaultDispositionBreakdown(
        requested_count=len(outcomes),
        model_success_count=counts["model_success"],
        model_failure_count=counts["model_failure"],
        unsupported_contract_count=counts["unsupported_contract"],
        infrastructure_failure_count=counts["infrastructure_failure"],
        validator_failure_count=counts["validator_failure"],
        missing_artifact_count=counts["missing_artifact"],
    )


def _metrics(scores: Sequence[bool]) -> tuple[int, int, Optional[float]]:
    denominator = len(scores)
    successes = sum(scores)
    return successes, denominator, None if denominator == 0 else successes / denominator


def _comparison(
    clean_rate: Optional[float], faulted_rate: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    if clean_rate is None or faulted_rate is None:
        return None, None
    delta = clean_rate - faulted_rate
    retention = None if clean_rate == 0.0 else faulted_rate / clean_rate
    return delta, retention


def _operator_cells(
    outcomes: Sequence[CellOutcome],
) -> Tuple[ACTFaultOperatorCellReport, ...]:
    clean = {
        (item.cell.task, item.cell.pair_key): item
        for item in outcomes
        if item.cell.condition is Condition.CLEAN
    }
    groups: dict[tuple[str, int], list[CellOutcome]] = defaultdict(list)
    for item in outcomes:
        if item.cell.condition is Condition.FAULTED:
            assert item.cell.operator_id is not None
            assert item.cell.severity_level is not None
            groups[(item.cell.operator_id, item.cell.severity_level)].append(item)
    reports = []
    for (operator_id, severity), group in sorted(groups.items()):
        pairs = []
        for fault in group:
            baseline = clean.get((fault.cell.task, fault.cell.pair_key))
            if (
                baseline is not None
                and baseline.score is not None
                and fault.score is not None
            ):
                pairs.append((baseline.score, fault.score))
        clean_success, clean_denominator, clean_rate = _metrics(
            [item[0] for item in pairs]
        )
        fault_success, fault_denominator, fault_rate = _metrics(
            [item[1] for item in pairs]
        )
        delta, retention = _comparison(clean_rate, fault_rate)
        reports.append(
            ACTFaultOperatorCellReport(
                operator_id=operator_id,
                severity_level=severity,
                planned_count=len(group),
                eligible_pair_count=len(pairs),
                clean_success_count=clean_success,
                clean_score_denominator=clean_denominator,
                faulted_success_count=fault_success,
                faulted_score_denominator=fault_denominator,
                clean_success_rate=clean_rate,
                faulted_success_rate=fault_rate,
                delta_success_rate=delta,
                retention=retention,
                dispositions=_breakdown(group),
            )
        )
    return tuple(reports)


def build_act_fault_campaign_report(
    manifest: object,
    artifacts_by_cell: Mapping[object, object],
) -> ACTFaultCampaignReport:
    """Aggregate one executed campaign without importing its concrete contracts."""

    inputs = load_campaign_inputs(manifest, artifacts_by_cell)
    outcomes = inputs.outcomes
    clean_outcomes = [
        item for item in outcomes if item.cell.condition is Condition.CLEAN
    ]
    fault_outcomes = [
        item for item in outcomes if item.cell.condition is Condition.FAULTED
    ]
    if not clean_outcomes or not fault_outcomes:
        raise ValueError("ACT reporting requires both Clean and Faulted cells")
    clean_success, clean_denominator, clean_rate = _metrics(
        [cast(bool, item.score) for item in clean_outcomes if item.score is not None]
    )
    fault_success, fault_denominator, fault_rate = _metrics(
        [cast(bool, item.score) for item in fault_outcomes if item.score is not None]
    )
    delta, retention = _comparison(clean_rate, fault_rate)
    breakdown = _breakdown(outcomes)
    blockers = []
    if breakdown.missing_artifact_count:
        blockers.append("missing_artifacts")
    if breakdown.infrastructure_failure_count:
        blockers.append("infrastructure_failures")
    if breakdown.validator_failure_count:
        blockers.append("validator_failures")
    if any(
        item.category == "unsupported_contract"
        and item.cell.disposition != "unsupported_contract"
        for item in outcomes
    ):
        blockers.append("unexpected_unsupported_contract")
    clean_pairs = {
        (item.cell.task, item.cell.pair_key)
        for item in clean_outcomes
        if item.score is not None
    }
    if any(
        item.score is not None
        and (item.cell.task, item.cell.pair_key) not in clean_pairs
        for item in fault_outcomes
    ):
        blockers.append("unpaired_scored_faults")
    if clean_denominator == 0:
        blockers.append("no_clean_scores")
    if fault_denominator == 0:
        blockers.append("no_faulted_scores")
    return ACTFaultCampaignReport(
        campaign_id=inputs.campaign_id,
        campaign_manifest_sha256=inputs.campaign_manifest_sha256,
        clean_success_count=clean_success,
        clean_score_denominator=clean_denominator,
        faulted_success_count=fault_success,
        faulted_score_denominator=fault_denominator,
        clean_success_rate=clean_rate,
        faulted_success_rate=fault_rate,
        delta_success_rate=delta,
        retention=retention,
        dispositions=breakdown,
        operator_cells=_operator_cells(outcomes),
        complete=not blockers,
        completeness_blockers=tuple(blockers),
    )
