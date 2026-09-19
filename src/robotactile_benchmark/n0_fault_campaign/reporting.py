from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Optional, Sequence, Tuple

from robotactile_benchmark.constants import (
    DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS,
    SEVERITY_REGISTRY_ID,
    SUPPORTED_SEVERITY_REGISTRY_IDS,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.reporting.contracts import operator_axis, require_sha256
from robotactile_benchmark.reporting.statistics import IntervalEstimate

N0_FAULT_SUMMARY_SEMANTIC_VERSION = "1.0"


def _rate(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _finite(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _retention(value: Optional[float], name: str) -> Optional[float]:
    result = _finite(value, name)
    if result is not None and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class N0FaultReportingSpec:
    system_id: str
    supported_operator_ids: Tuple[str, ...]
    contract_operator_ids: Tuple[str, ...]
    severity_levels: Tuple[int, ...]
    bootstrap_seed: int
    severity_registry: str = SEVERITY_REGISTRY_ID
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    semantic_version: str = N0_FAULT_SUMMARY_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        supported = tuple(sorted(self.supported_operator_ids))
        contract = tuple(sorted(self.contract_operator_ids))
        severities = tuple(sorted(self.severity_levels))
        if not isinstance(self.system_id, str) or not self.system_id:
            raise ValueError("system_id must be a non-empty string")
        if not supported or len(supported) != len(set(supported)):
            raise ValueError("supported_operator_ids must be non-empty and unique")
        if not contract or len(contract) != len(set(contract)):
            raise ValueError("contract_operator_ids must be non-empty and unique")
        for operator_id in contract:
            operator_axis(operator_id)
        if not set(supported) <= set(contract):
            raise ValueError("supported operators must be a contract subset")
        if not severities or len(severities) != len(set(severities)):
            raise ValueError("severity_levels must be non-empty and unique")
        if any(type(level) is not int or not 1 <= level <= 5 for level in severities):
            raise ValueError("severity levels must be exact integers in [1, 5]")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be a non-negative integer")
        if self.severity_registry not in SUPPORTED_SEVERITY_REGISTRY_IDS:
            raise ValueError("unsupported severity registry")
        if (
            self.severity_registry in DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS
            and severities != (5,)
        ):
            raise ValueError("diagnostic stress reporting requires level 5 only")
        if self.bootstrap_resamples != 10_000 or self.confidence_level != 0.95:
            raise ValueError(
                "N0 reporting requires 10,000 resamples and 95% confidence"
            )
        if self.semantic_version != N0_FAULT_SUMMARY_SEMANTIC_VERSION:
            raise ValueError("unsupported N0 reporting semantic version")
        object.__setattr__(self, "supported_operator_ids", supported)
        object.__setattr__(self, "contract_operator_ids", contract)
        object.__setattr__(self, "severity_levels", severities)

    @property
    def unsupported_operator_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(set(self.contract_operator_ids) - set(self.supported_operator_ids))
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeBreakdown:
    """Mutually exclusive dispositions; only model outcomes enter SR."""

    requested_count: int
    model_success_count: int
    model_failure_count: int
    infrastructure_failure_count: int
    validator_failure_count: int
    unsupported_contract_count: int
    other_ineligible_count: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("outcome counts must be non-negative integers")
        if self.requested_count != sum(values[1:]):
            raise ValueError("outcome breakdown does not cover every request")

    @property
    def score_denominator(self) -> int:
        return self.model_success_count + self.model_failure_count

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "score_denominator": self.score_denominator}


@dataclass(frozen=True)
class N0TaskSummary:
    task: str
    clean_outcome_count: int
    clean_success_rate: Optional[float]
    eligible_fault_pair_count: int
    scored_cell_count: int
    fault_success_rate: Optional[float]
    degradation: Optional[float]
    retention: Optional[float]

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must be non-empty")
        for name in (
            "clean_outcome_count",
            "eligible_fault_pair_count",
            "scored_cell_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("clean_success_rate", "fault_success_rate"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        object.__setattr__(self, "retention", _retention(self.retention, "retention"))
        object.__setattr__(
            self, "degradation", _finite(self.degradation, "degradation")
        )


@dataclass(frozen=True)
class N0OperatorCellSummary:
    operator_id: str
    severity_level: int
    native_dose: float
    native_unit: str
    requested_count: int
    eligible_pair_count: int
    clean_success_rate: Optional[float]
    fault_success_rate: Optional[float]
    degradation: Optional[float]
    retention: Optional[float]
    degradation_interval: Optional[IntervalEstimate]
    mcnemar_p_value: Optional[float]
    holm_adjusted_p_value: Optional[float]
    statistics_status: str
    outcomes: OutcomeBreakdown

    def __post_init__(self) -> None:
        operator_axis(self.operator_id)
        if not 1 <= self.severity_level <= 5 or not self.native_unit:
            raise ValueError("operator cell severity or unit is invalid")
        if not 0 <= self.eligible_pair_count <= self.requested_count:
            raise ValueError("operator cell pair coverage is invalid")
        for name in ("clean_success_rate", "fault_success_rate"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        object.__setattr__(self, "retention", _retention(self.retention, "retention"))
        object.__setattr__(
            self, "degradation", _finite(self.degradation, "degradation")
        )
        for name in ("mcnemar_p_value", "holm_adjusted_p_value"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        statistics = (
            self.clean_success_rate,
            self.fault_success_rate,
            self.degradation,
            self.degradation_interval,
            self.mcnemar_p_value,
            self.holm_adjusted_p_value,
        )
        if (self.statistics_status == "available") != all(
            item is not None for item in statistics
        ):
            raise ValueError("cell statistics and availability status disagree")
        if self.outcomes.requested_count != self.requested_count:
            raise ValueError("cell outcome breakdown count mismatch")

    @property
    def axis(self) -> str:
        return operator_axis(self.operator_id)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["axis"] = self.axis
        result["outcomes"] = self.outcomes.to_dict()
        return result


@dataclass(frozen=True)
class N0AxisSummary:
    axis: str
    operator_count: int
    scored_cell_count: int
    fault_success_rate: Optional[float]
    degradation: Optional[float]
    retention: Optional[float]


@dataclass(frozen=True)
class N0SeverityPoint:
    severity_level: int
    eligible_pair_count: int
    fault_success_rate: Optional[float]
    degradation: Optional[float]
    retention: Optional[float]


@dataclass(frozen=True)
class N0SeverityCurve:
    operator_id: str
    points: Tuple[N0SeverityPoint, ...]
    fault_success_rate_auc: Optional[float]
    degradation_auc: Optional[float]
    auc_status: str


@dataclass(frozen=True)
class N0FaultCampaignSummary:
    spec: N0FaultReportingSpec
    source_root_sha256: Tuple[str, ...]
    outcomes: OutcomeBreakdown
    tasks: Tuple[N0TaskSummary, ...]
    operator_cells: Tuple[N0OperatorCellSummary, ...]
    axes: Tuple[N0AxisSummary, ...]
    severity_curves: Tuple[N0SeverityCurve, ...]
    worst_cell: Optional[Tuple[str, int]]
    macro_clean_success_rate: Optional[float]
    macro_fault_success_rate: Optional[float]
    macro_degradation: Optional[float]
    macro_retention: Optional[float]
    degradation_interval: Optional[IntervalEstimate]
    statistically_complete: bool
    completeness_blockers: Tuple[str, ...]
    semantic_version: str = N0_FAULT_SUMMARY_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        sources = tuple(self.source_root_sha256)
        if not sources or sources != tuple(sorted(set(sources))):
            raise ValueError("source roots must be non-empty, unique, and sorted")
        for source in sources:
            require_sha256(source, "source_root_sha256")
        for name in (
            "macro_clean_success_rate",
            "macro_fault_success_rate",
        ):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        object.__setattr__(
            self, "macro_retention", _retention(self.macro_retention, "macro_retention")
        )
        object.__setattr__(
            self,
            "macro_degradation",
            _finite(self.macro_degradation, "macro_degradation"),
        )
        if self.statistically_complete != (not self.completeness_blockers):
            raise ValueError("statistical completeness disagrees with blockers")
        if self.semantic_version != N0_FAULT_SUMMARY_SEMANTIC_VERSION:
            raise ValueError("unsupported N0 summary semantic version")
        object.__setattr__(self, "source_root_sha256", sources)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "source_root_sha256": list(self.source_root_sha256),
            "outcomes": self.outcomes.to_dict(),
            "tasks": [asdict(item) for item in self.tasks],
            "operator_cells": [item.to_dict() for item in self.operator_cells],
            "axes": [asdict(item) for item in self.axes],
            "severity_curves": [asdict(item) for item in self.severity_curves],
            "worst_cell": None
            if self.worst_cell is None
            else {
                "operator_id": self.worst_cell[0],
                "severity_level": self.worst_cell[1],
            },
            "macro_clean_success_rate": self.macro_clean_success_rate,
            "macro_fault_success_rate": self.macro_fault_success_rate,
            "macro_degradation": self.macro_degradation,
            "macro_retention": self.macro_retention,
            "degradation_interval": None
            if self.degradation_interval is None
            else asdict(self.degradation_interval),
            "statistically_complete": self.statistically_complete,
            "completeness_blockers": list(self.completeness_blockers),
            "semantic_version": self.semantic_version,
        }


def _mean_available(
    cells: Sequence[N0OperatorCellSummary], name: str
) -> Optional[float]:
    values = [getattr(cell, name) for cell in cells if getattr(cell, name) is not None]
    return None if not values else mean(values)


def build_axis_summaries(
    cells: Sequence[N0OperatorCellSummary], spec: N0FaultReportingSpec
) -> Tuple[N0AxisSummary, ...]:
    result = []
    for axis in sorted({operator_axis(item) for item in spec.supported_operator_ids}):
        group = [cell for cell in cells if cell.axis == axis]
        result.append(
            N0AxisSummary(
                axis=axis,
                operator_count=sum(
                    operator_axis(item) == axis for item in spec.supported_operator_ids
                ),
                scored_cell_count=sum(
                    cell.statistics_status == "available" for cell in group
                ),
                fault_success_rate=_mean_available(group, "fault_success_rate"),
                degradation=_mean_available(group, "degradation"),
                retention=_mean_available(group, "retention"),
            )
        )
    return tuple(result)


def _normalized_auc(x: Sequence[int], y: Sequence[float]) -> float:
    width = x[-1] - x[0]
    return (
        sum((y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2.0 for i in range(len(x) - 1))
        / width
    )


def build_severity_curves(
    cells: Sequence[N0OperatorCellSummary], spec: N0FaultReportingSpec
) -> Tuple[N0SeverityCurve, ...]:
    result = []
    for operator_id in spec.supported_operator_ids:
        points = tuple(
            N0SeverityPoint(
                severity_level=cell.severity_level,
                eligible_pair_count=cell.eligible_pair_count,
                fault_success_rate=cell.fault_success_rate,
                degradation=cell.degradation,
                retention=cell.retention,
            )
            for cell in cells
            if cell.operator_id == operator_id
        )
        complete, enough = (
            all(point.fault_success_rate is not None for point in points),
            len(points) >= 2,
        )
        x = [point.severity_level for point in points]
        fault = [
            float(point.fault_success_rate)
            for point in points
            if point.fault_success_rate is not None
        ]
        degradation = [
            float(point.degradation)
            for point in points
            if point.degradation is not None
        ]
        result.append(
            N0SeverityCurve(
                operator_id=operator_id,
                points=points,
                fault_success_rate_auc=_normalized_auc(x, fault)
                if complete and enough
                else None,
                degradation_auc=_normalized_auc(x, degradation)
                if complete and enough
                else None,
                auc_status="available"
                if complete and enough
                else "fewer_than_two_levels"
                if complete
                else "incomplete_curve",
            )
        )
    return tuple(result)
