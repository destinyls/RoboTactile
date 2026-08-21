"""Frozen aggregate contracts shared by JSON, CSV, LaTeX, and SVG exports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.reporting.contracts import (
    REPORTING_SEMANTIC_VERSION,
    ReportingSpec,
    operator_axis,
    require_sha256,
)
from robotactile_benchmark.reporting.statistics import IntervalEstimate


def _rate(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _finite_optional(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CoverageSummary:
    """Coverage accounting that retains every requested terminal state."""

    requested_count: int
    eligible_count: int
    ineligible_count: int
    expected_primary_cell_count: int
    observed_primary_cell_count: int
    scored_primary_cell_count: int
    primary_grid_complete: bool
    primary_score_complete: bool
    terminal_counts: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        if min(self.requested_count, self.eligible_count, self.ineligible_count) < 0:
            raise ValueError("coverage counts must be non-negative")
        if self.eligible_count + self.ineligible_count != self.requested_count:
            raise ValueError("coverage counts do not sum to requested_count")
        if not (
            0
            <= self.scored_primary_cell_count
            <= self.observed_primary_cell_count
            <= self.expected_primary_cell_count
        ):
            raise ValueError("primary coverage counts are inconsistent")
        if (
            type(self.primary_grid_complete) is not bool
            or type(self.primary_score_complete) is not bool
        ):
            raise TypeError("primary completeness flags must be bool")
        if self.primary_grid_complete != (
            self.observed_primary_cell_count == self.expected_primary_cell_count
        ):
            raise ValueError("primary_grid_complete disagrees with coverage counts")
        if self.primary_score_complete != (
            self.scored_primary_cell_count == self.expected_primary_cell_count
        ):
            raise ValueError("primary_score_complete disagrees with coverage counts")
        counts = tuple(self.terminal_counts)
        if counts != tuple(sorted(counts)) or len({key for key, _ in counts}) != len(
            counts
        ):
            raise ValueError("terminal counts must be unique and sorted")
        if any(not key or value < 0 for key, value in counts):
            raise ValueError("terminal count entries are invalid")
        if sum(value for _, value in counts) != self.requested_count:
            raise ValueError("terminal counts do not cover all requested outcomes")
        object.__setattr__(self, "terminal_counts", counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_count": self.requested_count,
            "eligible_count": self.eligible_count,
            "ineligible_count": self.ineligible_count,
            "expected_primary_cell_count": self.expected_primary_cell_count,
            "observed_primary_cell_count": self.observed_primary_cell_count,
            "scored_primary_cell_count": self.scored_primary_cell_count,
            "primary_grid_complete": self.primary_grid_complete,
            "primary_score_complete": self.primary_score_complete,
            "terminal_counts": [list(item) for item in self.terminal_counts],
        }


@dataclass(frozen=True)
class TaskSummary:
    """One equally weighted task-level reliability estimate."""

    task: str
    pair_count: int
    clean_sr: float
    no_touch_sr: Optional[float]
    clean_gain: Optional[float]
    fault_sr: float
    paired_delta_sr: float
    no_touch_relative_fault_delta: Optional[float]
    tgr: Optional[float]
    tgr_ineligibility_reason: Optional[str]
    axis_fault_sr: Tuple[Tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.task or self.pair_count <= 0:
            raise ValueError("task summary requires a task and matched pairs")
        for name in ("clean_sr", "fault_sr"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        object.__setattr__(self, "no_touch_sr", _rate(self.no_touch_sr, "no_touch_sr"))
        for name in (
            "clean_gain",
            "paired_delta_sr",
            "no_touch_relative_fault_delta",
            "tgr",
        ):
            object.__setattr__(self, name, _finite_optional(getattr(self, name), name))
        if (self.tgr is None) == (self.tgr_ineligibility_reason is None):
            raise ValueError("TGR value and ineligibility reason must be exclusive")
        axes = tuple(self.axis_fault_sr)
        if axes != tuple(sorted(axes)) or len({axis for axis, _ in axes}) != len(axes):
            raise ValueError("axis summaries must be unique and sorted")
        for _, value in axes:
            _rate(value, "axis fault success rate")
        object.__setattr__(self, "axis_fault_sr", axes)

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "pair_count": self.pair_count,
            "clean_sr": self.clean_sr,
            "no_touch_sr": self.no_touch_sr,
            "clean_gain": self.clean_gain,
            "fault_sr": self.fault_sr,
            "paired_delta_sr": self.paired_delta_sr,
            "no_touch_relative_fault_delta": self.no_touch_relative_fault_delta,
            "tgr": self.tgr,
            "tgr_ineligibility_reason": self.tgr_ineligibility_reason,
            "axis_fault_sr": [list(item) for item in self.axis_fault_sr],
        }


@dataclass(frozen=True)
class OperatorCellSummary:
    """One operator-native severity point; doses are never pooled cross-operator."""

    operator_id: str
    severity_level: int
    native_dose: float
    native_unit: str
    requested_count: int
    eligible_count: int
    success_rate: Optional[float]
    paired_delta_sr: Optional[float]
    no_touch_relative_delta: Optional[float]
    paired_delta_lower: Optional[float]
    paired_delta_upper: Optional[float]
    mcnemar_p_value: Optional[float]
    holm_adjusted_p_value: Optional[float]

    def __post_init__(self) -> None:
        operator_axis(self.operator_id)
        if not 1 <= self.severity_level <= 5:
            raise ValueError("severity_level must be in [1, 5]")
        if not self.native_unit:
            raise ValueError("native_unit must be non-empty")
        if (
            self.requested_count <= 0
            or not 0 <= self.eligible_count <= self.requested_count
        ):
            raise ValueError("operator cell coverage is invalid")
        object.__setattr__(self, "native_dose", float(self.native_dose))
        object.__setattr__(
            self, "success_rate", _rate(self.success_rate, "success_rate")
        )
        for name in (
            "paired_delta_sr",
            "no_touch_relative_delta",
            "paired_delta_lower",
            "paired_delta_upper",
        ):
            object.__setattr__(self, name, _finite_optional(getattr(self, name), name))
        for name in ("mcnemar_p_value", "holm_adjusted_p_value"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        populated = self.success_rate is not None
        fields = (
            self.paired_delta_sr,
            self.paired_delta_lower,
            self.paired_delta_upper,
            self.mcnemar_p_value,
            self.holm_adjusted_p_value,
        )
        if populated != all(value is not None for value in fields):
            raise ValueError(
                "operator statistics must be all populated or all unavailable"
            )
        if not populated and self.no_touch_relative_delta is not None:
            raise ValueError("unscored operator cell cannot contain no-touch delta")

    @property
    def axis(self) -> str:
        return operator_axis(self.operator_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "axis": self.axis,
            "severity_level": self.severity_level,
            "native_dose": self.native_dose,
            "native_unit": self.native_unit,
            "requested_count": self.requested_count,
            "eligible_count": self.eligible_count,
            "success_rate": self.success_rate,
            "paired_delta_sr": self.paired_delta_sr,
            "no_touch_relative_delta": self.no_touch_relative_delta,
            "paired_delta_lower": self.paired_delta_lower,
            "paired_delta_upper": self.paired_delta_upper,
            "mcnemar_p_value": self.mcnemar_p_value,
            "holm_adjusted_p_value": self.holm_adjusted_p_value,
        }


@dataclass(frozen=True)
class RecoverySummary:
    """Post-restoration recovery without treating missing recovery as zero lag."""

    eligible_count: int
    recovered_count: int
    unrecovered_fraction: Optional[float]
    median_lag_steps: Optional[float]
    mean_lag_steps: Optional[float]

    def __post_init__(self) -> None:
        if (
            self.eligible_count < 0
            or not 0 <= self.recovered_count <= self.eligible_count
        ):
            raise ValueError("recovery counts are invalid")
        object.__setattr__(
            self,
            "unrecovered_fraction",
            _rate(self.unrecovered_fraction, "unrecovered_fraction"),
        )
        for name in ("median_lag_steps", "mean_lag_steps"):
            value = _finite_optional(getattr(self, name), name)
            if value is not None and value < 0.0:
                raise ValueError("recovery lag must be non-negative")
            object.__setattr__(self, name, value)
        if self.eligible_count == 0 and any(
            value is not None
            for value in (
                self.unrecovered_fraction,
                self.median_lag_steps,
                self.mean_lag_steps,
            )
        ):
            raise ValueError("ineligible recovery summary cannot contain estimates")

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible_count": self.eligible_count,
            "recovered_count": self.recovered_count,
            "unrecovered_fraction": self.unrecovered_fraction,
            "median_lag_steps": self.median_lag_steps,
            "mean_lag_steps": self.mean_lag_steps,
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    """Complete system-level reporting payload bound to source run receipts."""

    spec: ReportingSpec
    source_root_sha256: Tuple[str, ...]
    tasks: Tuple[TaskSummary, ...]
    operator_cells: Tuple[OperatorCellSummary, ...]
    worst_cell: OperatorCellSummary
    recovery: RecoverySummary
    coverage: CoverageSummary
    macro_clean_sr: float
    macro_no_touch_sr: Optional[float]
    macro_fault_sr: float
    macro_paired_delta_sr: float
    macro_tgr: Optional[float]
    paired_delta_interval: IntervalEstimate
    semantic_version: str = REPORTING_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        sources = tuple(self.source_root_sha256)
        if not sources or sources != tuple(sorted(set(sources))):
            raise ValueError("source roots must be non-empty, unique, and sorted")
        for value in sources:
            require_sha256(value, "source_root_sha256")
        tasks = tuple(self.tasks)
        cells = tuple(self.operator_cells)
        if not tasks or not cells:
            raise ValueError("benchmark summary requires task and operator results")
        if tasks != tuple(sorted(tasks, key=lambda item: item.task)):
            raise ValueError("task summaries must be sorted")
        if cells != tuple(
            sorted(cells, key=lambda item: (item.operator_id, item.severity_level))
        ):
            raise ValueError("operator cells must be sorted")
        if self.worst_cell not in cells or self.worst_cell.success_rate is None:
            raise ValueError("worst cell must identify a scored operator cell")
        for name in ("macro_clean_sr", "macro_fault_sr"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        object.__setattr__(
            self,
            "macro_no_touch_sr",
            _rate(self.macro_no_touch_sr, "macro_no_touch_sr"),
        )
        for name in ("macro_paired_delta_sr", "macro_tgr"):
            object.__setattr__(self, name, _finite_optional(getattr(self, name), name))
        if self.semantic_version != REPORTING_SEMANTIC_VERSION:
            raise ValueError("unsupported benchmark summary version")
        object.__setattr__(self, "source_root_sha256", sources)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "operator_cells", cells)

    @property
    def native_dose_curves(self) -> Tuple[OperatorCellSummary, ...]:
        return self.operator_cells

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "source_root_sha256": list(self.source_root_sha256),
            "tasks": [item.to_dict() for item in self.tasks],
            "operator_cells": [item.to_dict() for item in self.operator_cells],
            "worst_cell": {
                "operator_id": self.worst_cell.operator_id,
                "severity_level": self.worst_cell.severity_level,
            },
            "recovery": self.recovery.to_dict(),
            "coverage": self.coverage.to_dict(),
            "macro_clean_sr": self.macro_clean_sr,
            "macro_no_touch_sr": self.macro_no_touch_sr,
            "macro_fault_sr": self.macro_fault_sr,
            "macro_paired_delta_sr": self.macro_paired_delta_sr,
            "macro_tgr": self.macro_tgr,
            "paired_delta_interval": self.paired_delta_interval.__dict__,
            "semantic_version": self.semantic_version,
        }
