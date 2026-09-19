"""Strict JSON reconstruction for benchmark report summaries."""

from __future__ import annotations

from typing import Any

from robotactile_benchmark.reporting.contracts import ReportingSpec
from robotactile_benchmark.reporting.statistics import IntervalEstimate
from robotactile_benchmark.reporting.summary_contracts import (
    BenchmarkSummary,
    CoverageSummary,
    OperatorCellSummary,
    TaskSummary,
)


def _exact_dict(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields mismatch")
    return value


def _task(value: object) -> TaskSummary:
    fields = set(TaskSummary.__dataclass_fields__)
    data = _exact_dict(value, fields, "task summary")
    axes = data["axis_fault_sr"]
    if not isinstance(axes, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in axes
    ):
        raise ValueError("task axis summary must be a pair list")
    return TaskSummary(
        task=data["task"],
        pair_count=data["pair_count"],
        clean_sr=data["clean_sr"],
        no_touch_sr=data["no_touch_sr"],
        clean_gain=data["clean_gain"],
        fault_sr=data["fault_sr"],
        paired_delta_sr=data["paired_delta_sr"],
        no_touch_relative_fault_delta=data["no_touch_relative_fault_delta"],
        tgr=data["tgr"],
        tgr_ineligibility_reason=data["tgr_ineligibility_reason"],
        axis_fault_sr=tuple((item[0], item[1]) for item in axes),
    )


def _operator_cell(value: object) -> OperatorCellSummary:
    fields = set(OperatorCellSummary.__dataclass_fields__) | {"axis"}
    data = _exact_dict(value, fields, "operator cell")
    cell = OperatorCellSummary(
        operator_id=data["operator_id"],
        severity_level=data["severity_level"],
        native_dose=data["native_dose"],
        native_unit=data["native_unit"],
        requested_count=data["requested_count"],
        eligible_count=data["eligible_count"],
        success_rate=data["success_rate"],
        paired_delta_sr=data["paired_delta_sr"],
        no_touch_relative_delta=data["no_touch_relative_delta"],
        paired_delta_lower=data["paired_delta_lower"],
        paired_delta_upper=data["paired_delta_upper"],
        mcnemar_p_value=data["mcnemar_p_value"],
        holm_adjusted_p_value=data["holm_adjusted_p_value"],
    )
    if data["axis"] != cell.axis:
        raise ValueError("operator cell axis disagrees with operator ID")
    return cell


def summary_from_dict(value: object) -> BenchmarkSummary:
    """Reconstruct a summary while rejecting cached or unknown fields."""

    data = _exact_dict(value, set(BenchmarkSummary.__dataclass_fields__), "summary")
    sources = data["source_root_sha256"]
    tasks = data["tasks"]
    cells = data["operator_cells"]
    if (
        not isinstance(sources, list)
        or not isinstance(tasks, list)
        or not isinstance(cells, list)
    ):
        raise ValueError("summary inventories must be lists")
    worst_data = _exact_dict(
        data["worst_cell"],
        {"operator_id", "severity_level"},
        "worst cell pointer",
    )
    decoded_cells = tuple(_operator_cell(item) for item in cells)
    matches = tuple(
        item
        for item in decoded_cells
        if item.operator_id == worst_data["operator_id"]
        and item.severity_level == worst_data["severity_level"]
    )
    if len(matches) != 1:
        raise ValueError("worst cell pointer is not unique")
    coverage_data = _exact_dict(
        data["coverage"], set(CoverageSummary.__dataclass_fields__), "coverage"
    )
    terminal_counts = coverage_data["terminal_counts"]
    if not isinstance(terminal_counts, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in terminal_counts
    ):
        raise ValueError("terminal counts must be a pair list")
    interval_data = _exact_dict(
        data["paired_delta_interval"],
        set(IntervalEstimate.__dataclass_fields__),
        "paired interval",
    )
    return BenchmarkSummary(
        spec=ReportingSpec.from_dict(data["spec"]),
        source_root_sha256=tuple(sources),
        tasks=tuple(_task(item) for item in tasks),
        operator_cells=decoded_cells,
        worst_cell=matches[0],
        coverage=CoverageSummary(
            requested_count=coverage_data["requested_count"],
            eligible_count=coverage_data["eligible_count"],
            ineligible_count=coverage_data["ineligible_count"],
            expected_primary_cell_count=coverage_data["expected_primary_cell_count"],
            observed_primary_cell_count=coverage_data["observed_primary_cell_count"],
            scored_primary_cell_count=coverage_data["scored_primary_cell_count"],
            primary_grid_complete=coverage_data["primary_grid_complete"],
            primary_score_complete=coverage_data["primary_score_complete"],
            terminal_counts=tuple((item[0], item[1]) for item in terminal_counts),
        ),
        macro_clean_sr=data["macro_clean_sr"],
        macro_no_touch_sr=data["macro_no_touch_sr"],
        macro_fault_sr=data["macro_fault_sr"],
        macro_paired_delta_sr=data["macro_paired_delta_sr"],
        macro_tgr=data["macro_tgr"],
        paired_delta_interval=IntervalEstimate(**interval_data),
        semantic_version=data["semantic_version"],
    )
