"""Primary-grid completeness and terminal-state coverage accounting."""

from __future__ import annotations

from collections import Counter

from robotactile_benchmark.reporting.contracts import OutcomeRecord, ReportingSpec
from robotactile_benchmark.reporting.summary_contracts import CoverageSummary

BaselineKey = tuple[str, str]
FaultKey = tuple[str, int]


def _primary_records(
    faults: dict[FaultKey, list[OutcomeRecord]], spec: ReportingSpec
) -> tuple[OutcomeRecord, ...]:
    return tuple(
        record
        for (operator, severity), group in faults.items()
        if operator in spec.primary_operator_ids
        and severity in spec.primary_severity_levels
        for record in group
    )


def primary_task_score_complete(
    task: str,
    clean: dict[BaselineKey, OutcomeRecord],
    faults: dict[FaultKey, list[OutcomeRecord]],
    spec: ReportingSpec,
) -> bool:
    """Require every registered task/pair/operator/severity outcome to be scored."""

    expected = {
        (pair_key, operator, severity)
        for task_id, pair_key in clean
        if task_id == task
        for operator in spec.primary_operator_ids
        for severity in spec.primary_severity_levels
    }
    observed = {
        (record.pair_key, record.operator_id, record.severity_level)
        for record in _primary_records(faults, spec)
        if record.task == task
        and record.score_eligible
        and clean[(record.task, record.pair_key)].score_eligible
    }
    return observed == expected


def build_coverage_summary(
    records: tuple[OutcomeRecord, ...],
    clean: dict[BaselineKey, OutcomeRecord],
    faults: dict[FaultKey, list[OutcomeRecord]],
    spec: ReportingSpec,
) -> CoverageSummary:
    """Retain all terminal states and separately expose primary completeness."""

    expected = (
        len(clean) * len(spec.primary_operator_ids) * len(spec.primary_severity_levels)
    )
    primary = _primary_records(faults, spec)
    observed = len(primary)
    scored = sum(
        record.score_eligible and clean[(record.task, record.pair_key)].score_eligible
        for record in primary
    )
    terminal_counts = Counter(record.terminal_status.value for record in records)
    return CoverageSummary(
        requested_count=len(records),
        eligible_count=sum(record.score_eligible for record in records),
        ineligible_count=sum(not record.score_eligible for record in records),
        expected_primary_cell_count=expected,
        observed_primary_cell_count=observed,
        scored_primary_cell_count=scored,
        primary_grid_complete=observed == expected,
        primary_score_complete=scored == expected,
        terminal_counts=tuple(sorted(terminal_counts.items())),
    )
