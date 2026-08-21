"""Per-operator paired statistics, kept separate from macro aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from statistics import mean
from typing import Iterable, Optional

from robotactile_benchmark.reporting.contracts import OutcomeRecord, ReportingSpec
from robotactile_benchmark.reporting.statistics import (
    exact_mcnemar,
    holm_bonferroni,
    task_stratified_paired_bootstrap,
)
from robotactile_benchmark.reporting.summary_contracts import OperatorCellSummary
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.severity import severity_value

BaselineKey = tuple[str, str]
FaultKey = tuple[str, int]


def paired_values(
    group: Iterable[OutcomeRecord],
    clean: dict[BaselineKey, OutcomeRecord],
    no_touch: dict[BaselineKey, OutcomeRecord],
) -> list[tuple[OutcomeRecord, bool, Optional[bool], bool]]:
    """Return clean-fault pairs and an optional independently qualified control."""

    values = []
    for record in group:
        baseline_key = (record.task, record.pair_key)
        clean_record = clean[baseline_key]
        no_touch_record = no_touch[baseline_key]
        if not (record.score_eligible and clean_record.score_eligible):
            continue
        assert record.score_success is not None
        assert clean_record.score_success is not None
        no_touch_success = (
            no_touch_record.score_success if no_touch_record.score_eligible else None
        )
        values.append(
            (
                record,
                clean_record.score_success,
                no_touch_success,
                record.score_success,
            )
        )
    return values


def build_operator_cells(
    faults: dict[FaultKey, list[OutcomeRecord]],
    clean: dict[BaselineKey, OutcomeRecord],
    no_touch: dict[BaselineKey, OutcomeRecord],
    spec: ReportingSpec,
) -> tuple[OperatorCellSummary, ...]:
    """Compute per-cell effects, intervals, exact tests, and Holm adjustment."""

    severity_registry = load_severity_registry()["paths"]
    preliminary: list[OperatorCellSummary] = []
    raw_p_values: dict[str, float] = {}
    for (operator_id, severity), group in sorted(faults.items()):
        paired = paired_values(group, clean, no_touch)
        native_dose = float(severity_value(operator_id, severity))
        native_unit = str(severity_registry[operator_id]["unit"])
        if not paired:
            preliminary.append(
                OperatorCellSummary(
                    operator_id=operator_id,
                    severity_level=severity,
                    native_dose=native_dose,
                    native_unit=native_unit,
                    requested_count=len(group),
                    eligible_count=0,
                    success_rate=None,
                    paired_delta_sr=None,
                    no_touch_relative_delta=None,
                    paired_delta_lower=None,
                    paired_delta_upper=None,
                    mcnemar_p_value=None,
                    holm_adjusted_p_value=None,
                )
            )
            continue
        differences_by_task: dict[str, list[float]] = defaultdict(list)
        for record, clean_value, _, fault_value in paired:
            differences_by_task[record.task].append(
                float(fault_value) - float(clean_value)
            )
        interval = task_stratified_paired_bootstrap(
            differences_by_task,
            n_resamples=spec.bootstrap_resamples,
            confidence_level=spec.confidence_level,
            seed=spec.bootstrap_seed + severity,
        )
        mcnemar = exact_mcnemar(
            tuple(clean_value for _, clean_value, _, _ in paired),
            tuple(fault_value for _, _, _, fault_value in paired),
        )
        no_touch_differences = [
            float(fault_value) - float(no_touch_value)
            for _, _, no_touch_value, fault_value in paired
            if no_touch_value is not None
        ]
        no_touch_delta = (
            mean(no_touch_differences)
            if len(no_touch_differences) == len(paired)
            else None
        )
        hypothesis = f"{operator_id}:S{severity}"
        raw_p_values[hypothesis] = mcnemar.p_value
        preliminary.append(
            OperatorCellSummary(
                operator_id=operator_id,
                severity_level=severity,
                native_dose=native_dose,
                native_unit=native_unit,
                requested_count=len(group),
                eligible_count=len(paired),
                success_rate=mean(float(item[3]) for item in paired),
                paired_delta_sr=mean(
                    float(item[3]) - float(item[1]) for item in paired
                ),
                no_touch_relative_delta=no_touch_delta,
                paired_delta_lower=interval.lower,
                paired_delta_upper=interval.upper,
                mcnemar_p_value=mcnemar.p_value,
                holm_adjusted_p_value=mcnemar.p_value,
            )
        )
    adjusted = holm_bonferroni(raw_p_values) if raw_p_values else {}
    return tuple(
        replace(
            cell,
            holm_adjusted_p_value=adjusted[
                f"{cell.operator_id}:S{cell.severity_level}"
            ],
        )
        if cell.success_rate is not None
        else cell
        for cell in preliminary
    )
