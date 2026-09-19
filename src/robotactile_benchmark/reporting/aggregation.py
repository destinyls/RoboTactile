"""Paired aggregation for clean/faulted/no-touch result matrices."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, Optional

from robotactile_benchmark.metrics import tactile_gain_retention
from robotactile_benchmark.reporting.cells import build_operator_cells, paired_values
from robotactile_benchmark.reporting.contracts import OutcomeRecord, ReportingSpec
from robotactile_benchmark.reporting.coverage import (
    build_coverage_summary,
    primary_task_score_complete,
)
from robotactile_benchmark.reporting.statistics import (
    task_stratified_paired_bootstrap,
)
from robotactile_benchmark.reporting.summary_contracts import (
    BenchmarkSummary,
    TaskSummary,
)
from robotactile_benchmark.trials import Condition

BaselineKey = tuple[str, str]
FaultKey = tuple[str, int]


def _success(record: OutcomeRecord) -> Optional[float]:
    if not record.score_eligible:
        return None
    return 1.0 if record.score_success else 0.0


def _required_success(record: OutcomeRecord) -> float:
    value = _success(record)
    if value is None:
        raise ValueError("reporting outcome is not score eligible")
    return value


def _index_records(
    records: tuple[OutcomeRecord, ...], spec: ReportingSpec
) -> tuple[
    dict[BaselineKey, OutcomeRecord],
    dict[BaselineKey, OutcomeRecord],
    dict[FaultKey, list[OutcomeRecord]],
]:
    seen: set[tuple[object, ...]] = set()
    clean: dict[BaselineKey, OutcomeRecord] = {}
    no_touch: dict[BaselineKey, OutcomeRecord] = {}
    faults: dict[FaultKey, list[OutcomeRecord]] = defaultdict(list)
    for record in records:
        if record.system_id != spec.system_id:
            raise ValueError("outcome system does not match reporting spec")
        if record.cell_key in seen:
            raise ValueError("duplicate reporting cell")
        seen.add(record.cell_key)
        baseline_key = (record.task, record.pair_key)
        if record.condition is Condition.CLEAN:
            clean[baseline_key] = record
        elif record.condition is Condition.NO_TOUCH:
            no_touch[baseline_key] = record
        elif record.condition is Condition.FAULTED:
            assert record.operator_id is not None
            assert record.severity_level is not None
            faults[(record.operator_id, record.severity_level)].append(record)
    if set(clean) != set(no_touch):
        raise ValueError("every pair requires matched clean and no-touch baselines")
    for group in faults.values():
        if any((item.task, item.pair_key) not in clean for item in group):
            raise ValueError("fault outcome lacks matched clean and no-touch baselines")
    missing_operators = set(spec.primary_operator_ids) - {
        operator for operator, _ in faults
    }
    if missing_operators:
        raise ValueError(
            f"primary operator outcomes are missing: {sorted(missing_operators)}"
        )
    return clean, no_touch, faults


def _pair_fault_scores(
    faults: dict[FaultKey, list[OutcomeRecord]], spec: ReportingSpec
) -> dict[BaselineKey, float]:
    by_pair_operator: dict[tuple[BaselineKey, str], list[float]] = defaultdict(list)
    for (operator, severity), group in faults.items():
        if operator not in spec.primary_operator_ids:
            continue
        if severity not in spec.primary_severity_levels:
            continue
        for record in group:
            value = _success(record)
            if value is not None:
                by_pair_operator[((record.task, record.pair_key), operator)].append(
                    value
                )
    by_pair_axis: dict[tuple[BaselineKey, str], list[float]] = defaultdict(list)
    for (pair, operator), values in by_pair_operator.items():
        axis = next(
            record.axis
            for group in faults.values()
            for record in group
            if record.operator_id == operator
        )
        assert axis is not None
        by_pair_axis[(pair, axis)].append(mean(values))
    by_pair: dict[BaselineKey, list[float]] = defaultdict(list)
    for (pair, _), values in by_pair_axis.items():
        by_pair[pair].append(mean(values))
    return {pair: mean(values) for pair, values in by_pair.items()}


def _task_summaries(
    clean: dict[BaselineKey, OutcomeRecord],
    no_touch: dict[BaselineKey, OutcomeRecord],
    fault_scores: dict[BaselineKey, float],
    faults: dict[FaultKey, list[OutcomeRecord]],
    spec: ReportingSpec,
) -> tuple[TaskSummary, ...]:
    tasks = sorted({task for task, _ in clean})
    summaries = []
    for task in tasks:
        pair_keys = sorted(pair for pair in clean if pair[0] == task)
        scoreable = [
            pair
            for pair in pair_keys
            if _success(clean[pair]) is not None and pair in fault_scores
        ]
        if not scoreable:
            raise ValueError(f"task {task} has no fully paired scoreable outcomes")
        clean_sr = mean(_required_success(clean[pair]) for pair in scoreable)
        no_touch_values = [_success(no_touch[pair]) for pair in scoreable]
        no_touch_sr = (
            mean(value for value in no_touch_values if value is not None)
            if all(value is not None for value in no_touch_values)
            else None
        )
        fault_sr = mean(fault_scores[pair] for pair in scoreable)
        clean_gain = None if no_touch_sr is None else clean_sr - no_touch_sr
        if not spec.matched_control_qualified:
            tgr = None
            reason = "matched_control_not_qualified"
        elif no_touch_sr is None or clean_gain is None:
            tgr = None
            reason = "matched_control_outcomes_ineligible"
        elif not primary_task_score_complete(task, clean, faults, spec):
            tgr = None
            reason = "incomplete_or_ineligible_primary_grid"
        elif clean_gain <= spec.minimum_clean_gain:
            tgr = None
            reason = "small_or_nonpositive_clean_gain"
        else:
            tgr = tactile_gain_retention(
                clean_sr,
                no_touch_sr,
                fault_sr,
                minimum_gain=spec.minimum_clean_gain,
            )
            reason = None
        operator_values: dict[str, list[float]] = defaultdict(list)
        for (operator, severity), group in faults.items():
            if operator not in spec.primary_operator_ids:
                continue
            if severity not in spec.primary_severity_levels:
                continue
            values = [
                float(fault_value)
                for record, _, _, fault_value in paired_values(group, clean, no_touch)
                if record.task == task
            ]
            if values:
                operator_values[operator].append(mean(values))
        axis_values: dict[str, list[float]] = defaultdict(list)
        for operator, severity_means in operator_values.items():
            axis = next(
                record.axis
                for group in faults.values()
                for record in group
                if record.operator_id == operator
            )
            assert axis is not None
            axis_values[axis].append(mean(severity_means))
        summaries.append(
            TaskSummary(
                task=task,
                pair_count=len(scoreable),
                clean_sr=clean_sr,
                no_touch_sr=no_touch_sr,
                clean_gain=clean_gain,
                fault_sr=fault_sr,
                paired_delta_sr=mean(
                    _required_success(clean[pair]) - fault_scores[pair]
                    for pair in scoreable
                ),
                no_touch_relative_fault_delta=(
                    mean(
                        fault_scores[pair] - _required_success(no_touch[pair])
                        for pair in scoreable
                    )
                    if no_touch_sr is not None
                    else None
                ),
                tgr=tgr,
                tgr_ineligibility_reason=reason,
                axis_fault_sr=tuple(
                    sorted(
                        (axis, mean(operator_values))
                        for axis, operator_values in axis_values.items()
                    )
                ),
            )
        )
    return tuple(summaries)


def aggregate_benchmark(
    outcomes: Iterable[OutcomeRecord], spec: ReportingSpec
) -> BenchmarkSummary:
    """Aggregate outcomes; paired degradation is clean minus faulted SR."""

    records = tuple(outcomes)
    if not records:
        raise ValueError("reporting outcomes must be non-empty")
    clean, no_touch, faults = _index_records(records, spec)
    cells = build_operator_cells(faults, clean, no_touch, spec)
    scored_cells = [
        cell
        for cell in cells
        if cell.success_rate is not None
        and cell.operator_id in spec.primary_operator_ids
        and cell.severity_level in spec.primary_severity_levels
    ]
    if not scored_cells:
        raise ValueError("no operator cell has fully paired eligible outcomes")
    fault_scores = _pair_fault_scores(faults, spec)
    tasks = _task_summaries(clean, no_touch, fault_scores, faults, spec)
    differences_by_task = {
        task.task: tuple(
            _required_success(clean[pair]) - fault_scores[pair]
            for pair in sorted(clean)
            if pair[0] == task.task
            and pair in fault_scores
            and _success(clean[pair]) is not None
        )
        for task in tasks
    }
    interval = task_stratified_paired_bootstrap(
        differences_by_task,
        n_resamples=spec.bootstrap_resamples,
        confidence_level=spec.confidence_level,
        seed=spec.bootstrap_seed,
    )
    coverage = build_coverage_summary(records, clean, faults, spec)
    task_tgr = [task.tgr for task in tasks]
    macro_tgr = (
        mean(value for value in task_tgr if value is not None)
        if all(value is not None for value in task_tgr)
        else None
    )
    return BenchmarkSummary(
        spec=spec,
        source_root_sha256=tuple(
            sorted({record.source_root_sha256 for record in records})
        ),
        tasks=tasks,
        operator_cells=cells,
        worst_cell=min(
            scored_cells,
            key=lambda item: (
                item.success_rate if item.success_rate is not None else 1.0,
                item.operator_id,
                item.severity_level,
            ),
        ),
        coverage=coverage,
        macro_clean_sr=mean(task.clean_sr for task in tasks),
        macro_no_touch_sr=(
            mean(task.no_touch_sr for task in tasks if task.no_touch_sr is not None)
            if all(task.no_touch_sr is not None for task in tasks)
            else None
        ),
        macro_fault_sr=mean(task.fault_sr for task in tasks),
        macro_paired_delta_sr=mean(task.paired_delta_sr for task in tasks),
        macro_tgr=macro_tgr,
        paired_delta_interval=interval,
    )
