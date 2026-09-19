from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from statistics import mean
from typing import Iterable, Optional, Sequence

from robotactile_benchmark.reporting.contracts import OutcomeRecord
from robotactile_benchmark.reporting.statistics import (
    exact_mcnemar,
    holm_bonferroni,
    task_stratified_paired_bootstrap,
)
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.severity import severity_value
from robotactile_benchmark.trials import Condition, TerminalStatus

from .reporting import (
    N0FaultCampaignSummary,
    N0FaultReportingSpec,
    N0OperatorCellSummary,
    N0TaskSummary,
    OutcomeBreakdown,
    build_axis_summaries,
    build_severity_curves,
)

PairKey = tuple[str, str]
CellKey = tuple[str, int]
_MODEL_STATUSES = frozenset(
    {
        TerminalStatus.SUCCESS,
        TerminalStatus.TASK_FAILURE,
        TerminalStatus.EARLY_STOP,
        TerminalStatus.TIMEOUT,
    }
)


def _score(record: OutcomeRecord) -> Optional[bool]:
    if record.terminal_status not in _MODEL_STATUSES or not record.score_eligible:
        return None
    assert record.score_success is not None
    return record.score_success


def _known(value: Optional[float]) -> float:
    if value is None:
        raise ValueError("expected available statistic")
    return value


def _breakdown(records: Sequence[OutcomeRecord]) -> OutcomeBreakdown:
    counts: Counter[str] = Counter()
    for record in records:
        if record.terminal_status in _MODEL_STATUSES and record.score_eligible:
            counts["model_success" if record.score_success else "model_failure"] += 1
        elif record.terminal_status is TerminalStatus.CRASH:
            counts["infrastructure"] += 1
        elif record.terminal_status is TerminalStatus.VALIDATOR_REJECTED:
            counts["validator"] += 1
        elif record.terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT:
            counts["unsupported"] += 1
        else:
            counts["other"] += 1
    return OutcomeBreakdown(
        requested_count=len(records),
        model_success_count=counts["model_success"],
        model_failure_count=counts["model_failure"],
        infrastructure_failure_count=counts["infrastructure"],
        validator_failure_count=counts["validator"],
        unsupported_contract_count=counts["unsupported"],
        other_ineligible_count=counts["other"],
    )


def _index(
    records: tuple[OutcomeRecord, ...], spec: N0FaultReportingSpec
) -> tuple[dict[PairKey, OutcomeRecord], dict[CellKey, list[OutcomeRecord]]]:
    seen: set[tuple[object, ...]] = set()
    clean: dict[PairKey, OutcomeRecord] = {}
    faults: dict[CellKey, list[OutcomeRecord]] = defaultdict(list)
    for record in records:
        if record.system_id != spec.system_id:
            raise ValueError("outcome system does not match the N0 reporting spec")
        if record.condition is Condition.NO_TOUCH:
            raise ValueError("N0 fault reporting accepts only Clean and Faulted")
        if record.cell_key in seen:
            raise ValueError("duplicate N0 reporting cell")
        seen.add(record.cell_key)
        pair = (record.task, record.pair_key)
        if record.condition is Condition.CLEAN:
            clean[pair] = record
            continue
        if record.operator_id not in spec.contract_operator_ids:
            raise ValueError("fault outcome is outside the frozen operator contract")
        if record.severity_level not in spec.severity_levels:
            raise ValueError("fault outcome is outside the frozen severity contract")
        assert record.operator_id is not None
        assert record.severity_level is not None
        faults[(record.operator_id, record.severity_level)].append(record)
    if not clean:
        raise ValueError("N0 reporting requires at least one Clean outcome")
    return clean, faults


def _paired(
    group: Sequence[OutcomeRecord], clean: dict[PairKey, OutcomeRecord]
) -> list[tuple[OutcomeRecord, bool, bool]]:
    result = []
    for fault in group:
        baseline = clean.get((fault.task, fault.pair_key))
        clean_score = None if baseline is None else _score(baseline)
        fault_score = _score(fault)
        if clean_score is not None and fault_score is not None:
            result.append((fault, clean_score, fault_score))
    return result


def _task_macro(values: Sequence[tuple[str, float]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for task, value in values:
        by_task[task].append(value)
    return mean(mean(group) for group in by_task.values())


def _retention(
    clean_rate: Optional[float], fault_rate: Optional[float]
) -> Optional[float]:
    if clean_rate is None or fault_rate is None or clean_rate <= 0.0:
        return None
    return fault_rate / clean_rate


def _build_cells(
    faults: dict[CellKey, list[OutcomeRecord]],
    clean: dict[PairKey, OutcomeRecord],
    spec: N0FaultReportingSpec,
) -> tuple[N0OperatorCellSummary, ...]:
    registry = load_severity_registry(spec.severity_registry)["paths"]
    cells: list[N0OperatorCellSummary] = []
    raw_p_values: dict[str, float] = {}
    for operator_index, operator_id in enumerate(spec.supported_operator_ids):
        for severity in spec.severity_levels:
            group = faults.get((operator_id, severity), [])
            pairs = _paired(group, clean)
            outcomes = _breakdown(group)
            if not pairs:
                cells.append(
                    N0OperatorCellSummary(
                        operator_id=operator_id,
                        severity_level=severity,
                        native_dose=float(
                            severity_value(
                                operator_id,
                                severity,
                                registry_id=spec.severity_registry,
                            )
                        ),
                        native_unit=str(registry[operator_id]["unit"]),
                        requested_count=len(group),
                        eligible_pair_count=0,
                        clean_success_rate=None,
                        fault_success_rate=None,
                        degradation=None,
                        retention=None,
                        degradation_interval=None,
                        mcnemar_p_value=None,
                        holm_adjusted_p_value=None,
                        statistics_status=(
                            "missing" if not group else "unavailable_no_eligible_pairs"
                        ),
                        outcomes=outcomes,
                    )
                )
                continue
            clean_rate = _task_macro(
                [(record.task, float(clean_value)) for record, clean_value, _ in pairs]
            )
            fault_rate = _task_macro(
                [(record.task, float(fault_value)) for record, _, fault_value in pairs]
            )
            differences: dict[str, list[float]] = defaultdict(list)
            for record, clean_value, fault_value in pairs:
                differences[record.task].append(float(clean_value) - float(fault_value))
            interval = task_stratified_paired_bootstrap(
                differences,
                n_resamples=spec.bootstrap_resamples,
                confidence_level=spec.confidence_level,
                seed=spec.bootstrap_seed + 10 * operator_index + severity,
            )
            test = exact_mcnemar(
                tuple(clean_value for _, clean_value, _ in pairs),
                tuple(fault_value for _, _, fault_value in pairs),
            )
            hypothesis = f"{operator_id}:S{severity}"
            raw_p_values[hypothesis] = test.p_value
            cells.append(
                N0OperatorCellSummary(
                    operator_id=operator_id,
                    severity_level=severity,
                    native_dose=float(
                        severity_value(
                            operator_id,
                            severity,
                            registry_id=spec.severity_registry,
                        )
                    ),
                    native_unit=str(registry[operator_id]["unit"]),
                    requested_count=len(group),
                    eligible_pair_count=len(pairs),
                    clean_success_rate=clean_rate,
                    fault_success_rate=fault_rate,
                    degradation=interval.estimate,
                    retention=_retention(clean_rate, fault_rate),
                    degradation_interval=interval,
                    mcnemar_p_value=test.p_value,
                    holm_adjusted_p_value=test.p_value,
                    statistics_status="available",
                    outcomes=outcomes,
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
        if cell.statistics_status == "available"
        else cell
        for cell in cells
    )


def _build_tasks(
    clean: dict[PairKey, OutcomeRecord],
    faults: dict[CellKey, list[OutcomeRecord]],
    cells: Sequence[N0OperatorCellSummary],
) -> tuple[N0TaskSummary, ...]:
    summaries = []
    for task in sorted(
        {pair[0] for pair in clean}
        | {row.task for group in faults.values() for row in group}
    ):
        clean_values = [
            score
            for pair, record in clean.items()
            if pair[0] == task and (score := _score(record)) is not None
        ]
        cell_values: list[tuple[float, float]] = []
        eligible_pairs = 0
        for cell in cells:
            pairs = [
                item
                for item in _paired(
                    faults.get((cell.operator_id, cell.severity_level), []), clean
                )
                if item[0].task == task
            ]
            if pairs:
                eligible_pairs += len(pairs)
                cell_values.append(
                    (
                        mean(float(clean_value) for _, clean_value, _ in pairs),
                        mean(float(fault_value) for _, _, fault_value in pairs),
                    )
                )
        clean_rate = (
            mean(float(value) for value in clean_values) if clean_values else None
        )
        fault_rate = mean(value[1] for value in cell_values) if cell_values else None
        degradation = (
            mean(clean_value - fault_value for clean_value, fault_value in cell_values)
            if cell_values
            else None
        )
        summaries.append(
            N0TaskSummary(
                task=task,
                clean_outcome_count=len(clean_values),
                clean_success_rate=clean_rate,
                eligible_fault_pair_count=eligible_pairs,
                scored_cell_count=len(cell_values),
                fault_success_rate=fault_rate,
                degradation=degradation,
                retention=_retention(clean_rate, fault_rate),
            )
        )
    return tuple(summaries)


def _completeness_blockers(
    clean: dict[PairKey, OutcomeRecord],
    faults: dict[CellKey, list[OutcomeRecord]],
    spec: N0FaultReportingSpec,
    outcomes: OutcomeBreakdown,
) -> tuple[str, ...]:
    blockers = []
    clean_pairs = set(clean)
    for operator_id in spec.supported_operator_ids:
        for severity in spec.severity_levels:
            group = faults.get((operator_id, severity), [])
            if {(item.task, item.pair_key) for item in group} != clean_pairs or any(
                item.terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
                for item in group
            ):
                blockers.append(
                    f"supported_fault_inventory_or_contract_mismatch:{operator_id}:S{severity}"
                )
    for operator_id in spec.unsupported_operator_ids:
        for severity in spec.severity_levels:
            group = faults.get((operator_id, severity), [])
            pairs = {(item.task, item.pair_key) for item in group}
            if pairs != clean_pairs or any(
                item.terminal_status is not TerminalStatus.UNSUPPORTED_CONTRACT
                for item in group
            ):
                blockers.append(
                    f"unsupported_receipt_inventory_mismatch:{operator_id}:S{severity}"
                )
    if outcomes.infrastructure_failure_count:
        blockers.append("infrastructure_failures")
    if outcomes.validator_failure_count:
        blockers.append("validator_failures")
    if outcomes.other_ineligible_count:
        blockers.append("other_ineligible_outcomes")
    return tuple(sorted(blockers))


def aggregate_n0_fault_campaign(
    outcomes: Iterable[OutcomeRecord], spec: N0FaultReportingSpec
) -> N0FaultCampaignSummary:
    records = tuple(outcomes)
    if not records:
        raise ValueError("N0 reporting outcomes must be non-empty")
    clean, faults = _index(records, spec)
    cells = _build_cells(faults, clean, spec)
    tasks = _build_tasks(clean, faults, cells)
    axes = build_axis_summaries(cells, spec)
    curves = build_severity_curves(cells, spec)
    breakdown = _breakdown(records)
    available_cells = [cell for cell in cells if cell.statistics_status == "available"]
    valid_tasks = [
        task
        for task in tasks
        if task.clean_success_rate is not None and task.fault_success_rate is not None
    ]
    differences: dict[str, list[float]] = defaultdict(list)
    for group in faults.values():
        for record, clean_value, fault_value in _paired(group, clean):
            if record.operator_id in spec.supported_operator_ids:
                differences[record.task].append(float(clean_value) - float(fault_value))
    interval = (
        task_stratified_paired_bootstrap(
            differences,
            n_resamples=spec.bootstrap_resamples,
            confidence_level=spec.confidence_level,
            seed=spec.bootstrap_seed,
        )
        if differences
        else None
    )
    macro_clean = (
        mean(_known(task.clean_success_rate) for task in valid_tasks)
        if valid_tasks
        else None
    )
    macro_fault = (
        mean(_known(task.fault_success_rate) for task in valid_tasks)
        if valid_tasks
        else None
    )
    blockers = _completeness_blockers(clean, faults, spec, breakdown)
    worst = (
        min(
            available_cells,
            key=lambda cell: (
                _known(cell.fault_success_rate),
                -_known(cell.degradation),
                cell.operator_id,
                cell.severity_level,
            ),
        )
        if available_cells
        else None
    )
    return N0FaultCampaignSummary(
        spec=spec,
        source_root_sha256=tuple(
            sorted({record.source_root_sha256 for record in records})
        ),
        outcomes=breakdown,
        tasks=tasks,
        operator_cells=cells,
        axes=axes,
        severity_curves=curves,
        worst_cell=None if worst is None else (worst.operator_id, worst.severity_level),
        macro_clean_success_rate=macro_clean,
        macro_fault_success_rate=macro_fault,
        macro_degradation=(
            mean(_known(task.degradation) for task in valid_tasks)
            if valid_tasks
            else None
        ),
        macro_retention=_retention(macro_clean, macro_fault),
        degradation_interval=interval,
        statistically_complete=not blockers,
        completeness_blockers=blockers,
    )
