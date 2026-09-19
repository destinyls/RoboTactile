from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotactile_benchmark.n0_fault_campaign import report_adapter
from robotactile_benchmark.n0_fault_campaign.aggregation import (
    aggregate_n0_fault_campaign,
)
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCellDisposition,
)
from robotactile_benchmark.n0_fault_campaign.report_adapter import (
    load_n0_campaign_outcomes,
    reporting_spec_for_campaign,
)
from robotactile_benchmark.n0_fault_campaign.report_bundle import (
    write_n0_fault_report_bundle,
)
from robotactile_benchmark.n0_fault_campaign.reporting import N0FaultReportingSpec
from robotactile_benchmark.reporting.contracts import OutcomeRecord
from robotactile_benchmark.trials import Condition, TerminalStatus

_SYSTEM = "n0-twam"
_SUPPORTED = "F2_spatial_sensitivity_loss"
_UNSUPPORTED = "A1_stream_absence"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _outcome(
    task: str,
    pair: str,
    condition: Condition,
    success: bool | None,
    *,
    operator: str | None = None,
    severity: int | None = None,
    status: TerminalStatus | None = None,
) -> OutcomeRecord:
    terminal = status or (
        TerminalStatus.UNSUPPORTED_CONTRACT
        if success is None
        else TerminalStatus.SUCCESS
        if success
        else TerminalStatus.TASK_FAILURE
    )
    ineligible = terminal in {
        TerminalStatus.UNSUPPORTED_CONTRACT,
        TerminalStatus.VALIDATOR_REJECTED,
    }
    return OutcomeRecord(
        system_id=_SYSTEM,
        task=task,
        pair_key=_digest(pair),
        condition=condition,
        terminal_status=terminal,
        score_eligible=not ineligible,
        score_success=None if ineligible else bool(success),
        source_root_sha256=_digest(
            f"root:{task}:{pair}:{condition}:{operator}:{severity}"
        ),
        operator_id=operator,
        severity_level=severity,
    )


def _spec(*, severities: tuple[int, ...] = (1, 5)) -> N0FaultReportingSpec:
    return N0FaultReportingSpec(
        system_id=_SYSTEM,
        supported_operator_ids=(_SUPPORTED,),
        contract_operator_ids=(_UNSUPPORTED, _SUPPORTED),
        severity_levels=severities,
        bootstrap_seed=17,
    )


def _complete_fixture() -> tuple[OutcomeRecord, ...]:
    pairs = (
        ("task_a", "a1", True),
        ("task_a", "a2", True),
        ("task_b", "b1", True),
        ("task_b", "b2", False),
    )
    records = [
        _outcome(task, pair, Condition.CLEAN, clean) for task, pair, clean in pairs
    ]
    for severity, values in ((1, (True, True, False, False)), (5, (False,) * 4)):
        records.extend(
            _outcome(
                task,
                pair,
                Condition.FAULTED,
                success,
                operator=_SUPPORTED,
                severity=severity,
            )
            for (task, pair, _), success in zip(pairs, values)
        )
        records.extend(
            _outcome(
                task,
                pair,
                Condition.FAULTED,
                None,
                operator=_UNSUPPORTED,
                severity=severity,
            )
            for task, pair, _ in pairs
        )
    return tuple(records)


def test_paired_metrics_exclude_unsupported_and_publish_curves() -> None:
    summary = aggregate_n0_fault_campaign(_complete_fixture(), _spec())
    cells = {cell.severity_level: cell for cell in summary.operator_cells}

    assert summary.outcomes.requested_count == 20
    assert summary.outcomes.score_denominator == 12
    assert summary.outcomes.unsupported_contract_count == 8
    assert cells[1].clean_success_rate == pytest.approx(0.75)
    assert cells[1].fault_success_rate == pytest.approx(0.5)
    assert cells[1].degradation == pytest.approx(0.25)
    assert cells[1].retention == pytest.approx(2.0 / 3.0)
    assert cells[5].fault_success_rate == pytest.approx(0.0)
    assert cells[5].degradation == pytest.approx(0.75)
    assert summary.macro_clean_success_rate == pytest.approx(0.75)
    assert summary.macro_fault_success_rate == pytest.approx(0.25)
    assert summary.macro_degradation == pytest.approx(0.5)
    assert summary.degradation_interval is not None
    assert summary.degradation_interval.valid_resamples == 10_000
    assert summary.worst_cell == (_SUPPORTED, 5)
    assert summary.severity_curves[0].fault_success_rate_auc == pytest.approx(0.25)
    assert summary.severity_curves[0].degradation_auc == pytest.approx(0.5)
    assert summary.statistically_complete


def test_retention_can_exceed_one_without_clipping() -> None:
    pairs = (("p1", False), ("p2", True))
    records = [_outcome("task", pair, Condition.CLEAN, clean) for pair, clean in pairs]
    for pair, _ in pairs:
        records.append(
            _outcome(
                "task",
                pair,
                Condition.FAULTED,
                True,
                operator=_SUPPORTED,
                severity=1,
            )
        )
        records.append(
            _outcome(
                "task",
                pair,
                Condition.FAULTED,
                None,
                operator=_UNSUPPORTED,
                severity=1,
            )
        )
    summary = aggregate_n0_fault_campaign(records, _spec(severities=(1,)))
    assert summary.operator_cells[0].retention == pytest.approx(2.0)
    assert summary.macro_retention == pytest.approx(2.0)
    assert summary.macro_degradation == pytest.approx(-0.5)


def test_infrastructure_validator_and_supported_unsupported_are_distinct() -> None:
    records = list(_complete_fixture())
    indices = [
        index
        for index, item in enumerate(records)
        if item.operator_id == _SUPPORTED and item.severity_level == 1
    ]
    original = records[indices[0]]
    records[indices[0]] = _outcome(
        original.task,
        "a1",
        Condition.FAULTED,
        False,
        operator=_SUPPORTED,
        severity=1,
        status=TerminalStatus.CRASH,
    )
    original = records[indices[1]]
    records[indices[1]] = _outcome(
        original.task,
        "a2",
        Condition.FAULTED,
        None,
        operator=_SUPPORTED,
        severity=1,
        status=TerminalStatus.VALIDATOR_REJECTED,
    )
    summary = aggregate_n0_fault_campaign(records, _spec())
    assert summary.outcomes.infrastructure_failure_count == 1
    assert summary.outcomes.validator_failure_count == 1
    assert not summary.statistically_complete
    assert "infrastructure_failures" in summary.completeness_blockers
    assert "validator_failures" in summary.completeness_blockers

    records[indices[1]] = _outcome(
        original.task,
        "a2",
        Condition.FAULTED,
        None,
        operator=_SUPPORTED,
        severity=1,
    )
    summary = aggregate_n0_fault_campaign(records, _spec())
    assert any(
        item.startswith("supported_fault_inventory_or_contract_mismatch")
        for item in summary.completeness_blockers
    )


def test_no_touch_is_rejected() -> None:
    records = _complete_fixture() + (
        _outcome("task_a", "a1", Condition.NO_TOUCH, True),
    )
    with pytest.raises(ValueError, match="Clean and Faulted"):
        aggregate_n0_fault_campaign(records, _spec())


def test_report_bundle_is_deterministic_no_clobber_and_schema_allows_retention(
    tmp_path: Path,
) -> None:
    summary = aggregate_n0_fault_campaign(_complete_fixture(), _spec())
    output = tmp_path / "report"
    first = write_n0_fault_report_bundle(output, summary)
    second = write_n0_fault_report_bundle(output, summary)
    assert first.publication_status == "created"
    assert second.publication_status == "already_present"
    assert first.receipt.sha256 == second.receipt.sha256
    changed = aggregate_n0_fault_campaign(
        _complete_fixture(), replace(_spec(), bootstrap_seed=18)
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_n0_fault_report_bundle(output, changed)
    assert {path.name for path in output.iterdir()} == {
        "operator_cells.csv",
        "per_task.csv",
        "report_receipt.json",
        "summary.json",
    }
    schema_path = (
        Path(__file__).parents[1] / "schemas/n0_fault_campaign_summary.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    retention = schema["$defs"]["nullable_retention"]
    assert retention["minimum"] == 0
    assert "maximum" not in retention


def _cell(
    ordinal: int,
    condition: Condition,
    *,
    operator: str | None = None,
    severity: int | None = None,
    disposition: N0FaultCellDisposition = N0FaultCellDisposition.LIVE_REQUEST,
    artifact: str | None = None,
) -> N0FaultCampaignCellSpec:
    faulted = condition is Condition.FAULTED
    unsupported = disposition is N0FaultCellDisposition.UNSUPPORTED_CONTRACT
    return N0FaultCampaignCellSpec(
        ordinal=ordinal,
        task="task",
        initial_seed=1,
        exogenous_seed=2,
        pair_key=_digest("pair"),
        condition=condition,
        disposition=disposition,
        operator_id=operator,
        severity_level=severity,
        operator_template_seed=3 if faulted else None,
        request_relpath=None if unsupported else f"requests/{ordinal}.json",
        artifact_relpath=None if unsupported else artifact,
        request_file_sha256=None if unsupported else _digest(f"request:{ordinal}"),
        trial_manifest_sha256=_digest(f"trial:{ordinal}"),
        fault_manifest_sha256=_digest(f"fault:{ordinal}") if faulted else None,
        fault_manifest_relpath=f"fault_manifests/{ordinal}.json" if faulted else None,
        unsupported_receipt_relpath=(
            f"unsupported_contracts/{ordinal}.json" if unsupported else None
        ),
        unsupported_receipt_sha256=(
            _digest(f"unsupported:{ordinal}") if unsupported else None
        ),
    )


def test_campaign_adapter_preserves_missing_and_validates_crosslinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = _cell(0, Condition.CLEAN, artifact="artifacts/clean")
    fault = _cell(
        1,
        Condition.FAULTED,
        operator=_SUPPORTED,
        severity=1,
        artifact="artifacts/fault",
    )
    unsupported = _cell(
        2,
        Condition.FAULTED,
        operator=_UNSUPPORTED,
        severity=1,
        disposition=N0FaultCellDisposition.UNSUPPORTED_CONTRACT,
    )
    missing = _cell(
        3,
        Condition.FAULTED,
        operator=_SUPPORTED,
        severity=5,
        artifact="artifacts/missing",
    )
    for name in ("clean", "fault"):
        (tmp_path / "artifacts" / name).mkdir(parents=True)
    manifest = SimpleNamespace(
        campaign_id="campaign",
        sha256=_digest("manifest"),
        cells=(clean, fault, unsupported, missing),
        cell_count=4,
        operator_ids=(_UNSUPPORTED, _SUPPORTED),
        severity_levels=(1, 5),
    )
    campaign = SimpleNamespace(root=tmp_path, manifest=manifest)
    monkeypatch.setattr(
        report_adapter, "load_n0_fault_campaign_bundle", lambda _: campaign
    )
    monkeypatch.setattr(report_adapter, "_campaign_system_id", lambda _: _SYSTEM)

    def fake_artifact(path: Path) -> Any:
        cell = clean if path.name == "clean" else fault
        fault_manifest = (
            None
            if cell.condition is Condition.CLEAN
            else SimpleNamespace(
                sha256=cell.fault_manifest_sha256,
                operator_id=cell.operator_id,
                severity_level=cell.severity_level,
            )
        )
        trial = SimpleNamespace(
            base_system_id=_SYSTEM,
            sha256=cell.trial_manifest_sha256,
            task=cell.task,
            pair_key=cell.pair_key,
            condition=cell.condition,
        )
        result = SimpleNamespace(
            trial_manifest_sha256=cell.trial_manifest_sha256,
            pair_key=cell.pair_key,
            terminal_status=TerminalStatus.SUCCESS,
            score_eligible=True,
            score_success=True,
        )
        return SimpleNamespace(
            trial=trial,
            evidence=SimpleNamespace(result=result),
            fault_manifest=fault_manifest,
            root_receipt_sha256=_digest(f"artifact:{path.name}"),
        )

    monkeypatch.setattr(report_adapter, "load_live_univtac_artifact", fake_artifact)
    loaded = load_n0_campaign_outcomes(tmp_path)
    assert len(loaded.outcomes) == 3
    assert len(loaded.missing_live_cell_sha256s) == 1
    assert not loaded.complete
    assert loaded.outcomes[-1].terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
    monkeypatch.setattr(
        report_adapter,
        "_campaign_severity_registry_ids",
        lambda _: {"provisional_engineering_v2"},
    )
    spec = reporting_spec_for_campaign(loaded, bootstrap_seed=19)
    assert spec.supported_operator_ids == (_SUPPORTED,)

    def tampered(path: Path) -> Any:
        result = fake_artifact(path)
        if path.name == "fault":
            result.evidence.result.pair_key = _digest("wrong")
        return result

    monkeypatch.setattr(report_adapter, "load_live_univtac_artifact", tampered)
    with pytest.raises(N0FaultCampaignError, match="differs from campaign cell"):
        load_n0_campaign_outcomes(tmp_path)
