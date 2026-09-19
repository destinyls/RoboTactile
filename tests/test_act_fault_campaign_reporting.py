from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from robotactile_benchmark.act_fault_campaign.reporting import (
    build_act_fault_campaign_report,
)
from robotactile_benchmark.trials import Condition, TerminalStatus


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cell(
    ordinal: int,
    pair: str,
    condition: object,
    *,
    operator_id: str | None = None,
    severity_level: int | None = None,
    disposition: str = "live_request",
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "task": "pull_out_key",
        "pair_key": _digest(pair),
        "condition": condition,
        "disposition": disposition,
        "operator_id": operator_id,
        "severity_level": severity_level,
        "artifact_relpath": (
            None if disposition == "unsupported_contract" else f"artifacts/{ordinal}"
        ),
    }


def _manifest(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "campaign_id": "act-report-fixture-v1",
        "policy_kind": "act",
        "profile": "univtac",
        "sha256": _digest("manifest"),
        "cells": cells,
    }


def _result(status: TerminalStatus) -> dict[str, object]:
    if status is TerminalStatus.SUCCESS:
        eligible, success = True, True
    elif status in {
        TerminalStatus.TASK_FAILURE,
        TerminalStatus.EARLY_STOP,
        TerminalStatus.TIMEOUT,
        TerminalStatus.CRASH,
    }:
        eligible, success = True, False
    else:
        eligible, success = False, None
    return {
        "terminal_status": status,
        "score_eligible": eligible,
        "score_success": success,
    }


def test_report_has_only_clean_faulted_metrics_and_excludes_a1_a2() -> None:
    cells = [
        _cell(0, "p1", Condition.CLEAN),
        _cell(1, "p2", Condition.CLEAN),
        _cell(
            2,
            "p1",
            Condition.FAULTED,
            operator_id="F2_spatial_sensitivity_loss",
            severity_level=5,
        ),
        _cell(
            3,
            "p2",
            Condition.FAULTED,
            operator_id="F2_spatial_sensitivity_loss",
            severity_level=5,
        ),
        _cell(
            4,
            "p1",
            Condition.FAULTED,
            operator_id="A1_stream_absence",
            severity_level=5,
            disposition="unsupported_contract",
        ),
        _cell(
            5,
            "p2",
            Condition.FAULTED,
            operator_id="A1_stream_absence",
            severity_level=5,
            disposition="unsupported_contract",
        ),
    ]
    nested = SimpleNamespace(
        evidence=SimpleNamespace(
            result=SimpleNamespace(**_result(TerminalStatus.SUCCESS))
        )
    )
    artifacts: dict[object, object] = {
        "artifacts/0": _result(TerminalStatus.SUCCESS),
        "artifacts/1": _result(TerminalStatus.TASK_FAILURE),
        "artifacts/2": nested,
        3: _result(TerminalStatus.SUCCESS),
    }

    report = build_act_fault_campaign_report(_manifest(cells), artifacts)

    assert report.clean_score_denominator == 2
    assert report.clean_success_rate == pytest.approx(0.5)
    assert report.faulted_score_denominator == 2
    assert report.faulted_success_rate == pytest.approx(1.0)
    assert report.delta_success_rate == pytest.approx(-0.5)
    assert report.retention == pytest.approx(2.0)
    assert report.dispositions.unsupported_contract_count == 2
    assert report.dispositions.score_denominator == 4
    assert report.complete

    by_operator = {item.operator_id: item for item in report.operator_cells}
    supported = by_operator["F2_spatial_sensitivity_loss"]
    assert supported.eligible_pair_count == 2
    assert supported.clean_success_rate == pytest.approx(0.5)
    assert supported.faulted_success_rate == pytest.approx(1.0)
    unsupported = by_operator["A1_stream_absence"]
    assert unsupported.eligible_pair_count == 0
    assert unsupported.clean_success_rate is None
    assert unsupported.faulted_success_rate is None
    assert unsupported.dispositions.unsupported_contract_count == 2

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert "restored" not in serialized.lower()
    assert "no_touch" not in serialized.lower()
    assert len(report.sha256) == 64


def test_infrastructure_validator_and_missing_are_separate_from_sr() -> None:
    cells = []
    for ordinal, pair in enumerate(("p1", "p2", "p3")):
        cells.append(_cell(ordinal, pair, Condition.CLEAN))
        cells.append(
            _cell(
                ordinal + 3,
                pair,
                Condition.FAULTED,
                operator_id="T1_fixed_source_delay",
                severity_level=3,
            )
        )
    artifacts = {
        "artifacts/0": _result(TerminalStatus.SUCCESS),
        "artifacts/1": _result(TerminalStatus.SUCCESS),
        "artifacts/2": _result(TerminalStatus.SUCCESS),
        "artifacts/3": _result(TerminalStatus.CRASH),
        "artifacts/4": _result(TerminalStatus.VALIDATOR_REJECTED),
    }

    report = build_act_fault_campaign_report(_manifest(cells), artifacts)

    assert report.clean_score_denominator == 3
    assert report.clean_success_rate == pytest.approx(1.0)
    assert report.faulted_score_denominator == 0
    assert report.faulted_success_rate is None
    assert report.delta_success_rate is None
    assert report.retention is None
    assert report.dispositions.infrastructure_failure_count == 1
    assert report.dispositions.validator_failure_count == 1
    assert report.dispositions.missing_artifact_count == 1
    assert not report.complete
    assert set(report.completeness_blockers) == {
        "infrastructure_failures",
        "missing_artifacts",
        "no_faulted_scores",
        "validator_failures",
    }


@pytest.mark.parametrize("condition", [Condition.NO_TOUCH, "restored"])
def test_non_clean_faulted_conditions_are_rejected(condition: object) -> None:
    cells = [
        _cell(0, "p1", Condition.CLEAN),
        _cell(1, "p1", condition),
    ]
    with pytest.raises(ValueError, match="only Clean and Faulted"):
        build_act_fault_campaign_report(_manifest(cells), {})


def test_a1_a2_cannot_be_reported_as_live_model_outcomes() -> None:
    cells = [
        _cell(0, "p1", Condition.CLEAN),
        _cell(
            1,
            "p1",
            Condition.FAULTED,
            operator_id="A2_frame_erasure",
            severity_level=1,
        ),
    ]
    with pytest.raises(ValueError, match="A1/A2"):
        build_act_fault_campaign_report(_manifest(cells), {})


def test_incoherent_model_result_is_rejected() -> None:
    cells = [
        _cell(0, "p1", Condition.CLEAN),
        _cell(
            1,
            "p1",
            Condition.FAULTED,
            operator_id="F1_global_response_drift",
            severity_level=1,
        ),
    ]
    artifacts = {
        "artifacts/0": {
            "terminal_status": "success",
            "score_eligible": True,
            "score_success": False,
        }
    }
    with pytest.raises(ValueError, match="incoherent"):
        build_act_fault_campaign_report(_manifest(cells), artifacts)
