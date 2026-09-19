"""Do not turn invalid rows into failures or complete a partial campaign."""

import copy

import pytest

from scripts.decision_stress.combine_results import (
    MODELS,
    NATIVE_ORDER,
    SEEDS,
    SUPPLEMENT_ORDER,
    TASKS,
    category,
    counts,
    validate_reports,
)
from scripts.decision_stress.remaining import PREVIOUS


def documents():
    return {
        m: {
            "model": m,
            "new_completed": 66,
            "new_planned": 66,
            "rows": [
                {
                    "model": m,
                    "task": t,
                    "seed": s,
                    "protocol": protocol,
                    "condition": op,
                    "reused": op in PREVIOUS,
                    "status": "terminal",
                    "validation_passed": True,
                    "score_eligible": True,
                    "score_success": True,
                }
                for t in TASKS
                for s in SEEDS
                for protocol, order in (
                    ("native", NATIVE_ORDER),
                    ("zero_fill_v1", SUPPLEMENT_ORDER),
                )
                for op in order
            ],
        }
        for m in MODELS
    }


def test_complete_coverage() -> None:
    assert len(validate_reports(documents())) == 192


@pytest.mark.parametrize("mode", ["duplicate", "pending", "incomplete", "reuse"])
def test_rejects_incomplete_or_duplicated_reports(mode: str) -> None:
    d = copy.deepcopy(documents())
    if mode == "duplicate":
        d["n0"]["rows"][-1] = d["n0"]["rows"][0]
    elif mode == "pending":
        d["act"]["rows"][0]["status"] = "pending"
    elif mode == "incomplete":
        d["act"]["new_completed"] = 65
    else:
        d["n0"]["rows"][0]["reused"] = False
    with pytest.raises(ValueError):
        validate_reports(d)


def test_invalid_crash_is_not_a_model_failure() -> None:
    invalid = dict(
        validation_passed=False,
        score_eligible=True,
        score_success=False,
        terminal_status="crash",
        failure_code="execute_failed",
    )
    failed = dict(invalid, validation_passed=True)
    success = dict(
        failed, score_success=True, terminal_status="success", failure_code=None
    )
    assert category(invalid) == "invalid_or_ineligible"
    result = counts([invalid, failed, success])
    assert result["failed"] == result["success"] == result["invalid_or_ineligible"] == 1
