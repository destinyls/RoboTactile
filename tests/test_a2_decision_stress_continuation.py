"""A2 continuation preserves the schedule and scores only observed exposure."""

from pathlib import Path

import pytest

from robotactile_benchmark.manifests import FaultManifest, Observability
from scripts.decision_stress.a2_censored import (
    A2,
    END_POLICY,
    SCHEMA,
    censored_manifest,
    report,
)
from scripts.decision_stress.remaining import sha, write


def _fault() -> FaultManifest:
    return FaultManifest(
        operator_id=A2,
        severity_level=5,
        operator_seed=23,
        start_index=5,
        stop_index=301,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={"sample_period_s": 1 / 120},
        severity_registry="optical_decision_stress_v1",
    )


def test_censored_manifest_changes_only_endpoint_policy() -> None:
    original = _fault()
    updated = censored_manifest(original)
    assert updated.sha256 != original.sha256
    assert updated.parameters["a2_end_policy"] == END_POLICY
    assert updated.parameters["erased_offsets"] == original.parameters["erased_offsets"]
    assert updated.start_index == original.start_index
    assert updated.stop_index == original.stop_index
    assert updated.operator_seed == original.operator_seed
    assert FaultManifest.from_dict(updated.to_dict()) == updated
    with pytest.raises(ValueError, match="original strict A2"):
        censored_manifest(updated)


@pytest.mark.parametrize("query_after_erasure", [False, True])
def test_report_requires_actual_model_exposure(
    tmp_path: Path, query_after_erasure: bool
) -> None:
    fault = censored_manifest(_fault())
    output = tmp_path / "campaign"
    root = output / "seed-5-grasp_classify"
    artifact = root / "artifacts" / A2
    clean_artifact = tmp_path / "prior" / "availability_clean"
    old_artifact = tmp_path / "prior" / A2
    clean_terminal = clean_artifact / "terminal_result.json"
    old_terminal = old_artifact / "terminal_result.json"
    request = root / "requests" / f"{A2}.json"
    write(
        clean_terminal,
        {
            "initial_state_sha256": "a" * 64,
            "score_success": True,
        },
    )
    write(old_terminal, {"validation_failure_codes": ["A2_RESUME_MISSING"]})
    write(request, {"condition": A2})
    write(artifact / "fault_manifest.json", fault.to_dict())
    write(
        artifact / "terminal_result.json",
        {
            "initial_state_sha256": "a" * 64,
            "validation_passed": True,
            "validation_failure_codes": [],
            "score_eligible": True,
            "score_success": False,
            "terminal_status": "timeout",
            "execution_status": "timeout",
            "failure_code": None,
            "observation_count": 20,
        },
    )
    write(artifact / "validation_report.json", {"report": None})
    query_index = 19 if query_after_erasure else 0
    write(
        artifact / "action_trace.json",
        {"entries": [{"source_step_index": query_index}]},
    )
    write(
        output / "plan.json",
        {
            "schema": SCHEMA,
            "model": "n0",
            "boundary": "test",
            "groups": [
                {
                    "task": "grasp_classify",
                    "seed": 5,
                    "matched_clean_artifact": str(clean_artifact),
                    "original_a2_artifact": str(old_artifact),
                    "original_a2_terminal_file_sha256": sha(old_terminal),
                    "prior_cells": [{"terminal_file_sha256": sha(clean_terminal)}],
                    "cells": [
                        {
                            "artifact": str(artifact),
                            "request": str(request),
                            "request_sha256": sha(request),
                            "fault_sha256": fault.sha256,
                            "original_fault_sha256": "b" * 64,
                        }
                    ],
                }
            ],
        },
    )
    result = report(output)
    assert result["completed"] == 1
    assert result["rows"][0]["observed_erasure_count"] > 0
    assert result["rows"][0]["a2_resume_status"] == "right_censored"
    assert (
        result["rows"][0]["a2_endpoint_metrics_source"]
        == "frozen_manifest_and_terminal_length"
    )
    assert result["valid_eligible_exposed"] == int(query_after_erasure)
    assert result["success_rate"] == (0.0 if query_after_erasure else None)
