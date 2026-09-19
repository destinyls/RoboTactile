import copy

import pytest

from robotactile_benchmark.constants import (
    OPTICAL_DECISION_STRESS_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.manifests import FaultManifest, Observability
from scripts.act.decision_stress import (
    OPERATORS,
    prepare,
    report,
    validate_reference,
    write,
)


def reference():
    start = derive_early_random_onset(task="lift_can", seed=5, stop=301)
    faults = []
    for op in OPERATORS:
        p = {"sample_period_s": 1 / 120}
        if operator_requires_rest_reference(
            op, severity_registry=OPTICAL_DECISION_STRESS_REGISTRY_ID
        ):
            p["rest_reference_sha256"] = "1" * 64
        if op.startswith("T"):
            p["temporal_schedule"] = "window_to_end_v1"
        fault = FaultManifest(
            op,
            5,
            23,
            start,
            301,
            ("left", "right"),
            Observability.BLIND,
            p,
            severity_registry=OPTICAL_DECISION_STRESS_REGISTRY_ID,
        )
        faults.append(
            {
                "operator_id": op,
                "fault_manifest_sha256": fault.sha256,
                "document": fault.to_dict(),
            }
        )
    return {
        "schema": "robotactile_decision_stress_cross_model_reference_v1",
        "groups": [
            {
                "task": "lift_can",
                "seed": 5,
                "max_observation_steps": 301,
                "faults": faults,
            }
        ],
    }


def test_reference_preserves_exact_temporal_fault_not_act_chunk_rescaling():
    r = reference()
    validate_reference(r)
    t3 = r["groups"][0]["faults"][-1]["document"]
    assert t3["parameters"]["skew_frames"] == 48
    assert t3["parameters"]["sample_period_s"] == 1 / 120


@pytest.mark.parametrize("change", ["digest", "onset", "duplicates", "order"])
def test_reference_rejects_drift(change):
    r = reference()
    if change == "digest":
        r["groups"][0]["faults"][0]["fault_manifest_sha256"] = "0" * 64
    elif change == "onset":
        r["groups"][0]["seed"] = 7
    elif change == "duplicates":
        r["groups"].append(copy.deepcopy(r["groups"][0]))
    else:
        r["groups"][0]["faults"].reverse()
    with pytest.raises(ValueError):
        validate_reference(r)


def test_preparation_does_not_overwrite_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        prepare(
            reference=tmp_path / "missing",
            rest_root=tmp_path,
            output=tmp_path,
            deployment_root=tmp_path,
            config_root=tmp_path,
            isaac_python=tmp_path,
        )


def test_report_keeps_pending_and_invalid_out_of_sr(tmp_path):
    cells = [
        {"condition": x, "artifact": str(tmp_path / x), "fault_manifest_sha256": None}
        for x in ["clean", *OPERATORS]
    ]
    write(
        tmp_path / "pilot_plan.json",
        {"groups": [{"task": "lift_can", "seed": 5, "cells": cells}]},
    )
    clean = {
        "validation_passed": True,
        "score_eligible": True,
        "score_success": True,
        "initial_state_sha256": "a",
        "terminal_status": "success",
    }
    write(tmp_path / "clean/terminal_result.json", clean)
    write(
        tmp_path / OPERATORS[0] / "terminal_result.json",
        {**clean, "score_success": False, "terminal_status": "timeout"},
    )
    write(tmp_path / OPERATORS[0] / "fault_manifest.json", {"start_index": 5})
    write(
        tmp_path / OPERATORS[1] / "terminal_result.json",
        {
            **clean,
            "validation_passed": False,
            "score_eligible": False,
            "score_success": False,
        },
    )
    write(tmp_path / OPERATORS[1] / "fault_manifest.json", {"start_index": 5})
    r = report(tmp_path)
    assert r["completed"] == 3 and r["planned"] == 5
    metrics = {m["condition"]: m for m in r["metrics"]}
    assert metrics[OPERATORS[0]]["success_rate"] == 0
    assert metrics[OPERATORS[0]]["clean_success_to_fault_failure"] == 1
    assert metrics[OPERATORS[1]]["success_rate"] is None
    assert metrics[OPERATORS[2]]["completed"] == 0
    assert metrics[OPERATORS[2]]["success_rate"] is None
