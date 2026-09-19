"""Prepare and report a no-clobber, A2-only endpoint-censored continuation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operator_parameters import rematerialization_inputs
from robotactile_benchmark.policies.tactile_availability import TactileAvailabilityMode
from scripts.decision_stress.remaining import read, sha, write

A2 = "A2_frame_erasure"
SCHEMA = "a2_episode_censored_continuation_v1"
END_POLICY = "episode_censored_v1"


def censored_manifest(original: FaultManifest) -> FaultManifest:
    """Change only the endpoint rule, retaining the frozen erasure schedule."""
    if original.operator_id != A2 or "a2_end_policy" in original.parameters:
        raise ValueError("requires an original strict A2 manifest")
    document = original.to_dict()
    inputs = rematerialization_inputs(
        A2, original.parameters, preserve_erasure_schedule=True
    )
    inputs["a2_end_policy"] = END_POLICY
    document["parameters"] = inputs
    updated = FaultManifest.from_dict(document)
    if (
        updated.parameters["erased_offsets"] != original.parameters["erased_offsets"]
        or updated.start_index != original.start_index
        or updated.stop_index != original.stop_index
        or updated.operator_seed != original.operator_seed
        or updated.sha256 == original.sha256
    ):
        raise ValueError("A2 continuation changed the frozen schedule or identity")
    return updated


def _unique_cell(group: dict[str, Any], condition: str) -> dict[str, Any]:
    matches = [cell for cell in group["cells"] if cell["condition"] == condition]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {condition} cell")
    return cast(dict[str, Any], matches[0])


def prepare(prior: Path, output: Path) -> None:
    """Create fresh A2 requests without touching prior episodes or matched Clean."""
    if output.exists():
        raise FileExistsError(output)
    if not (prior / "finished.json").is_file():
        raise ValueError("prior campaign is not complete")
    plan = read(prior / "plan.json")
    if plan["schema"] != "remaining_decision_stress_campaign_v1":
        raise ValueError("wrong prior campaign schema")
    if plan["model"] not in {"n0", "act"} or len(plan["groups"]) != 6:
        raise ValueError("requires the completed six-cell N0 or ACT experiment")

    prepared = []
    for group in plan["groups"]:
        clean_cell = _unique_cell(group, "availability_clean")
        old_cell = _unique_cell(group, A2)
        clean_terminal = Path(clean_cell["artifact"]) / "terminal_result.json"
        old_terminal = Path(old_cell["artifact"]) / "terminal_result.json"
        clean_result = read(clean_terminal)
        old_result = read(old_terminal)
        if not (clean_result["validation_passed"] and clean_result["score_eligible"]):
            raise ValueError("matched zero-fill Clean is not valid and score-eligible")
        if old_result["validation_failure_codes"] != ["A2_RESUME_MISSING"]:
            raise ValueError("prior A2 did not fail solely at the resume boundary")
        old_request = load_live_univtac_request(Path(old_cell["request"]))
        if (
            old_request.tactile_availability_mode
            is not TactileAvailabilityMode.ZERO_FILL
            or old_request.output_dir != Path(old_cell["artifact"])
            or old_request.fault_manifest_path is None
        ):
            raise ValueError("prior A2 zero-fill request changed")
        original = FaultManifest.from_dict(read(old_request.fault_manifest_path))
        if original.sha256 != old_cell["fault_sha256"]:
            raise ValueError("prior A2 manifest SHA256 mismatch")
        prepared.append(
            (
                group,
                clean_cell,
                old_cell,
                clean_terminal,
                old_terminal,
                clean_result,
                old_request,
                censored_manifest(original),
            )
        )

    output.mkdir(parents=True, exist_ok=False)
    groups = []
    for (
        old_group,
        clean_cell,
        old_cell,
        clean_terminal,
        old_terminal,
        clean_result,
        old_request,
        fault,
    ) in prepared:
        root = output / f"seed-{old_group['seed']}-{old_group['task']}"
        fault_path = root / "faults" / f"{A2}.json"
        request_path = root / "requests" / f"{A2}.json"
        request = replace(
            old_request,
            fault_manifest_path=fault_path,
            output_dir=root / "artifacts" / A2,
            runtime_dir=root / "runtime",
        )
        write(fault_path, fault.to_dict())
        loaded = load_live_univtac_run(request)
        write(request_path, live_univtac_request_to_dict(request))
        cell = {
            "condition": A2,
            "protocol": "zero_fill_v1",
            "endpoint_policy": END_POLICY,
            "request": str(request_path),
            "request_sha256": sha(request_path),
            "artifact": str(request.output_dir),
            "run_content_sha256": loaded.content_sha256,
            "fault_sha256": fault.sha256,
            "original_fault_sha256": old_cell["fault_sha256"],
        }
        continuation = {
            "model": plan["model"],
            "task": old_group["task"],
            "seed": old_group["seed"],
            "path": str(root),
            "cells": [cell],
            "integration_config": old_group["integration_config"],
            "prior_cells": [
                {
                    "artifact": clean_cell["artifact"],
                    "terminal_file_sha256": sha(clean_terminal),
                }
            ],
            "prior_initial_state_sha256": clean_result["initial_state_sha256"],
            "matched_clean_artifact": clean_cell["artifact"],
            "original_a2_artifact": old_cell["artifact"],
            "original_a2_terminal_file_sha256": sha(old_terminal),
        }
        write(root / "group_plan.json", continuation)
        groups.append(continuation)
    write(
        output / "plan.json",
        {
            "schema": SCHEMA,
            "model": plan["model"],
            "prior": str(prior),
            "prior_plan_sha256": sha(prior / "plan.json"),
            "prior_finished_sha256": sha(prior / "finished.json"),
            "groups": groups,
            "isaac_python": plan["isaac_python"],
            "n0_port": plan["n0_port"],
            "planned_episode_count": len(groups),
            "boundary": "A2-only zero_fill_v1 diagnostic; 3 fixed seeds per task; old Clean reused only when initial state matches",
        },
    )


def report(output: Path) -> dict[str, Any]:
    """Score only valid, actually observed, model-exposed A2 episodes."""
    plan = read(output / "plan.json")
    if plan["schema"] != SCHEMA:
        raise ValueError("wrong A2 continuation schema")
    rows = []
    for group in plan["groups"]:
        cell = group["cells"][0]
        artifact = Path(cell["artifact"])
        clean_path = Path(group["matched_clean_artifact"]) / "terminal_result.json"
        old_path = Path(group["original_a2_artifact"]) / "terminal_result.json"
        if (
            sha(clean_path) != group["prior_cells"][0]["terminal_file_sha256"]
            or sha(old_path) != group["original_a2_terminal_file_sha256"]
            or sha(Path(cell["request"])) != cell["request_sha256"]
        ):
            raise ValueError("source episode or new A2 request changed")
        clean = read(clean_path)
        fault = FaultManifest.from_dict(read(artifact / "fault_manifest.json"))
        if (
            fault.sha256 != cell["fault_sha256"]
            or fault.parameters.get("a2_end_policy") != END_POLICY
        ):
            raise ValueError("executed A2 differs from requested endpoint policy")
        terminal = read(artifact / "terminal_result.json")
        validation = read(artifact / "validation_report.json")["report"]
        actions = read(artifact / "action_trace.json")["entries"]
        length = terminal["observation_count"]
        erased = [
            fault.start_index + int(offset)
            for offset in fault.parameters["erased_offsets"]
            if fault.start_index + int(offset) < length
        ]
        query_indices = [entry["source_step_index"] for entry in actions]
        exposed = (
            sum(index <= max(query_indices) for index in erased) if query_indices else 0
        )
        right_censored = bool(erased) and erased[-1] == length - 1
        if terminal["validation_passed"] is not True:
            resume_status = "invalid_delivery"
        elif not erased:
            resume_status = "no_erasure_observed"
        elif right_censored:
            resume_status = "right_censored"
        else:
            resume_status = "observed_resume"
        if validation is not None and (
            validation["metrics"].get("a2_resume_status") != resume_status
            or validation["metrics"].get("a2_right_censored") != right_censored
        ):
            raise ValueError("stored A2 endpoint metrics disagree with executed trace")
        initial_state_match = (
            terminal["initial_state_sha256"] is not None
            and terminal["initial_state_sha256"] == clean["initial_state_sha256"]
        )
        valid = (
            terminal["validation_passed"] is True
            and terminal["score_eligible"] is True
            and bool(erased)
            and exposed > 0
        )
        rows.append(
            {
                "model": plan["model"],
                "task": group["task"],
                "seed": group["seed"],
                "condition": A2,
                "protocol": "zero_fill_v1",
                "endpoint_policy": END_POLICY,
                "artifact": str(artifact),
                "fault_sha256": fault.sha256,
                "original_fault_sha256": cell["original_fault_sha256"],
                "terminal_file_sha256": sha(artifact / "terminal_result.json"),
                "validation_passed": terminal["validation_passed"],
                "validation_failure_codes": terminal["validation_failure_codes"],
                "score_eligible": terminal["score_eligible"],
                "score_success": terminal["score_success"],
                "terminal_status": terminal["terminal_status"],
                "execution_status": terminal["execution_status"],
                "failure_code": terminal["failure_code"],
                "observation_count": length,
                "inference_count": len(actions),
                "observed_erasure_count": len(erased),
                "erasure_before_or_at_last_inference_count": exposed,
                "a2_resume_status": resume_status,
                "a2_right_censored": right_censored,
                "a2_endpoint_metrics_source": (
                    "stored_validation_report"
                    if validation is not None
                    else "frozen_manifest_and_terminal_length"
                ),
                "initial_state_match": initial_state_match,
                "matched_clean_success": clean["score_success"],
                "valid_for_sr": valid,
                "valid_pair": valid and initial_state_match,
            }
        )
    valid_rows = [row for row in rows if row["valid_for_sr"]]
    success = sum(row["score_success"] is True for row in valid_rows)
    return {
        "schema": "a2_episode_censored_results_v1",
        "model": plan["model"],
        "planned": len(plan["groups"]),
        "completed": len(rows),
        "valid_eligible_exposed": len(valid_rows),
        "success_count": success,
        "success_rate": success / len(valid_rows) if valid_rows else None,
        "right_censored_count": sum(row["a2_right_censored"] is True for row in rows),
        "paired_initial_state_count": sum(row["initial_state_match"] for row in rows),
        "reporter_script_sha256": sha(Path(__file__)),
        "rows": rows,
        "boundary": plan["boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.prior.resolve(strict=True), args.output.absolute())


if __name__ == "__main__":
    main()
