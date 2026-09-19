"""Summarize fixed-seed three-model Isaac groups without treating gaps as failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.retrained_evaluation.group import (
    AVAILABILITY_OPERATORS,
    OPERATORS,
    write_json,
)

CAMPAIGNS = {
    "dream_tac": "dream-a800",
    "ftp1_policy": "ftp-a5000-v2",
    "n0_vtla": "vtla-a800",
}
TASKS = ("grasp_classify", "lift_can")
PHASES = {"native": OPERATORS, "zero_fill_v1": AVAILABILITY_OPERATORS}
INFRASTRUCTURE_STAGES = {"delivery", "artifact_export", "reset", "close"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summarize(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for model, campaign_name in CAMPAIGNS.items():
        for phase, operators in PHASES.items():
            for task in TASKS:
                selected_campaign = campaign_name
                if model == "dream_tac" and (phase, task) != (
                    "native",
                    "grasp_classify",
                ):
                    selected_campaign = "dream-completion-v2"
                group = (
                    root
                    / selected_campaign
                    / "phases"
                    / phase
                    / "groups"
                    / model
                    / task
                )
                plan_path = group / "group.json"
                planned = ("clean", *operators)
                expected = 1 + len(operators)
                plan = read_json(plan_path) if plan_path.is_file() else None
                if plan is not None:
                    if (plan.get("model"), plan.get("task"), plan.get("seed")) != (
                        model,
                        task,
                        3,
                    ):
                        raise ValueError(f"group identity mismatch: {plan_path}")
                    if len(plan.get("ordered_requests", [])) != expected:
                        raise ValueError(f"condition count mismatch: {plan_path}")
                    if plan.get("excluded"):
                        raise ValueError(f"unexpected exclusions: {plan_path}")
                clean_path = group / "results/00.json"
                clean = read_json(clean_path) if clean_path.is_file() else None
                initial = clean.get("initial_state_sha256") if clean else None
                for index, condition in enumerate(planned):
                    path = group / "results" / f"{index:02d}.json"
                    result = read_json(path) if path.is_file() else None
                    if result is not None:
                        actual = Path(result["request"]).stem
                        if actual != condition:
                            raise ValueError(f"result condition mismatch: {path}")
                    eligible = bool(
                        result is not None
                        and result.get("score_eligible") is True
                        and result.get("validation_passed") is True
                        and result.get("failure_stage") not in INFRASTRUCTURE_STAGES
                    )
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seed": 3,
                            "phase": phase,
                            "condition": condition,
                            "status": (
                                "pending"
                                if result is None
                                else "eligible"
                                if eligible
                                else "invalid_or_ineligible"
                            ),
                            "score_success": (
                                result.get("score_success")
                                if eligible and result is not None
                                else None
                            ),
                            "terminal_status": result.get("terminal_status")
                            if result
                            else None,
                            "failure_stage": result.get("failure_stage")
                            if result
                            else None,
                            "failure_code": result.get("failure_code")
                            if result
                            else None,
                            "validation_failure_codes": (
                                result.get("validation_failure_codes")
                                if result
                                else None
                            ),
                            "same_initial_state_as_phase_clean": (
                                None
                                if result is None or initial is None
                                else result.get("initial_state_sha256") == initial
                            ),
                            "observation_count": result.get("observation_count")
                            if result
                            else None,
                            "control_cycle_count": result.get("control_cycle_count")
                            if result
                            else None,
                            "delivery_validation_metrics": (
                                result.get("delivery_validation_metrics")
                                if result
                                else None
                            ),
                            "result_path": str(path),
                        }
                    )
                process = group / "process_exit.json"
                groups.append(
                    {
                        "model": model,
                        "task": task,
                        "phase": phase,
                        "planned": expected,
                        "published": sum(
                            (group / "results" / f"{index:02d}.json").is_file()
                            for index in range(expected)
                        ),
                        "process_exit": read_json(process)
                        if process.is_file()
                        else None,
                        "paired_receipt_present": (
                            group / "paired_receipt.json"
                        ).is_file(),
                    }
                )
    return {
        "schema": "robotactile-three-model-seed3-report-v1",
        "evidence_scope": "one_seed_diagnostic_not_paper_statistics",
        "root": str(root),
        "planned_episode_count": len(rows),
        "published_episode_count": sum(row["status"] != "pending" for row in rows),
        "eligible_episode_count": sum(row["status"] == "eligible" for row in rows),
        "success_count": sum(row["score_success"] is True for row in rows),
        "invalid_or_ineligible_count": sum(
            row["status"] == "invalid_or_ineligible" for row in rows
        ),
        "groups": groups,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    if args.output is None:
        print(json.dumps(report, sort_keys=True))
    else:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
