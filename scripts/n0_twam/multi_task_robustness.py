"""Serial N0 task/seed diagnostics with one policy worker per task."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.fault_timing import DEFAULT_EARLY_ONSET_MAX_INDEX
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.n0_twam.single_task_robustness import report, result_row
from scripts.retrained_evaluation.campaign import run_task_seeds
from scripts.retrained_evaluation.group import (
    OPERATORS,
    write_json,
)

UNSUPPORTED = ("A1_stream_absence", "A2_frame_erasure")


def event(campaign: Path, stage: str, **details: object) -> None:
    write_json(
        campaign / "events" / f"{time.time_ns()}.json",
        {"stage": stage, "unix_time": time.time(), **details},
    )


def summarize(campaign: Path, tasks: list[str], seeds: list[int]) -> dict[str, Any]:
    """Aggregate seeds within each task/condition; never pool fault categories."""
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for task in tasks:
        for condition in ("clean", *OPERATORS, *UNSUPPORTED):
            cell: dict[str, Any] = {
                "task": task,
                "condition": condition,
                "seeds": seeds,
                "attempted": 0,
                "eligible": 0,
                "success": 0,
                "invalid": 0,
                "missing": 0,
                "unsupported": 0,
            }
            for seed in seeds:
                group = (
                    campaign
                    / "tasks"
                    / task
                    / "seeds"
                    / f"seed-{seed:03d}"
                    / "groups/n0_twam"
                    / task
                )
                row: dict[str, Any] = {
                    "task": task,
                    "seed": seed,
                    "condition": condition,
                    "status": "missing",
                    "attempted": False,
                }
                if condition in UNSUPPORTED:
                    row["status"] = "unsupported"
                elif (group / "group.json").is_file():
                    plan = read_object(group / "group.json")
                    for index, request in enumerate(plan["ordered_requests"]):
                        if Path(request).stem != condition:
                            continue
                        result = group / "results" / f"{index:02d}.json"
                        if result.is_file():
                            row["attempted"] = True
                            try:
                                raw = read_object(result)
                                row.update(result_row(raw, condition))
                                row["official_result"] = raw
                                row["status"] = (
                                    "eligible"
                                    if row["valid_episode_count"]
                                    else "invalid"
                                )
                            except (OSError, ValueError, KeyError, TypeError) as exc:
                                row.update(status="invalid", error=str(exc))
                        else:
                            request_path = group / request
                            if request_path.is_file():
                                output = read_object(request_path).get("output_dir")
                                row["attempted"] = bool(
                                    output and Path(output).exists()
                                )
                        break
                cell["attempted"] += int(row["attempted"])
                cell[row["status"]] += 1
                if row["status"] == "eligible":
                    cell["success"] += int(row["success_count"])
                rows.append(row)
            cell["success_rate"] = (
                cell["success"] / cell["eligible"] if cell["eligible"] else None
            )
            cells.append(cell)
    return {
        "schema": "robotactile-n0-multi-task-robustness-v1",
        "evidence_scope": "small_sample_simulator_diagnostic_not_sufficient_paper_statistics",
        "seed_count": len(seeds),
        "fault_window_mode": "early_random_onset_v1",
        "tasks": tasks,
        "seeds": seeds,
        "cells": cells,
        "episodes": rows,
    }


def snapshot(campaign: Path, tasks: list[str], seeds: list[int]) -> Path:
    directory = campaign / "reports" / str(time.time_ns())
    summary = summarize(campaign, tasks, seeds)
    write_json(directory / "summary.json", summary)
    with (directory / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["cells"][0]))
        writer.writeheader()
        writer.writerows(summary["cells"])
    return directory


def run_campaign(
    binding_path: Path,
    campaign: Path,
    code: Path,
    package: Path,
    tasks: list[str],
    seeds: list[int],
) -> None:
    binding = read_object(binding_path)
    if binding["model"] != "n0_twam":
        raise ValueError("campaign requires n0_twam")
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(
            task not in binding["tasks"] or Path(task).name != task for task in tasks
        )
    ):
        raise ValueError("tasks must be unique task IDs in the binding")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    campaign.mkdir(parents=True, exist_ok=False)
    evaluation = dict(binding.get("evaluation", {}))
    evaluation.pop("fault_start_index", None)
    binding["evaluation"] = {
        **evaluation,
        "capture_profile": "paper_full_v1",
        "severity_registries": ["optical_marker_extreme_v1"],
        "fault_window_mode": "early_random_onset_v1",
        "fault_onset_max_index": DEFAULT_EARLY_ONSET_MAX_INDEX,
        "measure_n0_rest": True,
    }
    frozen_binding = campaign / "binding.json"
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = canonical_hash(binding)
    write_json(frozen_binding, binding)
    write_json(campaign / "plan.json", {"tasks": tasks, "seeds": seeds})
    event(campaign, "supervisor_started", tasks=tasks, seeds=seeds)
    for task in tasks:
        event(campaign, "task_started", task=task)

        def after_seed(seed: int, group: Path, task: str = task) -> None:
            event(campaign, "seed_exit", task=task, seed=seed, group=str(group))
            output = (
                campaign
                / "tasks"
                / task
                / "seeds"
                / f"seed-{seed:03d}"
                / "visualizations"
            )
            try:
                report(group, output)
                event(
                    campaign,
                    "report_finished",
                    task=task,
                    seed=seed,
                    output=str(output),
                )
            except Exception as exc:
                # Export errors never trigger another simulator episode.
                event(campaign, "report_error", task=task, seed=seed, error=str(exc))
            snapshot(campaign, tasks, seeds)

        try:
            run_task_seeds(
                frozen_binding,
                task,
                campaign / "tasks" / task,
                code,
                package,
                seeds,
                after_seed,
            )
        except Exception as exc:
            event(campaign, "task_infrastructure_error", task=task, error=str(exc))
        event(campaign, "task_exit", task=task)
        snapshot(campaign, tasks, seeds)
    final_report = snapshot(campaign, tasks, seeds)
    cells = read_object(final_report / "summary.json")["cells"]
    write_json(
        campaign / "supervisor_exit.json",
        {
            "status": "finished",
            "all_available_episodes_completed": all(
                cell["missing"] == 0 and cell["invalid"] == 0 for cell in cells
            ),
            "summary": str(final_report / "summary.json"),
            "finished_unix": time.time(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binding", "campaign", "code", "package"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    run_campaign(
        args.binding.absolute(),
        args.campaign.absolute(),
        args.code.absolute(),
        args.package.absolute(),
        args.tasks,
        args.seeds,
    )


if __name__ == "__main__":
    main()
