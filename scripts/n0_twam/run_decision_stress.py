"""Prepare and execute the frozen N0 decision-time pilot with existing runners."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.constants import OPTICAL_DECISION_STRESS_REGISTRY_ID
from robotactile_benchmark.n0_fault_campaign.action_effects import summarize_campaign
from scripts.n0_twam.run_official_early_fault_seed import prepare, run, write_json

OPERATORS = (
    "F1_global_response_drift",
    "F3_persistent_surface_artifact",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
)


def prepare_pilot(
    prior: Path, output: Path, seeds: tuple[int, ...], tasks: tuple[str, ...], port: int
) -> None:
    if output.exists():
        raise FileExistsError(output)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - {"grasp_classify", "lift_can"}
    ):
        raise ValueError("pilot tasks must be unique grasp_classify/lift_can IDs")
    old = json.loads((prior / "plan.json").read_text())
    groups = []
    for seed in seeds:
        for task in tasks:
            group = output / f"seed-{seed}-{task}"
            prepare(
                repo=Path(old["repo"]),
                output=group,
                reference_a=prior,
                reference_b=output / "unused-reference",
                tasks=(task,),
                seed=seed,
                integration_label=old["integration_label"],
                port=port,
                capture_profile="preview_v1",
                isaac_python=Path(old["isaac_python"]),
                severity_registry=OPTICAL_DECISION_STRESS_REGISTRY_ID,
                operator_ids=OPERATORS,
            )
            groups.append({"task": task, "seed": seed, "path": str(group)})
    write_json(
        output / "pilot_plan.json",
        {
            "schema": "n0_decision_stress_pilot_v1",
            "prior": str(prior),
            "repo": old["repo"],
            "groups": groups,
            "seeds": list(seeds),
            "operators": list(OPERATORS),
            "planned_episode_count": len(groups) * 5,
            "selection": "exploratory frozen seeds; all results retained; no outcome-driven reruns",
            "prepared_unix": time.time(),
        },
    )


def pilot_report(output: Path) -> dict[str, Any]:
    plan = json.loads((output / "pilot_plan.json").read_text())
    reports = [
        summarize_campaign(Path(g["path"]) / g["task"] / "fault_campaign")
        for g in plan["groups"]
    ]
    rows = [row for report in reports for row in report["rows"]]
    return {
        "schema": "n0_decision_stress_pilot_results_v1",
        "rows": rows,
        "completed": sum(r["status"] == "terminal" for r in rows),
        "planned": len(rows),
        "held_out_confirmation": False,
    }


def run_pilot(output: Path, package_root: Path, wait_seconds: int) -> None:
    plan = json.loads((output / "pilot_plan.json").read_text())
    lock = Path(plan["repo"]) / "deployment-sm120/runtime/decision-stress-gpu0.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = Path(plan["prior"])
        deadline = time.monotonic() + wait_seconds
        while not (prior / "finished.json").is_file():
            if any(prior.glob("*/error.json")):
                raise RuntimeError(
                    "prior campaign failed; inspect its active processes before launching"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "prior campaign has not finished within the wait budget"
                )
            time.sleep(10)
        for g in plan["groups"]:
            group = Path(g["path"])
            if not (group / "finished.json").exists():
                run(output=group, package_root=package_root)
            report_path = group / "decision_report.json"
            if not report_path.exists():
                write_json(
                    report_path,
                    summarize_campaign(group / g["task"] / "fault_campaign"),
                )
        write_json(output / "pilot_results.json", pilot_report(output))
        write_json(
            output / "finished.json",
            {
                "completed_unix": time.time(),
                "planned_episode_count": plan["planned_episode_count"],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[5, 3, 7])
    p.add_argument("--tasks", nargs="+", default=["grasp_classify", "lift_can"])
    p.add_argument("--port", type=int, default=29695)
    r = subs.add_parser("run")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--package-root", type=Path, required=True)
    r.add_argument("--wait-seconds", type=int, default=7200)
    s = subs.add_parser("report")
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_pilot(
            args.prior.resolve(strict=True),
            args.output.absolute(),
            tuple(args.seeds),
            tuple(args.tasks),
            args.port,
        )
    elif args.command == "run":
        run_pilot(
            args.output.resolve(strict=True),
            args.package_root.resolve(strict=True),
            args.wait_seconds,
        )
    else:
        print(
            json.dumps(
                pilot_report(args.output.resolve(strict=True)),
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
