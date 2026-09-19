"""Run disjoint N0 task/seed assignments, optionally draining a stopped supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.fault_timing import DEFAULT_EARLY_ONSET_MAX_INDEX
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.n0_twam.multi_task_robustness import event, summarize
from scripts.n0_twam.single_task_robustness import report
from scripts.retrained_evaluation.campaign import run_task_seeds
from scripts.retrained_evaluation.group import (
    write_json,
)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_assignments(value: object, tasks: object) -> dict[str, list[int]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("assignments must be a nonempty task-to-seeds object")
    if not isinstance(tasks, (list, dict)):
        raise ValueError("binding tasks must be a list or object")
    result: dict[str, list[int]] = {}
    for task, seeds in value.items():
        if not isinstance(task, str) or Path(task).name != task or task not in tasks:
            raise ValueError(f"disallowed task: {task}")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(f"{task}: seeds must be unique non-negative integers")
        result[task] = seeds
    return result


def process_state(identity: dict[str, Any]) -> str | None:
    """Verify Linux PID identity before observing or signalling it."""
    pid, ticks, marker = (
        identity.get("pid"),
        identity.get("start_ticks"),
        identity.get("cmd_marker"),
    )
    if type(pid) is not int or pid <= 1 or type(ticks) is not int or ticks <= 0:
        raise ValueError("process identity requires positive pid/start_ticks")
    if not isinstance(marker, str) or not marker.strip():
        raise ValueError("process identity requires a nonempty cmd_marker")
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if int(stat[19]) != ticks:
        raise RuntimeError(f"PID identity changed: {pid}")
    state = stat[0]
    if state != "Z":
        command = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        if marker not in command:
            raise RuntimeError(f"PID command mismatch: {pid}")
    pgid = identity.get("pgid")
    if pgid is not None and (
        type(pgid) is not int or pgid != pid or int(stat[2]) != pgid
    ):
        raise RuntimeError(f"worker is not the recorded process-group leader: {pid}")
    return state


def guarded_signal(identity: dict[str, Any], sig: int, *, group: bool = False) -> None:
    state = process_state(identity)
    if state is None or state == "Z":
        return
    if group:
        if identity.get("pgid") != identity["pid"]:
            raise RuntimeError("refusing unverified process-group signal")
        os.killpg(identity["pgid"], sig)
    else:
        os.kill(identity["pid"], sig)


def group_witness(group: Path) -> dict[str, str]:
    plan = read_object(group / "group.json")
    requests = plan.get("ordered_requests")
    if not isinstance(requests, list) or len(requests) != 13:
        raise RuntimeError("handoff group must contain exactly 13 ordered conditions")
    paths = [group / "paired_receipt.json", group / "group.json"]
    paths.extend(group / "results" / f"{index:02d}.json" for index in range(13))
    result: dict[str, str] = {}
    for path in paths:
        read_object(path)  # Invalid episodes are still completed execution witnesses.
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def drain_handoff(spec: dict[str, Any], campaign: Path, max_wait: float) -> None:
    if spec.get("hostname") != socket.gethostname():
        raise RuntimeError("handoff hostname mismatch")
    supervisor, isaac, worker = (spec[key] for key in ("supervisor", "isaac", "worker"))
    if len({item["pid"] for item in (supervisor, isaac, worker)}) != 3:
        raise ValueError("handoff requires three distinct process identities")
    group = Path(spec["current_group"])
    if not group.is_absolute():
        raise ValueError("current_group must be absolute")
    if read_object(group / "launch.json")["pid"] != isaac["pid"]:
        raise RuntimeError(
            "Isaac PID does not match the current group's launch receipt"
        )
    deadline = time.monotonic() + max_wait
    event(campaign, "waiting_for_current_group", group=str(group))
    while True:
        if process_state(supervisor) not in ("T", "t"):
            raise RuntimeError("old supervisor must remain stopped during drain")
        process_state(worker)
        state = process_state(isaac)
        if state in ("T", "t"):
            raise RuntimeError("active Isaac process is stopped; refusing handoff")
        if state is None or state == "Z":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "current Isaac group did not drain; no new episodes launched"
            )
        time.sleep(5)
    witnesses = group_witness(group)
    # Recheck every identity before any signal; never signal the Isaac child.
    if process_state(supervisor) not in ("T", "t"):
        raise RuntimeError("supervisor resumed before handoff")
    if process_state(isaac) not in (None, "Z"):
        raise RuntimeError("Isaac group is still active")
    process_state(worker)
    guarded_signal(supervisor, signal.SIGTERM)
    guarded_signal(supervisor, signal.SIGCONT)
    deadline = time.monotonic() + 30
    while process_state(supervisor) not in (None, "Z"):
        if time.monotonic() >= deadline:
            raise TimeoutError("old supervisor did not exit; no new episodes launched")
        time.sleep(1)
    guarded_signal(worker, signal.SIGTERM, group=True)
    deadline = time.monotonic() + 60
    while process_state(worker) not in (None, "Z"):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "old policy worker did not exit; no new episodes launched"
            )
        time.sleep(1)
    write_json(
        campaign / "adoption_receipt.json",
        {
            "schema": "n0-stopped-supervisor-adoption-v1",
            "handoff": spec,
            "group_witness_sha256": witnesses,
            "adopted_unix": time.time(),
            "note": "Original process_exit receipt is untouched; invalid results remain invalid.",
        },
    )
    event(campaign, "handoff_complete", group=str(group))
    try:
        report(group, campaign / "adopted" / "visualizations")
    except Exception as exc:
        event(campaign, "adoption_report_error", group=str(group), error=str(exc))


def shard_snapshot(campaign: Path, assignments: dict[str, list[int]]) -> Path:
    summaries = [
        summarize(campaign, [task], seeds) for task, seeds in assignments.items()
    ]
    path = campaign / "reports" / str(time.time_ns()) / "summary.json"
    write_json(
        path,
        {
            "schema": "n0-parallel-shard-summary-v1",
            "assignments": assignments,
            "evidence_scope": summaries[0]["evidence_scope"],
            "cells": [cell for summary in summaries for cell in summary["cells"]],
            "episodes": [
                episode for summary in summaries for episode in summary["episodes"]
            ],
        },
    )
    return path


def run_shard(
    binding_path: Path,
    campaign: Path,
    code: Path,
    package: Path,
    assignments: dict[str, list[int]],
    handoff: dict[str, Any] | None = None,
    max_wait: float = 21600,
) -> None:
    binding = read_object(binding_path)
    if binding.get("model") != "n0_twam":
        raise ValueError("parallel shard requires n0_twam")
    assignments = validate_assignments(assignments, binding.get("tasks"))
    if handoff is not None:
        adopted = read_object(Path(handoff["current_group"]) / "group.json")
        if adopted.get("seed") in assignments.get(adopted.get("task", ""), []):
            raise ValueError("assignments overlap the current group being adopted")
    if not 0 < max_wait <= 21600:
        raise ValueError("max_wait must be in (0, 21600]")
    campaign.mkdir(parents=True, exist_ok=False)
    write_json(
        campaign / "assignment.json",
        {
            "assignments": assignments,
            "hostname": socket.gethostname(),
            "binding_source": str(binding_path),
            "started_unix": time.time(),
        },
    )
    try:
        if handoff is not None:
            drain_handoff(handoff, campaign, max_wait)
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
        binding.pop("binding_sha256", None)
        binding["binding_sha256"] = canonical_hash(binding)
        frozen = campaign / "binding.json"
        write_json(frozen, binding)
        for task, seeds in assignments.items():
            event(campaign, "task_started", task=task, seeds=seeds)

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
                except Exception as exc:
                    event(
                        campaign, "report_error", task=task, seed=seed, error=str(exc)
                    )
                shard_snapshot(campaign, assignments)

            run_task_seeds(
                frozen,
                task,
                campaign / "tasks" / task,
                code,
                package,
                seeds,
                after_seed,
            )
            event(campaign, "task_exit", task=task)
        summary = shard_snapshot(campaign, assignments)
        cells = read_object(summary)["cells"]
        write_json(
            campaign / "supervisor_exit.json",
            {
                "status": "finished",
                "summary": str(summary),
                "finished_unix": time.time(),
                "all_available_episodes_completed": all(
                    cell["missing"] == 0 and cell["invalid"] == 0 for cell in cells
                ),
            },
        )
    except Exception as exc:
        write_json(
            campaign / "shard_error.json", {"error": str(exc), "unix_time": time.time()}
        )
        write_json(
            campaign / "supervisor_exit.json",
            {
                "status": "failed",
                "error": str(exc),
                "finished_unix": time.time(),
                "all_available_episodes_completed": False,
            },
        )
        raise


def run_drain_only(
    campaign: Path, handoff: dict[str, Any] | None, max_wait: float = 21600
) -> None:
    """Finish only the in-flight group; never dispatch another task or seed."""
    if handoff is None:
        raise ValueError("drain-only requires a handoff specification")
    if not 0 < max_wait <= 21600:
        raise ValueError("max_wait must be in (0, 21600]")
    campaign.mkdir(parents=True, exist_ok=False)
    write_json(
        campaign / "assignment.json",
        {
            "mode": "drain_only",
            "assignments": {},
            "hostname": socket.gethostname(),
            "started_unix": time.time(),
        },
    )
    try:
        drain_handoff(handoff, campaign, max_wait)
    except Exception as exc:
        write_json(
            campaign / "shard_error.json",
            {
                "mode": "drain_only",
                "error": str(exc),
                "unix_time": time.time(),
            },
        )
        write_json(
            campaign / "supervisor_exit.json",
            {
                "mode": "drain_only",
                "status": "failed",
                "error": str(exc),
                "finished_unix": time.time(),
                "new_episodes_launched": 0,
            },
        )
        raise
    write_json(
        campaign / "supervisor_exit.json",
        {
            "mode": "drain_only",
            "status": "finished",
            "finished_unix": time.time(),
            "new_episodes_launched": 0,
            "adoption_receipt": str(campaign / "adoption_receipt.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    for name in ("binding", "code", "package", "assignments"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--drain-only", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--max-wait-seconds", type=float, default=21600)
    args = parser.parse_args()
    if args.drain_only:
        if args.handoff is None:
            parser.error("--drain-only requires --handoff")
        if args.assignments is not None:
            parser.error("--drain-only does not accept --assignments")
        run_drain_only(
            args.campaign.absolute(), read_object(args.handoff), args.max_wait_seconds
        )
        return
    for name in ("binding", "code", "package", "assignments"):
        if getattr(args, name) is None:
            parser.error(f"normal shard execution requires --{name}")
    assignments = json.loads(
        args.assignments.read_text(), object_pairs_hook=unique_object
    )
    run_shard(
        args.binding.absolute(),
        args.campaign.absolute(),
        args.code.absolute(),
        args.package.absolute(),
        assignments,
        read_object(args.handoff) if args.handoff else None,
        args.max_wait_seconds,
    )


if __name__ == "__main__":
    main()
