"""Run a fresh Clean/A2 pair in one Isaac process; never recover an old snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from robotactile_benchmark.policies.n0_vtla_execution import (
    RECEDING_HORIZON_50X8,
    n0_vtla_execution_steps,
)
from scripts.retrained_evaluation.campaign import run_one
from scripts.retrained_evaluation.group import prepare_group, write_json

TASK = "put_bottle_in_shelf"
A2 = "A2_frame_erasure"


def prepare(
    source: Path, output: Path, seed: int, execution_profile: str | None = None
) -> None:
    if output.exists():
        raise FileExistsError(f"preserve existing run: {output}")
    binding = deepcopy(read_object(source))
    if binding["model"] != "n0_vtla" or TASK not in binding["tasks"]:
        raise ValueError("requires the N0-VTLA shelf checkpoint binding")
    if execution_profile is not None:
        n0_vtla_execution_steps(execution_profile, TASK)
        binding.setdefault("evaluation", {})["n0_vtla_execution_profile"] = (
            execution_profile
        )
    binding.setdefault("evaluation", {}).update(
        tactile_availability_mode="native_missing_v1",
        operators=[A2],
        fault_window_mode="full_episode_v1",
        fault_start_index=0,
        severity_registries=["optical_marker_extreme_v1"],
        severity_level=5,
        sensor_slots=["left", "right"],
        capture_profile="paper_full_v1",
        a2_end_policy="episode_censored_v1",
    )
    binding["evaluation"].pop("tactile_zero_shape", None)
    binding["parent_binding_sha256"] = binding["binding_sha256"]
    binding["binding_sha256"] = canonical_hash(
        {k: v for k, v in binding.items() if k != "binding_sha256"}
    )
    prepare_group(binding, TASK, output / "prepared", seed)
    write_json(output / "binding.json", binding)
    plan = {
        "task": TASK,
        "model": "n0_vtla",
        "seed": seed,
        "conditions": ["clean", A2],
        "planned_episodes": 2,
        "pairing": "same_process_canonical_snapshot_v1",
        "historical_reset_reference": None,
        "old_results_modified": False,
        "a1_repeated": False,
        "new_clean_reason": "independent_control_for_A2_not_replacement_of_historical_score",
        "evidence_scope": "one_seed_closed_loop_diagnostic_not_paper_statistics",
        "source_binding": str(source),
        "binding_sha256": binding["binding_sha256"],
    }
    profile = binding["evaluation"].get("n0_vtla_execution_profile")
    if profile is not None:
        plan["n0_vtla_execution_profile"] = profile
        plan["prediction_horizon"] = 50
        plan["execute_action_steps"] = n0_vtla_execution_steps(profile, TASK)
    write_json(output / "plan.json", plan)


def launch(output: Path, code: Path, package: Path) -> None:
    if (output / "launch.json").exists() or (output / "started.json").exists():
        raise FileExistsError("already started; do not repeat episodes")
    binding = read_object(output / "binding.json")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    ).strip():
        raise RuntimeError("GPU occupied; no process was interrupted")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", binding["port"]))
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(package), str(code))),
        PYTHONUNBUFFERED="1",
    )
    script = Path(__file__).absolute()
    command = [
        sys.executable,
        "-B",
        "-u",
        str(script),
        "run",
        "--output",
        str(output),
        "--code",
        str(code),
        "--package",
        str(package),
    ]
    with (output / "launch.json").open("x", encoding="utf-8") as receipt:
        with (output / "supervisor.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=code,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        ticks = (
            Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
        data = {
            "pid": process.pid,
            "start_ticks": ticks,
            "started_unix": time.time(),
            "command": command,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "code": str(code),
            "package": str(package),
            "status": "launched_not_completed",
        }
        json.dump(data, receipt, indent=2)
        receipt.flush()
        os.fsync(receipt.fileno())
    print(json.dumps(data), flush=True)


def run(output: Path, code: Path, package: Path) -> None:
    from robotactile_benchmark.visualization.live_artifact import (
        export_live_artifact_visualization,
    )

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit("stopped; preserve all artifacts")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    plan = read_object(output / "plan.json")
    write_json(output / "started.json", {"started_unix": time.time(), "plan": plan})
    try:
        # No recovery_group: both episodes share this process's canonical state.
        run_one(
            output / "binding.json",
            TASK,
            output / "campaign",
            code,
            package,
            plan["seed"],
        )
        group = output / "campaign/groups/n0_vtla" / TASK
        paired = read_object(group / "paired_receipt.json")
        results: list[dict[str, Any]] = []
        for index, condition in enumerate(plan["conditions"]):
            row = read_object(group / "results" / f"{index:02d}.json")
            valid = (
                row["validation_passed"] is True
                and row["score_eligible"] is True
                and row["failure_stage"] is None
                and row["terminal_status"] in {"success", "timeout", "early_stop"}
            )
            results.append({**row, "condition": condition, "valid": valid})
        summary = {
            "task": TASK,
            "model": "n0_vtla",
            "seed": plan["seed"],
            "paired_receipt": str(group / "paired_receipt.json"),
            "group_content_sha256": paired["group_content_sha256"],
            "valid_count": sum(row["valid"] for row in results),
            "results": results,
            "evidence_scope": plan["evidence_scope"],
        }
        if "n0_vtla_execution_profile" in plan:
            summary["n0_vtla_execution_profile"] = plan["n0_vtla_execution_profile"]
            summary["prediction_horizon"] = 50
            summary["execute_action_steps"] = plan["execute_action_steps"]
        # Publish metrics before the optional, slower media export.
        write_json(output / "summary.json", summary)
        if summary["valid_count"] != 2:
            raise RuntimeError(
                "Clean/A2 pair produced an invalid execution; inspect summary.json"
            )
        for row in results:
            media = export_live_artifact_visualization(
                Path(row["artifact"]),
                output / "videos" / row["condition"],
                fps=10,
                stride=1,
                video=True,
            )
            write_json(
                output / "videos" / f"{row['condition']}.json", media.to_cli_dict()
            )
        write_json(
            output / "completed.json",
            {
                "ended_unix": time.time(),
                "status": "valid_pair_and_videos",
                "summary": str(output / "summary.json"),
            },
        )
    except (Exception, SystemExit) as error:
        write_json(
            output / "interrupted.json",
            {
                "ended_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-profile", choices=(RECEDING_HORIZON_50X8,))
    args = parser.parse_args()
    if args.command == "prepare":
        if args.binding is None or args.seed < 0:
            parser.error("prepare requires --binding and nonnegative --seed")
        prepare(
            args.binding.absolute(),
            args.output.absolute(),
            args.seed,
            args.execution_profile,
        )
    else:
        if args.execution_profile is not None:
            parser.error("--execution-profile is fixed during prepare, not launch/run")
        if args.code is None or args.package is None:
            parser.error("launch/run require --code and --package")
        {"launch": launch, "run": run}[args.command](
            args.output.absolute(), args.code.absolute(), args.package.absolute()
        )


if __name__ == "__main__":
    main()
