"""Serial, no-clobber supervisor for retrained Isaac Clean / Robustness groups."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.retrained_evaluation.group import prepare_group, write_json


def wait_ready(
    process: subprocess.Popen[bytes], port: int, receipt: Path, timeout: int = 1800
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"policy server exited before ready: {process.returncode}"
            )
        if receipt.exists():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                pass
        time.sleep(2)
    raise TimeoutError("model server startup exceeded 1800 seconds")


def stop_owned(process: subprocess.Popen[bytes]) -> None:
    """Stop only this supervisor's newly created process group."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def resolve_isaac_python(binding: dict[str, Any]) -> Path:
    """Select a host runtime without moving the frozen deployment artifacts."""
    root = Path(binding["deployment_root"])
    isaac = Path(
        binding.get("isaac_python", root / "runtime/isaac-sim-4.5.0/python.sh")
    )
    if not root.is_absolute() or not isaac.is_absolute():
        raise ValueError("deployment_root and isaac_python must be absolute paths")
    # Constrain the launcher script, not its runtime's interpreter symlinks.
    if not isaac.resolve().is_relative_to(root.parent.resolve()):
        raise ValueError(f"Isaac launcher must remain within deployment repo: {isaac}")
    if not isaac.is_file():
        raise FileNotFoundError(f"repo-local Isaac runtime missing: {isaac}")
    return isaac


def build_process_environments(
    binding: dict[str, Any], code: Path, package: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Keep policy-runtime packages out of Isaac Sim's Python environment."""
    isaac_env = dict(os.environ)
    isaac_env.update(
        PYTHONPATH=os.pathsep.join((str(package), str(code))),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        TOKENIZERS_PARALLELISM="false",
    )
    server_env = dict(isaac_env)
    reset_time_limit_s = binding.get("evaluation", {}).get("reset_time_limit_s")
    if reset_time_limit_s is not None:
        isaac_env["ROBOTACTILE_UNIVTAC_RESET_TIME_LIMIT_S"] = str(reset_time_limit_s)
    if binding.get("shared_pythonpath"):
        server_env["PYTHONPATH"] += os.pathsep + binding["shared_pythonpath"]
    root = Path(binding["deployment_root"])
    model = binding["model"]
    if model == "ftp1_policy":
        server_env["OPENPI_DATA_HOME"] = str(root / "artifacts/openpi-data/ftp1-policy")
    elif model == "n0_vtla":
        server_env["OPENPI_DATA_HOME"] = str(
            root.parent / "deployment/artifacts/models/n0_vtla/data_cache"
        )
    elif model == "dream_tac":
        server_env["CUDNN_HOME"] = str(
            root / "runtime/ftp1-policy/lib/python3.11/site-packages/nvidia/cudnn"
        )
        server_env["CUDA_HOME"] = str(root / "runtime/cuda-toolkit-12.8")
        server_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (
                str(Path(server_env["CUDNN_HOME"]) / "lib"),
                server_env.get("LD_LIBRARY_PATH", ""),
            )
        )
    return server_env, isaac_env


def run_one(
    binding_path: Path,
    task: str,
    campaign: Path,
    code: Path,
    package: Path,
    seed: int,
    *,
    recovery_group: Path | None = None,
) -> None:
    _run_task_seeds(
        binding_path,
        task,
        campaign,
        code,
        package,
        [seed],
        recovery_group=recovery_group,
    )


def run_task_seeds(
    binding_path: Path,
    task: str,
    campaign: Path,
    code: Path,
    package: Path,
    seeds: list[int],
    after_seed: Callable[[int, Path], None],
) -> None:
    """Reuse one N0 worker while each seed owns a fresh Isaac process and group."""
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    if read_object(binding_path)["model"] != "n0_twam":
        raise ValueError("multi-seed worker reuse is N0-TWAM only")
    _run_task_seeds(
        binding_path, task, campaign, code, package, seeds, after_seed=after_seed
    )


def _run_task_seeds(
    binding_path: Path,
    task: str,
    campaign: Path,
    code: Path,
    package: Path,
    seeds: list[int],
    after_seed: Callable[[int, Path], None] | None = None,
    recovery_group: Path | None = None,
) -> None:
    binding = read_object(binding_path)
    seed = seeds[0]
    if recovery_group is not None and (len(seeds) != 1 or after_seed is not None):
        raise ValueError("reference recovery is a single missing episode only")

    def group_for(current_seed: int) -> Path:
        base = (
            campaign / "seeds" / f"seed-{current_seed:03d}"
            if after_seed is not None
            else campaign
        )
        return base / "groups" / str(binding["model"]) / task

    group = group_for(seed)
    if group.exists():
        print(
            json.dumps({"status": "preserved_existing_group", "group": str(group)}),
            flush=True,
        )
        return
    runtime = Path(binding["runtime_python"])
    if not runtime.is_file():
        raise FileNotFoundError(f"repo-local policy runtime missing: {runtime}")
    isaac = resolve_isaac_python(binding)
    # Never borrow an unrelated worker that happens to occupy our requested port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", binding["port"]))
    server_dir = campaign / "servers" / binding["model"] / task
    server_dir.mkdir(parents=True, exist_ok=False)
    receipt = server_dir / "worker" / "server_receipt.json"
    binding["server_receipt"] = str(receipt)
    runtime_binding = server_dir / "binding.json"
    write_json(runtime_binding, binding)
    calibrate = binding.get("evaluation", {}).get("measure_n0_rest", False)
    if type(calibrate) is not bool or (calibrate and binding["model"] != "n0_twam"):
        raise ValueError("measure_n0_rest must be boolean and is N0-only")
    if after_seed is not None and binding.get("rest_references", {}).get(task):
        calibrate = False
    plan: Path | None
    if recovery_group is not None:
        if calibrate:
            raise ValueError(
                "reference recovery must not recalibrate or change initial conditions"
            )
        from scripts.retrained_evaluation.availability_recovery import (
            prepare_recovery_group,
        )

        plan = prepare_recovery_group(binding, task, group, seed, recovery_group)
    else:
        plan = (
            prepare_group(binding, task, group, seed)
            if not calibrate and after_seed is None
            else None
        )
    model = binding["model"]
    server_env, isaac_env = build_process_environments(binding, code, package)
    if model == "n0_twam":
        command = [
            str(runtime),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            str(code / "scripts/n0_twam/serve_retrained.py"),
            "--artifact",
            binding["artifact"],
            "--task",
            task,
            "--output",
            str(receipt.parent),
            "--port",
            str(binding["port"]),
        ]
    else:
        module = {
            "n0_vtla": "serve_vtla",
            "ftp1_policy": "serve_ftp1",
            "dream_tac": "serve_dream",
        }[model]
        command = [
            str(runtime),
            "-m",
            f"scripts.retrained_evaluation.{module}",
            "--binding",
            str(runtime_binding),
            "--task",
            task,
            "--receipt",
            str(receipt),
        ]
        if model == "n0_vtla":
            command.extend(("--seed", str(seed)))
    started = time.monotonic()
    with (server_dir / "server.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=code,
            env=server_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            server_dir / "launch.json",
            {
                "pid": process.pid,
                "argv": command,
                "started_unix": time.time(),
                "hostname": socket.gethostname(),
                "isaac_python": str(isaac),
            },
        )
        try:
            print(
                json.dumps({"stage": "model_loading", "model": model, "task": task}),
                flush=True,
            )
            wait_ready(process, binding["port"], receipt)
            write_json(
                server_dir / "ready.json",
                {"startup_s": time.monotonic() - started, "receipt": str(receipt)},
            )
            if calibrate:
                calibration = campaign / "calibration" / task
                calibration_command = [
                    str(isaac),
                    "-m",
                    "scripts.n0_twam.single_task_robustness",
                    "calibrate",
                    "--binding",
                    str(runtime_binding),
                    "--task",
                    task,
                    "--output",
                    str(calibration),
                ]
                with (server_dir / "calibration.log").open("xb") as calibration_log:
                    measured = subprocess.Popen(
                        calibration_command,
                        cwd=code,
                        env=isaac_env,
                        stdout=calibration_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    write_json(
                        server_dir / "calibration_launch.json",
                        {
                            "pid": measured.pid,
                            "argv": calibration_command,
                            "hostname": socket.gethostname(),
                            "started_unix": time.time(),
                        },
                    )
                    try:
                        calibration_returncode = measured.wait()
                    finally:
                        stop_owned(measured)
                if calibration_returncode != 0:
                    raise RuntimeError(
                        f"N0 rest calibration exited {calibration_returncode}; see calibration.log"
                    )
                binding["rest_references"] = {
                    **binding.get("rest_references", {}),
                    task: str(calibration / "rest"),
                }
                if after_seed is None:
                    plan = prepare_group(binding, task, group, seed)
            for current_seed in seeds:
                current_group = group_for(current_seed)
                if after_seed is not None and current_group.exists():
                    after_seed(current_seed, current_group)
                    continue
                try:
                    current_plan = (
                        prepare_group(binding, task, current_group, current_seed)
                        if after_seed is not None
                        else plan
                    )
                    assert current_plan is not None
                    _run_group(
                        current_plan, current_group, isaac, code, isaac_env, started
                    )
                except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                    if after_seed is None:
                        raise
                    write_json(
                        current_group / "infrastructure_error.json",
                        {"seed": current_seed, "task": task, "error": str(error)},
                    )
                if after_seed is not None:
                    after_seed(current_seed, current_group)
        finally:
            stop_owned(process)


def _run_group(
    plan: Path,
    group: Path,
    isaac: Path,
    code: Path,
    env: dict[str, str],
    started: float,
) -> None:
    """Start a new Isaac process; seed snapshots never survive this process."""
    live_command = [
        str(isaac),
        "-m",
        "scripts.retrained_evaluation.group",
        "run",
        "--group",
        str(plan),
    ]
    with (group / "live.log").open("xb") as live_log:
        live = subprocess.Popen(
            live_command,
            cwd=code,
            env=env,
            stdout=live_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            group / "launch.json",
            {
                "pid": live.pid,
                "argv": live_command,
                "started_unix": time.time(),
                "hostname": socket.gethostname(),
                "isaac_python": str(isaac),
            },
        )
        try:
            returncode = live.wait()
        finally:
            stop_owned(live)
    result_count = len(list((group / "results").glob("*.json")))
    expected_count = len(read_object(plan)["ordered_requests"])
    has_receipt = (group / "paired_receipt.json").is_file() or (
        group / "recovery_receipt.json"
    ).is_file()
    complete = returncode == 0 and result_count == expected_count and has_receipt
    write_json(
        group / "process_exit.json",
        {
            "returncode": returncode,
            "published_result_count": result_count,
            "elapsed_s": time.monotonic() - started,
            "expected_result_count": expected_count,
            "complete": complete,
        },
    )
    print(
        json.dumps(
            {
                "stage": "group_exit",
                "group": str(group),
                "returncode": returncode,
                "published_result_count": result_count,
            }
        ),
        flush=True,
    )
    if not complete:
        raise RuntimeError(
            f"incomplete Isaac group: exit={returncode}, results={result_count}/{expected_count}, "
            f"receipt={has_receipt}; see {group / 'execution_failure.json'} and live.log"
        )


def report(campaign: Path) -> dict[str, Any]:
    rows = [
        read_object(path)
        for path in sorted((campaign / "groups").glob("*/*/results/*.json"))
    ]
    cells: dict[str, Any] = {}
    for row in rows:
        label = row["request"].removeprefix("requests/").removesuffix(".json")
        key = f"{row['model']}/{label}"
        cell = cells.setdefault(
            key, {"completed": 0, "eligible": 0, "success": 0, "tasks": []}
        )
        cell["completed"] += 1
        # Preserve raw receipts, but do not attribute benchmark injection errors
        # to the learned policy's robustness. The legacy runner marks crashes
        # eligible generically; this derived report records the narrower scope.
        eligible = row["score_eligible"] and row.get("failure_stage") not in {
            "delivery",
            "artifact_export",
            "reset",
            "close",
        }
        row["benchmark_score_eligible"] = bool(eligible)
        cell["eligible"] += int(eligible)
        cell["success"] += int(row["score_success"] is True and eligible)
        cell["tasks"].append(row["task"])
    for cell in cells.values():
        cell["success_rate"] = (
            cell["success"] / cell["eligible"] if cell["eligible"] else None
        )
    return {
        "evidence_scope": "one_seed_simulator_diagnostic_not_paper_statistics",
        "cells": cells,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["n0_twam", "ftp1_policy", "n0_vtla", "dream_tac"],
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--binding-root", type=Path)
    parser.add_argument("--wait-for-gpu-lock", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.absolute()
    with (campaign / "gpu.lock").open("a") as lock:
        fcntl.flock(
            lock, fcntl.LOCK_EX | (0 if args.wait_for_gpu_lock else fcntl.LOCK_NB)
        )
        bindings = {
            model: (args.binding_root or campaign / "bindings") / f"{model}.json"
            for model in args.models
        }
        available = [path for path in bindings.values() if path.is_file()]
        if not available:
            raise RuntimeError("no retrained binding ready")
        tasks = args.tasks or list(read_object(available[0])["tasks"])
        for task in tasks:
            for model, path in bindings.items():
                if not path.is_file():
                    print(
                        json.dumps(
                            {
                                "model": model,
                                "task": task,
                                "status": "not_run_binding_missing",
                            }
                        ),
                        flush=True,
                    )
                    continue
                try:
                    run_one(
                        path,
                        task,
                        campaign,
                        args.code.absolute(),
                        args.package.absolute(),
                        0,
                    )
                except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                    # Keep all live artifacts. An infrastructure failure is not
                    # a failed task and never triggers an automatic rerun.
                    errors = campaign / "errors" / model
                    write_json(
                        errors / f"{task}.json",
                        {
                            "task": task,
                            "model": model,
                            "status": "infrastructure_failure",
                            "error": str(error),
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "model": model,
                                "task": task,
                                "stage": "infrastructure_failure",
                                "error": str(error),
                            }
                        ),
                        flush=True,
                    )
                snapshot = campaign / "reports" / f"{time.time_ns()}.json"
                write_json(snapshot, report(campaign))
        write_json(
            campaign / f"campaign_exit-{'-'.join(args.models)}.json", report(campaign)
        )


if __name__ == "__main__":
    main()
