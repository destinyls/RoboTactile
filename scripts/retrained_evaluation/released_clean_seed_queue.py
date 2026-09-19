"""Run released ACT/N0-TWAM Clean seeds with fresh Isaac processes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import (
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    resolve_univtac_full_horizon_budget,
)
from robotactile_benchmark.closed_loop.contracts import WallTimeoutRole
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.act.requests import build_official_act_request
from robotactile_benchmark.integrations.n0_twam.requests import (
    build_official_n0_clean_request,
)
from robotactile_benchmark.integrations.n0_twam.retrained import file_sha256
from robotactile_benchmark.integrations.runtime_config import (
    resolve_act_runtime_artifacts,
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.trials import Condition
from scripts.n0_twam.run_official_early_fault_seed import (
    official_server_command,
    official_server_environment,
)


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def valid_terminal(value: dict[str, Any]) -> bool:
    """Infrastructure crashes and unvalidated artifacts are not SR outcomes."""
    return (
        value.get("score_eligible") is True
        and value.get("validation_passed") is True
        and value.get("terminal_status") != "crash"
        and isinstance(value.get("observation_count"), int)
        and value["observation_count"] > 0
    )


def stop_owned(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def worker_command(args: argparse.Namespace, request: Path) -> list[str]:
    command = [
        str(args.isaac_python),
        "-m",
        "robotactile_benchmark.cli",
        "live-univtac-run",
        "--root",
        str(args.root),
        "--request",
        str(request),
        "--config",
        str(args.config),
        "--capture-profile",
        "metrics_only_v1",
    ]
    if args.model == "n0_twam":
        command.extend(
            [
                "--n0-source-root",
                str(getattr(args, "n0_source", args.root / "sources/N0-TWAM")),
                "--n0-port",
                str(args.port),
                "--action-execution-contract",
                N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
            ]
        )
    return command


def build_request(args: argparse.Namespace, seed: int, output: Path) -> object:
    layout = DeploymentLayout(args.root)
    horizon, observations = resolve_univtac_full_horizon_budget(args.task)
    common = dict(
        layout=layout,
        dataset_sha256=args.dataset_sha256,
        initial_seed=seed,
        exogenous_seed=seed,
        max_control_cycles=horizon,
        max_observation_steps=observations,
        wall_timeout_s=args.episode_timeout_s,
        simulator_device="cuda:0",
        live_output_dir=output / "artifact",
    )
    if args.model == "n0_twam":
        resolved = resolve_n0_runtime_artifacts(args.config)
        request = build_official_n0_clean_request(
            manifest=resolved.manifest,
            wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
            **common,
        )
    else:
        resolved = resolve_act_runtime_artifacts(args.config)
        request = build_official_act_request(
            base_manifest=resolved.manifest,
            condition=Condition.CLEAN,
            act_device_name=resolved.device,
            **common,
        )
    if resolved.manifest.task_id != args.task:
        raise ValueError("configuration task differs from requested task")
    return replace(
        request,
        runtime_dir=output / "runtime",
        upstream_root=args.root / "sources/UniVTAC",
    )


def run(args: argparse.Namespace) -> None:
    if args.seed_start < 0 or args.seed_count < 1 or args.gpu_id < 0:
        raise ValueError("invalid seed range or GPU id")
    args.root = args.root.resolve(strict=True)
    args.code = args.code.resolve(strict=True)
    args.package = args.package.resolve(strict=True)
    args.config = args.config.resolve(strict=True)
    args.isaac_python = args.isaac_python.resolve(strict=True)
    args.use_official_server = args.model_root is not None
    args.model_root = (
        args.root if args.model_root is None else args.model_root.resolve(strict=True)
    )
    args.n0_python = (
        args.root / "runtime/n0-twam/bin/python"
        if args.n0_python is None
        else args.n0_python.resolve(strict=True)
    )
    args.n0_source = (
        args.root / "sources/N0-TWAM"
        if args.n0_source is None
        else args.n0_source.resolve(strict=True)
    )
    args.digest_cache = (
        args.root / "runtime/artifact-digest-cache/n0-twam"
        if args.digest_cache is None
        else args.digest_cache.resolve()
    )
    if not args.n0_python.is_file() or not os.access(args.n0_python, os.X_OK):
        raise FileNotFoundError(f"N0 runtime launcher is unavailable: {args.n0_python}")
    if not args.n0_source.is_dir():
        raise FileNotFoundError(f"N0 source root is unavailable: {args.n0_source}")
    args.digest_cache.mkdir(parents=True, exist_ok=True)
    args.campaign = args.campaign.absolute()
    args.campaign.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_id),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join(
            tuple(
                item
                for item in (str(args.package), str(args.code), str(args.n0_source))
                if item
            )
        ),
        ROBOTACTILE_REPOSITORY_ROOT=str(args.code),
        ROBOTACTILE_PACKAGE_PATH=str(args.package),
        ROBOTACTILE_N0_DIGEST_CACHE_DIR=str(args.digest_cache),
        TOKENIZERS_PARALLELISM="false",
    )
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    write_once(
        args.campaign / "plan.json",
        {
            "model": args.model,
            "task": args.task,
            "seeds": seeds,
            "gpu_id": args.gpu_id,
            "host": socket.gethostname(),
            "config": str(args.config),
            "config_sha256": file_sha256(args.config),
            "dataset_sha256": args.dataset_sha256,
            "capture_profile": "metrics_only_v1",
            "fresh_isaac_per_seed": True,
            "worker_sha256": file_sha256(Path(__file__)),
            "model_root": str(args.model_root),
            "n0_python": str(args.n0_python),
            "n0_source": str(args.n0_source),
            "digest_cache": str(args.digest_cache),
        },
    )

    def interrupted(_signum: int, _frame: object) -> None:
        raise SystemExit("paused; completed seeds are preserved")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    server: subprocess.Popen[bytes] | None = None
    worker: subprocess.Popen[bytes] | None = None
    results: list[dict[str, Any]] = []
    generation = 0

    def start_server() -> subprocess.Popen[bytes]:
        nonlocal generation
        generation += 1
        selected = args.campaign / "servers" / f"generation-{generation:03d}"
        selected.mkdir(parents=True, exist_ok=False)
        if not args.use_official_server:
            command = [
                "bash",
                str(args.code / "scripts/n0_twam/serve_univtac.sh"),
                "--root",
                str(args.root),
                "--task",
                args.task,
                "--gpus",
                str(args.gpu_id),
                "--port",
                str(args.port),
                "--master-port",
                str(args.master_port),
            ]
            server_env = env
        else:
            server_env = official_server_environment(
                root=args.model_root,
                task=args.task,
                save_root=selected / "dumps",
                inherited=env,
            )
            command = official_server_command(
                n0_python=args.n0_python,
                repo=args.code,
                port=args.port,
                save_root=selected / "dumps",
            )
        with (selected / "server.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                env=server_env,
                cwd=args.code,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        write_once(selected / "launch.json", {"pid": process.pid, "argv": command})
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"N0 server startup exited {process.returncode}")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                        return process
                except OSError:
                    time.sleep(2)
            raise TimeoutError("N0 server startup exceeded 1800 seconds")
        except BaseException:
            stop_owned(process)
            raise

    try:
        for seed in seeds:
            result: dict[str, Any] | None = None
            for attempt in range(1, args.max_infrastructure_attempts + 1):
                selected = (
                    args.campaign / "seeds" / f"seed-{seed:07d}" / f"attempt-{attempt}"
                )
                selected.mkdir(parents=True, exist_ok=False)
                if args.model == "n0_twam" and server is None:
                    server = start_server()
                request = build_request(args, seed, selected)
                request_path = selected / "request.json"
                write_once(request_path, live_univtac_request_to_dict(request))
                command = worker_command(args, request_path)
                started = time.monotonic()
                with (selected / "worker.log").open("xb") as log:
                    worker = subprocess.Popen(
                        command,
                        cwd=args.code,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    write_once(
                        selected / "launch.json", {"pid": worker.pid, "argv": command}
                    )
                    try:
                        returncode = worker.wait(timeout=args.episode_timeout_s + 900)
                    except subprocess.TimeoutExpired:
                        stop_owned(worker)
                        returncode = 124
                write_once(
                    selected / "worker_exit.json",
                    {
                        "returncode": returncode,
                        "wall_s": time.monotonic() - started,
                    },
                )
                terminal = selected / "artifact/terminal_result.json"
                if terminal.is_file():
                    value = json.loads(terminal.read_text())
                    if valid_terminal(value):
                        result = {
                            **value,
                            "seed": seed,
                            "artifact": str(terminal.parent),
                        }
                        write_once(selected.parent / "result.json", result)
                        results.append(result)
                        break
                write_once(
                    selected / "infrastructure_failure.json",
                    {
                        "seed": seed,
                        "attempt": attempt,
                        "returncode": returncode,
                        "terminal_present": terminal.exists(),
                    },
                )
                stop_owned(server)
                server = None
            if result is None:
                raise RuntimeError(
                    f"seed {seed} has no valid outcome after infrastructure retries"
                )
    finally:
        stop_owned(worker)
        stop_owned(server)
    successes = sum(row["score_success"] is True for row in results)
    write_once(
        args.campaign / "summary.json",
        {
            "model": args.model,
            "task": args.task,
            "planned_count": len(seeds),
            "completed_count": len(results),
            "eligible_count": len(results),
            "success_count": successes,
            "success_rate": successes / len(results),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("act", "n0_twam"), required=True)
    for name in ("root", "code", "package", "config", "isaac-python", "campaign"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, default=37200)
    parser.add_argument("--master-port", type=int, default=38200)
    parser.add_argument("--episode-timeout-s", type=float, default=7200)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--n0-python", type=Path)
    parser.add_argument("--n0-source", type=Path)
    parser.add_argument("--digest-cache", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
