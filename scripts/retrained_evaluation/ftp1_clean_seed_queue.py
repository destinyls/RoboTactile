#!/usr/bin/env python3
"""Run one FTP-1 Clean task over an independent seed range.

The FTP-1 policy server is loaded once per task/GPU. Every seed still starts
a fresh Isaac process, so simulator and task state are never reused.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExporter,
    LiveArtifactExportReceipt,
    execute_live_univtac_run,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.integrations.n0_twam.retrained import (
    file_sha256,
    read_object,
)
from scripts.retrained_evaluation.group import build_clean, policy_factory, write_json


def _wait_ready(
    process: subprocess.Popen[bytes], port: int, receipt: Path, timeout: int = 1800
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"FTP-1 server exited before ready: {process.returncode}"
            )
        if receipt.is_file():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                pass
        time.sleep(2)
    raise TimeoutError("FTP-1 startup exceeded 1800 seconds")


def _stop_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _freeze_binding(source: Path, output: Path, *, port: int) -> dict[str, Any]:
    original = read_object(source)
    binding = deepcopy(original)
    if binding.get("model") != "ftp1_policy":
        raise ValueError("FTP-1 Clean queue requires an FTP-1 binding")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    parent = binding.get("binding_sha256")
    tasks: dict[str, dict[str, Any]] = {}
    for task, entry in binding.get("tasks", {}).items():
        tasks[task] = {
            key: value for key, value in entry.items() if key != "transport_metadata"
        }
    binding.update(
        endpoint=f"tcp://127.0.0.1:{port}",
        port=port,
        execution_host=socket.gethostname(),
        parent_binding_sha256=parent,
        tasks=tasks,
    )
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = canonical_hash(binding)
    for task, entry in binding["tasks"].items():
        previous = original["tasks"][task]["transport_metadata"]
        entry["transport_metadata"] = {
            **previous,
            "serve_bundle_sha256": binding["binding_sha256"],
        }
    write_json(output, binding)
    return binding


def _process_environments(
    binding: dict[str, Any], code: Path, package: Path, gpu_id: int
) -> tuple[dict[str, str], dict[str, str]]:
    common = dict(os.environ)
    common.update(
        PYTHONPATH=os.pathsep.join((str(package), str(code))),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=str(gpu_id),
        TOKENIZERS_PARALLELISM="false",
    )
    server = dict(common)
    root = Path(binding["deployment_root"])
    server["OPENPI_DATA_HOME"] = str(root / "artifacts/openpi-data/ftp1-policy")
    return server, common


def _seed_dir(campaign: Path, seed: int) -> Path:
    return campaign / "seeds" / f"seed-{seed:07d}"


def _worker(binding_path: Path, task: str, seed: int, output: Path) -> None:
    binding = read_object(binding_path)
    if binding.get("model") != "ftp1_policy" or task not in binding.get("tasks", {}):
        raise ValueError("worker task is outside the frozen FTP-1 binding")
    receipt = read_object(output.parent.parent / "server/server_receipt.json")
    if receipt != binding["tasks"][task]["transport_metadata"]:
        raise ValueError("FTP-1 server receipt differs from the frozen binding")

    episode = output / "episode"
    episode.mkdir(parents=True, exist_ok=False)
    request = replace(
        build_clean(binding, task, episode, seed),
        runtime_dir=episode / "runtime",
        output_dir=episode / "artifact",
    )
    write_json(episode / "request.json", live_univtac_request_to_dict(request))
    started = time.monotonic()

    def publish(
        output: Path,
        loaded: LoadedLiveUniVTACRun,
        evidence: ClosedLoopExecutionEvidence,
    ) -> LiveArtifactExportReceipt:
        exported = write_live_univtac_artifact(
            output,
            loaded,
            evidence,
            capture_profile=LiveCaptureProfile.METRICS_ONLY,
        )
        reopened = load_live_univtac_artifact(output)
        write_json(
            episode / "result.json",
            {
                **result_to_dict(reopened.evidence.result),
                "artifact": str(output),
                "checkpoint_sha256": binding["checkpoint_sha256"],
                "model": "ftp1_policy",
                "root_receipt_sha256": reopened.root_receipt_sha256,
                "seed": seed,
                "task": task,
                "wall_episode_s": time.monotonic() - started,
            },
        )
        return exported

    exporter: LiveArtifactExporter = publish
    execute_live_univtac_run(
        request,
        policy_factory=partial(policy_factory, binding),
        artifact_exporter=exporter,
    )


def _run(args: argparse.Namespace) -> None:
    if args.seed_start < 0 or args.seed_count < 1 or args.gpu_id < 0:
        raise ValueError("seed range and gpu id must be non-negative")
    source = args.binding.resolve(strict=True)
    code = args.code.resolve(strict=True)
    package = args.package.resolve(strict=True)
    runtime = args.runtime_python.resolve(strict=True)
    isaac = args.isaac_python.resolve(strict=True)
    campaign = args.campaign.absolute()
    campaign.mkdir(parents=True, exist_ok=False)
    binding_path = campaign / "binding.json"
    binding = _freeze_binding(source, binding_path, port=args.port)
    if args.task not in binding["tasks"]:
        raise ValueError(f"task is absent from FTP-1 binding: {args.task}")
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    write_json(
        campaign / "plan.json",
        {
            "binding_file_sha256": file_sha256(binding_path),
            "capture_profile": LiveCaptureProfile.METRICS_ONLY.value,
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "evidence_scope": "clean_100_seed_closed_loop_diagnostic",
            "gpu_id": args.gpu_id,
            "isaac_lifecycle": "fresh_process_per_seed",
            "model": "ftp1_policy",
            "policy_server_lifecycle": "one_load_per_task_queue",
            "seeds": list(seeds),
            "task": args.task,
        },
    )

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit("stopped by signal; preserve all completed seeds")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server_dir = campaign / "server"
    server_dir.mkdir(exist_ok=False)
    receipt = server_dir / "server_receipt.json"
    server_env, isaac_env = _process_environments(binding, code, package, args.gpu_id)
    command = [
        str(runtime),
        "-m",
        "scripts.retrained_evaluation.serve_ftp1",
        "--binding",
        str(binding_path),
        "--task",
        args.task,
        "--receipt",
        str(receipt),
    ]
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    try:
        with (server_dir / "server.log").open("xb") as server_log:
            process = subprocess.Popen(
                command,
                cwd=code,
                env=server_env,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                server_dir / "launch.json",
                {"argv": command, "gpu_id": args.gpu_id, "pid": process.pid},
            )
            _wait_ready(process, args.port, receipt)
            write_json(
                server_dir / "ready.json",
                {"startup_s": time.monotonic() - started, "receipt": str(receipt)},
            )
            for sequence, seed in enumerate(seeds):
                selected = _seed_dir(campaign, seed)
                selected.mkdir(parents=True, exist_ok=False)
                write_json(
                    campaign / "events" / f"{sequence:03d}-seed-{seed}-started.json",
                    {"seed": seed, "sequence": sequence, "started_unix": time.time()},
                )
                worker_command = [
                    str(isaac),
                    "-m",
                    "scripts.retrained_evaluation.ftp1_clean_seed_queue",
                    "worker",
                    "--binding",
                    str(binding_path),
                    "--task",
                    args.task,
                    "--seed",
                    str(seed),
                    "--output",
                    str(selected),
                ]
                try:
                    with (selected / "worker.log").open("xb") as worker_log:
                        worker = subprocess.run(
                            worker_command,
                            cwd=code,
                            env=isaac_env,
                            stdin=subprocess.DEVNULL,
                            stdout=worker_log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    write_json(
                        selected / "worker_exit.json", {"returncode": worker.returncode}
                    )
                    result_path = selected / "episode/result.json"
                    if worker.returncode != 0 or not result_path.is_file():
                        raise RuntimeError(
                            f"Clean worker incomplete: returncode={worker.returncode}"
                        )
                except BaseException as error:
                    write_json(
                        campaign / "events" / f"{sequence:03d}-seed-{seed}-failed.json",
                        {
                            "error": str(error),
                            "error_type": type(error).__name__,
                            "failed_unix": time.time(),
                            "seed": seed,
                            "sequence": sequence,
                        },
                    )
                    traceback.print_exc()
                    raise
                result = read_object(result_path)
                write_json(
                    campaign / "events" / f"{sequence:03d}-seed-{seed}-completed.json",
                    {
                        "completed_unix": time.time(),
                        "score_eligible": result["score_eligible"],
                        "score_success": result["score_success"],
                        "seed": seed,
                        "sequence": sequence,
                    },
                )
    finally:
        if process is not None:
            _stop_owned(process)

    results = [
        read_object(_seed_dir(campaign, seed) / "episode/result.json") for seed in seeds
    ]
    eligible = [row for row in results if row["score_eligible"]]
    successes = sum(row["score_success"] is True for row in eligible)
    write_json(
        campaign / "summary.json",
        {
            "completed_count": len(results),
            "eligible_count": len(eligible),
            "model": "ftp1_policy",
            "planned_count": len(seeds),
            "success_count": successes,
            "success_rate": successes / len(eligible) if eligible else None,
            "task": args.task,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--binding", type=Path, required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--code", type=Path, required=True)
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--isaac-python", type=Path, required=True)
    run.add_argument("--seed-start", type=int, default=1_000_000)
    run.add_argument("--seed-count", type=int, default=100)
    run.add_argument("--gpu-id", type=int, default=0)
    run.add_argument("--port", type=int, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--binding", type=Path, required=True)
    worker.add_argument("--task", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "worker":
        _worker(args.binding.resolve(strict=True), args.task, args.seed, args.output)
    else:
        _run(args)


if __name__ == "__main__":
    main()
