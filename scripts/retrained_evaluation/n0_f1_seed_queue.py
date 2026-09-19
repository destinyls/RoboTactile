"""Run one official N0-TWAM F1 condition across a frozen seed range.

One invocation owns one task, one physical GPU, and one N0 server.  Every
rollout uses a fresh Isaac process and a freshly generated, hash-bound F1
request.  Clean requests are generated only as immutable pairing sources and
are never executed by this queue.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.fault_timing import EARLY_RANDOM_ONSET_MODE
from robotactile_benchmark.integrations.n0_twam.retrained import file_sha256
from robotactile_benchmark.n0_fault_campaign import (
    N0FaultCampaignGenerationSpec,
    N0FaultCellDisposition,
    generate_n0_fault_campaign_bundle,
)
from robotactile_benchmark.trials import Condition
from scripts.n0_twam.run_official_early_fault_seed import (
    official_server_command,
    official_server_environment,
)
from scripts.retrained_evaluation.released_clean_seed_queue import (
    build_request,
    stop_owned,
    valid_terminal,
    worker_command,
    write_once,
)

F1 = "F1_global_response_drift"
REGISTRY = "optical_marker_extreme_v1"
TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)


def _write_canonical_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def _check_port_available(port: int) -> None:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def _validate_rest_reference(path: Path, task: str) -> Path:
    selected = Path(path).resolve(strict=True)
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError("rest reference must be a regular directory")
    loaded = load_rest_reference_artifact(selected)
    if loaded.validation.task != task:
        raise ValueError("rest reference belongs to another task")
    return selected


def _select_f1_request(bundle: Any) -> Path:
    matches = [
        cell
        for cell in bundle.manifest.cells
        if cell.condition is Condition.FAULTED
        and cell.operator_id == F1
        and cell.severity_level == 5
        and cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
    ]
    if len(matches) != 1 or matches[0].request_relpath is None:
        raise ValueError("generated bundle does not contain one live F1 request")
    request = bundle.root / str(matches[0].request_relpath)
    loaded = load_live_univtac_request(request)
    if loaded.condition is not Condition.FAULTED:
        raise ValueError("selected request is not faulted")
    return request


def build_f1_request(
    args: argparse.Namespace,
    *,
    seed: int,
    attempt_root: Path,
    rest_reference: Path,
) -> Path:
    source_root = attempt_root / "source"
    source_root.mkdir(parents=True, exist_ok=False)
    clean = build_request(args, seed, source_root)
    clean_path = source_root / "base_clean_request.json"
    _write_canonical_once(clean_path, live_univtac_request_to_dict(clean))
    _, bundle = generate_n0_fault_campaign_bundle(
        attempt_root / "fault_campaign",
        N0FaultCampaignGenerationSpec(
            campaign_id=f"n0-f1-{args.task}-seed-{seed}",
            base_clean_request_paths=(clean_path,),
            operator_ids=(F1,),
            severity_levels=(5,),
            operator_seed_master=seed,
            fault_start_index=0,
            fault_stop_index=clean.max_observation_steps,
            rest_reference_artifacts={args.task: rest_reference},
            severity_registry=REGISTRY,
            fault_onset_mode=EARLY_RANDOM_ONSET_MODE,
            fault_onset_max_index=8,
        ),
    )
    return _select_f1_request(bundle)


def _run_calibration(
    args: argparse.Namespace,
    env: dict[str, str],
) -> Path:
    root = args.campaign / "calibration"
    root.mkdir(parents=True, exist_ok=False)
    base = build_request(args, args.calibration_seed, root / "source")
    base_path = root / "base_clean_request.json"
    _write_canonical_once(base_path, live_univtac_request_to_dict(base))
    rest = root / "rest"
    command = [
        str(args.isaac_python),
        str(args.code / "scripts/n0_twam/run_rest_calibration.py"),
        "--root",
        str(args.root),
        "--base-clean-request",
        str(base_path),
        "--integration-config",
        str(args.config),
        "--n0-source-root",
        str(args.n0_source),
        "--n0-port",
        str(args.port),
        "--initial-seed",
        str(args.calibration_seed),
        "--exogenous-seed",
        str(args.calibration_seed + 1),
        "--request-output",
        str(root / "request.json"),
        "--live-output",
        str(root / "live"),
        "--rest-output",
        str(rest),
    ]
    with (root / "worker.log").open("xb") as log:
        result = subprocess.run(
            command,
            cwd=args.code,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.episode_timeout_s + 900,
            check=False,
        )
    write_once(root / "worker_exit.json", {"returncode": result.returncode})
    if result.returncode != 0:
        raise RuntimeError("rest calibration failed; inspect calibration/worker.log")
    return _validate_rest_reference(rest, args.task)


def run(args: argparse.Namespace) -> None:
    if args.task not in TASKS:
        raise ValueError("task must be one of the eight official UniVTAC tasks")
    if args.seed_start < 0 or args.seed_count < 1 or args.gpu_id < 0:
        raise ValueError("invalid seed range or GPU id")
    if args.rest_reference is None and not args.calibrate_rest:
        raise ValueError("provide --rest-reference or --calibrate-rest")
    if args.rest_reference is not None and args.calibrate_rest:
        raise ValueError("rest reference and calibration are mutually exclusive")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")

    for name in (
        "root",
        "model_root",
        "code",
        "package",
        "config",
        "isaac_python",
        "n0_python",
        "n0_source",
        "digest_cache",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    args.model = "n0_twam"
    args.campaign = Path(args.campaign).absolute()
    _check_port_available(args.port)
    args.campaign.mkdir(parents=True, exist_ok=False)

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_id),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join(
            (str(args.package), str(args.code), str(args.n0_source))
        ),
        ROBOTACTILE_REPOSITORY_ROOT=str(args.code),
        ROBOTACTILE_PACKAGE_PATH=str(args.package),
        ROBOTACTILE_N0_DIGEST_CACHE_DIR=str(args.digest_cache),
        TOKENIZERS_PARALLELISM="false",
    )
    write_once(
        args.campaign / "plan.json",
        {
            "schema": "robotactile-n0-f1-seed-queue-v1",
            "model": "n0_twam",
            "task": args.task,
            "condition": F1,
            "severity_registry": REGISTRY,
            "severity_level": 5,
            "fault_window_mode": EARLY_RANDOM_ONSET_MODE,
            "fault_onset_max_index": 8,
            "seeds": seeds,
            "planned_rollouts": len(seeds),
            "gpu_id": args.gpu_id,
            "host": socket.gethostname(),
            "config": str(args.config),
            "config_sha256": file_sha256(args.config),
            "dataset_sha256": args.dataset_sha256,
            "digest_cache": str(args.digest_cache),
            "capture_profile": "metrics_only_v1",
            "fresh_isaac_per_seed": True,
            "persistent_n0_server": True,
            "clean_requests_executed": False,
            "worker_sha256": file_sha256(Path(__file__)),
        },
    )

    def interrupted(_signum: int, _frame: object) -> None:
        raise SystemExit("paused; completed seeds are preserved")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    server: subprocess.Popen[bytes] | None = None
    worker: subprocess.Popen[bytes] | None = None
    generation = 0
    results: list[dict[str, Any]] = []

    def start_server() -> subprocess.Popen[bytes]:
        nonlocal generation
        generation += 1
        selected = args.campaign / "servers" / f"generation-{generation:03d}"
        selected.mkdir(parents=True, exist_ok=False)
        server_env = official_server_environment(
            root=args.model_root,
            task=args.task,
            save_root=selected / "dumps",
            inherited=env,
        )
        server_env["PYTHONPATH"] = env["PYTHONPATH"]
        command = official_server_command(
            n0_python=args.n0_python,
            repo=args.code,
            port=args.port,
            save_root=selected / "dumps",
        )
        with (selected / "server.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=args.code,
                env=server_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_once(selected / "launch.json", {"pid": process.pid, "argv": command})
        deadline = time.monotonic() + 1800
        try:
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
        server = start_server()
        rest_reference = (
            _run_calibration(args, env)
            if args.calibrate_rest
            else _validate_rest_reference(args.rest_reference, args.task)
        )
        for seed in seeds:
            result: dict[str, Any] | None = None
            for attempt in range(1, args.max_infrastructure_attempts + 1):
                selected = (
                    args.campaign / "seeds" / f"seed-{seed:07d}" / f"attempt-{attempt}"
                )
                selected.mkdir(parents=True, exist_ok=False)
                if server is None:
                    server = start_server()
                request = build_f1_request(
                    args,
                    seed=seed,
                    attempt_root=selected,
                    rest_reference=rest_reference,
                )
                command = worker_command(args, request)
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
                        selected / "launch.json",
                        {"pid": worker.pid, "argv": command},
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
                terminal = (
                    selected
                    / "fault_campaign"
                    / "artifacts"
                    / args.task
                    / f"seed-{seed:010d}"
                    / F1
                    / "severity-5"
                    / "terminal_result.json"
                )
                if not terminal.is_file():
                    candidates = list(
                        (selected / "fault_campaign/artifacts").glob(
                            "**/terminal_result.json"
                        )
                    )
                    terminal = candidates[0] if len(candidates) == 1 else terminal
                if terminal.is_file():
                    value = json.loads(terminal.read_text(encoding="utf-8"))
                    if valid_terminal(value):
                        result = {
                            **value,
                            "seed": seed,
                            "condition": F1,
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
            "model": "n0_twam",
            "task": args.task,
            "condition": F1,
            "planned_count": len(seeds),
            "completed_count": len(results),
            "eligible_count": len(results),
            "success_count": successes,
            "success_rate": successes / len(results),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "model-root",
        "code",
        "package",
        "config",
        "isaac-python",
        "n0-python",
        "n0-source",
        "digest-cache",
        "campaign",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rest-reference", type=Path)
    parser.add_argument("--calibrate-rest", action="store_true")
    parser.add_argument("--calibration-seed", type=int, default=9_100_000)
    parser.add_argument("--episode-timeout-s", type=float, default=7200)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
