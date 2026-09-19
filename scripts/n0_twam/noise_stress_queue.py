"""Own one N0 server and launch fresh Isaac processes for frozen stress seeds."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.n0_fault_campaign.io import file_sha256
from robotactile_benchmark.n0_fault_campaign.stress_group import (
    prepare_stress_group,
    read_stress_rows,
    write_once,
)
from robotactile_benchmark.n0_fault_campaign.stress_metrics import summarize_stress
from robotactile_benchmark.n0_fault_campaign.stress_protocol import validate_protocol
from robotactile_benchmark.n0_fault_campaign.stress_provenance import (
    verify_runtime_code,
)
from scripts.n0_twam.run_official_early_fault_seed import (
    official_server_command,
    official_server_environment,
)
from scripts.retrained_evaluation.released_clean_seed_queue import (
    build_request,
    stop_owned,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def shard_seeds(plan: dict[str, Any], selected: list[int] | None) -> list[int]:
    """Restrict execution without changing the frozen statistical cohort."""
    if selected is None:
        return list(plan["seeds"])
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(seed) is not int or seed not in plan["seeds"] for seed in selected)
    ):
        raise ValueError("shard seeds must be unique members of the frozen protocol")
    wanted = set(selected)
    return [seed for seed in plan["seeds"] if seed in wanted]


@contextmanager
def gpu_lock(path: Path):
    """Cooperative lock at an explicit shared path; never remove the lock inode."""
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("GPU lock must be an absolute nonsymlink shared path")
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_port(port: int) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def validate_runtime_binding(
    args: argparse.Namespace, plan: dict[str, Any]
) -> dict[str, Any]:
    """Validate frozen config, model, source code and imported package content."""
    verify_runtime_code(args.code, args.package, plan["binding"]["code_sha256"])
    if file_sha256(args.config) != plan["binding"]["integration_config_sha256"]:
        raise ValueError("integration config differs from frozen protocol")
    runtime = resolve_n0_runtime_artifacts(args.config)
    model_root = args.model_root / "artifacts/models/n0_twam"
    expected_paths = {
        "serve_pool_root": model_root / "serve-pools/lift_bottle",
        "serve_bundle_root": model_root / "serve-bundle",
        "serve_bundle_manifest_path": model_root
        / "serve-pools/lift_bottle/serve_bundle_manifest.json",
    }
    for field, expected in expected_paths.items():
        if Path(getattr(runtime.manifest, field)).resolve() != expected.resolve():
            raise ValueError(f"official server {field} differs from integration config")
    if (
        runtime.manifest.task_id != "lift_bottle"
        or runtime.manifest.checkpoint_sha256 != plan["binding"]["model_sha256"]
    ):
        raise ValueError("runtime task/checkpoint differs from frozen protocol")
    return {
        "code_sha256": plan["binding"]["code_sha256"],
        "config_sha256": file_sha256(args.config),
        "artifact_manifest_sha256": file_sha256(runtime.manifest_path),
        "checkpoint_sha256": runtime.manifest.checkpoint_sha256,
    }


def server_command(args: argparse.Namespace, plan: dict[str, Any]) -> list[str]:
    command = official_server_command(
        n0_python=args.n0_python,
        repo=args.code,
        port=args.port,
        save_root=args.campaign / "server/dumps",
    )
    if plan["stage"] in {"screening", "calibration"}:
        command += [
            "--diagnostic-input-trace-root",
            str(args.campaign / "server/input-traces"),
            "--diagnostic-input-trace-limit",
            "32",
        ]
    return command


def worker_command(args: argparse.Namespace, group: Path) -> list[str]:
    return [
        str(args.isaac_python),
        str(args.code / "scripts/n0_twam/run_noise_stress.py"),
        "run",
        "--group",
        str(group),
        "--integration-config",
        str(args.config),
        "--n0-source-root",
        str(args.n0_source),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]


def wait_server(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("owned N0 server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("owned N0 server startup timed out")


def launch_file_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Hash real executable files while preserving venv launcher paths in argv."""
    isaac_target = args.isaac_python.resolve(strict=True)
    n0_target = args.n0_python.resolve(strict=True)
    return {
        "isaac_python_resolved": str(isaac_target),
        "n0_python_resolved": str(n0_target),
        "isaac_launcher_sha256": file_sha256(isaac_target),
        "n0_launcher_sha256": file_sha256(n0_target),
        "queue_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "worker_sha256": file_sha256(args.code / "scripts/n0_twam/run_noise_stress.py"),
        "n0_server_sha256": file_sha256(args.n0_source / "n0_twam/n0_twam_server.py"),
    }


def run(args: argparse.Namespace) -> None:
    if args.max_attempts != 1:
        raise ValueError(
            "this queue only permits one frozen attempt; no automatic reruns"
        )
    if (
        args.gpu_id < 0
        or not 1 <= args.port <= 65535
        or args.episode_timeout_s <= 0
        or args.startup_timeout_s <= 0
    ):
        raise ValueError("invalid GPU, port, or timeout")
    for name in (
        "protocol",
        "root",
        "model_root",
        "code",
        "package",
        "config",
        "n0_source",
        "digest_cache",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    for name in ("isaac_python", "n0_python"):
        launcher = Path(getattr(args, name)).absolute()
        if not launcher.is_file():
            raise FileNotFoundError(f"runtime launcher unavailable: {launcher}")
        # Preserve venv launcher symlinks: resolving them can switch interpreter
        # environments despite pointing at the same underlying executable.
        setattr(args, name, launcher)
    args.campaign = Path(args.campaign).absolute()
    args.gpu_lock = Path(args.gpu_lock).absolute()
    args.model, args.task = "n0_twam", "lift_bottle"
    plan = validate_protocol(read_json(args.protocol))
    selected_seeds = shard_seeds(plan, getattr(args, "seeds", None))
    args.dataset_sha256 = plan["binding"]["dataset_sha256"]
    provenance = validate_runtime_binding(args, plan)
    # Fail before creating a campaign if any launch provenance cannot be read.
    launch_provenance = launch_file_provenance(args)
    spatial = read_json(args.spatial_calibration) if args.spatial_calibration else None
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
    server = worker = None
    rows: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    handlers = {}

    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(f"queue interrupted by signal {signum}")

    with gpu_lock(args.gpu_lock):
        check_port(args.port)
        args.campaign.mkdir(parents=True, exist_ok=False)
        write_once(args.campaign / "protocol.json", plan)
        write_once(
            args.campaign / "launch.json",
            {
                "schema": "n0_noise_stress_queue_v1",
                "protocol_sha256": plan["protocol_sha256"],
                "gpu_id": args.gpu_id,
                "gpu_lock": str(args.gpu_lock),
                "host": socket.gethostname(),
                "code": str(args.code),
                "package": str(args.package),
                "n0_source": str(args.n0_source),
                "isaac_python": str(args.isaac_python),
                "n0_python": str(args.n0_python),
                "queue_python_version": sys.version,
                **launch_provenance,
                "max_attempts": 1,
                "fresh_isaac_per_seed": True,
                "persistent_n0_server": True,
                "shard_seeds": selected_seeds,
                "shard_expected_seed_groups": len(selected_seeds),
                **provenance,
            },
        )
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupted)
            selected = args.campaign / "server"
            selected.mkdir()
            server_env = official_server_environment(
                root=args.model_root,
                task=args.task,
                save_root=selected / "dumps",
                inherited=env,
            )
            server_env["PYTHONPATH"] = env["PYTHONPATH"]
            command = server_command(args, plan)
            with (selected / "server.log").open("xb") as log:
                server = subprocess.Popen(
                    command,
                    cwd=args.code,
                    env=server_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            write_once(selected / "launch.json", {"pid": server.pid, "argv": command})
            wait_server(server, args.port, args.startup_timeout_s)
            for seed in selected_seeds:
                if server.poll() is not None:
                    raise RuntimeError("owned server exited before next seed")
                attempt = args.campaign / "seeds" / f"seed-{seed:07d}" / "attempt-1"
                source = attempt / "source"
                source.mkdir(parents=True, exist_ok=False)
                base = build_request(args, seed, source)
                clean_path = source / "base_clean_request.json"
                write_once(clean_path, live_univtac_request_to_dict(base))
                group = attempt / "group"
                prepare_stress_group(
                    group,
                    protocol=plan,
                    clean_request=clean_path,
                    rest_reference=args.rest_reference,
                    spatial_calibration=spatial,
                )
                command = worker_command(args, group)
                with (attempt / "worker.log").open("xb") as log:
                    worker = subprocess.Popen(
                        command,
                        cwd=args.code,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    write_once(
                        attempt / "launch.json", {"pid": worker.pid, "argv": command}
                    )
                    try:
                        code = worker.wait(
                            timeout=args.episode_timeout_s * (1 + len(plan["variants"]))
                            + 900
                        )
                    except subprocess.TimeoutExpired:
                        stop_owned(worker)
                        code = 124
                exit_row = {
                    "seed": seed,
                    "returncode": code,
                    "pre_close_group_result_present": (
                        group / "group_result.json"
                    ).is_file(),
                }
                exits.append(exit_row)
                write_once(attempt / "worker_exit.json", exit_row)
                try:
                    current = read_stress_rows(group)
                except Exception as exc:
                    write_once(
                        attempt / "failure.json",
                        {
                            **exit_row,
                            "error_type": type(exc).__name__,
                            "automatic_retry": False,
                        },
                    )
                    raise
                rows.extend(current)
                accepted = (
                    bool(current)
                    and len(current) == 1 + len(plan["variants"])
                    and all(row["group_accepted"] is True for row in current)
                )
                write_once(
                    attempt.parent / "status.json",
                    {
                        **exit_row,
                        "group_accepted": accepted,
                        "verified_row_count": len(current),
                    },
                )
                if code != 0 or not accepted:
                    write_once(
                        attempt / "failure.json",
                        {
                            **exit_row,
                            "group_accepted": accepted,
                            "automatic_retry": False,
                        },
                    )
                    raise RuntimeError(
                        f"seed {seed} stopped queue; inspect immutable attempt artifacts"
                    )
        except BaseException as exc:
            write_once(
                args.campaign / "failure.json",
                {"error_type": type(exc).__name__, "automatic_retry": False},
            )
            raise
        finally:
            try:
                stop_owned(worker)
            finally:
                stop_owned(server)
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
            summary = summarize_stress(plan, rows)
            summary["worker_exits"] = exits
            summary["shard_seeds"] = selected_seeds
            summary["shard_expected_seed_groups"] = len(selected_seeds)
            summary["shard_finished_seed_groups"] = sum(
                all(
                    any(
                        row.get("seed") == seed
                        and row.get("condition") == label
                        and row.get("group_accepted") is True
                        for row in rows
                    )
                    for label in ["clean", *(v["label"] for v in plan["variants"])]
                )
                for seed in selected_seeds
            )
            summary["shard_execution_complete"] = (
                summary["shard_finished_seed_groups"] == len(selected_seeds)
                and len(exits) == len(selected_seeds)
                and all(row["returncode"] == 0 for row in exits)
            )
            summary["abnormal_worker_exit_count"] = sum(
                row["returncode"] != 0 for row in exits
            )
            write_once(args.campaign / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
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
        "gpu-lock",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rest-reference", type=Path)
    parser.add_argument("--spatial-calibration", type=Path)
    parser.add_argument("--episode-timeout-s", type=float, default=7200)
    parser.add_argument("--startup-timeout-s", type=float, default=1800)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
