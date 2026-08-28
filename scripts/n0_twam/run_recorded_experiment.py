#!/usr/bin/env python3
"""Own one official N0 server and run one recorded robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--n0-port", type=int, default=29601)
    parser.add_argument("--master-port", type=int, default=29988)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--runner-timeout-s", type=float, default=3600.0)
    parser.add_argument("--episode-id", default="univtac-lift_bottle-clean-90")
    parser.add_argument("--anchor-index", type=int, default=186)
    parser.add_argument("--release-index", type=int, default=280)
    parser.add_argument("--rest-index", type=int, default=0)
    parser.add_argument("--fault-start-index", type=int, default=17)
    parser.add_argument("--initial-seed", type=int, default=90)
    parser.add_argument("--exogenous-seed", type=int, default=20260823)
    parser.add_argument("--operator", action="append", dest="operators")
    parser.add_argument("--severity", action="append", dest="severities", type=int)
    parser.add_argument("--receipt", type=Path)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, document: object) -> None:
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("recorded runner receipt cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("recorded runner receipt already differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _below(root: Path, path: Path, name: str, *, must_exist: bool) -> Path:
    selected = Path(path).absolute()
    try:
        resolved = selected.resolve(strict=must_exist)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(f"{name} must remain below deployment root") from error
    if selected.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    return selected


def _runtime_executable(path: Path, runtime_root: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(runtime_root.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(f"{name} escapes its repo-local runtime") from error
    if not resolved.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{name} is not executable")
    return path.absolute()


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _terminate(process: subprocess.Popen[bytes]) -> str:
    if process.poll() is not None:
        return "already_exited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        process.wait(timeout=30.0)
        return "sigterm"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)
        return "sigkill_after_timeout"


def _wait_for_server(
    process: subprocess.Popen[bytes], command: Sequence[str], timeout_s: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_error = "metadata probe has not run"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"N0 server exited before readiness: {return_code}")
        try:
            completed = subprocess.run(
                tuple(command),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30.0,
            )
        except subprocess.TimeoutExpired:
            last_error = "metadata probe timed out"
        else:
            output = completed.stdout.decode("utf-8", errors="replace").strip()
            if completed.returncode == 0:
                value = json.loads(output)
                if not isinstance(value, dict):
                    raise TypeError("N0 readiness output must be an object")
                return value
            last_error = output[-1000:]
        time.sleep(5.0)
    raise TimeoutError(f"N0 server readiness timed out: {last_error}")


def _validate_args(args: argparse.Namespace) -> None:
    if _IDENTIFIER.fullmatch(args.task) is None:
        raise ValueError("task contains unsupported characters")
    if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpus) is None:
        raise ValueError("gpus must be a comma-separated integer list")
    if not 1 <= args.n0_port <= 65535 or not 1 <= args.master_port <= 65535:
        raise ValueError("ports must be in [1,65535]")
    if args.server_ready_timeout_s <= 0 or args.runner_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    if args.severities is not None and any(
        item not in range(1, 6) for item in args.severities
    ):
        raise ValueError("severity levels must be in [1,5]")
    if any(
        item < 0
        for item in (
            args.anchor_index,
            args.release_index,
            args.rest_index,
            args.fault_start_index,
            args.initial_seed,
            args.exogenous_seed,
        )
    ):
        raise ValueError("indices and seeds must be non-negative")


def run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    """Run one server-owned experiment and always publish a lifecycle receipt."""

    _validate_args(args)
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    repository_root = Path(__file__).resolve().parents[2]
    hdf5_path = _below(layout.root, args.hdf5, "hdf5", must_exist=True)
    output = _below(layout.root, args.output, "output", must_exist=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError("recorded experiment output already exists")
    source_root = layout.sources / "N0-TWAM"
    integration_config = (
        layout.model_artifacts
        / "n0_twam"
        / "configs"
        / args.task
        / "integration_config.json"
    )
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("pinned N0 source is unavailable")
    if integration_config.is_symlink() or not integration_config.is_file():
        raise ValueError("N0 integration config is unavailable")
    isaac_python = _runtime_executable(
        layout.runtime / "isaac-sim-4.5.0/python.sh",
        layout.runtime / "isaac-sim-4.5.0",
        "Isaac Python",
    )
    if _port_open(args.n0_port):
        raise RuntimeError("N0 port is already occupied")
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = layout.logs / "recorded-n0" / args.task / f"{run_id}.log"
    receipt_path = args.receipt or (
        layout.outputs / "recorded-n0-runs" / args.task / f"{run_id}.json"
    )
    receipt_path = _below(layout.root, receipt_path, "receipt", must_exist=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server_command = (
        "bash",
        str(repository_root / "scripts/n0_twam/serve_univtac.sh"),
        "--root",
        str(layout.root),
        "--task",
        args.task,
        "--gpus",
        args.gpus,
        "--port",
        str(args.n0_port),
        "--master-port",
        str(args.master_port),
    )
    check_command = (
        str(isaac_python),
        str(repository_root / "scripts/n0_twam/check_server.py"),
        "--source-root",
        str(source_root),
        "--port",
        str(args.n0_port),
    )
    runner_command = [
        str(isaac_python),
        "-m",
        "robotactile_benchmark.cli",
        "recorded-n0",
        "--hdf5",
        str(hdf5_path),
        "--integration-config",
        str(integration_config),
        "--source-root",
        str(source_root),
        "--output",
        str(output),
        "--task",
        args.task,
        "--episode-id",
        args.episode_id,
        "--anchor-index",
        str(args.anchor_index),
        "--release-index",
        str(args.release_index),
        "--rest-index",
        str(args.rest_index),
        "--fault-start-index",
        str(args.fault_start_index),
        "--initial-seed",
        str(args.initial_seed),
        "--exogenous-seed",
        str(args.exogenous_seed),
        "--port",
        str(args.n0_port),
    ]
    for operator in args.operators or ():
        runner_command.extend(("--operator", operator))
    for severity in args.severities or ():
        runner_command.extend(("--severity", str(severity)))
    server: subprocess.Popen[bytes] | None = None
    readiness: dict[str, object] | None = None
    runner_return_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    shutdown = "not_started"
    with log_path.open("xb") as log_stream:
        try:
            server = subprocess.Popen(
                server_command,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            readiness = _wait_for_server(
                server, check_command, args.server_ready_timeout_s
            )
            completed = subprocess.run(
                runner_command,
                check=False,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                timeout=args.runner_timeout_s,
            )
            runner_return_code = completed.returncode
        except (
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
            TimeoutError,
            TypeError,
            ValueError,
        ) as error:
            error_type = type(error).__name__
            error_message = str(error)
            log_stream.write(
                f"\nROBOTACTILE_RECORDED_ERROR {error_type}: {error_message}\n".encode(
                    "utf-8", errors="replace"
                )
            )
            log_stream.flush()
        finally:
            if server is not None:
                shutdown = _terminate(server)
    finished = datetime.now(timezone.utc)
    document = {
        "artifact_path": str(output),
        "artifact_present": output.is_dir(),
        "artifact_root_receipt_present": (output / "root_receipt.json").is_file(),
        "error_message": error_message,
        "error_type": error_type,
        "finished_at_utc": finished.isoformat(),
        "gpus": args.gpus,
        "hdf5_sha256": _sha256_file(hdf5_path),
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
        "n0_port": args.n0_port,
        "readiness": readiness,
        "runner_return_code": runner_return_code,
        "schema_version": "robotactile-recorded-n0-runner-v1",
        "server_shutdown": shutdown,
        "started_at_utc": started.isoformat(),
        "task_id": args.task,
    }
    _write_once(receipt_path, document)
    summary = {**document, "receipt": str(receipt_path)}
    success = (
        runner_return_code == 0
        and error_type is None
        and document["artifact_root_receipt_present"] is True
    )
    return summary, 0 if success else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, exit_code = run(args)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
