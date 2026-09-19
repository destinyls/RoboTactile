#!/usr/bin/env python3
"""Launch one persistent, source-bound N0-VTLA mixed8 4x8 HCU run."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import IO, Final, Mapping, Sequence

PROTOCOL: Final[str] = "robotactile.n0_vtla.formal_mixed8_4node.v1"
SUPERVISOR_PROTOCOL: Final[str] = (
    "robotactile.n0_vtla.formal_mixed8_4node_supervisor.v1"
)
NNODES: Final[int] = 4
NPROC_PER_NODE: Final[int] = 8
WORLD_SIZE: Final[int] = 32
NUM_TRAIN_STEPS: Final[int] = 160_000
GLOBAL_BATCH_SIZE: Final[int] = 64
MICRO_BATCH_PER_RANK: Final[int] = 1
GRADIENT_ACCUMULATION_STEPS: Final[int] = 2
SAVE_INTERVAL: Final[int] = 2_000
BASE_SHA256: Final[str] = (
    "4aafb1da22a2671e637884d175ddc1569344ac5fd2dde8b53f3a34720533051c"
)
SSH_OPTIONS: Final[tuple[str, ...]] = (
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "ConnectTimeout=15",
    "ServerAliveInterval=30",
    "ServerAliveCountMax=4",
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}")
_SSH_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,31}")
_SOURCE_NAMES: Final[tuple[str, ...]] = (
    "distributed_container_runtime",
    "distributed_rank_entrypoint",
    "distributed_collective_probe",
    "gradient_accumulation_patch",
    "hcu_train_entry",
    "launch_formal_mixed8_4node",
    "multitask_sampler",
    "rank_entrypoint",
    "runtime_patches",
    "runtime_shims_sitecustomize",
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: object) -> str:
    data = _json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _write_text_new(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = _json_bytes(payload)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shared_root(value: Path) -> Path:
    path = value.absolute()
    if path == Path("/mnt/data") or Path("/mnt/data") not in path.parents:
        raise ValueError("project root must be a child of /mnt/data")
    return path


def _ipv4(value: str, label: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an IPv4 address") from exc
    if address.version != 4:
        raise ValueError(f"{label} must be an IPv4 address")
    return str(address)


@dataclass(frozen=True)
class LaunchRequest:
    """Immutable topology and source identity for one fresh-base launch."""

    project_root: Path
    run_id: str
    nodes: tuple[str, str, str, str]
    ssh_user: str
    ssh_port: int
    master_addr: str
    master_port: int
    source_bindings: Mapping[str, str]

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        run_id: str,
        nodes: Sequence[str],
        ssh_user: str,
        ssh_port: int,
        master_addr: str,
        master_port: int,
        source_bindings: Mapping[str, str],
    ) -> LaunchRequest:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run ID is invalid")
        if len(nodes) != NNODES:
            raise ValueError("nodes must contain exactly four IPv4 addresses")
        parsed = tuple(_ipv4(node, "node") for node in nodes)
        if len(set(parsed)) != NNODES:
            raise ValueError("nodes must contain exactly four unique addresses")
        master = _ipv4(master_addr, "master_addr")
        if master != parsed[0]:
            raise ValueError("master_addr must exactly equal nodes[0]")
        if _SSH_USER.fullmatch(ssh_user) is None:
            raise ValueError("SSH user is invalid")
        if not 1 <= ssh_port <= 65535:
            raise ValueError("SSH port is invalid")
        if not 1024 <= master_port <= 65535:
            raise ValueError("master port must be in [1024, 65535]")
        if set(source_bindings) != set(_SOURCE_NAMES) or any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in source_bindings.values()
        ):
            raise ValueError("source bindings are incomplete or invalid")
        return cls(
            project_root=_shared_root(project_root),
            run_id=run_id,
            nodes=(parsed[0], parsed[1], parsed[2], parsed[3]),
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            master_addr=master,
            master_port=master_port,
            source_bindings=dict(source_bindings),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "base_checkpoint": {
                "path": str(self.project_root / "artifacts/base/n0-vtla-base-ec12548"),
                "sha256": BASE_SHA256,
            },
            "dataset_split": "train759",
            "fresh_base": True,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "micro_batch_per_rank": MICRO_BATCH_PER_RANK,
            "nnodes": NNODES,
            "nodes": list(self.nodes),
            "nproc_per_node": NPROC_PER_NODE,
            "num_train_steps": NUM_TRAIN_STEPS,
            "project_root": str(self.project_root),
            "protocol_id": PROTOCOL,
            "resume_checkpoint": None,
            "run_id": self.run_id,
            "save_interval": SAVE_INTERVAL,
            "scope": "mixed8",
            "source_bindings": dict(self.source_bindings),
            "ssh_port": self.ssh_port,
            "ssh_user": self.ssh_user,
            "task_balanced": True,
            "validation_data_used": False,
            "world_size": WORLD_SIZE,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LaunchRequest:
        expected = {
            "protocol_id": PROTOCOL,
            "scope": "mixed8",
            "dataset_split": "train759",
            "validation_data_used": False,
            "fresh_base": True,
            "resume_checkpoint": None,
            "task_balanced": True,
            "nnodes": NNODES,
            "nproc_per_node": NPROC_PER_NODE,
            "world_size": WORLD_SIZE,
            "num_train_steps": NUM_TRAIN_STEPS,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "micro_batch_per_rank": MICRO_BATCH_PER_RANK,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "save_interval": SAVE_INTERVAL,
        }
        for name, required in expected.items():
            if value.get(name) != required:
                raise ValueError(f"launch request field {name} must be {required!r}")
        nodes = value.get("nodes")
        bindings = value.get("source_bindings")
        if not isinstance(nodes, list) or not isinstance(bindings, dict):
            raise ValueError("launch request nodes/source bindings are invalid")
        base = value.get("base_checkpoint")
        if not isinstance(base, dict) or base.get("sha256") != BASE_SHA256:
            raise ValueError("launch request base checkpoint is not pinned")
        request = cls.create(
            project_root=Path(str(value.get("project_root"))),
            run_id=str(value.get("run_id")),
            nodes=[str(node) for node in nodes],
            ssh_user=str(value.get("ssh_user")),
            ssh_port=int(str(value.get("ssh_port"))),
            master_addr=str(value.get("master_addr")),
            master_port=int(str(value.get("master_port"))),
            source_bindings={str(key): str(item) for key, item in bindings.items()},
        )
        if base != request.to_dict()["base_checkpoint"]:
            raise ValueError("launch request base checkpoint path mismatch")
        allowed_fields = set(request.to_dict()) | {"started_at_utc"}
        if set(value) != allowed_fields:
            raise ValueError("launch request fields mismatch")
        return request


def _source_paths(project_root: Path) -> dict[str, Path]:
    tooling = project_root / "tooling/hpu_training"
    return {
        "distributed_container_runtime": tooling / "distributed_container_runtime.sh",
        "distributed_rank_entrypoint": tooling / "distributed_rank_entrypoint.sh",
        "distributed_collective_probe": tooling / "distributed_collective_probe.py",
        "gradient_accumulation_patch": tooling / "gradient_accumulation_patch.py",
        "hcu_train_entry": tooling / "hcu_train_entry.py",
        "launch_formal_mixed8_4node": tooling / "launch_formal_mixed8_4node.py",
        "multitask_sampler": tooling / "multitask_sampler.py",
        "rank_entrypoint": tooling / "rank_entrypoint.sh",
        "runtime_patches": tooling / "runtime_patches.py",
        "runtime_shims_sitecustomize": tooling / "runtime_shims/sitecustomize.py",
    }


def _verify_sources(request: LaunchRequest) -> dict[str, Path]:
    paths = _source_paths(request.project_root)
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source-bound file is missing or unsafe: {path}")
        if _file_sha256(path) != request.source_bindings[name]:
            raise ValueError(f"source-bound file changed: {path}")
    return paths


def _encode_payload(
    request: LaunchRequest,
    *,
    mode: str,
    node_rank: int,
    request_sha256: str,
) -> str:
    if mode not in {"preflight", "train"} or not 0 <= node_rank < NNODES:
        raise ValueError("invalid remote phase or node rank")
    paths = _source_paths(request.project_root)
    payload = {
        "collective_probe": {
            "path": str(paths["distributed_collective_probe"]),
            "sha256": request.source_bindings["distributed_collective_probe"],
        },
        "container_name": (
            f"robotactile-n0-vtla-4node-{request.run_id}-{mode}-node-{node_rank:02d}"
        ),
        "distributed_entrypoint": {
            "path": str(paths["distributed_rank_entrypoint"]),
            "sha256": request.source_bindings["distributed_rank_entrypoint"],
        },
        "fresh_base": True,
        "launch_request": {
            "path": str(
                request.project_root
                / "logs/supervisor"
                / request.run_id
                / "request.json"
            ),
            "sha256": request_sha256,
        },
        "master_addr": request.master_addr,
        "master_port": request.master_port,
        "mode": mode,
        "nnodes": NNODES,
        "node_addr": request.nodes[node_rank],
        "node_rank": node_rank,
        "nproc_per_node": NPROC_PER_NODE,
        "project_root": str(request.project_root),
        "rank_entrypoint": {
            "path": str(paths["rank_entrypoint"]),
            "sha256": request.source_bindings["rank_entrypoint"],
        },
        "run_id": request.run_id,
        "world_size": WORLD_SIZE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(encoded).decode("ascii")


def _ssh_command(request: LaunchRequest, node: str, payload: str) -> list[str]:
    command = ["ssh", "-p", str(request.ssh_port)]
    for option in SSH_OPTIONS:
        command.extend(("-o", option))
    command.extend((f"{request.ssh_user}@{node}", "bash", "-s", "--", payload))
    return command


def _terminate(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and any(
        process.poll() is None for process in processes
    ):
        time.sleep(0.25)
    for process in processes:
        if process.poll() is None:
            process.kill()


def _run_remote_phase(
    request: LaunchRequest,
    *,
    mode: str,
    request_sha256: str,
    log_dir: Path,
    timeout_seconds: float | None,
) -> dict[str, int]:
    runtime = _verify_sources(request)["distributed_container_runtime"].read_bytes()
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[IO[bytes]] = []
    started = time.monotonic()
    try:
        for node_rank, node in enumerate(request.nodes):
            payload = _encode_payload(
                request,
                mode=mode,
                node_rank=node_rank,
                request_sha256=request_sha256,
            )
            handle: IO[bytes] = (
                log_dir / f"{mode}-node-{node_rank:02d}-{node}.log"
            ).open("xb")
            handles.append(handle)
            process = subprocess.Popen(
                _ssh_command(request, node, payload),
                stdin=subprocess.PIPE,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            assert process.stdin is not None
            process.stdin.write(runtime)
            process.stdin.close()
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                _terminate(processes)
                break
            if all(code == 0 for code in codes):
                break
            if (
                timeout_seconds is not None
                and time.monotonic() - started > timeout_seconds
            ):
                _terminate(processes)
                break
            time.sleep(1.0)
        return {
            node: processes[index].wait() for index, node in enumerate(request.nodes)
        }
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def _state(
    *,
    status: str,
    phase: str,
    exit_code: int,
    started_at: str,
    returncodes: Mapping[str, int] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "exit_code": exit_code,
        "finished_at_utc": _utc_now() if status in {"completed", "failed"} else "",
        "phase": phase,
        "protocol_id": SUPERVISOR_PROTOCOL,
        "started_at_utc": started_at,
        "status": status,
    }
    if returncodes is not None:
        result["node_returncodes"] = dict(returncodes)
    if error is not None:
        result["error"] = error
    return result


def _worker(request_path: Path, expected_sha256: str) -> int:
    supervisor_dir = request_path.parent
    state_path = supervisor_dir / "state.json"
    started_at = _utc_now()

    def stop_worker(signum: int, _frame: FrameType | None) -> None:
        raise InterruptedError(f"supervisor received signal {signum}")

    signal.signal(signal.SIGINT, stop_worker)
    signal.signal(signal.SIGTERM, stop_worker)
    try:
        if _file_sha256(request_path) != expected_sha256:
            raise ValueError("launch request SHA256 mismatch")
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("launch request must be a JSON object")
        request = LaunchRequest.from_dict(payload)
        _verify_sources(request)
        started_at = str(payload["started_at_utc"])
        _write_text_new(supervisor_dir / "worker.pid", f"{os.getpid()}\n")
        _replace_json(
            state_path,
            _state(
                status="running",
                phase="preflight",
                exit_code=-1,
                started_at=started_at,
            ),
        )
        preflight = _run_remote_phase(
            request,
            mode="preflight",
            request_sha256=expected_sha256,
            log_dir=supervisor_dir,
            timeout_seconds=900.0,
        )
        if any(code != 0 for code in preflight.values()):
            raise RuntimeError(f"32-rank preflight failed: {preflight}")
        _replace_json(
            state_path,
            _state(
                status="running",
                phase="train",
                exit_code=-1,
                started_at=started_at,
                returncodes=preflight,
            ),
        )
        train = _run_remote_phase(
            request,
            mode="train",
            request_sha256=expected_sha256,
            log_dir=supervisor_dir,
            timeout_seconds=None,
        )
        succeeded = all(code == 0 for code in train.values())
        _replace_json(
            state_path,
            _state(
                status="completed" if succeeded else "failed",
                phase="finished",
                exit_code=0 if succeeded else 1,
                started_at=started_at,
                returncodes=train,
                error=None if succeeded else f"training node failure: {train}",
            ),
        )
        return 0 if succeeded else 1
    except BaseException as exc:
        _replace_json(
            state_path,
            _state(
                status="failed",
                phase="failed",
                exit_code=1,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nodes", nargs=4, required=True)
    parser.add_argument("--ssh-user", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "__worker":
        if len(values) != 3:
            raise ValueError("worker expects REQUEST_PATH REQUEST_SHA256")
        return _worker(Path(values[1]), values[2])
    args = _parse_args(values)
    project_root = _shared_root(args.project_root)
    paths = _source_paths(project_root)
    bindings = {name: _file_sha256(path) for name, path in paths.items()}
    request = LaunchRequest.create(
        project_root=project_root,
        run_id=args.run_id,
        nodes=args.nodes,
        ssh_user=args.ssh_user,
        ssh_port=args.ssh_port,
        master_addr=args.master_addr,
        master_port=args.master_port,
        source_bindings=bindings,
    )
    _verify_sources(request)
    supervisor_dir = project_root / "logs/supervisor" / request.run_id
    supervisor_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    request_document = {**request.to_dict(), "started_at_utc": started_at}
    request_path = supervisor_dir / "request.json"
    request_sha256 = _write_new(request_path, request_document)
    _write_new(
        supervisor_dir / "state.json",
        _state(
            status="running",
            phase="launching",
            exit_code=-1,
            started_at=started_at,
        ),
    )
    log_path = supervisor_dir / "launcher.log"
    with log_path.open("xb") as launcher_log:
        worker = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "__worker",
                str(request_path),
                request_sha256,
            ],
            stdin=subprocess.DEVNULL,
            stdout=launcher_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    launch = {
        **request_document,
        "background_persistent": True,
        "launch_request_path": str(request_path),
        "launch_request_sha256": request_sha256,
        "launcher_pid": worker.pid,
        "no_clobber": True,
        "no_system_modifications": True,
        "per_node_logs": [
            f"preflight-node-{rank:02d}-{node}.log"
            for rank, node in enumerate(request.nodes)
        ]
        + [
            f"train-node-{rank:02d}-{node}.log"
            for rank, node in enumerate(request.nodes)
        ],
        "transient_containers": True,
    }
    _write_new(supervisor_dir / "launch.json", launch)
    _write_text_new(supervisor_dir / "launcher.pid", f"{worker.pid}\n")
    if worker.poll() not in (None, 0):
        raise RuntimeError("4-node supervisor exited during launch")
    sys.stdout.write(
        f"launched run_id={request.run_id} pid={worker.pid} supervisor={supervisor_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
