"""Concurrent SSH transport for one no-clobber Dream-Tac HCU launch."""

from __future__ import annotations

import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import IO, Sequence

from .distributed_request import DistributedTrainingRequest
from .distributed_runtime import build_rank_payload
from .optimizer_step_request import HcuOptimizerStepRequest

SCRIPT_DIR = Path(__file__).resolve().parent
CONTAINER_RUNTIME = SCRIPT_DIR / "distributed_container_runtime.sh"
RANK_ENTRYPOINT = SCRIPT_DIR / "distributed_rank_entrypoint.sh"
SSH_OPTIONS = (
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "ConnectTimeout=15",
    "ServerAliveInterval=30",
    "ServerAliveCountMax=4",
)


def ssh_command(
    request: DistributedTrainingRequest, node: str, payload: str
) -> tuple[str, ...]:
    command = ["ssh", "-p", str(request.ssh_port)]
    for option in SSH_OPTIONS:
        command.extend(("-o", option))
    command.extend((f"{request.ssh_user}@{node}", "bash", "-s", "--", payload))
    return tuple(command)


def _preflight_node(
    request: DistributedTrainingRequest,
    source: HcuOptimizerStepRequest,
    *,
    node: str,
    rank: int,
    script: bytes,
) -> tuple[int, int, bytes]:
    payload = build_rank_payload(request, source, node_rank=rank, mode="preflight")
    result = subprocess.run(
        ssh_command(request, node, payload),
        input=script,
        capture_output=True,
        check=False,
        timeout=300,
    )
    return rank, result.returncode, result.stdout + result.stderr


def preflight_all(
    request: DistributedTrainingRequest,
    source: HcuOptimizerStepRequest,
    *,
    log_dir: Path,
) -> dict[str, int]:
    """Run the exact container/runtime gate concurrently on both nodes."""

    script = CONTAINER_RUNTIME.read_bytes()
    returncodes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _preflight_node,
                request,
                source,
                node=node,
                rank=rank,
                script=script,
            ): (rank, node)
            for rank, node in enumerate(request.nodes)
        }
        for future in as_completed(futures):
            expected_rank, node = futures[future]
            rank, returncode, output = future.result()
            if rank != expected_rank:
                raise RuntimeError("internal node-rank mismatch")
            log_path = log_dir / f"preflight-node-{rank:02d}-{node}.log"
            with log_path.open("xb") as stream:
                stream.write(output)
            returncodes[node] = returncode
    failures = {node: code for node, code in returncodes.items() if code != 0}
    if failures:
        raise RuntimeError(f"remote Dream-Tac preflight failed: {failures}")
    return returncodes


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


def launch_all(
    request: DistributedTrainingRequest,
    source: HcuOptimizerStepRequest,
    *,
    log_dir: Path,
) -> dict[str, int]:
    """Start both torchrun node ranks concurrently and wait as one job."""

    script = CONTAINER_RUNTIME.read_bytes()
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[IO[bytes]] = []
    try:
        for rank, node in enumerate(request.nodes):
            payload = build_rank_payload(request, source, node_rank=rank, mode="train")
            handle: IO[bytes] = (log_dir / f"train-node-{rank:02d}-{node}.log").open(
                "xb"
            )
            handles.append(handle)
            process = subprocess.Popen(
                ssh_command(request, node, payload),
                stdin=subprocess.PIPE,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            assert process.stdin is not None
            process.stdin.write(script)
            process.stdin.close()
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                _terminate(processes)
                break
            if all(code == 0 for code in codes):
                break
            time.sleep(2.0)
        final = [process.wait() for process in processes]
        return {node: final[rank] for rank, node in enumerate(request.nodes)}
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


__all__ = [
    "CONTAINER_RUNTIME",
    "RANK_ENTRYPOINT",
    "launch_all",
    "preflight_all",
    "ssh_command",
]
