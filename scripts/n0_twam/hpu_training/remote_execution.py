"""SSH transport for one no-clobber official N0-TWAM launch."""

from __future__ import annotations

import base64
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import IO, Sequence

from .contract import ClusterSpec, json_bytes
from .launch_profiles import LaunchProfile

SCRIPT_DIR = Path(__file__).resolve().parent
DOCKER_RUNTIME = SCRIPT_DIR / "docker_runtime.sh"
RANK_ENTRYPOINT = SCRIPT_DIR / "rank_entrypoint.sh"
FSDP_NAMESPACE_SHIM = SCRIPT_DIR / "runtime_shims" / "sitecustomize.py"
N0_TRAIN_COMPAT = SCRIPT_DIR / "runtime_shims" / "n0_train_compat.py"
FLEX25_COMPAT = SCRIPT_DIR / "runtime_shims" / "flex25_compat.py"
SSH_OPTIONS = (
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "ConnectTimeout=15",
    "ServerAliveInterval=30",
    "ServerAliveCountMax=4",
)


def encode_payload(
    *,
    mode: str,
    rank: int,
    cluster: ClusterSpec,
    profile: LaunchProfile,
    repo: Path,
    save_root: Path,
    latent_inventory_path: Path,
    latent_inventory_sha256: str,
    config_sha256: str,
    run_id: str,
) -> str:
    values: dict[str, object] = {
        "mode": mode,
        "launch_mode": profile.mode,
        "rank": rank,
        "nodes": list(profile.nodes),
        "processes_per_node": profile.processes_per_node,
        "expected_num_steps": profile.num_steps,
        "visible_devices": profile.visible_devices,
        "fsdp_topology": profile.fsdp_topology,
        "fsdp_shard_size": profile.fsdp_shard_size,
        "master_addr": cluster.master_addr,
        "master_port": cluster.master_port,
        "repo": str(repo),
        "save_root": str(save_root),
        "latent_inventory_path": str(latent_inventory_path),
        "latent_inventory_sha256": latent_inventory_sha256,
        "config_sha256": config_sha256,
        "container_name": (
            f"robotactile-n0-{run_id}-{profile.mode}-{mode}-rank-{rank:02d}"
        ),
        "container_entrypoint": str(RANK_ENTRYPOINT),
        "entrypoint_kind": "bash",
    }
    return base64.b64encode(json_bytes(values)).decode("ascii")


def ssh_command(cluster: ClusterSpec, node: str, payload: str) -> list[str]:
    command = ["ssh", "-p", str(cluster.ssh_port)]
    for option in SSH_OPTIONS:
        command.extend(("-o", option))
    command.extend((f"{cluster.ssh_user}@{node}", "bash", "-s", "--", payload))
    return command


def _preflight_node(
    *,
    cluster: ClusterSpec,
    profile: LaunchProfile,
    node: str,
    rank: int,
    repo: Path,
    save_root: Path,
    latent_inventory_path: Path,
    latent_inventory_sha256: str,
    config_sha256: str,
    run_id: str,
    script: bytes,
) -> tuple[str, int, str]:
    payload = encode_payload(
        mode="preflight",
        rank=rank,
        cluster=cluster,
        profile=profile,
        repo=repo,
        save_root=save_root,
        latent_inventory_path=latent_inventory_path,
        latent_inventory_sha256=latent_inventory_sha256,
        config_sha256=config_sha256,
        run_id=run_id,
    )
    result = subprocess.run(
        ssh_command(cluster, node, payload),
        input=script,
        capture_output=True,
        timeout=180,
    )
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    return node, result.returncode, output


def preflight_all(
    *,
    cluster: ClusterSpec,
    profile: LaunchProfile,
    repo: Path,
    save_root: Path,
    latent_inventory_path: Path,
    latent_inventory_sha256: str,
    config_sha256: str,
    log_dir: Path,
    run_id: str,
) -> None:
    script = DOCKER_RUNTIME.read_bytes()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(profile.nodes)) as executor:
        futures = {
            executor.submit(
                _preflight_node,
                cluster=cluster,
                profile=profile,
                node=node,
                rank=rank,
                repo=repo,
                save_root=save_root,
                latent_inventory_path=latent_inventory_path,
                latent_inventory_sha256=latent_inventory_sha256,
                config_sha256=config_sha256,
                run_id=run_id,
                script=script,
            ): (rank, node)
            for rank, node in enumerate(profile.nodes)
        }
        for future in as_completed(futures):
            rank, node = futures[future]
            checked_node, returncode, output = future.result()
            if checked_node != node:
                raise RuntimeError("internal preflight node mismatch")
            (log_dir / f"preflight-rank-{rank:02d}-{node}.log").write_text(output)
            if returncode != 0:
                failures.append(f"{node}: exit={returncode}")
    if failures:
        raise RuntimeError("remote preflight failed: " + ", ".join(sorted(failures)))


def _terminate_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
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
    *,
    cluster: ClusterSpec,
    profile: LaunchProfile,
    repo: Path,
    save_root: Path,
    latent_inventory_path: Path,
    latent_inventory_sha256: str,
    config_sha256: str,
    log_dir: Path,
    run_id: str,
) -> list[int]:
    script = DOCKER_RUNTIME.read_bytes()
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[IO[bytes]] = []
    try:
        for rank, node in enumerate(profile.nodes):
            payload = encode_payload(
                mode="train",
                rank=rank,
                cluster=cluster,
                profile=profile,
                repo=repo,
                save_root=save_root,
                latent_inventory_path=latent_inventory_path,
                latent_inventory_sha256=latent_inventory_sha256,
                config_sha256=config_sha256,
                run_id=run_id,
            )
            log_handle: IO[bytes] = (
                log_dir / f"train-rank-{rank:02d}-{node}.log"
            ).open("wb")
            handles.append(log_handle)
            process = subprocess.Popen(
                ssh_command(cluster, node, payload),
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            assert process.stdin is not None
            process.stdin.write(script)
            process.stdin.close()

        while True:
            returncodes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in returncodes):
                _terminate_processes(processes)
                return [process.wait() for process in processes]
            if all(code == 0 for code in returncodes):
                return [0] * len(processes)
            time.sleep(2.0)
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        for open_handle in handles:
            open_handle.close()
