#!/usr/bin/env python3
"""Run official N0-TWAM latent encoders on 2 nodes x 8 workers."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import IO, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hpu_training.contract import (  # noqa: E402
    OFFICIAL_COMMIT,
    PROVEN_DOCKER_IMAGE,
    PROVEN_DOCKER_IMAGE_ID,
    PROVEN_RCCL_PLUGIN_FILE,
    PROVEN_RCCL_PLUGIN_SHA256,
    RUNTIME_PYTHON_OVERLAYS,
    RUNTIME_PYTHON_PATH,
    ClusterSpec,
    load_cluster_spec,
    sha256_file,
    validate_run_id,
    verify_official_checkout,
)
from hpu_training.latent.contracts import (  # noqa: E402
    WorkUnit,
    build_work_units,
    canonical_sha256,
    load_certified_train759,
    load_latent_spec,
    verify_official_encoders,
)
from hpu_training.latent.inventory import aggregate_inventory  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "worker.py"
CONTAINER_ENTRYPOINT = SCRIPT_DIR / "container_worker_entrypoint.sh"
DOCKER_RUNTIME = SCRIPT_DIR.parent / "docker_runtime.sh"
SSH_OPTIONS = (
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "ConnectTimeout=15",
    "ServerAliveInterval=30",
    "ServerAliveCountMax=4",
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def _update_status(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _encoded_payload(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _ssh_command(cluster: ClusterSpec, node: str, payload: str) -> list[str]:
    command = ["ssh", "-p", str(cluster.ssh_port)]
    for option in SSH_OPTIONS:
        command.extend(("-o", option))
    command.extend(
        (
            f"{cluster.ssh_user}@{node}",
            "bash",
            "-s",
            "--",
            payload,
        )
    )
    return command


def _preflight_payload(
    *,
    node: str,
    dataset_root: Path,
    official_repo: Path,
    model_path: Path,
    run_id: str,
) -> str:
    return _encoded_payload(
        {
            "mode": "node_preflight",
            "node": node,
            "dataset_root": str(dataset_root),
            "official_repo": str(official_repo),
            "model_path": str(model_path),
            "repo": str(official_repo),
            "container_name": f"robotactile-n0-{run_id}-latent-preflight",
            "container_entrypoint": str(CONTAINER_ENTRYPOINT),
            "entrypoint_kind": "bash",
        }
    )


def _preflight_nodes(
    *,
    cluster: ClusterSpec,
    dataset_root: Path,
    official_repo: Path,
    model_path: Path,
    log_dir: Path,
    run_id: str,
) -> None:
    failures: list[str] = []

    def run(node: str) -> tuple[str, int, str]:
        payload = _preflight_payload(
            node=node,
            dataset_root=dataset_root,
            official_repo=official_repo,
            model_path=model_path,
            run_id=run_id,
        )
        result = subprocess.run(
            _ssh_command(cluster, node, payload),
            input=DOCKER_RUNTIME.read_bytes(),
            capture_output=True,
            timeout=180,
        )
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        return node, result.returncode, output

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run, node): node for node in cluster.nodes}
        for future in as_completed(futures):
            node, returncode, output = future.result()
            (log_dir / f"preflight-{node}.log").write_text(output, encoding="utf-8")
            if returncode != 0:
                failures.append(f"{node}: exit={returncode}")
    if failures:
        raise RuntimeError("latent node preflight failed: " + ", ".join(failures))


def _worker_payload(
    *,
    unit: WorkUnit,
    mode: str,
    dataset_root: Path,
    official_repo: Path,
    model_path: Path,
    receipt_path: Path,
    run_id: str,
) -> str:
    payload = unit.to_json()
    payload.update(
        {
            "mode": mode,
            "python_path": str(RUNTIME_PYTHON_PATH),
            "dataset_root": str(dataset_root),
            "official_repo": str(official_repo),
            "model_path": str(model_path),
            "receipt_path": str(receipt_path),
            "repo": str(official_repo),
            "container_name": f"robotactile-n0-{run_id}-latent-worker-{unit.worker_id:02d}",
            "container_entrypoint": str(CONTAINER_ENTRYPOINT),
            "entrypoint_kind": "bash",
        }
    )
    return _encoded_payload(payload)


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


def _run_workers(
    *,
    units: Sequence[WorkUnit],
    mode: str,
    cluster: ClusterSpec,
    dataset_root: Path,
    official_repo: Path,
    model_path: Path,
    receipt_dir: Path,
    log_dir: Path,
    run_id: str,
) -> list[int]:
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[IO[bytes]] = []
    try:
        for unit in units:
            receipt_path = receipt_dir / f"worker-{unit.worker_id:02d}.json"
            payload = _worker_payload(
                unit=unit,
                mode=mode,
                dataset_root=dataset_root,
                official_repo=official_repo,
                model_path=model_path,
                receipt_path=receipt_path,
                run_id=run_id,
            )
            log_handle: IO[bytes] = (
                log_dir
                / f"worker-{unit.worker_id:02d}-{unit.node}-{unit.task}-{unit.kind}.log"
            ).open("wb")
            handles.append(log_handle)
            processes.append(
                subprocess.Popen(
                    _ssh_command(cluster, unit.node, payload),
                    stdin=subprocess.PIPE,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            )
            assert processes[-1].stdin is not None
            processes[-1].stdin.write(DOCKER_RUNTIME.read_bytes())
            processes[-1].stdin.close()
        while True:
            returncodes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in returncodes):
                _terminate(processes)
                return [process.wait() for process in processes]
            if all(code == 0 for code in returncodes):
                return [0] * len(processes)
            time.sleep(2.0)
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-spec", type=Path, required=True)
    parser.add_argument("--cluster-spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = validate_run_id(args.run_id)
    spec_path = args.latent_spec.resolve()
    cluster_path = args.cluster_spec.resolve()
    run_root = args.run_root.resolve()
    spec = load_latent_spec(spec_path)
    cluster = load_cluster_spec(cluster_path)
    if len(cluster.nodes) != 2 or cluster.master_addr != cluster.nodes[0]:
        raise ValueError(
            "latent preprocessing requires two nodes with node 0 as master"
        )
    for runtime_file in (DOCKER_RUNTIME, CONTAINER_ENTRYPOINT, WORKER_SCRIPT):
        if not runtime_file.is_file():
            raise FileNotFoundError(runtime_file)
    official_source = verify_official_checkout(spec.official_repo)
    encoder_digests = verify_official_encoders(spec.official_repo)
    records = load_certified_train759(spec)
    units = build_work_units(records, cluster.nodes)
    if len(units) != 16:
        raise RuntimeError("latent preprocessing requires exactly 16 work units")

    lock_dir = run_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    dataset_lock_id = canonical_sha256(str(spec.dataset_root))[:20]
    lock_path = lock_dir / f"train759-{dataset_lock_id}.lock"
    lock_handle: IO[bytes] = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another latent run owns {spec.dataset_root}") from exc

    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "logs"
    receipt_dir = run_dir / "receipts"
    shard_dir = run_dir / "shards"
    for path in (log_dir, receipt_dir, shard_dir):
        path.mkdir()
    for unit in units:
        _write_json(shard_dir / f"worker-{unit.worker_id:02d}.json", unit.to_json())
    plan: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": "inventory" if args.inventory_only else "encode_missing",
        "official_source": official_source,
        "official_encoder_sha256": encoder_digests,
        "dataset_root": str(spec.dataset_root),
        "source_manifest_path": str(spec.source_manifest_path),
        "conversion_receipt_path": str(spec.conversion_receipt_path),
        "source_episode_count": len(records),
        "nodes": list(cluster.nodes),
        "workers_per_node": 8,
        "worker_count": len(units),
        "container_runtime": {
            "image": PROVEN_DOCKER_IMAGE,
            "image_id": PROVEN_DOCKER_IMAGE_ID,
            "rccl_plugin_file": str(PROVEN_RCCL_PLUGIN_FILE),
            "rccl_plugin_sha256": PROVEN_RCCL_PLUGIN_SHA256,
            "python_path": str(RUNTIME_PYTHON_PATH),
            "python_overlays": [str(path) for path in RUNTIME_PYTHON_OVERLAYS],
            "docker_runtime_sha256": sha256_file(DOCKER_RUNTIME),
            "container_entrypoint_sha256": sha256_file(CONTAINER_ENTRYPOINT),
            "worker_sha256": sha256_file(WORKER_SCRIPT),
            "transient_no_clobber_containers": True,
        },
        "workers": [unit.to_json() for unit in units],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    _write_json(run_dir / "plan.json", plan)
    status: dict[str, object] = {"run_id": run_id, "status": "preflighting"}
    status_path = run_dir / "status.json"
    _update_status(status_path, status)

    try:
        _preflight_nodes(
            cluster=cluster,
            dataset_root=spec.dataset_root,
            official_repo=spec.official_repo,
            model_path=spec.model_path,
            log_dir=log_dir,
            run_id=run_id,
        )
        status["status"] = "running"
        _update_status(status_path, status)
        returncodes = _run_workers(
            units=units,
            mode="inventory" if args.inventory_only else "encode",
            cluster=cluster,
            dataset_root=spec.dataset_root,
            official_repo=spec.official_repo,
            model_path=spec.model_path,
            receipt_dir=receipt_dir,
            log_dir=log_dir,
            run_id=run_id,
        )
        status["worker_returncodes"] = returncodes
        if any(code != 0 for code in returncodes):
            raise RuntimeError(f"latent worker failures: {returncodes}")
        inventory = aggregate_inventory(
            units=units,
            receipt_dir=receipt_dir,
            source_manifest_path=spec.source_manifest_path,
            conversion_receipt_path=spec.conversion_receipt_path,
            official_commit=OFFICIAL_COMMIT,
            encoder_sha256=encoder_digests,
        )
        _write_json(run_dir / "final_inventory.json", inventory)
        status["status"] = "complete"
        status["episode_count"] = inventory["episode_count"]
        status["inventory_sha256"] = inventory["inventory_sha256"]
        _update_status(status_path, status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        _update_status(status_path, status)
        raise


if __name__ == "__main__":
    sys.exit(main())
