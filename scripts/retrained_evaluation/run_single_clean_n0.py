#!/usr/bin/env python3
"""Launch one frozen N0-TWAM Clean cell on an isolated A800 node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stop_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_ready(process: subprocess.Popen[bytes], receipt: Path, port: int) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"N0 server exited before ready: {process.returncode}")
        if receipt.is_file():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                pass
        time.sleep(2)
    raise TimeoutError("N0 model server startup exceeded 1800 seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "robotactile-a800-single-clean-cell-v1":
        raise ValueError("unexpected cell manifest schema")
    if (manifest.get("model"), manifest.get("task"), manifest.get("seed")) != (
        "n0_twam",
        "lift_can",
        91111,
    ):
        raise ValueError("runner is restricted to the requested N0 Clean cell")
    if file_sha256(Path(__file__)) != manifest["launcher_sha256"]:
        raise ValueError("launcher differs from frozen manifest")
    cell = manifest_path.parent
    node_root = Path(manifest["node_deployment_root"]).resolve(strict=True)
    source = Path(manifest["source_root"]).resolve(strict=True)
    artifact = Path(manifest["artifact_path"]).resolve(strict=True)
    dataset = Path(manifest["dataset_manifest"]).resolve(strict=True)
    if file_sha256(artifact) != manifest["artifact_file_sha256"]:
        raise ValueError("prepared artifact file changed")
    if file_sha256(dataset) != manifest["dataset_manifest_sha256"]:
        raise ValueError("seeded dataset identity changed")
    prepared = json.loads(artifact.read_text(encoding="utf-8"))
    if (
        prepared["artifact_sha256"] != manifest["prepared_artifact_sha256"]
        or prepared["checkpoint_sha256"] != manifest["checkpoint_sha256"]
        or prepared["source_tree_sha256"] != manifest["source_tree_sha256"]
    ):
        raise ValueError("prepared N0 source or weight identity changed")
    runtime = node_root / "runtime/n0-twam/bin/python"
    isaac = node_root / "runtime/isaac-sim-4.5.0/python.sh"
    if not runtime.is_file() or not isaac.is_file():
        raise FileNotFoundError("node-local runtime is missing")
    port = manifest["server_port"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    server_dir = cell / "server"
    episode_dir = cell / "episode"
    if any(p.exists() or p.is_symlink() for p in (server_dir, episode_dir)):
        raise FileExistsError("this cell has already been launched")
    server_log = (cell / "server.stdout.log").open("xb", buffering=0)
    episode_log = (cell / "episode.stdout.log").open("xb", buffering=0)
    started = time.monotonic()
    server: subprocess.Popen[bytes] | None = None
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        TOKENIZERS_PARALLELISM="false",
    )
    try:
        server = subprocess.Popen(
            [
                str(runtime),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=1",
                str(source / "scripts/n0_twam/serve_retrained.py"),
                "--artifact",
                str(artifact),
                "--task",
                "lift_can",
                "--output",
                str(server_dir),
                "--port",
                str(port),
            ],
            cwd=source,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        receipt = server_dir / "server_receipt.json"
        wait_ready(server, receipt, port)
        write_json_once(
            cell / "server_ready.json",
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "server_pid": server.pid,
                "server_receipt_sha256": file_sha256(receipt),
                "server_port": port,
            },
        )
        episode = subprocess.run(
            [
                str(isaac),
                str(source / "scripts/n0_twam/run_retrained_clean.py"),
                "--artifact",
                str(artifact),
                "--task",
                "lift_can",
                "--dataset-manifest",
                str(dataset),
                "--upstream",
                str(node_root / "sources/UniVTAC"),
                "--runtime",
                str(cell / "simulator_runtime"),
                "--server-receipt",
                str(receipt),
                "--output",
                str(episode_dir),
                "--seed",
                "91111",
                "--port",
                str(port),
                "--watchdog-s",
                "7200",
                "--capture-profile",
                "preview_v1",
            ],
            cwd=source,
            env=env,
            stdout=episode_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        write_json_once(
            cell / "launcher_result.json",
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_exit_code": episode.returncode,
                "episode_summary_exists": (episode_dir / "summary.json").is_file(),
                "wall_s": time.monotonic() - started,
            },
        )
        if episode.returncode:
            raise RuntimeError(f"N0 Clean runner exited {episode.returncode}")
    except BaseException as error:
        write_json_once(
            cell / "launcher_failure.json",
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "type": type(error).__name__,
                "message": str(error),
                "wall_s": time.monotonic() - started,
            },
        )
        raise
    finally:
        if server is not None:
            stop_owned(server)
        server_log.close()
        episode_log.close()


if __name__ == "__main__":
    main()
