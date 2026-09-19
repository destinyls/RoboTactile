#!/usr/bin/env python3
"""Own one retrained model server and one frozen Clean Isaac episode."""

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


def wait_ready(process: subprocess.Popen[bytes], receipt: Path, port: int) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"model server exited before ready: {process.returncode}"
            )
        if receipt.is_file():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                pass
        time.sleep(2)
    raise TimeoutError("model server startup exceeded 1800 seconds")


def stop_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest.get("model")
    if (
        manifest.get("schema") != "robotactile-a800-single-clean-cell-v1"
        or model not in {"n0_vtla", "ftp1_policy", "dream_tac"}
        or (manifest.get("task"), manifest.get("condition"), manifest.get("seed"))
        != ("lift_can", "Clean", 91111)
        or manifest.get("expected_episode_count") != 1
    ):
        raise ValueError("manifest is outside this supervisor's one-cell scope")
    if file_sha256(Path(__file__)) != manifest["supervisor_sha256"]:
        raise ValueError("supervisor changed after freezing")
    binding_path = Path(manifest["binding_path"]).resolve(strict=True)
    if file_sha256(binding_path) != manifest["binding_file_sha256"]:
        raise ValueError("frozen model binding changed")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding["model"] != model
        or binding["dataset_sha256"] != manifest["dataset_manifest_sha256"]
        or binding["checkpoint_sha256"] != manifest["checkpoint_sha256"]
    ):
        raise ValueError("binding model, weight, or dataset mismatch")
    node = Path(manifest["node_deployment_root"]).resolve(strict=True)
    code = Path(manifest["source_root"]).resolve(strict=True)
    runtime = Path(binding["runtime_python"])
    isaac = node / "runtime/isaac-sim-4.5.0/python.sh"
    worker = Path(manifest["worker_path"])
    for path in (runtime, isaac, worker):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen runtime: {path}")
    if file_sha256(worker) != manifest["worker_sha256"]:
        raise ValueError("Clean Isaac worker changed")
    port = binding["port"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    cell = manifest_path.parent
    server_dir = cell / "server"
    episode_dir = cell / "episode"
    if server_dir.exists() or episode_dir.exists():
        raise FileExistsError("cell was already launched")
    server_dir.mkdir(parents=True, exist_ok=False)
    receipt = server_dir / "server_receipt.json"
    if str(receipt) != manifest["server_receipt"]:
        raise ValueError("server receipt path differs from manifest")
    module = {
        "n0_vtla": "serve_vtla",
        "ftp1_policy": "serve_ftp1",
        "dream_tac": "serve_dream",
    }[model]
    command = [
        str(runtime),
        "-m",
        f"scripts.retrained_evaluation.{module}",
        "--binding",
        str(binding_path),
        "--task",
        "lift_can",
        "--receipt",
        str(receipt),
    ]
    if model == "n0_vtla":
        command.extend(("--seed", "91111"))
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(code / "src"), str(code))),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        TOKENIZERS_PARALLELISM="false",
    )
    if model == "n0_vtla":
        env["OPENPI_DATA_HOME"] = str(node / "artifacts/models/n0_vtla/data_cache")
    elif model == "ftp1_policy":
        env["OPENPI_DATA_HOME"] = str(node / "artifacts/openpi-data/ftp1-policy")
    else:
        env["PYTHONPATH"] += os.pathsep + binding["shared_pythonpath"]
        env["CUDNN_HOME"] = str(
            node / "runtime/ftp1-policy/lib/python3.11/site-packages/nvidia/cudnn"
        )
        env["CUDA_HOME"] = str(node / "runtime/cuda-toolkit-12.8")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(Path(env["CUDNN_HOME"]) / "lib"), env.get("LD_LIBRARY_PATH", ""))
        )
    isaac_env = dict(env)
    isaac_env["PYTHONPATH"] = os.pathsep.join((str(code / "src"), str(code)))
    reset_limit = binding.get("evaluation", {}).get("reset_time_limit_s")
    if reset_limit is not None:
        isaac_env["ROBOTACTILE_UNIVTAC_RESET_TIME_LIMIT_S"] = str(reset_limit)
    server: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    with (cell / "server.stdout.log").open("xb", buffering=0) as server_log:
        try:
            server = subprocess.Popen(
                command,
                cwd=code,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json_once(
                cell / "server_launch.json",
                {
                    "pid": server.pid,
                    "argv": command,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            wait_ready(server, receipt, port)
            write_json_once(
                cell / "server_ready.json",
                {
                    "server_receipt_sha256": file_sha256(receipt),
                    "startup_s": time.monotonic() - started,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            with (cell / "episode.stdout.log").open("xb", buffering=0) as episode_log:
                episode = subprocess.run(
                    [str(isaac), str(worker), "--manifest", str(manifest_path)],
                    cwd=code,
                    env=isaac_env,
                    stdout=episode_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            write_json_once(
                cell / "launcher_result.json",
                {
                    "episode_exit_code": episode.returncode,
                    "preclose_result_exists": (
                        episode_dir / "preclose_result.json"
                    ).is_file(),
                    "wall_s": time.monotonic() - started,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if episode.returncode:
                raise RuntimeError(f"Clean Isaac worker exited {episode.returncode}")
        except BaseException as error:
            write_json_once(
                cell / "launcher_failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "wall_s": time.monotonic() - started,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
        finally:
            if server is not None:
                stop_owned(server)


if __name__ == "__main__":
    main()
