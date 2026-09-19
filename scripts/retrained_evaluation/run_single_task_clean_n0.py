#!/usr/bin/env python3
"""Own one frozen N0 server and one taskwise Clean qualification episode."""

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

TASK_SEQUENCE = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
CAPTURE_PROFILE = "metrics_only_v1"


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
            raise RuntimeError(f"N0 server exited before ready: {process.returncode}")
        if receipt.is_file():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                pass
        time.sleep(2)
    raise TimeoutError("N0 model server startup exceeded 1800 seconds")


def stop_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def verify_episode(
    manifest: dict[str, Any],
    episode_dir: Path,
    server_receipt: Path,
) -> dict[str, Any]:
    from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
    from robotactile_benchmark.contracts import canonical_hash, thaw_value
    from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
    from robotactile_benchmark.execution.live_artifacts import (
        load_live_univtac_artifact,
    )

    live = episode_dir / "live"
    bundle = load_live_univtac_artifact(live)
    identity = thaw_value(bundle.request_identity)
    if (
        not isinstance(identity, dict)
        or bundle.trial.task != manifest["task"]
        or bundle.trial.condition.value != "clean"
        or bundle.capture_profile is not LiveCaptureProfile.METRICS_ONLY
        or identity.get("policy_kind") != "n0"
        or identity.get("initial_seed") != manifest["seed"]
        or identity.get("exogenous_seed") != manifest["seed"]
        or identity.get("dataset_sha256") != manifest["dataset_manifest_sha256"]
        or identity.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or identity.get("n0_serve_bundle_sha256")
        != manifest["prepared_artifact_sha256"]
    ):
        raise ValueError("completed N0 artifact differs from the frozen cell")
    result = bundle.evidence.result
    if result.validation_passed is not True or not result.score_eligible:
        raise ValueError("completed N0 artifact is not valid and score eligible")
    diagnostics = thaw_value(bundle.evidence.initial_diagnostics)
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("benchmark_step") != 0
        or not isinstance(diagnostics.get("n0_reset"), dict)
    ):
        raise ValueError("metrics-only artifact lacks a strict initial reset witness")
    reset_witness = {
        "schema": "robotactile-n0-taskwise-reset-witness-v1",
        "task": manifest["task"],
        "seed": manifest["seed"],
        "hostname": manifest["hostname"],
        "capture_profile": CAPTURE_PROFILE,
        "initial_state_sha256": result.initial_state_sha256,
        "initial_diagnostics": diagnostics,
        "initial_diagnostics_sha256": canonical_hash(diagnostics),
        "live_root_receipt_sha256": bundle.root_receipt_sha256,
        "live_artifact": str(live),
    }
    reset_path = episode_dir / "reset_witness.json"
    write_json_once(reset_path, reset_witness)
    action_entries = bundle.evidence.action_entries
    acceptance = {
        "schema": "robotactile-n0-taskwise-clean-acceptance-v1",
        "task": manifest["task"],
        "seed": manifest["seed"],
        "hostname": manifest["hostname"],
        "capture_profile": CAPTURE_PROFILE,
        "validation_passed": result.validation_passed,
        "score_eligible": result.score_eligible,
        "score_success": result.score_success,
        "terminal_status": result.terminal_status.value,
        "execution_status": (
            None if result.execution_status is None else result.execution_status.value
        ),
        "control_cycle_count": result.control_cycle_count,
        "observation_count": result.observation_count,
        "action_entry_count": len(action_entries),
        "executed_action_count": sum(
            len(entry.executed_actions) for entry in action_entries
        ),
        "live_root_receipt_sha256": bundle.root_receipt_sha256,
        "root_receipt_file_sha256": file_sha256(live / "root_receipt.json"),
        "reset_witness": str(reset_path),
        "reset_witness_sha256": file_sha256(reset_path),
        "server_receipt_sha256": file_sha256(server_receipt),
        "request_identity_sha256": canonical_hash(identity),
        "strict_reload_passed": True,
        "inference_count_evidence": "action_entry_count",
        "evidence_scope": "single_seed_clean_qualification_not_success_rate",
        "result": result_to_dict(result),
    }
    write_json_once(episode_dir / "acceptance.json", acceptance)
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = manifest.get("task")
    if (
        manifest.get("schema") != "robotactile-a800-n0-taskwise-clean-cell-v1"
        or manifest.get("model") != "n0_twam"
        or task not in TASK_SEQUENCE
        or manifest.get("task_order_index") != TASK_SEQUENCE.index(task) + 1
        or manifest.get("condition") != "Clean"
        or manifest.get("capture_profile") != CAPTURE_PROFILE
        or manifest.get("expected_episode_count") != 1
        or manifest.get("evaluation_scope") != "minimal_clean_qualification"
        or type(manifest.get("seed")) is not int
        or manifest["seed"] < 0
    ):
        raise ValueError("manifest is outside the taskwise N0 Clean scope")
    if file_sha256(Path(__file__)) != manifest["launcher_sha256"]:
        raise ValueError("launcher changed after the cell was frozen")
    if socket.gethostname() != manifest["hostname"]:
        raise ValueError("cell manifest was frozen for another host")
    cell = manifest_path.parent
    if Path(manifest["episode_output"]).absolute() != (cell / "episode").absolute():
        raise ValueError("episode output differs from the frozen cell path")
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
        or task not in prepared["tasks"]
    ):
        raise ValueError("prepared N0 source, weight, or task identity changed")
    runtime = node_root / "runtime/n0-twam/bin/python"
    isaac = node_root / "runtime/isaac-sim-4.5.0/python.sh"
    if not runtime.is_file() or not isaac.is_file():
        raise FileNotFoundError("node-local runtime is missing")
    port = manifest["server_port"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    server_dir = cell / "server"
    episode_dir = cell / "episode"
    if any(path.exists() or path.is_symlink() for path in (server_dir, episode_dir)):
        raise FileExistsError("this frozen cell was already launched")
    server_log = (cell / "server.stdout.log").open("xb", buffering=0)
    episode_log = (cell / "episode.stdout.log").open("xb", buffering=0)
    started = time.monotonic()
    server: subprocess.Popen[bytes] | None = None
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
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
                task,
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
                task,
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
                str(manifest["seed"]),
                "--port",
                str(port),
                "--watchdog-s",
                "7200",
                "--capture-profile",
                CAPTURE_PROFILE,
            ],
            cwd=source,
            env=env,
            stdout=episode_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        acceptance = verify_episode(manifest, episode_dir, receipt)
        write_json_once(
            cell / "launcher_result.json",
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_exit_code": episode.returncode,
                "acceptance_sha256": file_sha256(episode_dir / "acceptance.json"),
                "live_root_receipt_sha256": acceptance["live_root_receipt_sha256"],
                "wall_s": time.monotonic() - started,
            },
        )
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
