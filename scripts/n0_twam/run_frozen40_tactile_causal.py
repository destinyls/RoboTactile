#!/usr/bin/env python3
"""Run the complete frozen40 Clean/structural-tactile-absence diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.contracts import N0ObservedTactileMode
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import OfficialN0Policy
from robotactile_benchmark.recorded.evaluator import RecordedN0Policy
from robotactile_benchmark.recorded.selection import select_univtac_contact_anchor
from robotactile_benchmark.recorded.source import load_univtac_hdf5_episode
from robotactile_benchmark.recorded.tactile_causal import (
    EVIDENCE_LEVEL,
    RecordedTactileCausalResult,
    aggregate_recorded_tactile_causal,
    run_recorded_tactile_causal_episode,
)
from robotactile_benchmark.recorded.tactile_causal_artifact import (
    write_recorded_tactile_causal_artifact,
)
from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    load_official_n0_rpc,
)

EXPECTED_TASK_IDS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
EPISODES_PER_TASK = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument("--master-port", type=int, default=29988)
    parser.add_argument("--exogenous-seed", type=int, default=20260829)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, path: Path, name: str) -> Path:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{name} escapes its declared root") from error
    return source


def _load_split(manifest_path: Path, data_root: Path) -> Mapping[str, tuple[Path, ...]]:
    path = _safe_file(manifest_path.parent, manifest_path, "split manifest")
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("split manifest must be a JSON list")
    grouped: dict[str, list[Path]] = defaultdict(list)
    relative_paths: set[str] = set()
    for raw_entry in raw:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "hdf5_path",
            "task",
        }:
            raise ValueError("each split entry must contain hdf5_path and task")
        task = raw_entry["task"]
        relative = raw_entry["hdf5_path"]
        if not isinstance(task, str) or not isinstance(relative, str):
            raise ValueError("split task/path fields must be strings")
        parts = Path(relative).parts
        if len(parts) != 3 or parts[:2] != (task, "clean"):
            raise ValueError("split path must be task/clean/episode.hdf5")
        if Path(relative).suffix != ".hdf5" or relative in relative_paths:
            raise ValueError("split HDF5 paths must be unique .hdf5 files")
        relative_paths.add(relative)
        grouped[task].append(_safe_file(data_root, data_root / relative, relative))
    if set(grouped) != set(EXPECTED_TASK_IDS) or any(
        len(grouped[task]) != EPISODES_PER_TASK for task in EXPECTED_TASK_IDS
    ):
        raise ValueError("frozen40 requires exactly five episodes for every task")
    return {
        task: tuple(sorted(grouped[task], key=lambda item: int(item.stem)))
        for task in EXPECTED_TASK_IDS
    }


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _wait_for_server(
    process: subprocess.Popen[bytes],
    *,
    repository_root: Path,
    source_root: Path,
    port: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    command = (
        sys.executable,
        str(repository_root / "scripts/n0_twam/check_server.py"),
        "--source-root",
        str(source_root),
        "--port",
        str(port),
    )
    last_error = "metadata probe has not completed"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"N0 server exited before readiness: {process.returncode}"
            )
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30.0,
        )
        if completed.returncode == 0:
            return
        last_error = completed.stdout.decode("utf-8", errors="replace")[-1000:]
        time.sleep(5.0)
    raise TimeoutError(f"N0 server readiness timed out: {last_error}")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def _identity(base: PolicyIdentity, *, absence: bool) -> PolicyIdentity:
    if not absence:
        return base
    return replace(
        base,
        system_id=f"{base.system_id}:observed-tactile-absent-v1",
        supports_structural_absence=True,
    )


def _policy_factory(
    *,
    identity: PolicyIdentity,
    source_root: Path,
    port: int,
    mode: N0ObservedTactileMode,
    api_key: str | None,
) -> RecordedN0Policy:
    return OfficialN0Policy(
        identity,
        lambda: OfficialN0Client(
            load_official_n0_rpc(
                source_root=source_root,
                host="127.0.0.1",
                port=port,
                api_key=api_key,
            )
        ),
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
        observed_tactile_mode=mode,
    )


def _make_policy_factory(
    *,
    identity: PolicyIdentity,
    source_root: Path,
    port: int,
    mode: N0ObservedTactileMode,
    api_key: str | None,
) -> Callable[[], RecordedN0Policy]:
    def factory() -> RecordedN0Policy:
        return _policy_factory(
            identity=identity,
            source_root=source_root,
            port=port,
            mode=mode,
            api_key=api_key,
        )

    return factory


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.exogenous_seed < 0 or args.server_ready_timeout_s <= 0.0:
        raise ValueError("seed must be non-negative and timeout must be positive")
    if not 1 <= args.port <= 65535 or not 1 <= args.master_port <= 65535:
        raise ValueError("ports must be in [1,65535]")
    if _port_open(args.port):
        raise RuntimeError("N0 port is already occupied")
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    repository_root = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root).absolute()
    split = _load_split(Path(args.split_manifest).absolute(), data_root)
    source_root = layout.sources / "N0-TWAM"
    source_receipt = verify_external_checkout(
        load_integration_lock().by_id("n0_twam"), source_root
    )
    api_key = os.environ.get("ROBOTACTILE_N0_API_KEY")
    results: list[RecordedTactileCausalResult] = []
    log_root = layout.logs / "recorded-n0-tactile-causal" / Path(args.output).stem
    log_root.mkdir(parents=True, exist_ok=True)
    for task_index, task in enumerate(EXPECTED_TASK_IDS):
        integration_config = (
            layout.model_artifacts
            / "n0_twam"
            / "configs"
            / task
            / "integration_config.json"
        )
        runtime = resolve_n0_runtime_artifacts(integration_config)
        manifest = runtime.manifest
        if manifest.task_id != task:
            raise ValueError("task-specific N0 integration config mismatch")
        base_identity = PolicyIdentity(
            system_id=(
                f"n0-twam:{source_receipt.commit_sha}:{manifest.serve_task_id}:"
                f"{N0_RECORDED_CHECKPOINT_INPUT_PROFILE.profile_id}"
            ),
            checkpoint_sha256=manifest.checkpoint_sha256,
            config_sha256=manifest.config_sha256,
            action_spec=EE8_ACTION_SPEC,
            consumes_tactile=True,
            supports_structural_absence=False,
        )
        server_command = (
            "bash",
            str(repository_root / "scripts/n0_twam/serve_univtac.sh"),
            "--root",
            str(layout.root),
            "--task",
            task,
            "--gpus",
            args.gpus,
            "--port",
            str(args.port),
            "--master-port",
            str(args.master_port),
            "--enable-observed-tactile-absence",
        )
        server: subprocess.Popen[bytes] | None = None
        with (log_root / f"{task}.log").open("xb") as log_stream:
            try:
                server = subprocess.Popen(
                    server_command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_for_server(
                    server,
                    repository_root=repository_root,
                    source_root=source_root,
                    port=args.port,
                    timeout_s=args.server_ready_timeout_s,
                )
                for episode_offset, hdf5_path in enumerate(split[task]):
                    seed = int(hdf5_path.stem)
                    selection = select_univtac_contact_anchor(hdf5_path, task_id=task)
                    episode = load_univtac_hdf5_episode(
                        hdf5_path,
                        task_id=task,
                        episode_id=f"univtac-{task}-clean-{seed}",
                        initial_seed=seed,
                        anchor_index=selection.anchor_index,
                        rest_index=selection.rest_index,
                    )
                    clean_identity = _identity(base_identity, absence=False)
                    absence_identity = _identity(base_identity, absence=True)
                    pair_seed = args.exogenous_seed + (
                        task_index * EPISODES_PER_TASK + episode_offset
                    )

                    results.append(
                        run_recorded_tactile_causal_episode(
                            episode=episode,
                            anchor_selection=selection,
                            clean_identity=clean_identity,
                            absence_identity=absence_identity,
                            clean_policy_factory=_make_policy_factory(
                                identity=clean_identity,
                                source_root=source_root,
                                port=args.port,
                                mode=N0ObservedTactileMode.REQUIRED,
                                api_key=api_key,
                            ),
                            absence_policy_factory=_make_policy_factory(
                                identity=absence_identity,
                                source_root=source_root,
                                port=args.port,
                                mode=N0ObservedTactileMode.ABSENT,
                                api_key=api_key,
                            ),
                            exogenous_seed=pair_seed,
                        )
                    )
            finally:
                if server is not None:
                    _terminate(server)
    summary = aggregate_recorded_tactile_causal(
        results,
        expected_task_ids=EXPECTED_TASK_IDS,
        episodes_per_task=EPISODES_PER_TASK,
    )
    output_sha256 = write_recorded_tactile_causal_artifact(
        Path(args.output).absolute(),
        results=results,
        summary=summary,
        source_commit=source_receipt.commit_sha,
        split_manifest_sha256=_sha256(Path(args.split_manifest).absolute()),
    )
    return {
        "completed_episode_count": len(results),
        "evidence_level": EVIDENCE_LEVEL,
        "output": str(Path(args.output).absolute()),
        "output_sha256": output_sha256,
        "success_rate_claimed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
