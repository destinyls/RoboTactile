"""Qualify exact in-process UniVTAC snapshot replay without a learned policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from robotactile_benchmark.action_specs import (
    EE8_ACTION_SPEC,
    QPOS8_ACTION_SPEC,
    SUPPORTED_ACTION_SPECS,
)
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_pairing import (
    PAIRING_EVIDENCE_LEVEL,
    UniVTACPairedBackendSession,
)
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import Array
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)


def _absolute_directory(variable: str) -> Path:
    path = Path(os.environ.get(variable, ""))
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError(f"{variable} must be an existing absolute directory")
    return path.resolve(strict=True)


def _result_path() -> Path:
    path = Path(os.environ.get("ROBOTACTILE_TASK_PAIRING_RESULT", ""))
    if not path.is_absolute() or not path.parent.is_dir():
        raise RuntimeError(
            "ROBOTACTILE_TASK_PAIRING_RESULT must have an existing absolute parent"
        )
    if path.exists() or path.is_symlink():
        raise RuntimeError("task pairing result path must not exist")
    return path


def _nonnegative_integer(variable: str) -> int:
    raw = os.environ.get(variable, "")
    if not raw.isdigit():
        raise RuntimeError(f"{variable} must be a non-negative integer")
    return int(raw)


def _publish(path: Path, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return serialized


def _context(
    config: UniVTACBackendConfig, initial_seed: int, exogenous_seed: int
) -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id=(
            "qualification-pairing-"
            f"{config.task.task_id}-{config.action_spec}-{initial_seed}"
        ),
        task=config.task.task_id,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        instruction=config.task.prompt,
        action_spec=config.action_spec,
    )


def _divergence_action(
    initial_action8: Array | None,
    config: UniVTACBackendConfig,
) -> Array:
    """Create one small bounds-safe target in the selected action contract."""

    if initial_action8 is None or initial_action8.shape != (8,):
        raise RuntimeError("live reset did not expose the selected action state")
    action = np.array(initial_action8, dtype=np.float32, order="C", copy=True)
    lower = np.asarray(config.action_lower_bounds, dtype=np.float32)
    upper = np.asarray(config.action_upper_bounds, dtype=np.float32)
    movable_indices = range(3) if config.action_spec == EE8_ACTION_SPEC else range(7)
    for index in movable_indices:
        positive_room = float(upper[index] - action[index])
        negative_room = float(action[index] - lower[index])
        if max(positive_room, negative_room) <= 0.0:
            continue
        direction = 1.0 if positive_room >= negative_room else -1.0
        room = positive_room if direction > 0.0 else negative_room
        delta = min(1e-3, room / 2.0)
        if delta > 0.0:
            action[index] = np.float32(action[index] + direction * delta)
            return np.ascontiguousarray(action.reshape(1, 8))
    raise RuntimeError("live reset action has no movable coordinate within bounds")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pull_out_key")
    parser.add_argument(
        "--action-spec",
        choices=sorted(SUPPORTED_ACTION_SPECS),
        default=QPOS8_ACTION_SPEC,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Diverge once, restore once, and publish the exact reset witness gate."""

    arguments = _argument_parser().parse_args(argv)
    task_id = arguments.task
    action_spec = arguments.action_spec
    result_path = _result_path()
    upstream_root = _absolute_directory("ROBOTACTILE_UNIVTAC_ROOT")
    runtime_dir = _absolute_directory("ROBOTACTILE_TASK_RUNTIME_DIR")
    initial_seed = _nonnegative_integer("ROBOTACTILE_INITIAL_SEED")
    exogenous_seed = _nonnegative_integer("ROBOTACTILE_EXOGENOUS_SEED")
    session: UniVTACPairedBackendSession | None = None
    canonical_backend = None
    exit_code = 0
    started = time.monotonic()
    try:
        config = build_univtac_backend_config(task_id, action_spec=action_spec)
        runtime = launch_univtac_runtime(
            config,
            upstream_root=upstream_root,
            runtime_dir=runtime_dir,
            initial_seed=initial_seed,
            launcher_args=production_univtac_launcher_args(),
            device="cuda:0",
        )
        session = UniVTACPairedBackendSession(config, runtime)
        context = _context(config, initial_seed, exogenous_seed)

        canonical_backend = session.new_backend()
        canonical_reset = canonical_backend.reset(context)
        canonical_record = canonical_backend.observe()
        initial_action8 = (
            canonical_record.observation.proprio
            if config.action_spec == EE8_ACTION_SPEC
            else canonical_backend.initial_model_visible_qpos8
        )
        action = _divergence_action(
            initial_action8,
            config,
        )
        transition = canonical_backend.execute(action).transitions[0]
        divergent_state_sha256 = canonical_backend.latest_state_sha256
        canonical_backend.close()
        if divergent_state_sha256 in {None, canonical_reset.simulator_state_sha256}:
            raise RuntimeError("qualification action did not create divergent state")

        replay_backend = session.new_backend()
        replay_reset = replay_backend.reset(context)
        replay_record = replay_backend.observe()
        replay_backend.close()
        receipt = session.reset_receipt
        if (
            len(receipt.witnesses) != 2
            or not receipt.all_exact
            or receipt.witnesses[0].reset_mode != "canonical_reset"
            or receipt.witnesses[1].reset_mode != "snapshot_replay"
            or canonical_reset.simulator_state_sha256
            != replay_reset.simulator_state_sha256
            or canonical_record.clean_record_sha256 != replay_record.clean_record_sha256
        ):
            raise RuntimeError("paired reset equivalence gate did not pass exactly")
        payload = {
            "action_commands_executed": 1,
            "action_mode": config.action_mode,
            "action_spec": config.action_spec,
            "canonical_clean_record_sha256": canonical_record.clean_record_sha256,
            "canonical_state_sha256": canonical_reset.simulator_state_sha256,
            "divergence_proven": True,
            "divergent_native_step_id": transition.native_step_id,
            "divergent_state_sha256": divergent_state_sha256,
            "duration_s": time.monotonic() - started,
            "evidence_level": PAIRING_EVIDENCE_LEVEL,
            "paired_reset_receipt": receipt.to_dict(),
            "paired_reset_receipt_sha256": receipt.sha256,
            "policy_loaded": False,
            "replay_clean_record_sha256": replay_record.clean_record_sha256,
            "replay_state_sha256": replay_reset.simulator_state_sha256,
            "robotactile_version": importlib.metadata.version("robotactile-benchmark"),
            "runtime_close_requested": True,
            "simulator_qualification_claimed": False,
            "status": "passed",
            "task_id": config.task.task_id,
            "task_source_sha256": config.task.task_source_sha256,
            "task_success_evaluated": False,
            "witness_count": len(receipt.witnesses),
        }
    except Exception as error:
        exit_code = 1
        payload = {
            "action_commands_executed": (
                0
                if canonical_backend is None
                else canonical_backend.executed_action_count
            ),
            "action_spec": action_spec,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "evidence_level": PAIRING_EVIDENCE_LEVEL,
            "policy_loaded": False,
            "runtime_close_requested": session is not None,
            "simulator_qualification_claimed": False,
            "status": "failed",
            "task_id": task_id,
            "task_success_evaluated": False,
        }
        if session is not None:
            try:
                failed_receipt = session.reset_receipt
            except Exception:
                pass
            else:
                payload["paired_reset_receipt"] = failed_receipt.to_dict()
                payload["paired_reset_receipt_sha256"] = failed_receipt.sha256
    serialized = _publish(result_path, payload)
    print(serialized, file=sys.stdout if exit_code == 0 else sys.stderr, flush=True)
    if session is not None:
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
