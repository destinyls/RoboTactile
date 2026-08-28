"""Instantiate and reset one pinned UniVTAC task through production wiring."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.action_specs import (
    QPOS8_ACTION_SPEC,
    SUPPORTED_ACTION_SPECS,
)
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import thaw_value
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)


def _absolute_directory(variable: str) -> Path:
    raw = os.environ.get(variable, "")
    path = Path(raw)
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError(f"{variable} must be an existing absolute directory")
    return path.resolve(strict=True)


def _result_path() -> Path:
    raw = os.environ.get("ROBOTACTILE_TASK_RESET_RESULT", "")
    path = Path(raw)
    if not path.is_absolute() or not path.parent.is_dir():
        raise RuntimeError(
            "ROBOTACTILE_TASK_RESET_RESULT must have an existing absolute parent"
        )
    if path.exists() or path.is_symlink():
        raise RuntimeError("task reset result path must not exist")
    return path


def _nonnegative_integer(variable: str) -> int:
    raw = os.environ.get(variable, "")
    if not raw.isdigit():
        raise RuntimeError(f"{variable} must be a non-negative integer")
    return int(raw)


def _publish_result(path: Path, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return serialized


def _array_contract(value: Any) -> dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _observation_contract(record: Any) -> dict[str, object]:
    observation = record.observation
    return {
        "proprio": _array_contract(observation.proprio),
        "tactile": {
            sensor.slot_id: _array_contract(sensor.payload)
            for sensor in observation.tactile
        },
        "vision": {
            name: _array_contract(value)
            for name, value in sorted(observation.vision.items())
        },
    }


def _success_payload(
    backend: UniVTACIsaacBackend,
    reset_receipt: Any,
    record: Any,
    reset_duration_s: float,
    plan_success: bool,
    construction_seed: int,
) -> dict[str, object]:
    import numpy
    import torch  # type: ignore[import-not-found]

    reset_diagnostics = thaw_value(reset_receipt.diagnostics)
    task_diagnostics = reset_diagnostics.get("task", {})
    if not isinstance(task_diagnostics, dict):
        raise RuntimeError("reset task diagnostics must be a mapping")
    return {
        "action_commands_executed": 0,
        "action_mode": backend.handshake.action_mode,
        "action_spec": backend.handshake.action_spec,
        "closed_loop_control_cycles": 0,
        "config_sha256": backend.handshake.config_sha256,
        "construction_seed": construction_seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "handshake_sha256": backend.handshake.sha256,
        "initial_clean_record_sha256": record.clean_record_sha256,
        "joint_reorder_witness_sha256": backend.joint_reorder_witness_sha256,
        "numpy_version": numpy.__version__,
        "observation_captured": True,
        "observation_contract": _observation_contract(record),
        "observation_step_index": record.observation.step_index,
        "phase_by_slot": {item.slot_id: item.phase.value for item in record.provenance},
        "plan_success": plan_success,
        "policy_loaded": False,
        "post_reset_native_step_id": backend.initial_native_step_id,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "reset_completed": True,
        "reset_diagnostics": reset_diagnostics,
        "placement_reset_assessment": task_diagnostics.get(
            "placement_reset_assessment"
        ),
        "reset_duration_s": reset_duration_s,
        "reset_receipt_sha256": reset_receipt.sha256,
        "robotactile_version": importlib.metadata.version("robotactile-benchmark"),
        "runtime_close_requested": True,
        "simulator_advanced_during_reset": True,
        "simulator_state_sha256": reset_receipt.simulator_state_sha256,
        "status": "passed",
        "task_id": backend.handshake.task_id,
        "task_instantiated": True,
        "task_source_sha256": backend.handshake.task_source_sha256,
        "torch_cuda_version": str(torch.version.cuda),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_version": torch.__version__,
    }


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
    """Run one reset/observation gate without loading or executing a policy."""
    arguments = _argument_parser().parse_args(argv)
    task_id = arguments.task
    action_spec = arguments.action_spec
    result_path = _result_path()
    upstream_root = _absolute_directory("ROBOTACTILE_UNIVTAC_ROOT")
    runtime_dir = _absolute_directory("ROBOTACTILE_TASK_RUNTIME_DIR")
    initial_seed = _nonnegative_integer("ROBOTACTILE_INITIAL_SEED")
    exogenous_seed = _nonnegative_integer("ROBOTACTILE_EXOGENOUS_SEED")
    backend: UniVTACIsaacBackend | None = None
    exit_code = 0
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
        backend = UniVTACIsaacBackend(config, runtime)
        context = PolicyEpisodeContext(
            episode_id=(f"qualification-reset-{task_id}-{action_spec}-{initial_seed}"),
            task=config.task.task_id,
            initial_seed=initial_seed,
            exogenous_seed=exogenous_seed,
            instruction=config.task.prompt,
            action_spec=config.action_spec,
        )
        started = time.monotonic()
        reset_receipt = backend.reset(context)
        reset_duration_s = time.monotonic() - started
        payload = _success_payload(
            backend,
            reset_receipt,
            backend.observe(),
            reset_duration_s,
            bool(runtime.task.plan_success),
            initial_seed,
        )
    except Exception as error:
        exit_code = 1
        payload = {
            "action_commands_executed": 0,
            "action_spec": action_spec,
            "closed_loop_control_cycles": 0,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "policy_loaded": False,
            "runtime_close_requested": backend is not None,
            "status": "failed",
            "task_id": task_id,
        }
    serialized = _publish_result(result_path, payload)
    print(serialized, file=sys.stdout if exit_code == 0 else sys.stderr, flush=True)
    if backend is not None:
        backend.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
