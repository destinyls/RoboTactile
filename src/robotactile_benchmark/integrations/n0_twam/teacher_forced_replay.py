"""Official sequential qpos replay for simulator teacher-forced probes."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from robotactile_benchmark.contracts import Array
from robotactile_benchmark.integrations.n0_twam.dynamic_contract import (
    state_match_at_index,
)


def strict_action_result(value: object) -> tuple[bool, bool]:
    """Validate the exact two-bool result returned by UniVTAC take_action."""

    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not bool
        or type(value[1]) is not bool
    ):
        raise RuntimeError("official qpos replay returned a malformed result")
    return value


def native_step(task: Any) -> int:
    """Read one non-negative native task step counter."""

    value = getattr(task, "step_count", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("live task did not expose a valid native step")
    return value


def _qpos_tensor(task: Any, joint9: Array) -> Any:
    torch = importlib.import_module("torch")
    as_tensor = getattr(torch, "as_tensor", None)
    float32 = getattr(torch, "float32", None)
    if not callable(as_tensor) or float32 is None:
        raise RuntimeError("runtime torch lacks float32 tensor conversion")
    qpos8 = np.array(joint9[:8], dtype=np.float32, order="C", copy=True)
    return as_tensor(qpos8, dtype=float32, device=task.device)


@dataclass(frozen=True)
class TeacherForcedReplay:
    """One immutable receipt plus live target/previous conversions."""

    indices: Tuple[int, ...]
    trace: Tuple[dict[str, object], ...]
    duration_s: float
    execution_success: bool
    returned_success: bool
    plan_success: bool
    native_step_before: int
    native_step_after: int
    target_raw: Any
    target_conversion: Any
    previous_index: Optional[int]
    previous_conversion: Any


def replay_official_qpos_sequence(
    *,
    task: Any,
    backend: Any,
    joint9_trajectory: Array,
    expert_states: Array,
    target_index: int,
    stride: int,
    physics_steps_per_target: int,
) -> TeacherForcedReplay:
    """Replay rows with qpos control and the collection render cadence."""

    if physics_steps_per_target < 1:
        raise ValueError("physics_steps_per_target must be positive")
    if getattr(task, "mode", None) != "eval":
        raise RuntimeError("official qpos replay requires an eval-mode task")
    update_render = getattr(task, "_update_render", None)
    step = getattr(task, "_step", None)
    if not callable(update_render) or not callable(step):
        raise RuntimeError("official qpos replay lacks the endpoint render surface")
    indices = tuple(range(0, target_index + 1, stride))
    start_step = native_step(task)
    started = time.monotonic()
    trace: list[dict[str, object]] = []
    target_raw: Any = None
    target_conversion: Any = None
    previous_index: Optional[int] = None
    previous_conversion: Any = None
    returned_success = False
    execution_success = True
    for replay_order, replay_index in enumerate(indices, start=1):
        step_before = native_step(task)
        previous_mode = task.mode
        try:
            task.mode = "eval_test"
            result = task.take_action(
                _qpos_tensor(task, joint9_trajectory[replay_index]),
                action_type="qpos",
                force=True,
            )
            for _ in range(physics_steps_per_target - 1):
                step(is_save=False)
        finally:
            task.mode = previous_mode
        update_render()
        action_success, returned_success = strict_action_result(result)
        plan_success = getattr(task, "plan_success", None)
        if not action_success or plan_success is not True:
            raise RuntimeError(
                f"official qpos replay failed at HDF5 index {replay_index}"
            )
        execution_success = execution_success and action_success
        current_raw = backend._fresh_raw()
        current_conversion = backend._convert(current_raw, benchmark_step=replay_order)
        trace.append(
            {
                "hdf5_index": replay_index,
                "native_step_after": native_step(task),
                "native_step_before": step_before,
                "native_step_delta": native_step(task) - step_before,
                "render_contract": "one_endpoint_render_no_intermediate_render_v1",
                "state_match": state_match_at_index(
                    expert_states,
                    current_conversion.record.observation.proprio,
                    replay_index,
                ).to_dict(),
            }
        )
        if replay_index == target_index:
            target_raw = current_raw
            target_conversion = current_conversion
        else:
            previous_index = replay_index
            previous_conversion = current_conversion
    final_plan_success = getattr(task, "plan_success", None)
    if target_conversion is None or target_raw is None:
        raise RuntimeError("official qpos replay did not reach the target row")
    if final_plan_success is not True:
        raise RuntimeError("official qpos replay ended without plan success")
    return TeacherForcedReplay(
        indices=indices,
        trace=tuple(trace),
        duration_s=time.monotonic() - started,
        execution_success=execution_success,
        returned_success=returned_success,
        plan_success=final_plan_success,
        native_step_before=start_step,
        native_step_after=native_step(task),
        target_raw=target_raw,
        target_conversion=target_conversion,
        previous_index=previous_index,
        previous_conversion=previous_conversion,
    )


__all__ = [
    "TeacherForcedReplay",
    "native_step",
    "replay_official_qpos_sequence",
    "strict_action_result",
]
