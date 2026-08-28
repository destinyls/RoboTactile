"""Source-bound diagnostics for failures inside the pinned UniVTAC planner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError

_DIAGNOSTIC_TASKS = frozenset({"grasp_classify", "insert_hole", "insert_tube"})
_RESULT_FIELDS = (
    "status",
    "valid_query",
    "solve_time",
    "total_time",
    "ik_time",
    "trajopt_time",
    "attempts",
)


def _host_array(value: Any) -> NDArray[Any]:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return cast(NDArray[Any], np.asarray(value))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (RuntimeError, TypeError, ValueError):
            pass
    array = _host_array(value)
    if array.ndim == 0:
        return _json_value(array.item())
    return array.tolist()


def _array_summary(value: Any) -> Mapping[str, Any]:
    array = _host_array(value)
    finite = np.isfinite(array)
    summary: dict[str, Any] = {
        "dtype": str(array.dtype),
        "finite_count": int(finite.sum()),
        "shape": list(array.shape),
        "size": int(array.size),
    }
    if array.size and finite.any():
        finite_values = array[finite]
        summary["finite_max"] = float(finite_values.max())
        summary["finite_min"] = float(finite_values.min())
    return summary


def _actor_summaries(task: Any) -> list[Mapping[str, Any]]:
    manager = getattr(task, "_actor_manager", None)
    actors = getattr(manager, "actors", None)
    if not isinstance(actors, Mapping):
        raise UniVTACContractError("upstream actor registry is unavailable")
    summaries: list[Mapping[str, Any]] = []
    for name, actor in sorted(actors.items(), key=lambda item: str(item[0])):
        summaries.append(
            {
                "name": str(name),
                "object_id": _json_value(getattr(actor, "obj_id", None)),
                "vertices": _array_summary(actor.vertices),
            }
        )
    return summaries


def _surface_offsets(task: Any) -> Any:
    simulator = getattr(task, "uipc_sim", None)
    offsets = getattr(simulator, "_surf_vertex_offsets", None)
    return _json_value(offsets)


def _result_summary(result: Any) -> Mapping[str, Any]:
    summary: dict[str, Any] = {"success": _json_value(getattr(result, "success", None))}
    for field in _RESULT_FIELDS:
        if hasattr(result, field):
            summary[field] = _json_value(getattr(result, field))
    return summary


def _result_succeeded(result: Any) -> bool:
    value = _json_value(getattr(result, "success", None))
    if isinstance(value, list):
        return bool(value) and all(bool(item) for item in value)
    return bool(value)


def install_planner_failure_diagnostics(task: Any, task_id: str) -> bool:
    """Log one structured snapshot only when the pinned planner fails."""

    if task_id not in _DIAGNOSTIC_TASKS:
        return False
    robot_manager = getattr(task, "_robot_manager", None)
    planner = getattr(robot_manager, "planner", None)
    upstream_plan = getattr(planner, "plan_path", None)
    logger = getattr(task, "logger", None)
    log_error = getattr(logger, "error", None)
    if planner is None or not callable(upstream_plan) or not callable(log_error):
        raise UniVTACContractError("upstream planner diagnostic contract is incomplete")

    def diagnosed_plan_path(
        curr_joint_pos: Any,
        curr_joint_vel: Any,
        target_ee_pose: Any,
        real_robot_pose: Any,
        pre_dis: Optional[float] = None,
        constraint_pose: Any = None,
        time_dilation_factor: Optional[float] = None,
    ) -> Any:
        result = upstream_plan(
            curr_joint_pos=curr_joint_pos,
            curr_joint_vel=curr_joint_vel,
            target_ee_pose=target_ee_pose,
            real_robot_pose=real_robot_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor,
        )
        if _result_succeeded(result):
            return result
        target_values = getattr(target_ee_pose, "tolist", None)
        payload = {
            "actors": _actor_summaries(task),
            "constraint_pose": _json_value(constraint_pose),
            "curr_joint_pos": _array_summary(curr_joint_pos),
            "curr_joint_vel": _array_summary(curr_joint_vel),
            "pre_dis": pre_dis,
            "result": _result_summary(result),
            "surface_offsets": _surface_offsets(task),
            "target_ee_pose": (
                _json_value(target_values())
                if callable(target_values)
                else _json_value(target_ee_pose)
            ),
            "task_id": task_id,
            "time_dilation_factor": time_dilation_factor,
        }
        log_error(
            "ROBOTACTILE_PLAN_DIAGNOSTIC "
            + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        )
        return result

    planner.plan_path = diagnosed_plan_path
    return True
