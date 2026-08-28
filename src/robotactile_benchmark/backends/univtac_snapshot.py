"""Pinned in-memory snapshot/restore support for one live UniVTAC runtime."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Tuple, cast

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.contracts import Array, canonical_hash

_TASK_STATE_FIELDS = (
    "instruction",
    "plan_success",
    "eval_success",
    "step_count",
    "save_count",
    "take_action_cnt",
    "current_goal_idx",
    "render_outdated",
    "last_render",
    "last_qpos",
    "keep_still_times",
    "metadata",
    "log",
    "atom_id",
    "atom_tag",
    "in_pre_move",
)


@dataclass(frozen=True)
class _LiveUniVTACSnapshot:
    """Opaque process-local state; never serialized or exposed as evidence."""

    runtime_token: object
    uipc_frame_id: int
    robot_root_state: Any
    robot_joint_position: Any
    robot_joint_velocity: Any
    robot_joint_position_target: Any
    robot_joint_velocity_target: Any
    plate_root_state: Any
    rng_state: object
    task_state: Mapping[str, object]


def _required(value: object, name: str) -> Any:
    if value is None:
        raise UniVTACContractError(f"snapshot surface is unavailable: {name}")
    return value


def _callable(value: object, name: str) -> Callable[..., Any]:
    if not callable(value):
        raise UniVTACContractError(f"snapshot callback is unavailable: {name}")
    return value


def _clone_tensor(value: object, name: str) -> Any:
    clone = _callable(getattr(value, "clone", None), f"{name}.clone")
    return clone()


def _tensor_array(value: object, name: str) -> Array:
    current = value
    for method_name in ("detach", "cpu", "contiguous", "numpy"):
        method = getattr(current, method_name, None)
        if not callable(method):
            raise UniVTACContractError(
                f"snapshot tensor conversion is unavailable: {name}.{method_name}"
            )
        current = method()
    if not isinstance(current, np.ndarray):
        raise UniVTACContractError(
            f"snapshot tensor conversion did not produce numpy: {name}"
        )
    return cast(Array, np.ascontiguousarray(current))


def _witness_value(value: object, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, Mapping):
        return {
            str(key): _witness_value(item, f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(
            _witness_value(item, f"{name}[{index}]") for index, item in enumerate(value)
        )
    if callable(getattr(value, "detach", None)):
        return _tensor_array(value, name)
    raise UniVTACContractError(f"snapshot witness contains unsupported state: {name}")


def _task_state(task: object) -> Mapping[str, object]:
    state = {
        name: copy.deepcopy(getattr(task, name))
        for name in _TASK_STATE_FIELDS
        if hasattr(task, name)
    }
    return MappingProxyType(state)


def _rng_state(task: object) -> object:
    rng = _required(getattr(task, "rng", None), "task.rng")
    generator = _required(getattr(rng, "bit_generator", None), "rng.bit_generator")
    if not hasattr(generator, "state"):
        raise UniVTACContractError("snapshot RNG state is unavailable")
    return copy.deepcopy(generator.state)


def _capture_snapshot(task: object, runtime_token: object) -> _LiveUniVTACSnapshot:
    uipc_sim = _required(getattr(task, "uipc_sim", None), "task.uipc_sim")
    world = _required(getattr(uipc_sim, "world", None), "uipc_sim.world")
    frame = _callable(getattr(world, "frame", None), "world.frame")()
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise UniVTACContractError("UIPC frame ID must be a non-negative integer")
    _callable(getattr(uipc_sim, "save_frame", None), "uipc_sim.save_frame")()

    manager = _required(getattr(task, "_robot_manager", None), "task._robot_manager")
    robot = _required(getattr(manager, "robot", None), "robot_manager.robot")
    robot_data = _required(getattr(robot, "data", None), "robot.data")
    plate = _required(getattr(task, "plate", None), "task.plate")
    plate_data = _required(getattr(plate, "data", None), "plate.data")
    return _LiveUniVTACSnapshot(
        runtime_token=runtime_token,
        uipc_frame_id=frame,
        robot_root_state=_clone_tensor(robot_data.root_state_w, "robot root state"),
        robot_joint_position=_clone_tensor(
            robot_data.joint_pos, "robot joint position"
        ),
        robot_joint_velocity=_clone_tensor(
            robot_data.joint_vel, "robot joint velocity"
        ),
        robot_joint_position_target=_clone_tensor(
            robot_data.joint_pos_target, "robot joint position target"
        ),
        robot_joint_velocity_target=_clone_tensor(
            robot_data.joint_vel_target, "robot joint velocity target"
        ),
        plate_root_state=_clone_tensor(plate_data.root_state_w, "plate root state"),
        rng_state=_rng_state(task),
        task_state=_task_state(task),
    )


def _restore_task_state(task: object, snapshot: _LiveUniVTACSnapshot) -> None:
    for name, value in snapshot.task_state.items():
        setattr(task, name, copy.deepcopy(value))
    rng = _required(getattr(task, "rng", None), "task.rng")
    generator = _required(getattr(rng, "bit_generator", None), "rng.bit_generator")
    generator.state = copy.deepcopy(snapshot.rng_state)
    now = time.perf_counter()
    if hasattr(task, "start_time"):
        task.start_time = now
    if hasattr(task, "last_step"):
        task.last_step = now


def _restore_physx(task: object, snapshot: _LiveUniVTACSnapshot) -> None:
    manager = _required(getattr(task, "_robot_manager", None), "task._robot_manager")
    robot = _required(getattr(manager, "robot", None), "robot_manager.robot")
    _callable(
        getattr(robot, "write_root_state_to_sim", None),
        "robot.write_root_state_to_sim",
    )(snapshot.robot_root_state)
    _callable(
        getattr(robot, "write_joint_state_to_sim", None),
        "robot.write_joint_state_to_sim",
    )(snapshot.robot_joint_position, snapshot.robot_joint_velocity)
    _callable(
        getattr(robot, "set_joint_position_target", None),
        "robot.set_joint_position_target",
    )(snapshot.robot_joint_position_target)
    _callable(
        getattr(robot, "set_joint_velocity_target", None),
        "robot.set_joint_velocity_target",
    )(snapshot.robot_joint_velocity_target)

    plate = _required(getattr(task, "plate", None), "task.plate")
    _callable(
        getattr(plate, "write_root_state_to_sim", None),
        "plate.write_root_state_to_sim",
    )(snapshot.plate_root_state)


def _snapshot_state_sha256(task: object) -> str:
    uipc_sim = _required(getattr(task, "uipc_sim", None), "task.uipc_sim")
    world = _required(getattr(uipc_sim, "world", None), "uipc_sim.world")
    frame = _callable(getattr(world, "frame", None), "world.frame")()
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise UniVTACContractError("UIPC frame ID must be a non-negative integer")
    manager = _required(getattr(task, "_robot_manager", None), "task._robot_manager")
    robot = _required(getattr(manager, "robot", None), "robot_manager.robot")
    robot_data = _required(getattr(robot, "data", None), "robot.data")
    plate = _required(getattr(task, "plate", None), "task.plate")
    plate_data = _required(getattr(plate, "data", None), "plate.data")
    return canonical_hash(
        {
            "uipc_frame_id": frame,
            "robot_root_state": _tensor_array(
                robot_data.root_state_w, "robot root state"
            ),
            "robot_joint_position": _tensor_array(
                robot_data.joint_pos, "robot joint position"
            ),
            "robot_joint_velocity": _tensor_array(
                robot_data.joint_vel, "robot joint velocity"
            ),
            "robot_joint_position_target": _tensor_array(
                robot_data.joint_pos_target, "robot joint position target"
            ),
            "robot_joint_velocity_target": _tensor_array(
                robot_data.joint_vel_target, "robot joint velocity target"
            ),
            "plate_root_state": _tensor_array(
                plate_data.root_state_w, "plate root state"
            ),
            "rng_state": _witness_value(_rng_state(task), "rng_state"),
            "task_state": _witness_value(_task_state(task), "task_state"),
        }
    )


def _restore_snapshot(
    task: object,
    runtime_token: object,
    value: object,
) -> None:
    if not isinstance(value, _LiveUniVTACSnapshot):
        raise UniVTACContractError("runtime snapshot type mismatch")
    if value.runtime_token is not runtime_token:
        raise UniVTACContractError("runtime snapshot belongs to another task")
    uipc_sim = _required(getattr(task, "uipc_sim", None), "task.uipc_sim")
    world = _required(getattr(uipc_sim, "world", None), "uipc_sim.world")
    recovered = _callable(getattr(world, "recover", None), "world.recover")(
        value.uipc_frame_id
    )
    if not recovered:
        raise UniVTACContractError("UIPC canonical frame recovery failed")
    _callable(getattr(world, "retrieve", None), "world.retrieve")()
    _restore_physx(task, value)
    _restore_task_state(task, value)


def build_live_snapshot_callbacks(
    task: object,
) -> Tuple[Callable[[], object], Callable[[object], None], Callable[[], str]]:
    """Bind snapshot callbacks to one pinned UniVTAC task instance."""

    runtime_token = object()

    def capture() -> object:
        return _capture_snapshot(task, runtime_token)

    def restore(snapshot: object) -> None:
        _restore_snapshot(task, runtime_token, snapshot)

    def state_sha256() -> str:
        return _snapshot_state_sha256(task)

    return capture, restore, state_sha256
