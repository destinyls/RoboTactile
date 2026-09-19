"""Live capture/replay hooks for source-bound UniVTAC pre-move trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import wraps
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Callable, Tuple, cast

import numpy as np

from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveSegment,
    UniVTACPreMoveTrajectory,
    UniVTACPreMoveTrajectoryError,
)
from robotactile_benchmark.contracts import canonical_hash

if TYPE_CHECKING:
    from robotactile_benchmark.backends.univtac_reset_trajectory import (
        PreMoveTrajectoryCapture,
    )

_START_QPOS_ATOL = 1e-5


def _seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise UniVTACPreMoveTrajectoryError(f"{name} must be non-negative integer")
    return int(value)


def _optional_expected_native_step(value: object) -> int | None:
    if value is None:
        return None
    return _seed(value, "expected_native_step")


def _native_step(task: Any) -> int:
    return _seed(getattr(task, "step_count", None), "task.step_count")


def _array(value: object, name: str) -> np.ndarray[Any, Any]:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        result = np.asarray(current, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UniVTACPreMoveTrajectoryError(f"{name} is not numeric") from exc
    if not np.isfinite(result).all():
        raise UniVTACPreMoveTrajectoryError(f"{name} must be finite")
    return cast(np.ndarray[Any, Any], result)


def _joint9(manager: Any) -> Tuple[float, ...]:
    get_qpos = getattr(manager, "get_qpos", None)
    if not callable(get_qpos):
        raise UniVTACPreMoveTrajectoryError("robot manager get_qpos is unavailable")
    value = _array(get_qpos(), "joint9")
    if value.shape == (1, 9):
        value = value[0]
    if value.shape != (9,):
        raise UniVTACPreMoveTrajectoryError("live joint state must have shape [9]")
    return tuple(float(item) for item in value)


def _json_value(value: object, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise UniVTACPreMoveTrajectoryError(f"{name} must be finite")
        return normalized
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _json_value(to_list(), name)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{name}.{key}") for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item, f"{name}[]") for item in value]
    raise UniVTACPreMoveTrajectoryError(f"{name} is not fingerprintable")


def _fingerprint(
    kind: str, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> str:
    if kind == "gripper":
        values = {
            "pos": args[0] if args else kwargs.get("pos"),
            "type": args[1] if len(args) > 1 else kwargs.get("type", "percent"),
        }
    else:
        values = {
            "target_pose": args[0] if args else kwargs.get("target_pose"),
            "constraint_pose": (
                args[1] if len(args) > 1 else kwargs.get("constraint_pose")
            ),
            "pre_dis": args[2] if len(args) > 2 else kwargs.get("pre_dis"),
            "time_dilation_factor": (
                args[3] if len(args) > 3 else kwargs.get("time_dilation_factor")
            ),
        }
    return canonical_hash(
        {
            "kind": kind,
            **{key: _json_value(item, key) for key, item in values.items()},
        }
    )


def _capture_segment(
    kind: str,
    fingerprint: str,
    start: Tuple[float, ...],
    result: object,
) -> UniVTACPreMoveSegment:
    if (
        not isinstance(result, Mapping)
        or str(result.get("status", "")).lower() != "success"
    ):
        raise UniVTACPreMoveTrajectoryError(f"{kind} planner did not succeed")
    steps = result.get("num_steps")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) <= 0:
        raise UniVTACPreMoveTrajectoryError(f"{kind} num_steps is invalid")
    width = 1 if kind == "gripper" else 7
    position = _array(result.get("position"), f"{kind}.position")
    velocity = _array(result.get("velocity"), f"{kind}.velocity")
    if kind == "gripper":
        if position.shape == (int(steps),):
            position = position.reshape(-1, 1)
        if velocity.shape == (int(steps),):
            velocity = velocity.reshape(-1, 1)
    expected = (int(steps), width)
    if position.shape != expected or velocity.shape != expected:
        raise UniVTACPreMoveTrajectoryError(
            f"{kind} planner tensors must have shape {expected}"
        )

    def rows(value: np.ndarray[Any, Any]) -> Tuple[Tuple[float, ...], ...]:
        return tuple(tuple(float(item) for item in row) for row in value)

    return UniVTACPreMoveSegment(
        kind=kind,
        request_fingerprint=fingerprint,
        start_joint9=start,
        position=rows(position),
        velocity=rows(velocity),
    )


def _surface(
    task: Any,
) -> tuple[Any, Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    manager = getattr(task, "_robot_manager", None)
    reset = getattr(task, "reset", None)
    gripper = getattr(manager, "plan_gripper", None)
    arm = getattr(manager, "plan_arm", None)
    if (
        manager is None
        or not callable(reset)
        or not callable(gripper)
        or not callable(arm)
        or not isinstance(getattr(task, "in_pre_move", None), bool)
    ):
        raise UniVTACPreMoveTrajectoryError("task planner surface is incomplete")
    if getattr(task, "_robotactile_pre_move_trajectory_mode", None) is not None:
        raise UniVTACPreMoveTrajectoryError("trajectory hook already installed")
    return (
        manager,
        cast(Callable[..., Any], reset),
        cast(Callable[..., Any], gripper),
        cast(Callable[..., Any], arm),
    )


def install_pre_move_trajectory_capture(
    task: Any, expected_native_step: int | None = None
) -> PreMoveTrajectoryCapture:
    """Capture planner results during one explicit-seed task reset."""
    from robotactile_benchmark.backends.univtac_reset_trajectory import (
        PreMoveTrajectoryCapture,
    )

    expected_native_step = _optional_expected_native_step(expected_native_step)
    manager, upstream_reset, upstream_gripper, upstream_arm = _surface(task)
    capture = PreMoveTrajectoryCapture(task, manager)

    def wrap_plan(kind: str, upstream: Callable[..., Any]) -> Callable[..., Any]:
        def planned(*args: object, **kwargs: object) -> Any:
            if not capture._active or getattr(task, "in_pre_move", None) is not True:
                return upstream(*args, **kwargs)
            fingerprint = _fingerprint(kind, args, kwargs)
            start = _joint9(manager)
            result = upstream(*args, **kwargs)
            capture._record(_capture_segment(kind, fingerprint, start, result))
            return result

        return planned

    @wraps(upstream_reset)
    def reset(*args: object, **kwargs: object) -> Any:
        capture._begin(args, kwargs)
        try:
            result = upstream_reset(*args, **kwargs)
            capture._validate_segments()
            hold_steps = 0
            if expected_native_step is not None:
                hold_steps = expected_native_step - _native_step(task)
                if hold_steps < 0 or hold_steps > 2:
                    raise UniVTACPreMoveTrajectoryError(
                        "expected_native_step delta must be between 0 and 2"
                    )
            if hold_steps:
                _post_reset_hold(task, manager, capture._segments[-1], hold_steps)
            capture._end(hold_steps)
            return result
        except BaseException:
            capture._abort()
            raise

    manager.plan_gripper = wrap_plan("gripper", upstream_gripper)
    manager.plan_arm = wrap_plan("arm", upstream_arm)
    task.reset = reset
    task._robotactile_pre_move_trajectory_mode = "capture"
    task._robotactile_pre_move_trajectory_capture = capture
    return capture


def _materialize(manager: Any, segment: UniVTACPreMoveSegment) -> dict[str, object]:
    robot = getattr(manager, "robot", None)
    data = getattr(robot, "data", None)
    template = getattr(data, "joint_pos", None)
    new_tensor = getattr(template, "new_tensor", None)
    if not callable(new_tensor):
        raise UniVTACPreMoveTrajectoryError("cannot materialize live replay tensor")
    position = new_tensor([list(row) for row in segment.position])
    velocity = new_tensor([list(row) for row in segment.velocity])
    if segment.kind == "gripper":
        position = position.reshape(-1)
        velocity = velocity.reshape(-1)
    return {
        "status": "Success",
        "num_steps": len(segment.position),
        "position": position,
        "velocity": velocity,
    }


def _post_reset_hold(
    task: Any,
    manager: Any,
    segment: UniVTACPreMoveSegment,
    hold_steps: int,
) -> tuple[int, int]:
    if hold_steps < 0 or hold_steps > 2:
        raise UniVTACPreMoveTrajectoryError("post-reset hold must be between 0 and 2")
    before = _native_step(task)
    if segment.kind != "arm":
        raise UniVTACPreMoveTrajectoryError(
            "post-reset hold requires final arm segment"
        )
    if hold_steps == 0:
        return before, before
    set_arm = getattr(manager, "set_arm", None)
    step = getattr(task, "_step", None)
    template = getattr(
        getattr(getattr(manager, "robot", None), "data", None), "joint_pos", None
    )
    new_tensor = getattr(template, "new_tensor", None)
    if not callable(set_arm) or not callable(step) or not callable(new_tensor):
        raise UniVTACPreMoveTrajectoryError("post-reset hold surface is incomplete")
    position = new_tensor(list(segment.position[-1]))
    velocity = new_tensor(list(segment.velocity[-1]))
    for _ in range(hold_steps):
        set_arm(position, velocity, force=False)
        step(is_save=False)
    after = _native_step(task)
    if after != before + hold_steps:
        raise UniVTACPreMoveTrajectoryError("post-reset hold cadence mismatch")
    return before, after


def install_pre_move_trajectory_replay(
    task: Any, trajectory: UniVTACPreMoveTrajectory | None
) -> bool:
    """Replay dense plans while retaining upstream move/delay/physics."""
    if trajectory is None:
        return False
    if not isinstance(trajectory, UniVTACPreMoveTrajectory):
        raise TypeError("trajectory must be a UniVTACPreMoveTrajectory or None")
    manager, upstream_reset, upstream_gripper, upstream_arm = _surface(task)
    active = False
    cursor = 0
    request_fingerprint_checks: list[dict[str, object]] = []

    def wrap_plan(kind: str, upstream: Callable[..., Any]) -> Callable[..., Any]:
        def planned(*args: object, **kwargs: object) -> Any:
            nonlocal cursor
            if not active or getattr(task, "in_pre_move", None) is not True:
                return upstream(*args, **kwargs)
            if cursor >= 4 or trajectory.segments[cursor].kind != kind:
                raise UniVTACPreMoveTrajectoryError("trajectory segment order mismatch")
            segment = trajectory.segments[cursor]
            actual_fingerprint = _fingerprint(kind, args, kwargs)
            request_fingerprint_checks.append(
                {
                    "segment_index": cursor,
                    "kind": kind,
                    "expected": segment.request_fingerprint,
                    "actual": actual_fingerprint,
                    "exact_match": actual_fingerprint == segment.request_fingerprint,
                }
            )
            # Dynamic target poses include physics-derived floating values.
            # Their exact hash is diagnostic; source identity, call order,
            # start joints, and the capture-derived endpoint remain hard gates.
            if not np.allclose(
                _joint9(manager),
                segment.start_joint9,
                rtol=0.0,
                atol=_START_QPOS_ATOL,
            ):
                raise UniVTACPreMoveTrajectoryError("trajectory start_joint9 mismatch")
            cursor += 1
            return _materialize(manager, segment)

        return planned

    @wraps(upstream_reset)
    def reset(*args: object, **kwargs: object) -> Any:
        nonlocal active, cursor
        reset_seed = _seed(kwargs.get("seed", args[0] if args else None), "reset seed")
        if reset_seed != trajectory.initial_seed:
            raise UniVTACPreMoveTrajectoryError("trajectory reset seed mismatch")
        active, cursor = True, 0
        request_fingerprint_checks.clear()
        try:
            result = upstream_reset(*args, **kwargs)
            if cursor != 4:
                raise UniVTACPreMoveTrajectoryError("trajectory was not fully consumed")
            before_hold, after_hold = _post_reset_hold(
                task,
                manager,
                trajectory.segments[-1],
                trajectory.post_reset_hold_steps,
            )
            task._robotactile_pre_move_trajectory_replay_witness = {
                "consumed_segment_count": cursor,
                "post_reset_hold_steps": trajectory.post_reset_hold_steps,
                "pre_hold_native_step": before_hold,
                "post_hold_native_step": after_hold,
                "request_fingerprint_all_match": all(
                    item["exact_match"] is True for item in request_fingerprint_checks
                ),
                "request_fingerprint_checks": tuple(
                    dict(item) for item in request_fingerprint_checks
                ),
                "trajectory_sha256": trajectory.sha256,
            }
            return result
        finally:
            active = False

    manager.plan_gripper = wrap_plan("gripper", upstream_gripper)
    manager.plan_arm = wrap_plan("arm", upstream_arm)
    task.reset = reset
    task._robotactile_pre_move_trajectory_mode = "replay"
    task._robotactile_pre_move_trajectory_sha256 = trajectory.sha256
    return True


__all__ = [
    "install_pre_move_trajectory_capture",
    "install_pre_move_trajectory_replay",
]
