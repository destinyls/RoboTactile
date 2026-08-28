"""Explicit HDF5 actor placement for teacher-forced simulator diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.integrations.n0_twam.teacher_forced_replay import (
    native_step,
)

_ACTOR_NAMES = frozenset({"bottle", "wall"})
_ANCHOR_TRANSLATION_GATE_M = 0.002
_ANCHOR_ROTATION_GATE_DEG = 1.0
_PLACEMENT_TRANSLATION_GATE_M = 0.003
_PLACEMENT_ROTATION_GATE_DEG = 2.0
_CONSTRAINED_SETTLE_STEPS = 20
_RELEASE_STEPS = 5


def _expected_pose(value: object, label: str) -> NDArray[np.float64]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must be a finite 7D pose")
    norm = float(np.linalg.norm(pose[3:]))
    if not np.isclose(norm, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(f"{label} quaternion must have unit norm")
    return cast(NDArray[np.float64], pose)


def _live_pose(actor: Any, label: str) -> tuple[Any, NDArray[np.float64]]:
    get_pose = getattr(actor, "get_pose", None)
    if not callable(get_pose):
        raise RuntimeError(f"{label} actor does not expose get_pose")
    pose = get_pose()
    to_list = getattr(pose, "tolist", None)
    if not callable(to_list):
        raise RuntimeError(f"{label} live pose does not expose tolist")
    return pose, _expected_pose(to_list(), f"{label} live pose")


def _pose_error(
    expected: NDArray[np.float64], observed: NDArray[np.float64]
) -> dict[str, float]:
    translation = float(np.linalg.norm(expected[:3] - observed[:3]))
    dot = abs(float(np.dot(expected[3:], observed[3:])))
    rotation = float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))
    return {
        "rotation_geodesic_deg": rotation,
        "translation_l2_m": translation,
    }


def _pose_document(value: NDArray[np.float64]) -> dict[str, list[float]]:
    return {
        "position_xyz": [float(item) for item in value[:3]],
        "quaternion_wxyz": [float(item) for item in value[3:]],
    }


def apply_lift_bottle_hdf5_initial_placement(
    task: Any,
    actor_poses: Mapping[str, object],
) -> dict[str, object]:
    """Place the bottle at its expert initial pose, then release dynamics.

    The fixed wall is never moved. Its HDF5 pose is a coordinate-frame gate.
    The settle/release sequence mirrors the pinned UniVTAC reset lifecycle and
    is intentionally available only to a teacher-forced diagnostic.
    """

    if set(actor_poses) != _ACTOR_NAMES:
        raise ValueError("lift_bottle placement requires bottle and wall poses")
    if getattr(task, "mode", None) != "eval":
        raise RuntimeError("teacher-forced actor placement requires eval mode")
    bottle = getattr(task, "bottle", None)
    wall = getattr(task, "wall", None)
    actor_manager = getattr(task, "_actor_manager", None)
    step = getattr(task, "_step", None)
    render = getattr(task, "_update_render", None)
    remove_animate = getattr(actor_manager, "remove_animate", None)
    if (
        bottle is None
        or wall is None
        or not callable(step)
        or not callable(render)
        or not callable(remove_animate)
    ):
        raise RuntimeError("lift_bottle actor placement surface is incomplete")

    expected_bottle = _expected_pose(actor_poses["bottle"], "HDF5 bottle pose")
    expected_wall = _expected_pose(actor_poses["wall"], "HDF5 wall pose")
    bottle_pose, before_bottle = _live_pose(bottle, "bottle")
    _, before_wall = _live_pose(wall, "wall")
    wall_error = _pose_error(expected_wall, before_wall)
    anchor_passed = bool(
        wall_error["translation_l2_m"] <= _ANCHOR_TRANSLATION_GATE_M
        and wall_error["rotation_geodesic_deg"] <= _ANCHOR_ROTATION_GATE_DEG
    )
    if not anchor_passed:
        raise RuntimeError("live wall does not match the HDF5 coordinate frame")

    pose_type = type(bottle_pose)
    from_list = getattr(pose_type, "from_list", None)
    set_pose = getattr(bottle, "set_pose", None)
    if not callable(from_list) or not callable(set_pose):
        raise RuntimeError("bottle does not expose the pinned pose placement API")

    native_before = native_step(task)
    set_pose(from_list(expected_bottle))
    for _ in range(_CONSTRAINED_SETTLE_STEPS):
        step(is_save=False)
    render()
    _, constrained_bottle = _live_pose(bottle, "constrained bottle")
    constrained_error = _pose_error(expected_bottle, constrained_bottle)

    remove_animate()
    for _ in range(_RELEASE_STEPS):
        step(is_save=False)
    render()
    _, released_bottle = _live_pose(bottle, "released bottle")
    _, released_wall = _live_pose(wall, "released wall")
    released_error = _pose_error(expected_bottle, released_bottle)
    released_wall_error = _pose_error(expected_wall, released_wall)
    native_after = native_step(task)
    dynamics_released = getattr(bottle, "next_status", None) is None
    constrained_passed = bool(
        constrained_error["translation_l2_m"] <= _PLACEMENT_TRANSLATION_GATE_M
        and constrained_error["rotation_geodesic_deg"] <= _PLACEMENT_ROTATION_GATE_DEG
    )
    released_passed = bool(
        released_error["translation_l2_m"] <= _PLACEMENT_TRANSLATION_GATE_M
        and released_error["rotation_geodesic_deg"] <= _PLACEMENT_ROTATION_GATE_DEG
        and released_wall_error["translation_l2_m"] <= _ANCHOR_TRANSLATION_GATE_M
        and released_wall_error["rotation_geodesic_deg"] <= _ANCHOR_ROTATION_GATE_DEG
    )
    step_contract_passed = (
        native_after - native_before == _CONSTRAINED_SETTLE_STEPS + _RELEASE_STEPS
    )
    alignment_passed = bool(
        anchor_passed
        and constrained_passed
        and released_passed
        and dynamics_released
        and step_contract_passed
    )
    witness: dict[str, object] = {
        "alignment_passed": alignment_passed,
        "anchor_actor": "wall",
        "anchor_error_before": wall_error,
        "anchor_error_released": released_wall_error,
        "bottle_error_constrained": constrained_error,
        "bottle_error_released": released_error,
        "bottle_pose_before": _pose_document(before_bottle),
        "bottle_pose_constrained": _pose_document(constrained_bottle),
        "bottle_pose_hdf5": _pose_document(expected_bottle),
        "bottle_pose_released": _pose_document(released_bottle),
        "constrained_settle_steps": _CONSTRAINED_SETTLE_STEPS,
        "dynamics_released": dynamics_released,
        "evidence_level": "hdf5_initial_actor_placement_diagnostic_v1",
        "movement_from_reset_m": float(
            np.linalg.norm(released_bottle[:3] - before_bottle[:3])
        ),
        "native_step_after": native_after,
        "native_step_before": native_before,
        "native_step_delta": native_after - native_before,
        "release_steps": _RELEASE_STEPS,
        "render_updates": 2,
        "task_id": "lift_bottle",
        "wall_pose_hdf5": _pose_document(expected_wall),
        "wall_pose_live": _pose_document(released_wall),
    }
    if not alignment_passed:
        raise RuntimeError(f"HDF5 actor placement failed its witness gates: {witness}")
    return witness


__all__ = ["apply_lift_bottle_hdf5_initial_placement"]
