"""Predicate witnesses for UniVTAC tasks not covered by the core diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError

_TACTILE_CONTACT_THRESHOLD = 20.0


def _pose(value: object, name: str) -> dict[str, object]:
    position = np.asarray(getattr(value, "p", None), dtype=np.float64)
    quaternion = np.asarray(getattr(value, "q", None), dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
    ):
        raise UniVTACContractError(f"{name} pose is unavailable for diagnostics")
    return {
        "position_xyz": [float(item) for item in position],
        "quaternion_wxyz": [float(item) for item in quaternion],
    }


def _matrix(value: object, name: str) -> NDArray[np.float64]:
    method = getattr(value, "to_transformation_matrix", None)
    if not callable(method):
        raise UniVTACContractError(f"{name} transform is unavailable")
    matrix = np.asarray(method(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise UniVTACContractError(f"{name} transform must be finite [4,4]")
    return cast(NDArray[np.float64], matrix)


def _relative_pose(value: object, target: object, name: str) -> object:
    rebase = getattr(value, "rebase", None)
    if not callable(rebase):
        raise UniVTACContractError(f"{name} pose cannot be rebased")
    relative = rebase(target)
    position = np.asarray(getattr(relative, "p", None), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise UniVTACContractError(f"{name} relative position is invalid")
    return relative


def _host_array(value: object) -> NDArray[np.float64]:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return cast(NDArray[np.float64], np.asarray(value, dtype=np.float64))


def _tactile_contact(task: Any) -> Mapping[str, object]:
    manager = getattr(task, "_tactile_manager", None)
    getter = getattr(manager, "get_min_depth", None)
    if not callable(getter):
        return {
            "available": False,
            "unavailable_reason": "tactile_min_depth_unavailable",
        }
    values = _host_array(getter())
    if values.size == 0 or not np.isfinite(values).all():
        raise UniVTACContractError("tactile minimum depth is empty or non-finite")
    minimum = float(np.min(values))
    return {
        "available": True,
        "contact_qualified": minimum >= _TACTILE_CONTACT_THRESHOLD,
        "minimum_depth": minimum,
        "threshold": _TACTILE_CONTACT_THRESHOLD,
    }


def _ee_z(robot_manager: object) -> float:
    getter = getattr(robot_manager, "get_ee_pose", None)
    if not callable(getter):
        raise UniVTACContractError("robot EE pose is unavailable")
    value = getter()
    position = getattr(value, "p", None)
    if position is None:
        position = value
    values = _host_array(position).reshape(-1)
    if values.size < 3 or not np.isfinite(values[:3]).all():
        raise UniVTACContractError("robot EE position is invalid")
    return float(values[2])


def _insert_hdmi(task: Any) -> Mapping[str, object]:
    prism = getattr(task, "prism", None)
    get_prism_pose = getattr(prism, "get_pose", None)
    target_pose = getattr(task, "target_pose", None)
    robot_manager = getattr(task, "_robot_manager", None)
    if not callable(get_prism_pose) or target_pose is None or robot_manager is None:
        return {
            "diagnostic_schema": "univtac-insert-hdmi-predicate-v1",
            "available": False,
            "unavailable_reason": "prism_target_or_robot_pose_unavailable",
        }
    prism_pose = get_prism_pose()
    relative_pose = _relative_pose(prism_pose, target_pose, "insert_HDMI prism")
    relative = np.asarray(getattr(relative_pose, "p", None), dtype=np.float64)
    orientation = float(np.dot(_matrix(relative_pose, "insert_HDMI")[:3, 2], (0, 0, 1)))
    ee_z = _ee_z(robot_manager)
    upstream_y_pass = bool(abs(relative[1]) < 0.005)
    corrected_xy_pass = bool(np.all(np.abs(relative[:2]) < (0.005, 0.005)))
    z_pass = bool(relative[2] < 0.005)
    ee_pass = bool(ee_z > 0.145)
    orientation_pass = bool(orientation > 0.965)
    common = z_pass and ee_pass and orientation_pass
    return {
        "diagnostic_schema": "univtac-insert-hdmi-predicate-v1",
        "available": True,
        "ee_z_m": ee_z,
        "orientation_alignment_to_world_z": orientation,
        "predicate_success": upstream_y_pass and common,
        "predicate_success_corrected_xy": corrected_xy_pass and common,
        "prism_pose": _pose(prism_pose, "insert_HDMI prism"),
        "relative_position_xyz": [float(item) for item in relative],
        "success_conditions": {
            "upstream_abs_relative_y_lt_0_005": upstream_y_pass,
            "corrected_abs_relative_xy_lt_0_005": corrected_xy_pass,
            "relative_z_lt_0_005": z_pass,
            "ee_z_gt_0_145": ee_pass,
            "orientation_alignment_gt_0_965": orientation_pass,
        },
        "target_pose": _pose(target_pose, "insert_HDMI target"),
        "tactile_contact": dict(_tactile_contact(task)),
        "upstream_x_coordinate_omitted": True,
    }


def _lift_can(task: Any) -> Mapping[str, object]:
    can = getattr(task, "can", None)
    get_can_pose = getattr(can, "get_pose", None)
    origin_inhand = getattr(task, "origin_inhand_pose", None)
    robot_manager = getattr(task, "_robot_manager", None)
    get_inhand = getattr(robot_manager, "get_inhand_pose", None)
    if not callable(get_can_pose) or origin_inhand is None or not callable(get_inhand):
        return {
            "diagnostic_schema": "univtac-lift-can-predicate-v1",
            "available": False,
            "unavailable_reason": "can_origin_or_robot_pose_unavailable",
        }
    can_pose = get_can_pose()
    current_inhand = get_inhand(can)
    origin_z = float(np.asarray(origin_inhand.p, dtype=np.float64)[2])
    current_z = float(np.asarray(current_inhand.p, dtype=np.float64)[2])
    inhand_z_bias = abs(current_z - origin_z)
    matrix = _matrix(can_pose, "lift_can")
    x_to_world_z = abs(float(np.dot(matrix[:3, 0], (0, 0, 1))))
    z_to_world_z = abs(float(np.dot(matrix[:3, 2], (0, 0, 1))))
    tactile = dict(_tactile_contact(task))
    tactile_stop = bool(tactile.get("available") and not tactile["contact_qualified"])
    inhand_stop = bool(inhand_z_bias > 0.05 and z_to_world_z > 0.99)
    reasons = [
        reason
        for reason, active in (
            ("tactile_min_depth_lt_20", tactile_stop),
            ("inhand_z_bias_gt_0_05_while_upright", inhand_stop),
        )
        if active
    ]
    z_pass = bool(float(can_pose.p[2]) < 0.01)
    orientation_pass = bool(x_to_world_z > 0.99)
    return {
        "diagnostic_schema": "univtac-lift-can-predicate-v1",
        "available": True,
        "can_pose": _pose(can_pose, "lift_can can"),
        "current_inhand_pose": _pose(current_inhand, "lift_can current in-hand"),
        "early_stop_predicate": bool(reasons),
        "early_stop_reasons": reasons,
        "inhand_z_bias_m": inhand_z_bias,
        "orientation_alignment_x_to_world_z": x_to_world_z,
        "orientation_alignment_z_to_world_z": z_to_world_z,
        "origin_inhand_pose": _pose(origin_inhand, "lift_can origin in-hand"),
        "predicate_success": z_pass and orientation_pass,
        "success_conditions": {
            "can_z_lt_0_01": z_pass,
            "abs_x_axis_alignment_gt_0_99": orientation_pass,
        },
        "tactile_contact": tactile,
    }


def _pull_out_key(task: Any) -> Mapping[str, object]:
    key = getattr(task, "key", None)
    slot = getattr(task, "slot", None)
    get_key_pose = getattr(key, "get_pose", None)
    get_slot_pose = getattr(slot, "get_pose", None)
    slot_init_pose = getattr(task, "slot_init_pose", None)
    robot_manager = getattr(task, "_robot_manager", None)
    if (
        not callable(get_key_pose)
        or not callable(get_slot_pose)
        or slot_init_pose is None
        or robot_manager is None
    ):
        return {
            "diagnostic_schema": "univtac-pull-out-key-predicate-v1",
            "available": False,
            "unavailable_reason": "key_slot_or_robot_pose_unavailable",
        }
    key_pose = get_key_pose()
    slot_pose = get_slot_pose()
    slot_relative = _relative_pose(slot_pose, slot_init_pose, "pull_out_key slot")
    slot_alignment = float(
        np.dot(_matrix(slot_relative, "pull_out_key slot")[:3, 0], (1, 0, 0))
    )
    key_alignment = float(
        np.dot(_matrix(key_pose, "pull_out_key key")[:3, 2], (0, 0, 1))
    )
    z_distance = abs(_ee_z(robot_manager) - float(key_pose.p[2]))
    distance_stop = bool(z_distance >= 0.14)
    slot_stop = bool(slot_alignment < 0.99)
    reasons = [
        reason
        for reason, active in (
            ("ee_key_z_distance_gte_0_14", distance_stop),
            ("slot_x_alignment_lt_0_99", slot_stop),
        )
        if active
    ]
    height_pass = bool(float(key_pose.p[2]) > 0.09)
    key_orientation_pass = bool(key_alignment > 0.965)
    slot_pass = bool(slot_alignment > 0.99)
    distance_pass = bool(z_distance < 0.14)
    return {
        "diagnostic_schema": "univtac-pull-out-key-predicate-v1",
        "available": True,
        "early_stop_predicate": bool(reasons),
        "early_stop_reasons": reasons,
        "ee_key_z_distance_m": z_distance,
        "key_orientation_alignment_to_world_z": key_alignment,
        "key_pose": _pose(key_pose, "pull_out_key key"),
        "predicate_success": height_pass
        and key_orientation_pass
        and slot_pass
        and distance_pass,
        "slot_init_pose": _pose(slot_init_pose, "pull_out_key initial slot"),
        "slot_pose": _pose(slot_pose, "pull_out_key slot"),
        "slot_x_alignment_to_initial": slot_alignment,
        "success_conditions": {
            "key_z_gt_0_09": height_pass,
            "key_orientation_alignment_gt_0_965": key_orientation_pass,
            "slot_x_alignment_gt_0_99": slot_pass,
            "ee_key_z_distance_lt_0_14": distance_pass,
        },
        "tactile_contact": dict(_tactile_contact(task)),
    }


def _put_bottle_in_shelf(task: Any) -> Mapping[str, object]:
    bottle = getattr(task, "bottle", None)
    get_bottle_pose = getattr(bottle, "get_pose", None)
    target_pose = getattr(task, "place_target", None)
    if not callable(get_bottle_pose) or target_pose is None:
        return {
            "diagnostic_schema": "univtac-put-bottle-in-shelf-predicate-v1",
            "available": False,
            "unavailable_reason": "bottle_or_place_target_unavailable",
        }
    bottle_pose = get_bottle_pose()
    relative_pose = _relative_pose(bottle_pose, target_pose, "shelf bottle")
    relative = np.asarray(getattr(relative_pose, "p", None), dtype=np.float64)
    orientation = float(
        np.dot(_matrix(relative_pose, "shelf bottle")[:3, 2], (0, 0, 1))
    )
    limits = np.asarray((0.02, 0.1, 0.02), dtype=np.float64)
    position_pass = np.abs(relative) < limits
    orientation_pass = bool(orientation > 0.965)
    tactile = dict(_tactile_contact(task))
    tactile_stop = bool(tactile.get("available") and not tactile["contact_qualified"])
    return {
        "diagnostic_schema": "univtac-put-bottle-in-shelf-predicate-v1",
        "available": True,
        "bottle_pose": _pose(bottle_pose, "shelf bottle"),
        "early_stop_predicate": tactile_stop,
        "early_stop_reasons": ["tactile_min_depth_lt_20"] if tactile_stop else [],
        "orientation_alignment_to_world_z": orientation,
        "place_target_pose": _pose(target_pose, "shelf place target"),
        "predicate_success": bool(np.all(position_pass) and orientation_pass),
        "relative_position_xyz": [float(item) for item in relative],
        "success_conditions": {
            "abs_relative_x_lt_0_02": bool(position_pass[0]),
            "abs_relative_y_lt_0_1": bool(position_pass[1]),
            "abs_relative_z_lt_0_02": bool(position_pass[2]),
            "orientation_alignment_gt_0_965": orientation_pass,
        },
        "tactile_contact": tactile,
    }


def capture_extended_task_diagnostics(
    task: Any, task_id: str
) -> Mapping[str, object] | None:
    """Return a pinned predicate witness, or ``None`` for an unknown task."""

    handlers = {
        "insert_HDMI": _insert_hdmi,
        "lift_can": _lift_can,
        "pull_out_key": _pull_out_key,
        "put_bottle_in_shelf": _put_bottle_in_shelf,
    }
    handler = handlers.get(task_id)
    return None if handler is None else handler(task)


__all__ = ["capture_extended_task_diagnostics"]
