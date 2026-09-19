"""Read-only task-specific witnesses for pinned UniVTAC success predicates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.backends.univtac_extended_task_diagnostics import (
    capture_extended_task_diagnostics,
)
from robotactile_benchmark.backends.univtac_success_profiles import (
    INSERT_HOLE_STRICT_THRESHOLDS,
)

_INSERTION_PLACEMENT_PHASES = (
    "approach_complete",
    "final_continuation_complete",
)


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


def _lift_bottle(task: Any) -> Mapping[str, object]:
    bottle = getattr(task, "bottle", None)
    get_pose = getattr(bottle, "get_pose", None)
    target_pose = getattr(task, "target_pose", None)
    tactile_attachment = getattr(task, "_robotactile_tactile_attachment", None)
    if (
        not callable(get_pose)
        or target_pose is None
        or not isinstance(tactile_attachment, Mapping)
    ):
        return {
            "diagnostic_schema": "univtac-lift-bottle-predicate-v1",
            "available": False,
            "unavailable_reason": "task_actors_or_tactile_attachment_unavailable",
        }
    bottle_pose = get_pose()
    rebase = getattr(bottle_pose, "rebase", None)
    if not callable(rebase):
        raise UniVTACContractError("lift_bottle pose cannot be rebased")
    relative_pose = rebase(target_pose)
    relative = np.asarray(getattr(relative_pose, "p", None), dtype=np.float64)
    matrix_method = getattr(relative_pose, "to_transformation_matrix", None)
    if (
        relative.shape != (3,)
        or not np.isfinite(relative).all()
        or not callable(matrix_method)
    ):
        raise UniVTACContractError("lift_bottle relative pose is invalid")
    matrix = np.asarray(matrix_method(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise UniVTACContractError("lift_bottle relative transform is invalid")
    orientation_alignment = abs(float(np.dot(matrix[:3, 0], (0.0, 0.0, 1.0))))
    early_stop_alignment = abs(float(np.dot(matrix[:3, 0], (-1.0, 0.0, 0.0))))
    x_pass = bool(relative[0] > -0.02)
    y_pass = bool(abs(relative[1]) < 0.1)
    z_pass = bool(abs(relative[2]) < 0.001)
    orientation_pass = bool(orientation_alignment > 0.99)
    return {
        "diagnostic_schema": "univtac-lift-bottle-predicate-v1",
        "available": True,
        "bottle_pose": _pose(bottle_pose, "bottle"),
        "target_pose": _pose(target_pose, "target"),
        "tactile_attachment": dict(tactile_attachment),
        "relative_position_xyz": [float(item) for item in relative],
        "orientation_alignment_to_world_z": orientation_alignment,
        "early_stop_alignment_to_negative_world_x": early_stop_alignment,
        "success_conditions": {
            "relative_x_gt_negative_0_02": x_pass,
            "abs_relative_y_lt_0_1": y_pass,
            "abs_relative_z_lt_0_001": z_pass,
            "orientation_alignment_gt_0_99": orientation_pass,
        },
        "predicate_success": x_pass and y_pass and z_pass and orientation_pass,
    }


def _grasp_classify(task: Any) -> Mapping[str, object]:
    prism = getattr(task, "prism", None)
    get_pose = getattr(prism, "get_pose", None)
    target_pose = getattr(task, "target_pose", None)
    initialization = getattr(task, "_robotactile_grasp_initialization", None)
    tactile_attachment = getattr(task, "_robotactile_tactile_attachment", None)
    robot_manager = getattr(task, "_robot_manager", None)
    get_gripper_pose = getattr(robot_manager, "get_gripper_center_pose", None)
    if not callable(get_pose) or target_pose is None or not callable(get_gripper_pose):
        return {
            "diagnostic_schema": "univtac-grasp-classify-predicate-v1",
            "available": False,
            "unavailable_reason": "task_actors_or_robot_pose_unavailable",
        }
    if initialization is not None and not isinstance(initialization, Mapping):
        raise UniVTACContractError(
            "grasp_classify initialization diagnostics must be a mapping"
        )
    if tactile_attachment is not None and not isinstance(tactile_attachment, Mapping):
        raise UniVTACContractError(
            "grasp_classify tactile attachment diagnostics must be a mapping"
        )
    prism_pose = get_pose()
    rebase = getattr(prism_pose, "rebase", None)
    if not callable(rebase):
        raise UniVTACContractError("grasp_classify pose cannot be rebased")
    relative_pose = rebase(target_pose)
    relative = np.asarray(getattr(relative_pose, "p", None), dtype=np.float64)
    matrix_method = getattr(relative_pose, "to_transformation_matrix", None)
    if (
        relative.shape != (3,)
        or not np.isfinite(relative).all()
        or not callable(matrix_method)
    ):
        raise UniVTACContractError("grasp_classify relative pose is invalid")
    matrix = np.asarray(matrix_method(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise UniVTACContractError("grasp_classify relative transform is invalid")
    orientation_alignment = float(np.dot(matrix[:3, 2], (0.0, 0.0, 1.0)))
    limits = np.asarray((0.02, 0.02, 0.01), dtype=np.float64)
    position_pass = np.abs(relative) < limits
    orientation_pass = orientation_alignment > 0.965
    predicate_success = bool(np.all(position_pass) and orientation_pass)
    return {
        "diagnostic_schema": "univtac-grasp-classify-predicate-v1",
        "available": True,
        "gripper_center_pose": _pose(get_gripper_pose(), "gripper center"),
        "initialization": (None if initialization is None else dict(initialization)),
        "orientation_alignment_to_world_z": orientation_alignment,
        "predicate_success": predicate_success,
        "prism_pose": _pose(prism_pose, "prism"),
        "relative_position_xyz": [float(item) for item in relative],
        "success_conditions": {
            "abs_relative_x_lt_0_02": bool(position_pass[0]),
            "abs_relative_y_lt_0_02": bool(position_pass[1]),
            "abs_relative_z_lt_0_01": bool(position_pass[2]),
            "orientation_alignment_gt_0_965": orientation_pass,
        },
        "target_pose": _pose(target_pose, "target"),
        "tactile_attachment": (
            None if tactile_attachment is None else dict(tactile_attachment)
        ),
    }


def _insertion_inhand(
    task: Any,
    *,
    task_id: str,
    early_stop_threshold_m: float,
    include_placement_assessment: bool,
) -> Mapping[str, object]:
    prism = getattr(task, "prism", None)
    get_prism_pose = getattr(prism, "get_pose", None)
    origin_inhand_pose = getattr(task, "origin_inhand_pose", None)
    robot_manager = getattr(task, "_robot_manager", None)
    get_gripper_pose = getattr(robot_manager, "get_gripper_center_pose", None)
    if (
        not callable(get_prism_pose)
        or origin_inhand_pose is None
        or not callable(get_gripper_pose)
    ):
        return {
            "diagnostic_schema": f"univtac-{task_id.replace('_', '-')}-inhand-v1",
            "available": False,
            "unavailable_reason": "prism_origin_or_gripper_pose_unavailable",
        }
    prism_pose = get_prism_pose()
    gripper_pose = get_gripper_pose()
    rebase = getattr(prism_pose, "rebase", None)
    if not callable(rebase):
        raise UniVTACContractError(f"{task_id} prism pose cannot be rebased")
    current_inhand_pose = rebase(gripper_pose)
    origin_position = np.asarray(
        getattr(origin_inhand_pose, "p", None), dtype=np.float64
    )
    current_position = np.asarray(
        getattr(current_inhand_pose, "p", None), dtype=np.float64
    )
    if (
        origin_position.shape != (3,)
        or current_position.shape != (3,)
        or not np.isfinite(origin_position).all()
        or not np.isfinite(current_position).all()
    ):
        raise UniVTACContractError(f"{task_id} in-hand pose is invalid")
    inhand_z_bias_m = abs(float(origin_position[2] - current_position[2]))
    result: dict[str, object] = {
        "diagnostic_schema": f"univtac-{task_id.replace('_', '-')}-inhand-v1",
        "available": True,
        "current_inhand_pose": _pose(current_inhand_pose, "current in-hand"),
        "early_stop_predicate": inhand_z_bias_m > early_stop_threshold_m,
        "early_stop_threshold_m": early_stop_threshold_m,
        "gripper_center_pose": _pose(gripper_pose, "gripper center"),
        "inhand_z_bias_m": inhand_z_bias_m,
        "origin_inhand_pose": _pose(origin_inhand_pose, "origin in-hand"),
        "prism_pose": _pose(prism_pose, "prism"),
        "source_expression": (
            "abs(origin_inhand_pose[2] - current_inhand_pose[2]) > threshold"
        ),
    }
    if task_id == "insert_hole":
        result.update(
            _insert_hole_success_metrics(
                task=task,
                prism_pose=prism_pose,
                inhand_z_drift_m=inhand_z_bias_m,
            )
        )
    placement_witnesses = getattr(task, "_robotactile_placement_witnesses", None)
    if include_placement_assessment and placement_witnesses is not None:
        if not isinstance(placement_witnesses, tuple) or not all(
            isinstance(item, Mapping) for item in placement_witnesses
        ):
            raise UniVTACContractError(f"{task_id} placement witnesses are invalid")
        normalized = tuple(dict(item) for item in placement_witnesses)
        result["placement_witnesses"] = list(normalized)
        result["placement_reset_assessment"] = _placement_reset_assessment(
            task_id=task_id,
            current_early_stop=bool(result["early_stop_predicate"]),
            witnesses=normalized,
        )
    return result


def _insert_hole_success_metrics(
    *,
    task: Any,
    prism_pose: object,
    inhand_z_drift_m: float,
) -> Mapping[str, object]:
    """Recompute official and strict insertion geometry from one live pose."""

    target_pose = getattr(task, "target_pose", None)
    rebase = getattr(prism_pose, "rebase", None)
    if target_pose is None or not callable(rebase):
        return {
            "success_metrics_available": False,
            "success_metrics_unavailable_reason": "target_pose_unavailable",
        }
    relative_pose = rebase(target_pose)
    relative = np.asarray(getattr(relative_pose, "p", None), dtype=np.float64)
    matrix_method = getattr(relative_pose, "to_transformation_matrix", None)
    if (
        relative.shape != (3,)
        or not np.isfinite(relative).all()
        or not callable(matrix_method)
    ):
        raise UniVTACContractError("insert_hole relative pose is invalid")
    matrix = np.asarray(matrix_method(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise UniVTACContractError("insert_hole relative transform is invalid")

    xy_error_m = float(np.linalg.norm(relative[:2]))
    insertion_depth_m = -float(relative[2])
    alignment_dot = float(np.dot(matrix[:3, 2], (0.0, 0.0, 1.0)))
    official_conditions = {
        "abs_relative_x_lt_0_01": bool(abs(relative[0]) < 0.01),
        "abs_relative_y_lt_0_01": bool(abs(relative[1]) < 0.01),
        "insertion_depth_m_gt_0_04": bool(insertion_depth_m > 0.04),
        "alignment_dot_gt_0_99": bool(alignment_dot > 0.99),
        "inhand_z_drift_m_lt_0_04": bool(inhand_z_drift_m < 0.04),
    }
    thresholds = INSERT_HOLE_STRICT_THRESHOLDS
    strict_conditions = {
        "xy_error_m_lt_0_005": bool(xy_error_m < thresholds.xy_error_m),
        "insertion_depth_m_gt_0_050": bool(
            insertion_depth_m > thresholds.insertion_depth_m
        ),
        "alignment_dot_gt_0_999": bool(alignment_dot > thresholds.alignment_dot),
        "inhand_z_drift_m_lt_0_025": bool(
            inhand_z_drift_m < thresholds.inhand_z_drift_m
        ),
    }
    return {
        "diagnostic_schema": "univtac-insert-hole-dual-success-v1",
        "success_metrics_available": True,
        "target_pose": _pose(target_pose, "target"),
        "relative_position_xyz": [float(item) for item in relative],
        "xy_error_m": xy_error_m,
        "insertion_depth_m": insertion_depth_m,
        "alignment_dot": alignment_dot,
        "inhand_z_drift_m": inhand_z_drift_m,
        "official_success_conditions": official_conditions,
        "predicate_success": all(official_conditions.values()),
        "strict_success_conditions": strict_conditions,
        "strict_instantaneous_success": all(strict_conditions.values()),
        "strict_thresholds": thresholds.to_dict(),
    }


def _placement_reset_assessment(
    *,
    task_id: str,
    current_early_stop: bool,
    witnesses: tuple[dict[str, object], ...],
) -> Mapping[str, object]:
    phases = tuple(item.get("phase") for item in witnesses)
    if phases != _INSERTION_PLACEMENT_PHASES[: len(phases)]:
        raise UniVTACContractError(
            f"{task_id} placement witness phases are invalid: {phases!r}"
        )
    summaries: list[dict[str, object]] = []
    failure_phase: str | None = None
    for witness in witnesses:
        phase = str(witness["phase"])
        after_bias = witness.get("inhand_z_bias_m")
        before_bias = witness.get("before_inhand_z_bias_m", after_bias)
        after_early_stop = witness.get("early_stop_predicate")
        before_early_stop = witness.get("early_stop_predicate_before", after_early_stop)
        plan_success = witness.get("plan_success_after", True)
        if (
            not isinstance(before_bias, (int, float))
            or isinstance(before_bias, bool)
            or not np.isfinite(float(before_bias))
            or not isinstance(after_bias, (int, float))
            or isinstance(after_bias, bool)
            or not np.isfinite(float(after_bias))
            or type(before_early_stop) is not bool
            or type(after_early_stop) is not bool
            or type(plan_success) is not bool
        ):
            raise UniVTACContractError(
                f"{task_id} placement witness values are invalid at {phase}"
            )
        threshold_crossed = not before_early_stop and after_early_stop
        if failure_phase is None:
            if not plan_success:
                failure_phase = f"{phase}:plan_failure"
            elif after_early_stop:
                failure_phase = phase
        summaries.append(
            {
                "phase": phase,
                "native_step_before": witness.get("native_step_before"),
                "native_step_after": witness.get("native_step_after"),
                "before_inhand_z_bias_m": float(before_bias),
                "after_inhand_z_bias_m": float(after_bias),
                "inhand_z_bias_delta_m": float(after_bias) - float(before_bias),
                "early_stop_before": before_early_stop,
                "early_stop_after": after_early_stop,
                "threshold_crossed": threshold_crossed,
                "plan_success_after": plan_success,
            }
        )
    phase_sequence_complete = phases == _INSERTION_PLACEMENT_PHASES
    if failure_phase is None and not phase_sequence_complete:
        failure_phase = (
            "placement_not_started" if not witnesses else "final_continuation_missing"
        )
    if failure_phase is None and current_early_stop:
        failure_phase = "post_final_settle"
    return {
        "assessment_schema": "univtac-insertion-placement-reset-v1",
        "available": True,
        "current_early_stop": current_early_stop,
        "expected_phase_count": len(_INSERTION_PLACEMENT_PHASES),
        "failure_phase": failure_phase,
        "observed_phase_count": len(phases),
        "phase_sequence_complete": phase_sequence_complete,
        "phase_summaries": summaries,
        "reset_viable": phase_sequence_complete and failure_phase is None,
    }


def capture_task_diagnostics(
    task: Any,
    task_id: str,
    *,
    include_placement_assessment: bool = True,
) -> Mapping[str, object]:
    """Capture task-specific read-only state or an explicit empty mapping."""

    result: Mapping[str, object]
    if task_id == "lift_bottle":
        result = _lift_bottle(task)
    elif task_id == "grasp_classify":
        result = _grasp_classify(task)
    elif task_id == "insert_hole":
        result = _insertion_inhand(
            task,
            task_id=task_id,
            early_stop_threshold_m=0.04,
            include_placement_assessment=include_placement_assessment,
        )
    elif task_id == "insert_tube":
        result = _insertion_inhand(
            task,
            task_id=task_id,
            early_stop_threshold_m=0.03,
            include_placement_assessment=include_placement_assessment,
        )
    else:
        extended = capture_extended_task_diagnostics(task, task_id)
        result = {} if extended is None else extended
    calibration = getattr(task, "_robotactile_rest_calibration", None)
    if calibration is None:
        return result
    if not isinstance(calibration, Mapping):
        raise UniVTACContractError("rest calibration witness must be a mapping")
    return {**result, "rest_calibration": dict(calibration)}


__all__ = ["capture_task_diagnostics"]
