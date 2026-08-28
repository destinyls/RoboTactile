"""Stable grasp initialization for the pinned UniVTAC classification task."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.contracts import Array

_SUPPORTED_TASKS = frozenset({"grasp_classify"})
_GRASP_QPOS_M = 0.0065
_LOADED_ADAPTIVE_QPOS_M = 0.0045
_LOADED_ADAPTIVE_DEPTH_MM = 25.0
_TACTILE_FAR_PLANE_MM = 34.0
_BILATERAL_CONTACT_INDENTATION_MM = 0.8
_BILATERAL_CONTACT_TOLERANCE_MM = 0.05
_LOADED_LIFT_TIME_DILATION = 0.2
_MIN_LIFT_M = 0.03
_SETTLE_STEPS = 20
_MAX_RELATIVE_DRIFT_M = 0.003


def _position(pose: Any, label: str) -> Array:
    value = getattr(pose, "p", None)
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    position = cast(Array, np.asarray(value, dtype=np.float64))
    if position.shape != (3,) or not np.isfinite(position).all():
        raise UniVTACContractError(f"{label} position must be finite shape [3]")
    return position


def _minimum_tactile_depths_mm(task: Any) -> list[float] | None:
    manager = getattr(task, "_tactile_manager", None)
    get_min_depth = getattr(manager, "get_min_depth", None)
    if not callable(get_min_depth):
        return None
    value = get_min_depth()
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    depths = np.asarray(value, dtype=np.float64).reshape(-1)
    if depths.size != 2 or not np.isfinite(depths).all():
        raise UniVTACContractError(
            "grasp_classify tactile minimum depths must contain two finite values"
        )
    return [float(item) for item in depths]


def install_grasp_initialization_compatibility(task: Any, task_id: str) -> bool:
    """Clamp pre-move closure and verify a stable lifted prism after reset."""

    if task_id not in _SUPPORTED_TASKS:
        return False
    upstream_pre_move = getattr(task, "pre_move", None)
    upstream_reset = getattr(task, "reset", None)
    atom = getattr(task, "atom", None)
    upstream_close = getattr(atom, "close_gripper", None)
    upstream_displacement = getattr(atom, "move_by_displacement", None)
    robot_manager = getattr(task, "_robot_manager", None)
    get_gripper_pose = getattr(robot_manager, "get_gripper_center_pose", None)
    get_gripper_qpos = getattr(robot_manager, "get_gripper_qpos", None)
    move = getattr(task, "move", None)
    delay = getattr(task, "delay", None)
    cfg = getattr(task, "cfg", None)
    logger = getattr(task, "logger", None)
    log_info = getattr(logger, "info", None)
    log_error = getattr(logger, "error", None)
    gripper_max_qpos = getattr(robot_manager, "gripper_max_qpos", None)
    if (
        not callable(upstream_pre_move)
        or not callable(upstream_reset)
        or atom is None
        or not callable(upstream_close)
        or not callable(upstream_displacement)
        or not callable(get_gripper_pose)
        or not callable(get_gripper_qpos)
        or not callable(move)
        or not callable(delay)
        or not callable(log_info)
        or not callable(log_error)
        or not isinstance(gripper_max_qpos, (int, float))
        or gripper_max_qpos <= 0.0
        or cfg is None
        or not isinstance(getattr(cfg, "use_adaptive_grasp", None), bool)
        or not isinstance(getattr(task, "in_pre_move", None), bool)
    ):
        raise UniVTACContractError(
            "grasp_classify initialization contract is incomplete"
        )
    grasp_ratio = _GRASP_QPOS_M / float(gripper_max_qpos)
    initial_position: Array | None = None
    retry_mode = "official"

    def compatible_close_gripper(
        pos: float = 0.0,
        depth_threshold: Any = "auto",
    ) -> Any:
        if retry_mode == "adaptive":
            target = 0.0
        elif retry_mode == "loaded_adaptive":
            target = _LOADED_ADAPTIVE_QPOS_M / float(gripper_max_qpos)
            depth_threshold = _LOADED_ADAPTIVE_DEPTH_MM
        else:
            target = min(float(pos), grasp_ratio)
        return upstream_close(target, depth_threshold)

    def compatible_move_by_displacement(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        actions = upstream_displacement(*args, **kwargs)
        if retry_mode == "loaded_adaptive":
            for action in actions:
                action.args["time_dilation_factor"] = _LOADED_LIFT_TIME_DILATION
        return actions

    def compatible_pre_move() -> Any:
        nonlocal initial_position
        prism = getattr(task, "prism", None)
        get_prism_pose = getattr(prism, "get_pose", None)
        if not callable(get_prism_pose):
            raise UniVTACContractError("grasp_classify selected prism is unavailable")
        initial_position = _position(get_prism_pose(), "initial prism")
        atom.close_gripper = compatible_close_gripper
        atom.move_by_displacement = compatible_move_by_displacement
        try:
            return upstream_pre_move()
        finally:
            atom.close_gripper = upstream_close
            atom.move_by_displacement = upstream_displacement

    def capture_attempt(label: str) -> dict[str, object]:
        if initial_position is None:
            raise UniVTACContractError(
                "grasp_classify reset did not execute the pre-move grasp"
            )
        prism = getattr(task, "prism", None)
        get_prism_pose = getattr(prism, "get_pose", None)
        if not callable(get_prism_pose):
            raise UniVTACContractError("grasp_classify selected prism is unavailable")
        before_position = _position(get_prism_pose(), "pre-settle prism")
        before_gripper = _position(get_gripper_pose(), "pre-settle gripper")
        delay(_SETTLE_STEPS, is_save=False)
        after_position = _position(get_prism_pose(), "post-settle prism")
        after_gripper = _position(get_gripper_pose(), "post-settle gripper")
        lift_m = float(after_position[2] - initial_position[2])
        relative_drift_m = float(
            np.linalg.norm(
                (after_position - after_gripper) - (before_position - before_gripper)
            )
        )
        minimum_depths_mm = _minimum_tactile_depths_mm(task)
        kinematics_passed = (
            lift_m >= _MIN_LIFT_M and relative_drift_m <= _MAX_RELATIVE_DRIFT_M
        )
        maximum_contact_depth_mm = (
            _TACTILE_FAR_PLANE_MM
            - _BILATERAL_CONTACT_INDENTATION_MM
            + _BILATERAL_CONTACT_TOLERANCE_MM
        )
        contact_count = sum(
            depth <= maximum_contact_depth_mm for depth in (minimum_depths_mm or [])
        )
        any_contact = contact_count >= 1
        bilateral_contact = contact_count == 2
        passed = kinematics_passed and any_contact
        return {
            "any_contact": any_contact,
            "attempt": label,
            "bilateral_contact": bilateral_contact,
            "contact_count": contact_count,
            "maximum_contact_depth_mm": maximum_contact_depth_mm,
            "gripper_qpos_m": float(get_gripper_qpos()),
            "initial_prism": initial_position.tolist(),
            "kinematics_passed": kinematics_passed,
            "lift_m": lift_m,
            "minimum_tactile_depths_mm": minimum_depths_mm,
            "passed": passed,
            "post_settle_gripper": after_gripper.tolist(),
            "post_settle_prism": after_position.tolist(),
            "pre_settle_gripper": before_gripper.tolist(),
            "pre_settle_prism": before_position.tolist(),
            "relative_drift_m": relative_drift_m,
            "settle_steps": _SETTLE_STEPS,
        }

    def run_retry(mode: str, *, in_place: bool) -> None:
        nonlocal retry_mode
        if mode not in {"adaptive", "loaded_adaptive"}:
            raise ValueError("unsupported grasp retry mode")
        prior_adaptive = cfg.use_adaptive_grasp
        prior_in_pre_move = task.in_pre_move
        retry_mode = mode
        cfg.use_adaptive_grasp = mode in {"adaptive", "loaded_adaptive"}
        task.in_pre_move = True
        try:
            if in_place:
                move(compatible_close_gripper())
            else:
                compatible_pre_move()
        finally:
            task.in_pre_move = prior_in_pre_move
            cfg.use_adaptive_grasp = prior_adaptive
            retry_mode = "official"

    def compatible_reset(*args: Any, **kwargs: Any) -> Any:
        nonlocal initial_position
        initial_position = None
        result = upstream_reset(*args, **kwargs)
        attempts = [capture_attempt("official_lower_bound")]
        if bool(attempts[-1]["kinematics_passed"]):
            run_retry("loaded_adaptive", in_place=True)
            attempts.append(capture_attempt("loaded_hold_preload"))
        else:
            run_retry("adaptive", in_place=False)
            attempts.append(capture_attempt("adaptive_contact_retry"))
        if (
            not bool(attempts[-1]["passed"])
            and attempts[-1]["attempt"] != "loaded_hold_preload"
        ):
            run_retry(
                "loaded_adaptive",
                in_place=bool(attempts[-1]["kinematics_passed"]),
            )
            attempts.append(capture_attempt("loaded_hold_preload"))
        witness = {
            "attempt_count": len(attempts),
            "attempts": attempts,
            "grasp_qpos_m": _GRASP_QPOS_M,
            "grasp_ratio": grasp_ratio,
            "bilateral_contact_indentation_mm": (_BILATERAL_CONTACT_INDENTATION_MM),
            "bilateral_contact_tolerance_mm": _BILATERAL_CONTACT_TOLERANCE_MM,
            "passed": bool(attempts[-1]["passed"]),
            "loaded_adaptive_depth_mm": _LOADED_ADAPTIVE_DEPTH_MM,
            "loaded_adaptive_qpos_m": _LOADED_ADAPTIVE_QPOS_M,
            "loaded_lift_time_dilation": _LOADED_LIFT_TIME_DILATION,
            "preload_required": True,
            "preload_reason": "dynamic_hold_stability_v1",
            "selected_attempt": attempts[-1]["attempt"],
            "task_id": task_id,
            "tactile_far_plane_mm": _TACTILE_FAR_PLANE_MM,
        }
        task._robotactile_grasp_initialization = witness
        payload = json.dumps(
            witness,
            separators=(",", ":"),
            sort_keys=True,
        )
        message = "ROBOTACTILE_GRASP_INITIALIZATION " + payload
        if not witness["passed"]:
            log_error(message)
            raise UniVTACContractError(
                "grasp_classify deterministic regrasp did not establish a stable lift"
            )
        log_info(message)
        return result

    task.pre_move = compatible_pre_move
    task.reset = compatible_reset
    return True
