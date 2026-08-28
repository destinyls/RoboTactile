"""Narrow planner compatibility for the pinned UniVTAC/cuRobo pair."""

from __future__ import annotations

import importlib
import json
import math
import types
from collections.abc import Mapping
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError

LOCAL_IK_FALLBACK_MARKER = "robotactile_partial_ik_fallback"
_MARKED_LOCAL_IK_FALLBACK_TASKS = frozenset({"insert_hole"})
_DIRECT_EE_LOCAL_IK_FALLBACK_TASKS = frozenset({"grasp_classify"})
_LOCAL_IK_FALLBACK_TASKS = (
    _MARKED_LOCAL_IK_FALLBACK_TASKS | _DIRECT_EE_LOCAL_IK_FALLBACK_TASKS
)
_IK_FAILURE_STATUS = "IK Fail"
_TRAJOPT_FAILURE_STATUS = "Finetune TrajOpt Fail"
_MAX_LOCAL_TRANSLATION_M = 0.01
_MAX_LOCAL_ROTATION_DEGREES = 6.0
_MIN_LOCAL_QUATERNION_ALIGNMENT = math.cos(
    math.radians(_MAX_LOCAL_ROTATION_DEGREES) / 2.0
)
_MAX_LOCAL_JOINT_DELTA_RAD = 0.15
_JOINT_INTERPOLATION_STEP_RAD = 0.002
_MIN_INTERPOLATION_STEPS = 8
_MAX_INTERPOLATION_STEPS = 64


def _status_value(result: Any) -> str:
    status = getattr(result, "status", "")
    return str(getattr(status, "value", status))


def _succeeded(result: Any) -> bool:
    success = getattr(result, "success", False)
    item = getattr(success, "item", None)
    if callable(item):
        success = item()
    return bool(success)


def _marked_actions(actions: Any) -> list[Any]:
    if not isinstance(actions, list):
        return []
    return [
        action
        for action in actions
        if isinstance(getattr(action, "args", None), dict)
        and action.args.get(LOCAL_IK_FALLBACK_MARKER) is True
    ]


def _direct_ee_actions(actions: Any) -> list[Any]:
    if not isinstance(actions, list):
        return []
    return [
        action
        for action in actions
        if getattr(action, "action", None) == "all"
        and getattr(action, "target_pose", None) is not None
    ]


def _json_values(value: Any) -> Any:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_list = getattr(value, "tolist", None)
    return to_list() if callable(to_list) else value


def _clone_with_zero_dynamics(start_joint_states: Any) -> Any:
    clone = getattr(start_joint_states, "clone", None)
    if not callable(clone):
        raise UniVTACContractError("cuRobo start joint state cannot be cloned")
    retry_start = clone()
    for field_name in ("velocity", "acceleration", "jerk"):
        value = getattr(retry_start, field_name, None)
        if value is None:
            continue
        zero = getattr(value, "zero_", None)
        if not callable(zero):
            raise UniVTACContractError(
                f"cuRobo start joint state {field_name} cannot be zeroed"
            )
        zero()
    return retry_start


def _differential_ik_plan(
    task: Any, target_pose: Any
) -> tuple[Any | None, Mapping[str, Any]]:
    torch = importlib.import_module("torch")
    math_utils = importlib.import_module("isaaclab.utils.math")
    controller_module = importlib.import_module("isaaclab.controllers.differential_ik")
    config_module = importlib.import_module("isaaclab.controllers.differential_ik_cfg")
    controller_type = getattr(controller_module, "DifferentialIKController", None)
    config_type = getattr(config_module, "DifferentialIKControllerCfg", None)
    robot_manager = getattr(task, "_robot_manager", None)
    robot = getattr(robot_manager, "robot", None)
    if (
        not callable(controller_type)
        or not callable(config_type)
        or robot_manager is None
        or robot is None
    ):
        raise UniVTACContractError("IsaacLab differential IK contract is incomplete")

    rebase = getattr(target_pose, "rebase", None)
    if not callable(rebase):
        raise UniVTACContractError("local IK target pose cannot be rebased")
    target_base = rebase(to_coord=robot_manager.root_pose)
    target_values = torch.as_tensor(
        target_base.tolist(),
        dtype=torch.float32,
        device=robot_manager.device,
    ).reshape(1, 7)
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        robot.data.body_link_pos_w[:, robot_manager._body_idx],
        robot.data.body_link_quat_w[:, robot_manager._body_idx],
    )
    translation_m = float(
        torch.linalg.vector_norm(target_values[:, :3] - ee_pos_base).item()
    )
    quaternion_alignment = float(
        torch.abs(torch.sum(target_values[:, 3:7] * ee_quat_base, dim=1)).min().item()
    )
    metadata: dict[str, Any] = {
        "current_ee_base": _json_values(
            torch.cat((ee_pos_base, ee_quat_base), dim=1)[0]
        ),
        "current_gripper_center_base": _json_values(
            robot_manager.get_gripper_center_pose().tolist()
        ),
        "current_joint_pos": _json_values(
            robot.data.joint_pos[:, robot_manager._arm_ids][0]
        ),
        "quaternion_alignment": quaternion_alignment,
        "raw_target": _json_values(target_pose.tolist()),
        "robot_root_pose": _json_values(robot_manager.root_pose.tolist()),
        "target_base": _json_values(target_values[0]),
        "translation_m": translation_m,
    }
    if (
        translation_m > _MAX_LOCAL_TRANSLATION_M
        or quaternion_alignment < _MIN_LOCAL_QUATERNION_ALIGNMENT
    ):
        metadata["rejected"] = "nonlocal_pose_delta"
        return None, metadata

    config = config_type(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    controller = controller_type(config, num_envs=1, device=robot_manager.device)
    controller.set_command(target_values)
    arm_ids = robot_manager._arm_ids
    jacobian = robot.root_physx_view.get_jacobians()[
        :, robot_manager._jacobi_body_idx, :, arm_ids
    ].clone()
    base_rotation = math_utils.matrix_from_quat(
        math_utils.quat_inv(robot.data.root_link_quat_w)
    )
    jacobian[:, :3, :] = torch.bmm(base_rotation, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rotation, jacobian[:, 3:, :])
    joint_pos = robot.data.joint_pos[:, arm_ids]
    joint_desired = controller.compute(
        ee_pos_base,
        ee_quat_base,
        jacobian,
        joint_pos,
    )
    if not bool(torch.isfinite(joint_desired).all().item()):
        metadata["rejected"] = "nonfinite_joint_target"
        return None, metadata
    joint_limits = robot.data.soft_joint_pos_limits[:, arm_ids, :]
    within_limits = (joint_desired >= joint_limits[..., 0]) & (
        joint_desired <= joint_limits[..., 1]
    )
    if not bool(within_limits.all().item()):
        metadata["rejected"] = "joint_limit"
        return None, metadata
    joint_delta = joint_desired - joint_pos
    max_joint_delta = float(torch.abs(joint_delta).max().item())
    metadata["max_joint_delta_rad"] = max_joint_delta
    if max_joint_delta > _MAX_LOCAL_JOINT_DELTA_RAD:
        metadata["rejected"] = "joint_delta"
        return None, metadata

    steps = max(
        _MIN_INTERPOLATION_STEPS,
        min(
            _MAX_INTERPOLATION_STEPS,
            math.ceil(max_joint_delta / _JOINT_INTERPOLATION_STEP_RAD),
        ),
    )
    alpha = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    )[1:]
    position = joint_pos[0] + alpha[:, None] * joint_delta[0]
    previous = torch.cat((joint_pos, position[:-1]), dim=0)
    velocity = (position - previous) / float(task.cfg.sim.dt)
    metadata["steps"] = steps
    result = types.SimpleNamespace(
        attempts=1,
        interpolated_plan=types.SimpleNamespace(
            position=position.detach(),
            velocity=velocity.detach(),
        ),
        status="Differential IK Success",
        success=torch.tensor(True, device=joint_pos.device),
        valid_query=True,
    )
    return result, metadata


def install_local_ik_fallback(task: Any, task_id: str) -> bool:
    """Retry one marked 8 mm placement with cuRobo partial IK optimization."""

    if task_id not in _LOCAL_IK_FALLBACK_TASKS:
        return False
    upstream_move = getattr(task, "move", None)
    robot_manager = getattr(task, "_robot_manager", None)
    planner = getattr(robot_manager, "planner", None)
    motion_gen = getattr(planner, "motion_gen", None)
    upstream_plan_single = getattr(motion_gen, "plan_single", None)
    logger = getattr(task, "logger", None)
    log_info = getattr(logger, "info", None)
    log_error = getattr(logger, "error", None)
    if (
        not callable(upstream_move)
        or motion_gen is None
        or not callable(upstream_plan_single)
        or not callable(log_info)
        or not callable(log_error)
    ):
        raise UniVTACContractError("upstream local IK fallback contract is incomplete")

    fallback_active = False
    fallback_target: Any = None
    fallback_mode: str | None = None
    retry_statuses = {_IK_FAILURE_STATUS}
    if task_id in _DIRECT_EE_LOCAL_IK_FALLBACK_TASKS:
        retry_statuses.add(_TRAJOPT_FAILURE_STATUS)

    def compatible_plan_single(
        start_joint_states: Any,
        goal_pose: Any,
        plan_config: Any,
    ) -> Any:
        result = upstream_plan_single(start_joint_states, goal_pose, plan_config)
        if (
            not fallback_active
            or _succeeded(result)
            or _status_value(result) not in retry_statuses
        ):
            return result
        differential: Any | None = None
        differential_metadata: Mapping[str, Any] | None = None
        retry_start = start_joint_states
        zero_dynamics_status: str | None = None
        if fallback_mode == "direct_ee":
            if fallback_target is None:
                raise UniVTACContractError("local IK fallback target is unavailable")
            differential, differential_metadata = _differential_ik_plan(
                task, fallback_target
            )
            if differential is None:
                retry = result
                payload = json.dumps(
                    {
                        "differential_ik": differential_metadata,
                        "initial_status": _status_value(result),
                        "mode": fallback_mode,
                        "retry_status": _status_value(retry),
                        "retry_success": False,
                        "task_id": task_id,
                        "zero_dynamics_status": zero_dynamics_status,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                log_error("ROBOTACTILE_LOCAL_IK_FALLBACK " + payload)
                return retry
            retry_start = _clone_with_zero_dynamics(start_joint_states)
            retry = upstream_plan_single(retry_start, goal_pose, plan_config)
            zero_dynamics_status = _status_value(retry)
            if _succeeded(retry):
                payload = json.dumps(
                    {
                        "differential_ik": differential_metadata,
                        "initial_status": _status_value(result),
                        "mode": fallback_mode,
                        "retry_status": _status_value(retry),
                        "retry_success": True,
                        "task_id": task_id,
                        "zero_dynamics_status": zero_dynamics_status,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                log_info("ROBOTACTILE_LOCAL_IK_FALLBACK " + payload)
                return retry

        clone = getattr(plan_config, "clone", None)
        if not callable(clone):
            raise UniVTACContractError("cuRobo plan config cannot be cloned")
        retry_config = clone()
        retry_config.partial_ik_opt = True
        retry = upstream_plan_single(retry_start, goal_pose, retry_config)
        if not _succeeded(retry):
            if differential is None:
                if fallback_target is None:
                    raise UniVTACContractError(
                        "local IK fallback target is unavailable"
                    )
                differential, differential_metadata = _differential_ik_plan(
                    task, fallback_target
                )
            if differential is not None:
                retry = differential
        payload = json.dumps(
            {
                "differential_ik": differential_metadata,
                "initial_status": _status_value(result),
                "mode": fallback_mode,
                "retry_status": _status_value(retry),
                "retry_success": _succeeded(retry),
                "task_id": task_id,
                "zero_dynamics_status": zero_dynamics_status,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        if _succeeded(retry):
            log_info("ROBOTACTILE_LOCAL_IK_FALLBACK " + payload)
        else:
            log_error("ROBOTACTILE_LOCAL_IK_FALLBACK " + payload)
        return retry

    def compatible_move(actions: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal fallback_active, fallback_mode, fallback_target
        marked = _marked_actions(actions)
        direct = (
            _direct_ee_actions(actions)
            if task_id in _DIRECT_EE_LOCAL_IK_FALLBACK_TASKS
            else []
        )
        selected = marked or direct
        if not selected:
            return upstream_move(actions, *args, **kwargs)
        if len(selected) != 1:
            raise UniVTACContractError("expected exactly one local IK fallback action")
        action = selected[0]
        if marked:
            compatible_args = dict(action.args)
            del compatible_args[LOCAL_IK_FALLBACK_MARKER]
            action.args = compatible_args
        fallback_active = True
        fallback_mode = "marked" if marked else "direct_ee"
        fallback_target = getattr(action, "target_pose", None)
        try:
            return upstream_move(actions, *args, **kwargs)
        finally:
            fallback_active = False
            fallback_mode = None
            fallback_target = None

    motion_gen.plan_single = compatible_plan_single
    task.move = compatible_move
    return True
