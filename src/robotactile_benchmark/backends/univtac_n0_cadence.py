"""Fixed-cadence N0 EE execution using UniVTAC's existing cuRobo plan."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACBackendConfig,
    UniVTACContractError,
    validate_n0_ee_action_execution_contract,
)


def install_n0_evaluation_reset(task: Any, config: UniVTACBackendConfig) -> bool:
    """Keep N0 reset under the released UniVTAC evaluator's ``eval`` mode."""

    if config.action_spec != EE8_ACTION_SPEC:
        return False
    upstream_reset = getattr(task, "reset", None)
    if not callable(upstream_reset) or getattr(task, "mode", None) != "eval":
        raise UniVTACContractError("UniVTAC N0 reset surface is incompatible")

    def evaluation_reset(*args: Any, **kwargs: Any) -> Any:
        mode_before = task.mode
        if mode_before != "eval":
            raise UniVTACContractError("UniVTAC N0 reset must start in eval mode")
        task._robotactile_n0_last_reset = {
            "mode_after": None,
            "mode_before": mode_before,
            "mode_during": task.mode,
            "reset_contract": "univtac_released_evaluator_eval_v1",
        }
        try:
            result = upstream_reset(*args, **kwargs)
        finally:
            mode_after = task.mode
            task._robotactile_n0_last_reset = {
                "mode_after": mode_after,
                "mode_before": mode_before,
                "mode_during": "eval",
                "reset_contract": "univtac_released_evaluator_eval_v1",
            }
        if mode_after != "eval":
            raise UniVTACContractError("UniVTAC N0 reset changed task mode")
        return result

    task.reset = evaluation_reset
    return True


def install_n0_stock_ee_execution(
    task: Any,
    config: UniVTACBackendConfig,
) -> bool:
    """Bind N0 to the pinned upstream ``BaseTask.take_action`` implementation.

    No wrapper is installed: every EE action therefore reaches UniVTAC's stock
    ``move`` loop and executes the complete cuRobo waypoint sequence.  The
    method-owner check prevents an instance patch or task-local override from
    being mislabeled as upstream stock execution.
    """

    if config.action_spec != EE8_ACTION_SPEC:
        return False
    take_action = getattr(task, "take_action", None)
    function = getattr(take_action, "__func__", None)
    owner = next(
        (
            base
            for base in type(task).__mro__
            if base.__dict__.get("take_action") is function
        ),
        None,
    )
    if (
        not callable(take_action)
        or owner is None
        or owner.__name__ != "BaseTask"
        or owner.__module__ != "envs._base_task"
    ):
        raise UniVTACContractError(
            "N0 stock EE execution requires upstream BaseTask.take_action"
        )
    if getattr(task, "mode", None) != "eval":
        raise UniVTACContractError("UniVTAC N0 stock EE execution requires eval mode")
    if getattr(task, "_robotactile_n0_fixed_cadence_enabled", False) is not False:
        raise UniVTACContractError("N0 stock EE execution conflicts with fixed cadence")
    task._robotactile_n0_stock_ee_enabled = True
    task._robotactile_n0_action_execution_contract = (
        N0_STOCK_EE_ACTION_EXECUTION_CONTRACT
    )
    return True


def install_n0_action_execution(
    task: Any,
    config: UniVTACBackendConfig,
    *,
    execution_contract: str,
    diagnostic_only: bool,
) -> bool:
    """Install exactly one explicit N0 EE execution surface."""

    selected = validate_n0_ee_action_execution_contract(execution_contract)
    if selected == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT:
        return install_n0_stock_ee_execution(task, config)
    return install_n0_fixed_cadence(
        task,
        config,
        diagnostic_only=diagnostic_only,
        execution_contract=selected,
    )


def install_n0_fixed_cadence(
    task: Any,
    config: UniVTACBackendConfig,
    *,
    diagnostic_only: bool,
    execution_contract: str = N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
) -> bool:
    """Execute each N0 endpoint in two collection-compatible UniVTAC ticks.

    The released HDF5 rows advance by two 120 Hz simulator steps.  UniVTAC's
    stock ``ee`` path instead executes every waypoint in a variable-length
    cuRobo trajectory.  This wrapper keeps cuRobo for IK/collision feasibility,
    applies only its final joint target, and advances two native environment
    steps with the upstream collection decimation of one.
    ``robotactile_n0_training_60hz_ee_v1`` is the production N0 path. The
    historical ``robotactile_fixed_endpoint_v1`` spelling remains available as
    a diagnostic-only alias so archived experiments stay readable.
    """

    selected = validate_n0_ee_action_execution_contract(execution_contract)
    if selected == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT:
        raise UniVTACContractError("stock EE cannot use the fixed cadence installer")
    legacy_diagnostic = selected == N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT
    if legacy_diagnostic and diagnostic_only is not True:
        raise UniVTACContractError(
            "N0 fixed cadence is restricted to training-endpoint diagnostics"
        )
    if not legacy_diagnostic and (
        selected
        not in {
            N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
            N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        }
        or diagnostic_only is not False
    ):
        raise UniVTACContractError("N0 training cadence contract scope mismatch")
    if config.action_spec != EE8_ACTION_SPEC:
        return False
    native_steps_per_action, remainder = divmod(
        config.physics_steps_per_action, config.decimation
    )
    expected_steps = (
        12 if selected == N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT else 2
    )
    if config.physics_steps_per_action != expected_steps:
        raise UniVTACContractError("N0 executor cadence does not match backend config")
    if remainder or native_steps_per_action < 1:
        raise UniVTACContractError(
            "N0 cadence requires an integral number of native control ticks"
        )
    upstream_take_action = getattr(task, "take_action", None)
    robot_manager = getattr(task, "_robot_manager", None)
    step = getattr(task, "_step", None)
    update_render = getattr(task, "_update_render", None)
    if (
        not callable(upstream_take_action)
        or robot_manager is None
        or not callable(step)
        or not callable(update_render)
    ):
        raise UniVTACContractError("UniVTAC N0 cadence surface is incomplete")
    if getattr(task, "mode", None) != "eval":
        raise UniVTACContractError("UniVTAC N0 cadence requires eval mode")
    if getattr(getattr(task, "cfg", None), "decimation", None) != config.decimation:
        raise UniVTACContractError("live UniVTAC decimation differs from N0 cadence")

    def fixed_take_action(
        action: Any,
        action_type: str = "qpos",
        force: bool = True,
    ) -> tuple[Any, Any]:
        if action_type != "ee":
            return cast(
                tuple[Any, Any],
                upstream_take_action(
                    action,
                    action_type=action_type,
                    force=force,
                ),
            )
        if force is not True:
            raise UniVTACContractError("N0 fixed cadence requires force=True")
        if task.take_action_cnt >= task.cfg.step_lim or task.eval_success:
            return True, task.eval_success

        task.take_action_cnt += 1
        target_ee8 = [float(action[index]) for index in range(8)]
        current_pose = robot_manager.get_ee_pose()
        pose_type = type(current_pose)
        target_pose = pose_type(p=action[:3], q=action[3:7])
        plan_started = time.monotonic()
        arm_plan = robot_manager.plan_arm(target_pose)
        plan_duration_s = time.monotonic() - plan_started
        if not isinstance(arm_plan, Mapping) or arm_plan.get("status") != "Success":
            task._robotactile_n0_last_plan = {
                "action_execution_contract": selected,
                "execution_purpose": (
                    "training_endpoint_diagnostic_v1"
                    if legacy_diagnostic
                    else "n0_training_aligned_execution_v1"
                ),
                "native_step_after": None,
                "native_step_before": int(task.step_count),
                "plan_duration_s": plan_duration_s,
                "status": str(
                    arm_plan.get("status", "invalid_plan")
                    if isinstance(arm_plan, Mapping)
                    else "invalid_plan"
                ),
                "stock_move_loop_used": False,
                "target_ee8": target_ee8,
                "waypoint_count": 0,
            }
            task.plan_success = False
            return False, task.eval_success
        positions = arm_plan.get("position")
        velocities = arm_plan.get("velocity")
        if positions is None or velocities is None or len(positions) < 1:
            raise UniVTACContractError("cuRobo plan lacks a final joint target")

        final_position = positions[-1]
        zero_velocity = velocities[-1] * 0.0
        robot_manager.set_arm(final_position, zero_velocity, force=True)
        robot_manager.set_gripper(action[7:], force=True)
        native_before = int(task.step_count)
        previous_mode = task.mode
        try:
            task.mode = "eval_test"
            for _ in range(native_steps_per_action):
                step(is_save=False)
        finally:
            task.mode = previous_mode
        try:
            update_render()
        finally:
            task.mode = previous_mode
        native_after = int(task.step_count)
        if native_after != native_before + native_steps_per_action:
            raise UniVTACContractError(
                "N0 action did not advance the required native steps"
            )
        task._robotactile_n0_last_plan = {
            "action_execution_contract": selected,
            "execution_purpose": (
                "training_endpoint_diagnostic_v1"
                if legacy_diagnostic
                else "n0_training_aligned_execution_v1"
            ),
            "native_step_after": native_after,
            "native_step_before": native_before,
            "plan_duration_s": plan_duration_s,
            "render_contract": "one_endpoint_render_no_intermediate_render_v1",
            "status": "Success",
            "stock_move_loop_used": False,
            "target_ee8": target_ee8,
            "waypoint_count": len(positions),
        }
        if task.check_success():
            task.eval_success = True
        return True, task.eval_success

    task._robotactile_n0_fixed_cadence_enabled = True
    task._robotactile_n0_action_execution_contract = selected
    task.take_action = fixed_take_action
    return True
