"""Empty-gripper reset contracts for measured UniVTAC rest calibration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACBackendConfig,
    UniVTACContractError,
)

N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT = "univtac_n0_empty_gripper_rest_calibration_v1"
ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT = "univtac_act_empty_gripper_rest_calibration_v1"


def _require_callable(task: Any, name: str) -> Any:
    value = getattr(task, name, None)
    if not callable(value):
        raise UniVTACContractError(
            f"empty-gripper calibration requires callable task.{name}"
        )
    return value


def _install_empty_gripper_rest_calibration(
    task: Any,
    config: UniVTACBackendConfig,
    *,
    calibration_contract: str,
    execution_purpose: str,
) -> Mapping[str, object]:
    """Skip task ``pre_move`` and hold an undeformed sensor state.

    The contract is calibration-only. It preserves the official task constructor
    and reset up to the task-specific ``pre_move`` boundary, then advances the
    same physics/render cadence without applying model actions. Formal Clean and
    Faulted requests never install this hook.
    """

    if not isinstance(config, UniVTACBackendConfig):
        raise TypeError("config must be a UniVTACBackendConfig")
    if getattr(task, "_robotactile_rest_calibration", None) is not None:
        raise UniVTACContractError("empty-gripper calibration is already installed")
    delay = _require_callable(task, "delay")
    step = _require_callable(task, "_step")
    update_render = _require_callable(task, "_update_render")
    _require_callable(task, "pre_move")
    _require_callable(task, "take_action")
    _require_callable(task, "check_success")
    if config.task.early_stop_capable:
        _require_callable(task, "check_early_stop")
    native_steps, remainder = divmod(
        config.physics_steps_per_action,
        config.decimation,
    )
    if remainder or native_steps < 1:
        raise UniVTACContractError(
            "empty-gripper calibration requires integral native 60 Hz cadence"
        )
    if getattr(task, "mode", None) != "eval":
        raise UniVTACContractError(
            "empty-gripper calibration requires UniVTAC eval mode"
        )

    witness: dict[str, object] = {
        "action_execution_contract": config.action_execution_contract,
        "calibration_contract": calibration_contract,
        "formal_evaluation_reset_modified": False,
        "model_actions_applied": False,
        "native_steps_per_endpoint": native_steps,
        "pre_move_skipped": True,
        "purpose": "measured_no_contact_rest_reference_only",
        "sim_hz": config.sim_hz,
    }
    task._robotactile_rest_calibration = witness
    task._robotactile_rest_calibration_executor = True

    def empty_gripper_pre_move() -> None:
        delay(10, is_save=False)

    def calibration_success(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False

    def calibration_early_stop(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False

    def hold_endpoint(
        action: Any,
        action_type: str = "qpos",
        force: bool = True,
    ) -> tuple[bool, bool]:
        if action_type != config.action_mode or force is not True:
            raise UniVTACContractError(
                "empty-gripper calibration requires the registered action contract"
            )
        task.take_action_cnt += 1
        native_before = int(task.step_count)
        previous_mode = task.mode
        try:
            task.mode = "eval_test"
            for _ in range(native_steps):
                step(is_save=False)
            update_render()
        finally:
            task.mode = previous_mode
        native_after = int(task.step_count)
        if native_after != native_before + native_steps:
            raise UniVTACContractError(
                "empty-gripper calibration cadence did not advance exactly"
            )
        task.plan_success = True
        task.eval_success = False
        hold_witness = {
            "action_execution_contract": config.action_execution_contract,
            "execution_purpose": execution_purpose,
            "native_step_after": native_after,
            "native_step_before": native_before,
            "plan_duration_s": 0.0,
            "render_contract": "one_endpoint_render_no_intermediate_render_v1",
            "status": "Success",
            "stock_move_loop_used": False,
            "target_ee8": [float(action[index]) for index in range(8)],
            "waypoint_count": 0,
        }
        task._robotactile_rest_calibration_last_hold = hold_witness
        if config.action_execution_contract in {
            N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
            N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        }:
            task._robotactile_n0_last_plan = hold_witness
        return True, False

    task.pre_move = empty_gripper_pre_move
    task.take_action = hold_endpoint
    task.check_success = calibration_success
    if config.task.early_stop_capable:
        task.check_early_stop = calibration_early_stop
    return witness


def install_n0_empty_gripper_rest_calibration(
    task: Any,
    config: UniVTACBackendConfig,
) -> Mapping[str, object]:
    """Measure N0 rest at the selected registered 60 Hz or retrained 10 Hz rate."""

    if config.action_execution_contract not in {
        N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    }:
        raise UniVTACContractError(
            "N0 empty-gripper calibration requires a registered N0 fixed cadence"
        )
    return _install_empty_gripper_rest_calibration(
        task,
        config,
        calibration_contract=N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
        execution_purpose="n0_empty_gripper_rest_calibration_v1",
    )


def install_act_empty_gripper_rest_calibration(
    task: Any,
    config: UniVTACBackendConfig,
) -> Mapping[str, object]:
    """Install an ACT qpos calibration hook without applying policy actions."""

    if config.action_execution_contract == (N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT):
        raise UniVTACContractError(
            "ACT empty-gripper calibration requires the qpos action contract"
        )
    return _install_empty_gripper_rest_calibration(
        task,
        config,
        calibration_contract=ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
        execution_purpose="act_empty_gripper_rest_calibration_v1",
    )


__all__ = [
    "ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT",
    "N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT",
    "install_act_empty_gripper_rest_calibration",
    "install_n0_empty_gripper_rest_calibration",
]
