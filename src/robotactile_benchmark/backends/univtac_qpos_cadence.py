"""Training-rate qpos delivery using the pinned UniVTAC actuator surface."""

from __future__ import annotations

from typing import Any

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    UniVTACContractError,
)


def install_retrained_qpos_cadence(task: Any, config: UniVTACBackendConfig) -> bool:
    """Hold one predicted target for exactly 2 or 12 physics ticks, render once."""
    if config.action_spec != QPOS8_ACTION_SPEC or config.physics_steps_per_action == 1:
        return False
    if config.physics_steps_per_action not in (2, 12) or config.decimation != 1:
        raise UniVTACContractError("unsupported retrained qpos cadence")
    if task.mode != "eval" or task.cfg.decimation != 1:
        raise UniVTACContractError("retrained qpos requires eval / decimation=1")
    upstream = task.take_action

    def take_action(
        action: Any, action_type: str = "qpos", force: bool = True
    ) -> tuple[Any, Any]:
        if action_type != "qpos":
            return upstream(action, action_type=action_type, force=force)  # type: ignore[no-any-return]
        if force is not True:
            raise UniVTACContractError("retrained qpos requires force=True")
        if task.take_action_cnt >= task.cfg.step_lim or task.eval_success:
            return True, task.eval_success
        task.take_action_cnt += 1
        task._robot_manager.set_arm(action[:-1], force=True)
        task._robot_manager.set_gripper(action[-1], force=True)
        before, previous_mode = int(task.step_count), task.mode
        try:
            task.mode = "eval_test"
            for _ in range(config.physics_steps_per_action):
                task._step(is_save=False)
        finally:
            task.mode = previous_mode
        task._update_render()
        if int(task.step_count) - before != config.physics_steps_per_action:
            raise UniVTACContractError("retrained qpos native-step count mismatch")
        if task.check_success():
            task.eval_success = True
        return True, task.eval_success

    task.take_action = take_action
    task._robotactile_retrained_qpos_contract = config.action_execution_contract
    return True
