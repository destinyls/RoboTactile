"""Reference-guided IK seeding used only to capture a reset trajectory."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError


def install_reference_guided_pre_move_ik(
    task: Any,
    expected_arm_qpos7: tuple[float, ...],
) -> bool:
    """Bias calibration IK to the branch of a qualified reset reference.

    The hook is active only while upstream ``pre_move`` calls ``plan_arm``.
    It does not execute a joint teleport and is never installed during formal
    replay; its sole output is a dense trajectory subsequently re-executed by
    the normal UniVTAC physics loop.
    """

    values = np.asarray(expected_arm_qpos7, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise UniVTACContractError(
            "reference-guided IK requires seven finite arm joints"
        )
    manager = getattr(task, "_robot_manager", None)
    upstream_plan_arm = getattr(manager, "plan_arm", None)
    planner = getattr(manager, "planner", None)
    motion_gen = getattr(planner, "motion_gen", None)
    ik_solver = getattr(motion_gen, "ik_solver", None)
    upstream_get_seed = getattr(ik_solver, "get_seed", None)
    if not callable(upstream_plan_arm) or not callable(upstream_get_seed):
        raise UniVTACContractError(
            "UniVTAC runtime does not expose the pinned CuRobo IK seed hook"
        )
    live_manager = cast(Any, manager)
    live_ik_solver = cast(Any, ik_solver)

    def guided_plan_arm(*args: Any, **kwargs: Any) -> Any:
        if getattr(task, "in_pre_move", False) is not True:
            return upstream_plan_arm(*args, **kwargs)

        def reference_seed(
            num_seeds: int,
            goal_pose: Any,
            use_nn_seed: Any,
            seed_config: Any = None,
        ) -> Any:
            del use_nn_seed
            if isinstance(num_seeds, bool) or not isinstance(num_seeds, int):
                raise UniVTACContractError("CuRobo IK seed count must be an integer")
            if num_seeds < 1:
                raise UniVTACContractError("CuRobo IK seed count must be positive")
            template = seed_config
            if template is None:
                robot = getattr(manager, "robot", None)
                data = getattr(robot, "data", None)
                joint_pos = getattr(data, "joint_pos", None)
                if joint_pos is None:
                    raise UniVTACContractError(
                        "CuRobo IK seed tensor template is unavailable"
                    )
                template = joint_pos[:, :7].reshape(1, 1, 7)
            new_tensor = getattr(template, "new_tensor", None)
            if not callable(new_tensor):
                raise UniVTACContractError(
                    "CuRobo IK seed tensor cannot materialize reference joints"
                )
            batch = getattr(goal_pose, "batch", 1)
            if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
                raise UniVTACContractError("CuRobo goal batch must be positive")
            seed = new_tensor(values.tolist()).reshape(1, 1, 7)
            seed = seed.expand(batch, num_seeds, 7).clone()
            return seed.reshape(-1, 1, 7)

        live_ik_solver.get_seed = reference_seed
        try:
            return upstream_plan_arm(*args, **kwargs)
        finally:
            live_ik_solver.get_seed = upstream_get_seed

    live_manager.plan_arm = guided_plan_arm
    task._robotactile_reference_guided_ik = {
        "expected_arm_qpos7": values.tolist(),
        "formal_replay": False,
        "mode": "reset_reference_branch_capture_v1",
        "teleport_used": False,
    }
    return True


__all__ = ["install_reference_guided_pre_move_ik"]
