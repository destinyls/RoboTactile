from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_n0_cadence import (
    install_n0_action_execution,
    install_n0_evaluation_reset,
    install_n0_fixed_cadence,
    install_n0_stock_ee_execution,
)


class _Pose:
    def __init__(self, p: object, q: object) -> None:
        self.p = np.asarray(p, dtype=np.float32)
        self.q = np.asarray(q, dtype=np.float32)


class _RobotManager:
    def __init__(self, *, planning_succeeds: bool = True) -> None:
        self.planning_succeeds = planning_succeeds
        self.target_pose: _Pose | None = None
        self.arm_calls: list[tuple[object, object, bool]] = []
        self.gripper_calls: list[tuple[object, bool]] = []

    @staticmethod
    def get_ee_pose() -> _Pose:
        return _Pose(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]))

    def plan_arm(self, target_pose: _Pose) -> dict[str, object]:
        self.target_pose = target_pose
        if not self.planning_succeeds:
            return {"status": "Fail", "position": None, "velocity": None}
        return {
            "status": "Success",
            "position": np.asarray([[0.0] * 7, [0.1] * 7], dtype=np.float32),
            "velocity": np.asarray([[1.0] * 7, [2.0] * 7], dtype=np.float32),
        }

    def set_arm(self, position: object, velocity: object, *, force: bool) -> None:
        self.arm_calls.append((position, velocity, force))

    def set_gripper(self, position: object, *, force: bool) -> None:
        self.gripper_calls.append((position, force))


def _task(*, planning_succeeds: bool = True) -> SimpleNamespace:
    task = SimpleNamespace(
        _robot_manager=_RobotManager(planning_succeeds=planning_succeeds),
        cfg=SimpleNamespace(decimation=1, step_lim=500),
        eval_success=False,
        mode="eval",
        plan_success=True,
        render_count=0,
        render_modes=[],
        step_count=17,
        step_modes=[],
        take_action_cnt=0,
    )

    def upstream_take_action(
        _action: object, action_type: str = "qpos", force: bool = True
    ) -> tuple[bool, bool]:
        return action_type == "qpos" and force, False

    def step(*, is_save: bool) -> None:
        assert not is_save
        task.step_modes.append(task.mode)
        task.step_count += 1

    def update_render() -> None:
        task.render_modes.append(task.mode)
        task.render_count += 1

    task.take_action = upstream_take_action
    task._step = step
    task._update_render = update_render
    task.check_success = lambda: False
    return task


def test_fixed_cadence_uses_curobo_final_target_and_two_native_ticks() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _task()
    assert install_n0_fixed_cadence(task, config, diagnostic_only=True)
    assert task._robotactile_n0_fixed_cadence_enabled is True
    action = np.asarray(
        [0.4, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02],
        dtype=np.float32,
    )

    assert task.take_action(action, action_type="ee", force=True) == (True, False)

    robot = task._robot_manager
    assert task.step_count == 19
    assert task.step_modes == ["eval_test", "eval_test"]
    assert task.render_count == 1
    assert task.render_modes == ["eval"]
    assert task.mode == "eval"
    assert task.take_action_cnt == 1
    assert robot.target_pose is not None
    np.testing.assert_array_equal(robot.target_pose.p, action[:3])
    np.testing.assert_array_equal(robot.target_pose.q, action[3:7])
    np.testing.assert_array_equal(
        robot.arm_calls[0][0], np.asarray([0.1] * 7, dtype=np.float32)
    )
    np.testing.assert_array_equal(robot.arm_calls[0][1], np.zeros(7, dtype=np.float32))
    assert robot.arm_calls[0][2] is True
    np.testing.assert_array_equal(robot.gripper_calls[0][0], action[7:])
    assert robot.gripper_calls[0][1] is True
    assert task._robotactile_n0_last_plan == {
        "action_execution_contract": "robotactile_fixed_endpoint_v1",
        "execution_purpose": "training_endpoint_diagnostic_v1",
        "native_step_after": 19,
        "native_step_before": 17,
        "plan_duration_s": task._robotactile_n0_last_plan["plan_duration_s"],
        "render_contract": "one_endpoint_render_no_intermediate_render_v1",
        "status": "Success",
        "stock_move_loop_used": False,
        "target_ee8": [float(item) for item in action],
        "waypoint_count": 2,
    }
    assert task._robotactile_n0_last_plan["plan_duration_s"] >= 0.0


def test_fixed_cadence_requires_explicit_diagnostic_scope() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)

    with pytest.raises(UniVTACContractError, match="training-endpoint diagnostics"):
        install_n0_fixed_cadence(_task(), config, diagnostic_only=False)


def test_training_60hz_contract_is_the_non_diagnostic_fixed_cadence() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _task()

    assert install_n0_action_execution(
        task,
        config,
        execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        diagnostic_only=False,
    )
    assert (
        task._robotactile_n0_action_execution_contract
        == N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )
    action = np.asarray(
        [0.4, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02],
        dtype=np.float32,
    )
    assert task.take_action(action, action_type="ee", force=True) == (True, False)
    assert task._robotactile_n0_last_plan["execution_purpose"] == (
        "n0_training_aligned_execution_v1"
    )

    with pytest.raises(UniVTACContractError, match="scope mismatch"):
        install_n0_action_execution(
            _task(),
            config,
            execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
            diagnostic_only=True,
        )


def _stock_task() -> object:
    calls: list[tuple[object, str, bool]] = []

    def take_action(
        self: object,
        action: object,
        action_type: str = "qpos",
        force: bool = True,
    ) -> tuple[bool, bool]:
        calls.append((action, action_type, force))
        return True, False

    base_task = type(
        "BaseTask",
        (),
        {
            "__module__": "envs._base_task",
            "take_action": take_action,
        },
    )
    task_type = type("Task", (base_task,), {"__module__": "envs.lift_bottle"})
    task = task_type()
    task.mode = "eval"
    task.calls = calls
    return task


def test_stock_ee_preserves_upstream_dense_waypoint_method_identity() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _stock_task()
    before = task.take_action.__func__

    assert install_n0_stock_ee_execution(task, config)

    assert task.take_action.__func__ is before
    assert task._robotactile_n0_stock_ee_enabled is True
    assert task._robotactile_n0_action_execution_contract == "univtac_stock_ee_v1"
    assert task.take_action("action", action_type="ee", force=True) == (True, False)
    assert task.calls == [("action", "ee", True)]


def test_stock_ee_rejects_task_override_and_fixed_is_diagnostic_only() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _stock_task()
    task.take_action = lambda *_args, **_kwargs: (True, False)

    with pytest.raises(UniVTACContractError, match="BaseTask.take_action"):
        install_n0_stock_ee_execution(task, config)
    with pytest.raises(UniVTACContractError, match="training-endpoint diagnostics"):
        install_n0_action_execution(
            _task(),
            config,
            execution_contract="robotactile_fixed_endpoint_v1",
            diagnostic_only=False,
        )


def test_fixed_cadence_fails_cleanly_when_curobo_has_no_plan() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _task(planning_succeeds=False)
    install_n0_fixed_cadence(task, config, diagnostic_only=True)

    result = task.take_action(
        np.asarray([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02]),
        action_type="ee",
        force=True,
    )

    assert result == (False, False)
    assert task.plan_success is False
    assert task.step_count == 17
    assert task._robotactile_n0_last_plan["status"] == "Fail"
    assert task._robotactile_n0_last_plan["stock_move_loop_used"] is False


@pytest.mark.parametrize(
    ("mode", "has_update_render", "message"),
    [
        ("collect", True, "requires eval mode"),
        ("eval", False, "surface is incomplete"),
    ],
)
def test_fixed_cadence_rejects_incompatible_render_surface(
    mode: str,
    has_update_render: bool,
    message: str,
) -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _task()
    task.mode = mode
    if not has_update_render:
        task._update_render = None

    with pytest.raises(UniVTACContractError, match=message):
        install_n0_fixed_cadence(task, config, diagnostic_only=True)


def test_fixed_cadence_restores_eval_mode_when_step_raises() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _task()
    observed_modes: list[str] = []

    def failing_step(*, is_save: bool) -> None:
        assert not is_save
        observed_modes.append(task.mode)
        task.mode = "broken"
        raise RuntimeError("step failed")

    task._step = failing_step
    install_n0_fixed_cadence(task, config, diagnostic_only=True)

    with pytest.raises(RuntimeError, match="step failed"):
        task.take_action(
            np.asarray([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02]),
            action_type="ee",
            force=True,
        )

    assert observed_modes == ["eval_test"]
    assert task.mode == "eval"
    assert task.render_count == 0


def test_fixed_cadence_restores_eval_mode_when_endpoint_render_raises() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    task = _task()
    observed_modes: list[str] = []

    def failing_render() -> None:
        observed_modes.append(task.mode)
        task.render_count += 1
        task.mode = "broken"
        raise RuntimeError("render failed")

    task._update_render = failing_render
    install_n0_fixed_cadence(task, config, diagnostic_only=True)

    with pytest.raises(RuntimeError, match="render failed"):
        task.take_action(
            np.asarray([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02]),
            action_type="ee",
            force=True,
        )

    assert task.step_modes == ["eval_test", "eval_test"]
    assert observed_modes == ["eval"]
    assert task.render_count == 1
    assert task.mode == "eval"


def test_n0_reset_preserves_released_evaluator_mode() -> None:
    config = build_univtac_backend_config("lift_bottle", action_spec=EE8_ACTION_SPEC)
    observed_modes: list[str] = []
    task = SimpleNamespace(mode="eval")

    def reset(*, seed: int, instructions: list[str]) -> dict[str, object]:
        observed_modes.append(task.mode)
        return {"seed": seed, "instructions": instructions}

    task.reset = reset
    assert install_n0_evaluation_reset(task, config)

    result = task.reset(seed=90, instructions=["lift"])

    assert observed_modes == ["eval"]
    assert task.mode == "eval"
    assert result == {"seed": 90, "instructions": ["lift"]}
    assert task._robotactile_n0_last_reset == {
        "mode_after": "eval",
        "mode_before": "eval",
        "mode_during": "eval",
        "reset_contract": "univtac_released_evaluator_eval_v1",
    }
