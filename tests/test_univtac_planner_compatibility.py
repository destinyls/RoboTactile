from __future__ import annotations

import types
from unittest.mock import patch

from robotactile_benchmark.backends.univtac_planner_compatibility import (
    LOCAL_IK_FALLBACK_MARKER,
    install_local_ik_fallback,
)


class _Scalar:
    def __init__(self, value: bool) -> None:
        self._value = value

    def item(self) -> bool:
        return self._value


class _PlanConfig:
    def __init__(self) -> None:
        self.partial_ik_opt = False

    def clone(self) -> _PlanConfig:
        clone = _PlanConfig()
        clone.partial_ik_opt = self.partial_ik_opt
        return clone


class _Zeroable:
    def __init__(self, value: float) -> None:
        self.value = value

    def zero_(self) -> _Zeroable:
        self.value = 0.0
        return self


class _StartState:
    def __init__(self, value: float = 1.0) -> None:
        self.velocity = _Zeroable(value)
        self.acceleration = _Zeroable(value)
        self.jerk = _Zeroable(value)

    def clone(self) -> _StartState:
        return _StartState(self.velocity.value)


def test_marked_insert_hole_action_retries_only_an_ik_failure() -> None:
    calls: list[bool] = []
    messages: list[str] = []

    def plan_single(_start: object, _goal: object, config: _PlanConfig) -> object:
        calls.append(config.partial_ik_opt)
        return types.SimpleNamespace(
            status=types.SimpleNamespace(
                value="Success" if config.partial_ik_opt else "IK Fail"
            ),
            success=_Scalar(config.partial_ik_opt),
        )

    motion_gen = types.SimpleNamespace(plan_single=plan_single)
    action = types.SimpleNamespace(
        args={LOCAL_IK_FALLBACK_MARKER: True, "pre_dis": None}
    )

    def move(actions: list[object]) -> object:
        assert LOCAL_IK_FALLBACK_MARKER not in action.args
        return motion_gen.plan_single(object(), object(), _PlanConfig())

    task = types.SimpleNamespace(
        _robot_manager=types.SimpleNamespace(
            planner=types.SimpleNamespace(motion_gen=motion_gen)
        ),
        logger=types.SimpleNamespace(info=messages.append, error=messages.append),
        move=move,
    )
    assert install_local_ik_fallback(task, "insert_hole")

    result = task.move([action])

    assert result.success.item()
    assert calls == [False, True]
    assert len(messages) == 1
    assert '"retry_success":true' in messages[0]


def test_unmarked_actions_and_other_tasks_keep_the_upstream_path() -> None:
    calls: list[bool] = []

    def plan_single(_start: object, _goal: object, config: _PlanConfig) -> object:
        calls.append(config.partial_ik_opt)
        return types.SimpleNamespace(
            status=types.SimpleNamespace(value="IK Fail"),
            success=_Scalar(False),
        )

    motion_gen = types.SimpleNamespace(plan_single=plan_single)
    task = types.SimpleNamespace(
        _robot_manager=types.SimpleNamespace(
            planner=types.SimpleNamespace(motion_gen=motion_gen)
        ),
        logger=types.SimpleNamespace(
            info=lambda _message: None, error=lambda _message: None
        ),
        move=lambda actions: motion_gen.plan_single(object(), object(), _PlanConfig()),
    )
    assert not install_local_ik_fallback(task, "pull_out_key")
    assert install_local_ik_fallback(task, "insert_hole")

    result = task.move([types.SimpleNamespace(args={})])

    assert not result.success.item()
    assert calls == [False]


def test_failed_partial_ik_uses_bounded_differential_plan() -> None:
    calls: list[bool] = []
    messages: list[str] = []
    differential = types.SimpleNamespace(
        status="Differential IK Success",
        success=_Scalar(True),
    )

    def plan_single(_start: object, _goal: object, config: _PlanConfig) -> object:
        calls.append(config.partial_ik_opt)
        return types.SimpleNamespace(
            status=types.SimpleNamespace(value="IK Fail"),
            success=_Scalar(False),
        )

    motion_gen = types.SimpleNamespace(plan_single=plan_single)
    action = types.SimpleNamespace(
        args={LOCAL_IK_FALLBACK_MARKER: True},
        target_pose=object(),
    )

    def move(actions: list[object]) -> object:
        return motion_gen.plan_single(object(), object(), _PlanConfig())

    task = types.SimpleNamespace(
        _robot_manager=types.SimpleNamespace(
            planner=types.SimpleNamespace(motion_gen=motion_gen)
        ),
        logger=types.SimpleNamespace(info=messages.append, error=messages.append),
        move=move,
    )
    with patch(
        "robotactile_benchmark.backends.univtac_planner_compatibility."
        "_differential_ik_plan",
        return_value=(differential, {"steps": 8, "translation_m": 0.008}),
    ):
        assert install_local_ik_fallback(task, "insert_hole")
        result = task.move([action])

    assert result is differential
    assert calls == [False, True]
    assert len(messages) == 1
    assert '"retry_success":true' in messages[0]
    assert '"steps":8' in messages[0]


def test_direct_grasp_action_retries_local_trajopt_failure() -> None:
    calls: list[tuple[bool, float]] = []
    messages: list[str] = []
    bounded_plan = types.SimpleNamespace(
        status="Differential IK Success",
        success=_Scalar(True),
    )

    def plan_single(start: _StartState, _goal: object, config: _PlanConfig) -> object:
        calls.append((config.partial_ik_opt, start.velocity.value))
        success = start.velocity.value == 0.0
        return types.SimpleNamespace(
            status=types.SimpleNamespace(
                value="Success" if success else "Finetune TrajOpt Fail"
            ),
            success=_Scalar(success),
        )

    motion_gen = types.SimpleNamespace(plan_single=plan_single)
    action = types.SimpleNamespace(
        action="all",
        args={"constraint_pose": None},
        target_pose=object(),
    )

    def move(actions: list[object]) -> object:
        assert actions == [action]
        return motion_gen.plan_single(_StartState(), object(), _PlanConfig())

    task = types.SimpleNamespace(
        _robot_manager=types.SimpleNamespace(
            planner=types.SimpleNamespace(motion_gen=motion_gen)
        ),
        logger=types.SimpleNamespace(info=messages.append, error=messages.append),
        move=move,
    )
    with patch(
        "robotactile_benchmark.backends.univtac_planner_compatibility."
        "_differential_ik_plan",
        return_value=(
            bounded_plan,
            {
                "max_joint_delta_rad": 0.04,
                "quaternion_alignment": 0.9989,
                "translation_m": 0.0085,
            },
        ),
    ):
        assert install_local_ik_fallback(task, "grasp_classify")
        result = task.move([action])

    assert result.success.item()
    assert calls == [(False, 1.0), (False, 0.0)]
    assert len(messages) == 1
    assert '"initial_status":"Finetune TrajOpt Fail"' in messages[0]
    assert '"mode":"direct_ee"' in messages[0]
    assert '"zero_dynamics_status":"Success"' in messages[0]


def test_direct_grasp_action_rejects_pose_outside_local_gate() -> None:
    calls: list[bool] = []
    messages: list[str] = []

    def plan_single(_start: _StartState, _goal: object, config: _PlanConfig) -> object:
        calls.append(config.partial_ik_opt)
        return types.SimpleNamespace(
            status=types.SimpleNamespace(value="Finetune TrajOpt Fail"),
            success=_Scalar(False),
        )

    motion_gen = types.SimpleNamespace(plan_single=plan_single)
    action = types.SimpleNamespace(action="all", args={}, target_pose=object())
    task = types.SimpleNamespace(
        _robot_manager=types.SimpleNamespace(
            planner=types.SimpleNamespace(motion_gen=motion_gen)
        ),
        logger=types.SimpleNamespace(info=messages.append, error=messages.append),
        move=lambda actions: motion_gen.plan_single(
            _StartState(), object(), _PlanConfig()
        ),
    )
    with patch(
        "robotactile_benchmark.backends.univtac_planner_compatibility."
        "_differential_ik_plan",
        return_value=(
            None,
            {
                "quaternion_alignment": 0.99,
                "rejected": "nonlocal_pose_delta",
                "translation_m": 0.02,
            },
        ),
    ):
        assert install_local_ik_fallback(task, "grasp_classify")
        result = task.move([action])

    assert not result.success.item()
    assert calls == [False]
    assert len(messages) == 1
    assert '"rejected":"nonlocal_pose_delta"' in messages[0]
    assert '"retry_success":false' in messages[0]
