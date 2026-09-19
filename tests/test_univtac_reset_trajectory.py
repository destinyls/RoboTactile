"""Tests for source-bound UniVTAC pre-move capture and replay."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    PRE_MOVE_CAPTURE_MODE,
    UniVTACPreMoveSegment,
    UniVTACPreMoveTrajectory,
    UniVTACPreMoveTrajectoryError,
    install_pre_move_trajectory_capture,
    install_pre_move_trajectory_replay,
)


class _Tensor:
    def __init__(self, value: object, token: str = "upstream") -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.token = token

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value.copy()

    def new_tensor(self, value: object) -> _Tensor:
        return _Tensor(value, token="live")

    def reshape(self, *shape: int) -> _Tensor:
        return _Tensor(self.value.reshape(*shape), token=self.token)


class _Pose:
    def __init__(self, bias: float = 0.0) -> None:
        self.value = [0.35 + bias, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]

    def tolist(self) -> list[float]:
        return list(self.value)


class _Manager:
    def __init__(self) -> None:
        self.qpos = _Tensor(np.zeros((1, 9), dtype=np.float32))
        self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=self.qpos))
        self.gripper_planner_calls = 0
        self.arm_planner_calls = 0
        self.arm_hold_calls: list[tuple[_Tensor, _Tensor, bool]] = []

    def get_qpos(self) -> _Tensor:
        return self.qpos

    def plan_gripper(self, pos: float, type: str = "percent") -> dict[str, object]:
        self.gripper_planner_calls += 1
        assert type == "percent"
        return {
            "status": "Success",
            "num_steps": 2,
            "position": _Tensor([0.0, pos]),
            "velocity": _Tensor([0.0, 0.01]),
        }

    def plan_arm(
        self,
        target_pose: _Pose,
        constraint_pose: object = None,
        pre_dis: float | None = None,
        time_dilation_factor: float | None = None,
    ) -> dict[str, object]:
        self.arm_planner_calls += 1
        del target_pose, constraint_pose, pre_dis, time_dilation_factor
        return {
            "status": "Success",
            "num_steps": 2,
            "position": _Tensor(np.arange(14, dtype=np.float32).reshape(2, 7)),
            "velocity": _Tensor(np.ones((2, 7), dtype=np.float32) * 0.1),
        }

    def set_arm(
        self, position: _Tensor, velocity: _Tensor, *, force: bool = True
    ) -> None:
        self.arm_hold_calls.append((position, velocity, force))


class _Task:
    def __init__(
        self,
        *,
        pose_bias: float = 0.0,
        wrong_order: bool = False,
        reset_native_step: int = 10,
    ) -> None:
        self._robot_manager = _Manager()
        self.in_pre_move = False
        self.pose_bias = pose_bias
        self.wrong_order = wrong_order
        self.reset_native_step = reset_native_step
        self.step_count = 0
        self.dense_calls = 0
        self.delay_calls = 0
        self.executed: list[dict[str, object]] = []

    def _execute(self, result: dict[str, object]) -> None:
        self.executed.append(result)
        self.dense_calls += 1
        self.delay_calls += 1

    def _step(self, *, is_save: bool) -> None:
        assert is_save is False
        self.step_count += 1

    def reset(self, seed: int, instructions: list[str] | None = None) -> str:
        del seed, instructions
        self.in_pre_move = True
        try:
            manager = self._robot_manager
            if self.wrong_order:
                self._execute(manager.plan_arm(_Pose(self.pose_bias), pre_dis=0.04))
                return "unreachable"
            self._execute(manager.plan_gripper(0.5))
            self._execute(manager.plan_arm(_Pose(self.pose_bias), pre_dis=0.04))
            self._execute(manager.plan_gripper(0.18))
            self._execute(manager.plan_arm(_Pose(0.05 + self.pose_bias)))
            self.step_count = self.reset_native_step
            return "reset"
        finally:
            self.in_pre_move = False


def _finish(capture: Any, *, initial_seed: int = 938) -> UniVTACPreMoveTrajectory:
    return capture.finish(
        task_id="grasp_classify",
        action_spec=QPOS8_ACTION_SPEC,
        initial_seed=initial_seed,
        exogenous_seed=1_000_000,
        pair_key="a" * 64,
        dataset_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        config_sha256="d" * 64,
        source_run_content_sha256="e" * 64,
        reset_reference_sha256="f" * 64,
        capture_reset_receipt_sha256="1" * 64,
        capture_simulator_state_sha256="4" * 64,
        capture_native_step=240,
        capture_qpos8=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.007),
        source_qpos8_max_abs_error=0.001,
        upstream_commit="2" * 40,
        task_source_sha256="3" * 64,
    )


def _captured(*, expected_native_step: int | None = None) -> UniVTACPreMoveTrajectory:
    task = _Task()
    capture = install_pre_move_trajectory_capture(
        task, expected_native_step=expected_native_step
    )
    assert task.reset(seed=938, instructions=["classify"]) == "reset"
    return _finish(capture)


def test_capture_freezes_exact_four_segment_source_bound_document() -> None:
    trajectory = _captured()

    assert [segment.kind for segment in trajectory.segments] == [
        "gripper",
        "arm",
        "gripper",
        "arm",
    ]
    assert len(trajectory.segments[0].position[0]) == 1
    assert len(trajectory.segments[1].position[0]) == 7
    assert trajectory.capture_mode == PRE_MOVE_CAPTURE_MODE
    assert trajectory.post_reset_hold_steps == 0
    assert trajectory.capture_native_step == 240
    assert trajectory.capture_qpos8[-1] == 0.007
    assert trajectory.source_qpos8_max_abs_error == 0.001
    assert len(trajectory.sha256) == 64
    assert UniVTACPreMoveTrajectory.from_dict(trajectory.to_dict()) == trajectory
    with pytest.raises(FrozenInstanceError):
        trajectory.task_id = "other"  # type: ignore[misc]


def test_replay_bypasses_planners_but_preserves_dense_and_delay_path() -> None:
    trajectory = _captured()
    task = _Task()
    assert install_pre_move_trajectory_replay(task, trajectory) is True

    assert task.reset(seed=938, instructions=["classify"]) == "reset"

    assert task._robot_manager.gripper_planner_calls == 0
    assert task._robot_manager.arm_planner_calls == 0
    assert task.dense_calls == 4
    assert task.delay_calls == 4
    assert all(result["position"].token == "live" for result in task.executed)
    assert task._robotactile_pre_move_trajectory_replay_witness == {
        "consumed_segment_count": 4,
        "post_reset_hold_steps": 0,
        "pre_hold_native_step": 10,
        "post_hold_native_step": 10,
        "request_fingerprint_all_match": True,
        "request_fingerprint_checks": tuple(
            {
                "segment_index": index,
                "kind": segment.kind,
                "expected": segment.request_fingerprint,
                "actual": segment.request_fingerprint,
                "exact_match": True,
            }
            for index, segment in enumerate(trajectory.segments)
        ),
        "trajectory_sha256": trajectory.sha256,
    }


def test_capture_and_replay_add_one_post_reset_physics_hold() -> None:
    source = _Task()
    capture = install_pre_move_trajectory_capture(source, expected_native_step=11)
    source.reset(seed=938)
    trajectory = _finish(capture)

    assert trajectory.post_reset_hold_steps == 1
    assert [segment.kind for segment in trajectory.segments] == [
        "gripper",
        "arm",
        "gripper",
        "arm",
    ]
    assert source.step_count == 11
    assert len(source._robot_manager.arm_hold_calls) == 1
    source_position, source_velocity, source_force = (
        source._robot_manager.arm_hold_calls[0]
    )
    assert source_position.token == source_velocity.token == "live"
    assert source_force is False
    np.testing.assert_allclose(source_position.value, np.arange(7, 14))

    replay = _Task()
    install_pre_move_trajectory_replay(replay, trajectory)
    replay.reset(seed=938)
    assert replay.step_count == 11
    assert len(replay._robot_manager.arm_hold_calls) == 1
    assert replay._robot_manager.arm_hold_calls[0][2] is False
    assert replay._robotactile_pre_move_trajectory_replay_witness == {
        "consumed_segment_count": 4,
        "post_reset_hold_steps": 1,
        "pre_hold_native_step": 10,
        "post_hold_native_step": 11,
        "request_fingerprint_all_match": True,
        "request_fingerprint_checks": tuple(
            {
                "segment_index": index,
                "kind": segment.kind,
                "expected": segment.request_fingerprint,
                "actual": segment.request_fingerprint,
                "exact_match": True,
            }
            for index, segment in enumerate(trajectory.segments)
        ),
        "trajectory_sha256": trajectory.sha256,
    }


def test_post_reset_hold_is_strictly_bounded_to_two_steps() -> None:
    trajectory = _captured()
    for invalid in (-1, 3):
        with pytest.raises(UniVTACPreMoveTrajectoryError, match="between 0 and 2"):
            replace(trajectory, post_reset_hold_steps=invalid)

    for expected_native_step in (9, 13):
        task = _Task()
        install_pre_move_trajectory_capture(
            task, expected_native_step=expected_native_step
        )
        with pytest.raises(UniVTACPreMoveTrajectoryError, match="between 0 and 2"):
            task.reset(seed=938)
        assert task._robot_manager.arm_hold_calls == []


def test_capture_anchor_fields_are_strictly_validated() -> None:
    trajectory = _captured()
    invalid_values = (
        ("capture_simulator_state_sha256", "A" * 64, "lowercase SHA256"),
        ("capture_native_step", -1, "non-negative integer"),
        ("capture_native_step", True, "non-negative integer"),
        ("capture_qpos8", (0.0,) * 7, "width 8"),
        ("capture_qpos8", (0.0,) * 7 + (float("nan"),), "finite"),
        ("source_qpos8_max_abs_error", -0.001, "between 0 and 0.002"),
        ("source_qpos8_max_abs_error", 0.0021, "between 0 and 0.002"),
        ("source_qpos8_max_abs_error", float("inf"), "between 0 and 0.002"),
    )
    for field, value, message in invalid_values:
        with pytest.raises(UniVTACPreMoveTrajectoryError, match=message):
            replace(trajectory, **{field: value})


def test_planners_outside_wrapped_reset_are_not_intercepted() -> None:
    task = _Task()
    install_pre_move_trajectory_replay(task, _captured())

    result = task._robot_manager.plan_gripper(0.5)

    assert result["position"].token == "upstream"
    assert task._robot_manager.gripper_planner_calls == 1


def test_replay_records_request_drift_but_rejects_start_state_mismatch() -> None:
    trajectory = _captured()
    request_task = _Task(pose_bias=0.001)
    install_pre_move_trajectory_replay(request_task, trajectory)
    request_task.reset(seed=938)
    assert request_task._robot_manager.arm_planner_calls == 0
    request_witness = request_task._robotactile_pre_move_trajectory_replay_witness
    assert request_witness["request_fingerprint_all_match"] is False
    assert any(
        item["exact_match"] is False
        for item in request_witness["request_fingerprint_checks"]
    )

    start_task = _Task()
    start_task._robot_manager.qpos = _Tensor(np.ones((1, 9)) * 0.1)
    start_task._robot_manager.robot.data.joint_pos = start_task._robot_manager.qpos
    install_pre_move_trajectory_replay(start_task, trajectory)
    with pytest.raises(UniVTACPreMoveTrajectoryError, match="start_joint9"):
        start_task.reset(seed=938)


def test_capture_rejects_wrong_order_and_mismatched_finish_seed() -> None:
    wrong = _Task(wrong_order=True)
    install_pre_move_trajectory_capture(wrong)
    with pytest.raises(UniVTACPreMoveTrajectoryError, match="order"):
        wrong.reset(seed=938)

    task = _Task()
    capture = install_pre_move_trajectory_capture(task)
    task.reset(seed=938)
    with pytest.raises(UniVTACPreMoveTrajectoryError, match="seed mismatch"):
        _finish(capture, initial_seed=939)


def test_none_replay_is_a_true_no_op_and_documents_are_strict() -> None:
    task = _Task()
    assert install_pre_move_trajectory_replay(task, None) is False
    assert not hasattr(task, "_robotactile_pre_move_trajectory_mode")

    trajectory = _captured()
    document = trajectory.to_dict()
    document["unknown"] = True
    with pytest.raises(UniVTACPreMoveTrajectoryError, match="fields mismatch"):
        UniVTACPreMoveTrajectory.from_dict(document)

    segment = trajectory.segments[0].to_dict()
    segment["kind"] = "arm"
    with pytest.raises(UniVTACPreMoveTrajectoryError):
        UniVTACPreMoveSegment.from_dict(segment)
