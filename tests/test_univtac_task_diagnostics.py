from __future__ import annotations

import numpy as np
import pytest

from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)


class _Pose:
    def __init__(
        self,
        position: tuple[float, float, float],
        *,
        z_alignment: float = 1.0,
    ) -> None:
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self.z_alignment = z_alignment

    def rebase(self, target: _Pose) -> _Pose:
        return _Pose(
            tuple(float(item) for item in self.p - target.p),
            z_alignment=self.z_alignment,
        )

    def to_transformation_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 0] = (0.0, 0.0, 1.0)
        matrix[:3, 2] = (0.0, 0.0, self.z_alignment)
        return matrix


class _Bottle:
    def __init__(self, pose: _Pose) -> None:
        self._pose = pose

    def get_pose(self) -> _Pose:
        return self._pose


class _Task:
    def __init__(self) -> None:
        self.target_pose = _Pose((0.5, 0.0, 0.2))
        self.bottle = _Bottle(_Pose((0.49, 0.05, 0.2)))
        self._robotactile_tactile_attachment = {
            "attachment_count_per_side": 83,
            "indices_sha256": "pinned",
        }


class _GraspTask:
    def __init__(self) -> None:
        self.target_pose = _Pose((0.4, -0.08, 0.025))
        self.prism = _Bottle(_Pose((0.41, -0.075, 0.03)))
        self._robot_manager = type(
            "RobotManager",
            (),
            {"get_gripper_center_pose": lambda _self: _Pose((0.41, -0.075, 0.08))},
        )()
        self._robotactile_grasp_initialization = {
            "grasp_qpos_m": 0.0065,
            "lift_m": 0.05,
            "passed": True,
            "relative_drift_m": 0.0002,
        }
        self._robotactile_tactile_attachment = {
            "attachment_count_per_side": 83,
            "indices_sha256": "pinned",
        }


class _InsertionTask:
    def __init__(self, *, prism_z: float) -> None:
        self.prism = _Bottle(_Pose((0.4, 0.0, prism_z)))
        self.origin_inhand_pose = _Pose((0.0, 0.0, -0.05))
        self._robot_manager = type(
            "RobotManager",
            (),
            {"get_gripper_center_pose": lambda _self: _Pose((0.4, 0.0, 0.08))},
        )()


def _placement_witness(
    phase: str,
    *,
    before_bias: float,
    after_bias: float,
    before_early_stop: bool,
    after_early_stop: bool,
) -> dict[str, object]:
    return {
        "phase": phase,
        "before_inhand_z_bias_m": before_bias,
        "inhand_z_bias_m": after_bias,
        "early_stop_predicate_before": before_early_stop,
        "early_stop_predicate": after_early_stop,
        "plan_success_after": True,
        "native_step_before": 100,
        "native_step_after": 120,
    }


def test_lift_bottle_diagnostics_reproduce_predicate_subconditions() -> None:
    diagnostics = capture_task_diagnostics(_Task(), "lift_bottle")

    assert diagnostics["predicate_success"] is True
    assert diagnostics["relative_position_xyz"] == [-0.010000000000000009, 0.05, 0.0]
    conditions = diagnostics["success_conditions"]
    assert isinstance(conditions, dict)
    assert all(conditions.values())
    attachment = diagnostics["tactile_attachment"]
    assert isinstance(attachment, dict)
    assert attachment["attachment_count_per_side"] == 83
    missing_key = capture_task_diagnostics(_Task(), "pull_out_key")
    assert missing_key["available"] is False
    assert missing_key["unavailable_reason"] == "key_slot_or_robot_pose_unavailable"

    unavailable = capture_task_diagnostics(object(), "lift_bottle")
    assert unavailable["available"] is False
    assert unavailable["unavailable_reason"] == (
        "task_actors_or_tactile_attachment_unavailable"
    )


def test_grasp_classify_diagnostics_persist_initialization_and_predicate() -> None:
    diagnostics = capture_task_diagnostics(_GraspTask(), "grasp_classify")

    assert diagnostics["available"] is True
    assert diagnostics["predicate_success"] is True
    assert diagnostics["relative_position_xyz"] == pytest.approx([0.01, 0.005, 0.005])
    initialization = diagnostics["initialization"]
    assert isinstance(initialization, dict)
    assert initialization["passed"] is True
    attachment = diagnostics["tactile_attachment"]
    assert isinstance(attachment, dict)
    assert attachment["attachment_count_per_side"] == 83

    unavailable = capture_task_diagnostics(object(), "grasp_classify")
    assert unavailable["available"] is False


def test_insertion_diagnostics_reproduce_official_inhand_early_stop() -> None:
    stable = capture_task_diagnostics(_InsertionTask(prism_z=0.03), "insert_hole")
    dropped_hole = capture_task_diagnostics(
        _InsertionTask(prism_z=-0.02), "insert_hole"
    )
    dropped_tube = capture_task_diagnostics(
        _InsertionTask(prism_z=0.005), "insert_tube"
    )
    failed_tube = capture_task_diagnostics(_InsertionTask(prism_z=-0.01), "insert_tube")

    assert stable["inhand_z_bias_m"] == pytest.approx(0.0)
    assert stable["early_stop_predicate"] is False
    assert dropped_hole["inhand_z_bias_m"] == pytest.approx(0.05)
    assert dropped_hole["early_stop_threshold_m"] == 0.04
    assert dropped_hole["early_stop_predicate"] is True
    assert dropped_tube["inhand_z_bias_m"] == pytest.approx(0.025)
    assert dropped_tube["early_stop_threshold_m"] == 0.03
    assert dropped_tube["early_stop_predicate"] is False
    assert failed_tube["inhand_z_bias_m"] == pytest.approx(0.04)
    assert failed_tube["early_stop_predicate"] is True

    unavailable = capture_task_diagnostics(object(), "insert_tube")
    assert unavailable["available"] is False


def test_insertion_diagnostics_locate_placement_reset_failure_phase() -> None:
    task = _InsertionTask(prism_z=-0.01)
    task._robotactile_placement_witnesses = (
        _placement_witness(
            "approach_complete",
            before_bias=0.0,
            after_bias=0.01,
            before_early_stop=False,
            after_early_stop=False,
        ),
        _placement_witness(
            "final_continuation_complete",
            before_bias=0.01,
            after_bias=0.04,
            before_early_stop=False,
            after_early_stop=True,
        ),
    )

    diagnostics = capture_task_diagnostics(task, "insert_tube")
    assessment = diagnostics["placement_reset_assessment"]

    assert assessment["reset_viable"] is False
    assert assessment["failure_phase"] == "final_continuation_complete"
    summaries = assessment["phase_summaries"]
    assert summaries[1]["threshold_crossed"] is True
    assert summaries[1]["inhand_z_bias_delta_m"] == pytest.approx(0.03)


def test_insertion_diagnostics_distinguish_post_final_settle_failure() -> None:
    task = _InsertionTask(prism_z=-0.01)
    task._robotactile_placement_witnesses = (
        _placement_witness(
            "approach_complete",
            before_bias=0.0,
            after_bias=0.01,
            before_early_stop=False,
            after_early_stop=False,
        ),
        _placement_witness(
            "final_continuation_complete",
            before_bias=0.01,
            after_bias=0.02,
            before_early_stop=False,
            after_early_stop=False,
        ),
    )

    diagnostics = capture_task_diagnostics(task, "insert_tube")
    assessment = diagnostics["placement_reset_assessment"]

    assert assessment["reset_viable"] is False
    assert assessment["failure_phase"] == "post_final_settle"


def test_insertion_diagnostics_report_incomplete_placement_sequence() -> None:
    task = _InsertionTask(prism_z=0.03)
    task._robotactile_placement_witnesses = (
        _placement_witness(
            "approach_complete",
            before_bias=0.0,
            after_bias=0.0,
            before_early_stop=False,
            after_early_stop=False,
        ),
    )

    diagnostics = capture_task_diagnostics(task, "insert_tube")
    assessment = diagnostics["placement_reset_assessment"]

    assert assessment["reset_viable"] is False
    assert assessment["failure_phase"] == "final_continuation_missing"
    assert assessment["observed_phase_count"] == 1
    assert assessment["expected_phase_count"] == 2
    assert assessment["phase_sequence_complete"] is False
