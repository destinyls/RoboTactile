from __future__ import annotations

import types

import numpy as np

from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)


class _Pose:
    def __init__(
        self,
        position: tuple[float, float, float],
        *,
        x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._x_axis = np.asarray(x_axis, dtype=np.float64)
        self._z_axis = np.asarray(z_axis, dtype=np.float64)

    def rebase(self, target: _Pose) -> _Pose:
        return _Pose(
            tuple(float(item) for item in self.p - target.p),
            x_axis=tuple(float(item) for item in self._x_axis),
            z_axis=tuple(float(item) for item in self._z_axis),
        )

    def to_transformation_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 0] = self._x_axis
        matrix[:3, 2] = self._z_axis
        matrix[:3, 3] = self.p
        return matrix


class _Actor:
    def __init__(self, pose: _Pose) -> None:
        self.pose = pose

    def get_pose(self) -> _Pose:
        return self.pose


def _tactile(minimum_depth: float) -> object:
    return types.SimpleNamespace(
        get_min_depth=lambda: np.asarray((minimum_depth, minimum_depth + 1.0))
    )


def test_insert_hdmi_records_upstream_x_omission_and_corrected_predicate() -> None:
    task = types.SimpleNamespace(
        prism=_Actor(_Pose((0.1, 0.0, 0.0))),
        target_pose=_Pose((0.0, 0.0, 0.0)),
        _robot_manager=types.SimpleNamespace(
            get_ee_pose=lambda: np.asarray((0.0, 0.0, 0.2, 1.0))
        ),
        _tactile_manager=_tactile(31.0),
    )

    diagnostics = capture_task_diagnostics(task, "insert_HDMI")

    assert diagnostics["available"] is True
    assert diagnostics["predicate_success"] is True
    assert diagnostics["predicate_success_corrected_xy"] is False
    assert diagnostics["upstream_x_coordinate_omitted"] is True
    assert diagnostics["tactile_contact"]["contact_qualified"] is True


def test_lift_can_reports_both_upstream_early_stop_reasons() -> None:
    can = _Actor(
        _Pose(
            (0.7, 0.0, 0.02),
            x_axis=(0.0, 0.0, 1.0),
            z_axis=(0.0, 0.0, 1.0),
        )
    )
    task = types.SimpleNamespace(
        can=can,
        origin_inhand_pose=_Pose((0.0, 0.0, 0.0)),
        _robot_manager=types.SimpleNamespace(
            get_inhand_pose=lambda _actor: _Pose((0.0, 0.0, 0.06))
        ),
        _tactile_manager=_tactile(19.0),
    )

    diagnostics = capture_task_diagnostics(task, "lift_can")

    assert diagnostics["available"] is True
    assert diagnostics["early_stop_predicate"] is True
    assert diagnostics["early_stop_reasons"] == [
        "tactile_min_depth_lt_20",
        "inhand_z_bias_gt_0_05_while_upright",
    ]
    assert diagnostics["predicate_success"] is False
    assert diagnostics["inhand_z_bias_m"] == 0.06


def test_pull_out_key_identifies_distance_and_slot_motion_failures() -> None:
    task = types.SimpleNamespace(
        key=_Actor(_Pose((0.5, 0.0, 0.1))),
        slot=_Actor(_Pose((0.5, 0.0, 0.0), x_axis=(0.5, 0.0, 0.0))),
        slot_init_pose=_Pose((0.5, 0.0, 0.0)),
        _robot_manager=types.SimpleNamespace(
            get_ee_pose=lambda: np.asarray((0.5, 0.0, 0.3, 1.0))
        ),
        _tactile_manager=_tactile(35.0),
    )

    diagnostics = capture_task_diagnostics(task, "pull_out_key")

    assert diagnostics["early_stop_predicate"] is True
    assert diagnostics["early_stop_reasons"] == [
        "ee_key_z_distance_gte_0_14",
        "slot_x_alignment_lt_0_99",
    ]
    assert diagnostics["ee_key_z_distance_m"] == 0.19999999999999998
    assert diagnostics["predicate_success"] is False


def test_put_bottle_in_shelf_separates_pose_success_from_contact_loss() -> None:
    task = types.SimpleNamespace(
        bottle=_Actor(_Pose((0.91, 0.02, 0.205))),
        place_target=_Pose((0.9, 0.0, 0.21)),
        _tactile_manager=_tactile(18.0),
    )

    diagnostics = capture_task_diagnostics(task, "put_bottle_in_shelf")

    assert diagnostics["predicate_success"] is True
    assert diagnostics["early_stop_predicate"] is True
    assert diagnostics["early_stop_reasons"] == ["tactile_min_depth_lt_20"]
    assert diagnostics["relative_position_xyz"] == [
        0.010000000000000009,
        0.02,
        -0.0050000000000000044,
    ]


def test_extended_diagnostics_report_missing_runtime_state_explicitly() -> None:
    for task_id in (
        "insert_HDMI",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    ):
        diagnostics = capture_task_diagnostics(object(), task_id)
        assert diagnostics["available"] is False
        assert "unavailable_reason" in diagnostics

    assert capture_task_diagnostics(object(), "unknown_task") == {}
