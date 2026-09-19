"""Tests for the N0 empty-gripper rest-calibration hook."""

from __future__ import annotations

import numpy as np
import pytest

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC, QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    QPOS_ACTION_EXECUTION_CONTRACT,
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_rest_calibration import (
    ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
    N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
    install_act_empty_gripper_rest_calibration,
    install_n0_empty_gripper_rest_calibration,
)
from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)


class FakeCalibrationTask:
    def __init__(self) -> None:
        self.mode = "eval"
        self.step_count = 0
        self.take_action_cnt = 0
        self.plan_success = True
        self.eval_success = False
        self.delay_calls: list[tuple[int, bool]] = []
        self.render_count = 0

    def delay(self, steps: int, *, is_save: bool) -> None:
        self.delay_calls.append((steps, is_save))

    def _step(self, *, is_save: bool) -> None:
        assert is_save is False
        self.step_count += 1

    def _update_render(self) -> None:
        self.render_count += 1

    def pre_move(self) -> None:
        raise AssertionError("upstream pre_move must be skipped")

    def take_action(self, action: object, action_type: str, force: bool) -> object:
        raise AssertionError("upstream action must not be applied")

    def check_success(self) -> bool:
        return True

    def check_early_stop(self) -> bool:
        return True


def test_empty_gripper_calibration_skips_pre_move_and_holds_60hz() -> None:
    config = build_univtac_backend_config("insert_tube", EE8_ACTION_SPEC)
    task = FakeCalibrationTask()
    witness = install_n0_empty_gripper_rest_calibration(task, config)

    task.pre_move()
    result = task.take_action(
        np.zeros(8, dtype=np.float32),
        action_type="ee",
        force=True,
    )

    assert result == (True, False)
    assert task.delay_calls == [(10, False)]
    assert task.step_count == 2
    assert task.render_count == 1
    assert task.check_success() is False
    assert task.check_early_stop() is False
    assert task._robotactile_n0_last_plan["execution_purpose"] == (
        "n0_empty_gripper_rest_calibration_v1"
    )
    assert witness["calibration_contract"] == (N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT)
    assert witness["action_execution_contract"] == (
        N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )
    diagnostics = capture_task_diagnostics(task, "insert_tube")
    assert diagnostics["rest_calibration"] == witness


def test_empty_gripper_calibration_rejects_double_install() -> None:
    config = build_univtac_backend_config("insert_tube", EE8_ACTION_SPEC)
    task = FakeCalibrationTask()
    install_n0_empty_gripper_rest_calibration(task, config)
    with pytest.raises(UniVTACContractError, match="already installed"):
        install_n0_empty_gripper_rest_calibration(task, config)


def test_act_empty_gripper_calibration_holds_qpos_at_120hz() -> None:
    config = build_univtac_backend_config("grasp_classify", QPOS8_ACTION_SPEC)
    task = FakeCalibrationTask()
    witness = install_act_empty_gripper_rest_calibration(task, config)

    task.pre_move()
    result = task.take_action(
        np.zeros(8, dtype=np.float32),
        action_type="qpos",
        force=True,
    )

    assert result == (True, False)
    assert task.delay_calls == [(10, False)]
    assert task.step_count == 1
    assert task.render_count == 1
    assert task.check_success() is False
    assert witness["calibration_contract"] == (ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT)
    assert witness["action_execution_contract"] == (QPOS_ACTION_EXECUTION_CONTRACT)
    assert task._robotactile_rest_calibration_last_hold["execution_purpose"] == (
        "act_empty_gripper_rest_calibration_v1"
    )
    assert not hasattr(task, "_robotactile_n0_last_plan")
