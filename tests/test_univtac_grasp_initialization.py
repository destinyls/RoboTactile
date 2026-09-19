from __future__ import annotations

import json
import types

import numpy as np
import pytest

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.backends.univtac_grasp_initialization import (
    install_grasp_initialization_compatibility,
)
from robotactile_benchmark.backends.univtac_rest_calibration import (
    N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
)


class _Pose:
    def __init__(self, position: list[float]) -> None:
        self.p = np.asarray(position, dtype=np.float64)


def _task(*, lift_on_attempt: int | None) -> tuple[object, list[float], list[str]]:
    close_values: list[float] = []
    messages: list[str] = []
    prism_pose = _Pose([0.35, 0.0, 0.01])
    gripper_pose = _Pose([0.35, 0.0, 0.05])
    pre_move_count = 0
    displacement_args: list[dict[str, float]] = []

    def close_gripper(pos: float, _threshold: object = "auto") -> list[float]:
        close_values.append(pos)
        return [pos]

    atom = types.SimpleNamespace(close_gripper=close_gripper)
    atom.move_by_displacement = lambda **_kwargs: [types.SimpleNamespace(args={})]
    prism = types.SimpleNamespace(get_pose=lambda: prism_pose)

    def pre_move() -> str:
        nonlocal pre_move_count
        pre_move_count += 1
        atom.close_gripper(0.0072 / 0.039)
        displacement = atom.move_by_displacement(z=0.05)
        displacement_args.append(dict(displacement[0].args))
        if lift_on_attempt is not None and pre_move_count >= lift_on_attempt:
            prism_pose.p = prism_pose.p + np.asarray([0.0, 0.0, 0.05])
            gripper_pose.p = gripper_pose.p + np.asarray([0.0, 0.0, 0.05])
        return "done"

    task = types.SimpleNamespace(
        _robot_manager=types.SimpleNamespace(
            get_gripper_center_pose=lambda: gripper_pose,
            get_gripper_qpos=lambda: 0.0065,
            gripper_max_qpos=0.039,
        ),
        _tactile_manager=types.SimpleNamespace(
            get_min_depth=lambda: np.asarray([24.7, 24.8], dtype=np.float64)
        ),
        atom=atom,
        cfg=types.SimpleNamespace(use_adaptive_grasp=False),
        delay=lambda _steps, is_save: is_save is False,
        in_pre_move=True,
        logger=types.SimpleNamespace(info=messages.append, error=messages.append),
        move=lambda _actions: None,
        pre_move=pre_move,
        prism=prism,
        displacement_args=displacement_args,
    )

    def reset(*_args: object, **_kwargs: object) -> str:
        task.pre_move()
        task.in_pre_move = False
        return "reset-done"

    task.reset = reset
    return task, close_values, messages


def test_grasp_initialization_clamps_close_and_accepts_stable_lift() -> None:
    task, close_values, messages = _task(lift_on_attempt=1)

    assert install_grasp_initialization_compatibility(task, "grasp_classify")
    assert task.reset(seed=17, instructions=["prompt"]) == "reset-done"

    assert close_values == pytest.approx([0.0065 / 0.039, 0.0045 / 0.039])
    assert len(messages) == 1
    payload = json.loads(messages[0].split(" ", 1)[1])
    assert payload["grasp_qpos_m"] == pytest.approx(0.0065)
    assert payload["attempts"][0]["lift_m"] == pytest.approx(0.05)
    assert payload["attempt_count"] == 2
    assert payload["selected_attempt"] == "loaded_hold_preload"


def test_grasp_initialization_rejects_an_unlifted_prism() -> None:
    task, close_values, messages = _task(lift_on_attempt=None)
    assert install_grasp_initialization_compatibility(task, "grasp_classify")

    with pytest.raises(
        UniVTACContractError,
        match="deterministic regrasp did not establish a stable lift",
    ):
        task.reset(seed=17, instructions=["prompt"])

    assert len(messages) == 1
    payload = json.loads(messages[0].split(" ", 1)[1])
    assert payload["attempt_count"] == 3
    assert payload["attempts"][2]["lift_m"] == pytest.approx(0.0)
    assert close_values == pytest.approx([0.0065 / 0.039, 0.0, 0.0045 / 0.039])


def test_grasp_initialization_retries_with_adaptive_bilateral_contact() -> None:
    task, close_values, messages = _task(lift_on_attempt=2)
    assert install_grasp_initialization_compatibility(task, "grasp_classify")

    assert task.reset(seed=17, instructions=["prompt"]) == "reset-done"

    assert close_values == pytest.approx([0.0065 / 0.039, 0.0])
    payload = json.loads(messages[0].split(" ", 1)[1])
    assert payload["passed"] is True
    assert payload["selected_attempt"] == "adaptive_contact_retry"
    assert payload["attempts"][1]["lift_m"] == pytest.approx(0.05)


def test_grasp_initialization_tightens_unilateral_lift_in_place() -> None:
    task, close_values, messages = _task(lift_on_attempt=1)
    task._tactile_manager.get_min_depth = lambda: np.asarray(
        [34.0, 20.0] if len(close_values) == 1 else [24.7, 24.8],
        dtype=np.float64,
    )
    assert install_grasp_initialization_compatibility(task, "grasp_classify")

    assert task.reset(seed=17, instructions=["prompt"]) == "reset-done"

    payload = json.loads(messages[0].split(" ", 1)[1])
    assert payload["selected_attempt"] == "loaded_hold_preload"
    assert payload["attempts"][0]["kinematics_passed"] is True
    assert payload["attempts"][0]["bilateral_contact"] is False
    assert payload["attempts"][1]["bilateral_contact"] is True
    assert payload["attempts"][1]["post_settle_prism"][2] == pytest.approx(0.06)
    assert task.displacement_args == [{}]
    assert close_values == pytest.approx([0.0065 / 0.039, 0.0045 / 0.039])


def test_grasp_initialization_uses_loaded_adaptive_preload_after_shallow_contact() -> (
    None
):
    task, close_values, messages = _task(lift_on_attempt=3)
    assert install_grasp_initialization_compatibility(task, "grasp_classify")

    assert task.reset(seed=17, instructions=["prompt"]) == "reset-done"

    assert close_values == pytest.approx([0.0065 / 0.039, 0.0, 0.0045 / 0.039])
    payload = json.loads(messages[0].split(" ", 1)[1])
    assert payload["passed"] is True
    assert payload["selected_attempt"] == "loaded_hold_preload"
    assert payload["attempts"][2]["lift_m"] == pytest.approx(0.05)
    assert payload["attempts"][2]["minimum_tactile_depths_mm"] == pytest.approx(
        [24.7, 24.8]
    )
    assert task.displacement_args == [
        {},
        {},
        {"time_dilation_factor": pytest.approx(0.2)},
    ]


def test_rest_calibration_bypasses_only_the_pre_move_grasp_witness() -> None:
    task, close_values, messages = _task(lift_on_attempt=None)
    assert install_grasp_initialization_compatibility(task, "grasp_classify")
    task._robotactile_rest_calibration = {
        "calibration_contract": N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
    }
    task.pre_move = lambda: None

    assert task.reset(seed=17, instructions=["prompt"]) == "reset-done"

    witness = task._robotactile_grasp_initialization
    assert witness["calibration_bypass"] is True
    assert witness["attempt_count"] == 0
    assert witness["selected_attempt"] == "rest_calibration_empty_gripper"
    assert close_values == []
    assert len(messages) == 1


def test_other_task_does_not_install_grasp_initialization() -> None:
    task, _close_values, _messages = _task(lift_on_attempt=1)

    assert not install_grasp_initialization_compatibility(task, "pull_out_key")
