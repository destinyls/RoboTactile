from __future__ import annotations

import json
import types

import numpy as np

from robotactile_benchmark.backends.univtac_diagnostics import (
    install_planner_failure_diagnostics,
)


class _Tensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def item(self) -> object:
        return self._values.item()

    def __array__(self) -> np.ndarray:
        return self._values


def test_planner_failure_publishes_finite_input_diagnostic() -> None:
    messages: list[str] = []
    result = types.SimpleNamespace(
        success=_Tensor(False),
        status="IK_FAIL",
        solve_time=_Tensor(0.5),
    )

    def plan_path(**_kwargs: object) -> object:
        return result

    planner = types.SimpleNamespace(plan_path=plan_path)
    actor = types.SimpleNamespace(obj_id=3, vertices=np.ones((4, 3)))
    task = types.SimpleNamespace(
        _actor_manager=types.SimpleNamespace(actors={"prism": actor}),
        _robot_manager=types.SimpleNamespace(planner=planner),
        logger=types.SimpleNamespace(error=messages.append),
        uipc_sim=types.SimpleNamespace(_surf_vertex_offsets=[0, 4]),
    )
    assert install_planner_failure_diagnostics(task, "insert_hole")

    observed = planner.plan_path(
        curr_joint_pos=_Tensor([1.0, 2.0]),
        curr_joint_vel=_Tensor([0.0, 0.0]),
        target_ee_pose=types.SimpleNamespace(tolist=lambda: [0.1] * 7),
        real_robot_pose=object(),
        pre_dis=None,
        constraint_pose=None,
        time_dilation_factor=None,
    )

    assert observed is result
    assert len(messages) == 1
    prefix = "ROBOTACTILE_PLAN_DIAGNOSTIC "
    assert messages[0].startswith(prefix)
    payload = json.loads(messages[0][len(prefix) :])
    assert payload["actors"] == [
        {
            "name": "prism",
            "object_id": 3,
            "vertices": {
                "dtype": "float64",
                "finite_count": 12,
                "finite_max": 1.0,
                "finite_min": 1.0,
                "shape": [4, 3],
                "size": 12,
            },
        }
    ]
    assert payload["result"] == {
        "solve_time": 0.5,
        "status": "IK_FAIL",
        "success": False,
    }
    assert payload["surface_offsets"] == [0, 4]


def test_planner_success_and_other_tasks_do_not_emit_diagnostics() -> None:
    messages: list[str] = []
    result = types.SimpleNamespace(success=_Tensor(True))
    planner = types.SimpleNamespace(plan_path=lambda **_kwargs: result)
    task = types.SimpleNamespace(
        _actor_manager=types.SimpleNamespace(actors={}),
        _robot_manager=types.SimpleNamespace(planner=planner),
        logger=types.SimpleNamespace(error=messages.append),
        uipc_sim=types.SimpleNamespace(_surf_vertex_offsets=[]),
    )
    assert not install_planner_failure_diagnostics(task, "pull_out_key")
    assert install_planner_failure_diagnostics(task, "insert_hole")
    assert (
        planner.plan_path(
            curr_joint_pos=_Tensor([1.0]),
            curr_joint_vel=_Tensor([0.0]),
            target_ee_pose=types.SimpleNamespace(tolist=lambda: [0.1] * 7),
            real_robot_pose=object(),
        )
        is result
    )
    assert messages == []
