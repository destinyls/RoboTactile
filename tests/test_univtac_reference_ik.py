"""Tests for reference-guided reset-only IK calibration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from robotactile_benchmark.backends.univtac_reference_ik import (
    install_reference_guided_pre_move_ik,
)


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value, dtype=np.float32)

    def new_tensor(self, value: object) -> _Tensor:
        return _Tensor(value)

    def reshape(self, *shape: int) -> _Tensor:
        return _Tensor(self.value.reshape(*shape))

    def expand(self, *shape: int) -> _Tensor:
        return _Tensor(np.broadcast_to(self.value, shape))

    def clone(self) -> _Tensor:
        return _Tensor(self.value.copy())


class _Manager:
    def __init__(self) -> None:
        self.default_seed_calls = 0
        self.seeds: list[np.ndarray[Any, Any]] = []
        self.ik_solver = SimpleNamespace(get_seed=self._default_seed)
        self.planner = SimpleNamespace(
            motion_gen=SimpleNamespace(ik_solver=self.ik_solver)
        )
        self.robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=_Tensor(np.zeros((1, 9))))
        )

    def _default_seed(self, *_args: object, **_kwargs: object) -> _Tensor:
        self.default_seed_calls += 1
        return _Tensor(np.zeros((1, 1, 7)))

    def plan_arm(self) -> str:
        seed = self.ik_solver.get_seed(
            2,
            SimpleNamespace(batch=1),
            False,
            _Tensor(np.zeros((1, 1, 7))),
        )
        self.seeds.append(seed.value.copy())
        return "planned"


def test_reference_seed_is_scoped_to_pre_move_and_restored() -> None:
    expected = tuple(float(value) for value in range(7))
    manager = _Manager()
    task = SimpleNamespace(_robot_manager=manager, in_pre_move=False)

    assert install_reference_guided_pre_move_ik(task, expected) is True
    assert manager.plan_arm() == "planned"
    assert manager.default_seed_calls == 1

    task.in_pre_move = True
    assert manager.plan_arm() == "planned"
    assert manager.default_seed_calls == 1
    assert manager.seeds[-1].shape == (2, 1, 7)
    np.testing.assert_allclose(
        manager.seeds[-1][:, 0, :],
        np.tile(np.asarray(expected), (2, 1)),
    )

    task.in_pre_move = False
    assert manager.plan_arm() == "planned"
    assert manager.default_seed_calls == 2
    assert task._robotactile_reference_guided_ik["teleport_used"] is False
