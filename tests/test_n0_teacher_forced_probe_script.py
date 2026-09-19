from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam import teacher_forced_replay
from robotactile_benchmark.integrations.n0_twam.teacher_forced_placement import (
    apply_lift_bottle_hdf5_initial_placement,
)
from robotactile_benchmark.integrations.n0_twam.teacher_forced_replay import (
    replay_official_qpos_sequence,
    strict_action_result,
)
from scripts.live_univtac.probe_n0_teacher_forced_alignment_isaac import (
    _n0_server_pixel_streams,
    _parser,
    _validate_args,
)


def test_teacher_forced_probe_defaults_to_complete_expert_trajectory() -> None:
    args = _parser().parse_args(
        [
            "--upstream-root",
            "/upstream",
            "--runtime-dir",
            "/runtime",
            "--hdf5",
            "/episode.hdf5",
            "--output-dir",
            "/output",
        ]
    )

    _validate_args(args)

    assert args.hdf5_index is None
    assert args.replay_stride == 1
    assert args.initial_seed == 90
    assert args.antialiasing_mode == "TAA"


def test_teacher_forced_probe_accepts_only_strict_bool_action_results() -> None:
    assert strict_action_result((True, False)) == (True, False)

    with pytest.raises(RuntimeError, match="malformed"):
        strict_action_result((1, False))


def test_teacher_forced_target_must_lie_on_replay_stride() -> None:
    args = _parser().parse_args(
        [
            "--upstream-root",
            "/upstream",
            "--runtime-dir",
            "/runtime",
            "--hdf5",
            "/episode.hdf5",
            "--hdf5-index",
            "1",
            "--replay-stride",
            "2",
            "--output-dir",
            "/output",
        ]
    )

    with pytest.raises(ValueError, match="replay stride"):
        _validate_args(args)


def test_hdf5_actor_placement_is_explicit_and_lift_bottle_only() -> None:
    args = _parser().parse_args(
        [
            "--upstream-root",
            "/upstream",
            "--runtime-dir",
            "/runtime",
            "--hdf5",
            "/episode.hdf5",
            "--actor-placement",
            "hdf5_initial",
            "--task",
            "insert_hole",
            "--output-dir",
            "/output",
        ]
    )

    with pytest.raises(ValueError, match="requires lift_bottle"):
        _validate_args(args)


def test_hdf5_lift_bottle_placement_matches_reset_settle_release_contract() -> None:
    class Pose:
        def __init__(self, values: object) -> None:
            self.values = np.asarray(values, dtype=np.float64)

        @classmethod
        def from_list(cls, values: object) -> Pose:
            return cls(values)

        def tolist(self) -> list[float]:
            return [float(item) for item in self.values]

    class Actor:
        def __init__(self, values: object) -> None:
            self.pose = Pose(values)
            self.target: Pose | None = None
            self.next_status: str | None = None

        def get_pose(self) -> Pose:
            return self.pose

        def set_pose(self, pose: Pose) -> None:
            self.target = pose
            self.next_status = "set"

    class Manager:
        def __init__(self, bottle: Actor) -> None:
            self.bottle = bottle

        def remove_animate(self) -> None:
            self.bottle.next_status = "unset"

    class Task:
        def __init__(self) -> None:
            self.mode = "eval"
            self.step_count = 10
            self.render_count = 0
            self.bottle = Actor([0.67, 0.04, 0.026, 1.0, 0.0, 0.0, 0.0])
            self.wall = Actor([0.75, 0.0, 0.005, 1.0, 0.0, 0.0, 0.0])
            self._actor_manager = Manager(self.bottle)

        def _step(self, *, is_save: bool) -> None:
            assert is_save is False
            if self.bottle.next_status == "set":
                assert self.bottle.target is not None
                self.bottle.pose = self.bottle.target
            elif self.bottle.next_status == "unset":
                self.bottle.next_status = None
            self.step_count += 1

        def _update_render(self) -> None:
            self.render_count += 1

    task = Task()
    expected_bottle = np.array(
        [0.674, -0.009, 0.026, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    witness = apply_lift_bottle_hdf5_initial_placement(
        task,
        {
            "bottle": expected_bottle,
            "wall": np.array([0.75, 0.0, 0.005, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
    )

    assert witness["alignment_passed"] is True
    assert witness["dynamics_released"] is True
    assert witness["native_step_delta"] == 25
    assert witness["render_updates"] == 2
    assert witness["movement_from_reset_m"] == pytest.approx(
        np.linalg.norm(expected_bottle[:3] - np.array([0.67, 0.04, 0.026]))
    )
    assert task.step_count == 35
    assert task.render_count == 2
    np.testing.assert_allclose(task.bottle.pose.values, expected_bottle)


def test_teacher_forced_probe_mirrors_n0_server_area_resize() -> None:
    cv2 = pytest.importorskip("cv2")
    streams = {
        name: np.arange(18 * 24 * 3, dtype=np.uint16)
        .reshape(18, 24, 3)
        .astype(np.uint8)
        for name in ("top", "wrist_l", "tactile_a", "tactile_b")
    }

    resized = _n0_server_pixel_streams(streams)

    assert resized["top"].shape == (256, 256, 3)
    assert resized["wrist_l"].shape == (256, 256, 3)
    assert resized["tactile_a"].shape == (128, 128, 3)
    assert resized["tactile_b"].shape == (128, 128, 3)
    np.testing.assert_array_equal(
        resized["top"],
        cv2.resize(streams["top"], (256, 256), interpolation=cv2.INTER_AREA),
    )
    assert not resized["top"].flags.writeable


def test_teacher_forced_probe_rejects_incomplete_server_streams() -> None:
    with pytest.raises(ValueError, match="missing N0 image stream"):
        _n0_server_pixel_streams({"top": np.zeros((8, 8, 3), dtype=np.uint8)})


def test_timestamped_replay_holds_each_qpos_for_two_native_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_states = np.zeros((3, 8), dtype=np.float32)
    expert_states[:, 3] = 1.0

    class Task:
        step_count = 10
        plan_success = True
        device = "cpu"
        mode = "eval"
        render_count = 0
        step_modes: list[str] = []

        def take_action(self, *_args: object, **_kwargs: object) -> tuple[bool, bool]:
            self._step(is_save=True)
            return True, False

        def _step(self, *, is_save: bool) -> None:
            assert self.mode == "eval_test"
            self.step_modes.append(self.mode)
            self.step_count += 1

        def _update_render(self) -> None:
            assert self.mode == "eval"
            self.render_count += 1

    class Backend:
        def _fresh_raw(self) -> object:
            return object()

        def _convert(self, _raw: object, benchmark_step: int) -> SimpleNamespace:
            assert benchmark_step in (1, 2)
            observation = SimpleNamespace(
                proprio=expert_states[benchmark_step - 1].copy()
            )
            record = SimpleNamespace(observation=observation)
            return SimpleNamespace(record=record)

    monkeypatch.setattr(
        teacher_forced_replay,
        "_qpos_tensor",
        lambda _task, joint9: joint9[:8],
    )
    task = Task()
    replay = replay_official_qpos_sequence(
        task=task,
        backend=Backend(),
        joint9_trajectory=np.zeros((3, 9), dtype=np.float32),
        expert_states=expert_states,
        target_index=1,
        stride=1,
        physics_steps_per_target=2,
    )

    assert replay.indices == (0, 1)
    assert replay.native_step_after - replay.native_step_before == 4
    assert [item["native_step_delta"] for item in replay.trace] == [2, 2]
    assert task.mode == "eval"
    assert task.render_count == 2
    assert task.step_modes == ["eval_test"] * 4
    assert {item["render_contract"] for item in replay.trace} == {
        "one_endpoint_render_no_intermediate_render_v1"
    }


def test_timestamped_replay_restores_eval_mode_when_action_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        step_count = 0
        mode = "eval"
        device = "cpu"

        @staticmethod
        def take_action(*_args: object, **_kwargs: object) -> tuple[bool, bool]:
            raise RuntimeError("action failed")

        @staticmethod
        def _step(*, is_save: bool) -> None:
            raise AssertionError("extra step must not execute")

        @staticmethod
        def _update_render() -> None:
            raise AssertionError("failed endpoint must not render")

    monkeypatch.setattr(
        teacher_forced_replay,
        "_qpos_tensor",
        lambda _task, joint9: joint9[:8],
    )
    task = Task()
    states = np.zeros((2, 8), dtype=np.float32)
    states[:, 3] = 1.0

    with pytest.raises(RuntimeError, match="action failed"):
        replay_official_qpos_sequence(
            task=task,
            backend=object(),
            joint9_trajectory=np.zeros((2, 9), dtype=np.float32),
            expert_states=states,
            target_index=0,
            stride=1,
            physics_steps_per_target=2,
        )

    assert task.mode == "eval"
