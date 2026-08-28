"""Upstream-faithful CPU doubles used by the shipped UniVTAC qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import UniVTACBackendConfig
from robotactile_benchmark.backends.univtac_isaac import UniVTACTaskRuntime
from robotactile_benchmark.contracts import Array, canonical_hash, freeze_array


@dataclass(frozen=True)
class FakeUpstreamScenario:
    """Deterministic fault/terminal controls for the CPU upstream double."""

    initial_native_step: int = 417
    native_step_increment: int = 1
    success_steps: Tuple[int, ...] = ()
    execution_failure_steps: Tuple[int, ...] = ()
    plan_failure_steps: Tuple[int, ...] = ()
    early_stop_steps: Tuple[int, ...] = ()
    missing_field: Optional[str] = None
    malformed_field: Optional[str] = None
    nan_field: Optional[str] = None
    none_when_early_stop_false: bool = False
    initial_left_depth_mm: float = 34.0
    initial_right_depth_mm: float = 32.5

    def __post_init__(self) -> None:
        if self.initial_native_step < 0 or self.native_step_increment < 0:
            raise ValueError("fake native steps must be non-negative")
        if type(self.none_when_early_stop_false) is not bool:
            raise ValueError("none_when_early_stop_false must be bool")
        for field_name in (
            "success_steps",
            "execution_failure_steps",
            "plan_failure_steps",
            "early_stop_steps",
        ):
            values = tuple(getattr(self, field_name))
            if len(values) != len(set(values)) or any(value < 1 for value in values):
                raise ValueError(f"{field_name} must contain unique positive steps")
            object.__setattr__(self, field_name, values)
        allowed_missing = {None, "head", "left_tactile", "right_tactile", "joint"}
        allowed_malformed = {None, "head_rgb", "left_rgb", "right_depth", "joint"}
        allowed_nan = {None, "head_rgb", "right_depth", "joint"}
        if self.missing_field not in allowed_missing:
            raise ValueError("unknown fake missing field")
        if self.malformed_field not in allowed_malformed:
            raise ValueError("unknown fake malformed field")
        if self.nan_field not in allowed_nan:
            raise ValueError("unknown fake NaN field")


@dataclass(frozen=True)
class FakeTakeActionCall:
    """One exact call observed at the upstream action boundary."""

    action: Array
    action_type: str
    force: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", freeze_array(self.action))


@dataclass(frozen=True)
class _FakeRuntimeSnapshot:
    native_step: int
    action_count: int
    state: float
    plan_success: bool
    eval_success: bool


class StrictFakeCudaTensor:
    """CPU value that rejects any conversion order other than the CUDA contract."""

    def __init__(
        self,
        array: Array,
        completion: Callable[[Tuple[str, ...]], None],
    ) -> None:
        self._array = array
        self._completion = completion
        self._calls: list[str] = []

    def _advance(self, method_name: str) -> StrictFakeCudaTensor:
        expected = ("detach", "cpu", "contiguous", "numpy")[len(self._calls)]
        if method_name != expected:
            raise RuntimeError(
                f"fake CUDA conversion expected {expected}, received {method_name}"
            )
        self._calls.append(method_name)
        return self

    def detach(self) -> StrictFakeCudaTensor:
        return self._advance("detach")

    def cpu(self) -> StrictFakeCudaTensor:
        return self._advance("cpu")

    def contiguous(self) -> StrictFakeCudaTensor:
        return self._advance("contiguous")

    def numpy(self) -> Array:
        self._advance("numpy")
        self._completion(tuple(self._calls))
        return self._array


class FakeUpstreamTask:
    """Stateful fake matching the UniVTAC reset/observe/take_action surface."""

    def __init__(
        self,
        config: UniVTACBackendConfig,
        live_joint_names: Tuple[str, ...],
        scenario: FakeUpstreamScenario,
    ) -> None:
        self._config = config
        self.live_joint_names = live_joint_names
        self.scenario = scenario
        self.plan_success = True
        self.eval_success = False
        self.reset_count = 0
        self.observation_count = 0
        self.close_count = 0
        self.capture_count = 0
        self.restore_count = 0
        self.reset_calls: list[tuple[int, tuple[str, ...]]] = []
        self.reset_seed_arguments: list[int] = []
        self.take_action_calls: list[FakeTakeActionCall] = []
        self.cuda_conversion_sequences: list[Tuple[str, ...]] = []
        self.stale_reset_return: dict[str, Any] = {}
        self.stale_return_sha256 = canonical_hash({"uninitialized": True})
        self._native_step = scenario.initial_native_step
        self._action_count = 0
        self._state = 0.0

    def reset(self, seed: int, instructions: list[str]) -> dict[str, Any]:
        self.reset_count += 1
        self.reset_seed_arguments.append(seed)
        self.reset_calls.append((seed, tuple(instructions)))
        self._native_step = self.scenario.initial_native_step
        self._action_count = 0
        self._state = float(seed) / 1000.0
        self.plan_success = True
        self.eval_success = False
        self.stale_reset_return = {
            "step": 0,
            "stale": True,
            "seed": seed,
            "head_pixel": 255,
        }
        self.stale_return_sha256 = canonical_hash(self.stale_reset_return)
        return dict(self.stale_reset_return)

    def _get_observations(self) -> dict[str, Any]:
        self.observation_count += 1
        return self._raw_observation()

    def take_action(
        self,
        action: Any,
        action_type: str = "qpos",
        force: bool = True,
    ) -> Tuple[bool, bool]:
        if not isinstance(action, np.ndarray):
            raise TypeError("fake upstream action must be a numpy array")
        if action.dtype != np.float32 or action.shape != (8,):
            raise ValueError("fake upstream action must be float32 shape [8]")
        self.take_action_calls.append(
            FakeTakeActionCall(action=action, action_type=action_type, force=force)
        )
        self._action_count += 1
        self._state += float(action[0]) + 0.001
        self._native_step += self.scenario.native_step_increment
        self.plan_success = self._action_count not in self.scenario.plan_failure_steps
        self.eval_success = self._action_count in self.scenario.success_steps
        execution_success = (
            self._action_count not in self.scenario.execution_failure_steps
        )
        return execution_success, self.eval_success

    def check_success(self) -> bool:
        return self.eval_success

    def check_early_stop(self) -> Optional[bool]:
        active = self._action_count in self.scenario.early_stop_steps
        if not active and self.scenario.none_when_early_stop_false:
            return None
        return active

    def close(self) -> None:
        self.close_count += 1

    def capture_state(self) -> object:
        self.capture_count += 1
        return _FakeRuntimeSnapshot(
            native_step=self._native_step,
            action_count=self._action_count,
            state=self._state,
            plan_success=self.plan_success,
            eval_success=self.eval_success,
        )

    def restore_state(self, value: object) -> None:
        if not isinstance(value, _FakeRuntimeSnapshot):
            raise TypeError("fake snapshot type mismatch")
        self._native_step = value.native_step
        self._action_count = value.action_count
        self._state = value.state
        self.plan_success = value.plan_success
        self.eval_success = value.eval_success
        self.restore_count += 1

    def snapshot_state_sha256(self) -> str:
        return canonical_hash(
            _FakeRuntimeSnapshot(
                native_step=self._native_step,
                action_count=self._action_count,
                state=self._state,
                plan_success=self.plan_success,
                eval_success=self.eval_success,
            )
        )

    def _wrap(self, value: Array) -> StrictFakeCudaTensor:
        return StrictFakeCudaTensor(
            value,
            completion=self.cuda_conversion_sequences.append,
        )

    def _canonical_joint9(self) -> Array:
        values = np.asarray(
            [self._state, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0, 0.02, 0.02],
            dtype=np.float32,
        )
        if self.scenario.malformed_field == "joint":
            values = values[:-1]
        if self.scenario.nan_field == "joint":
            values[0] = np.nan
        return values

    def _raw_observation(self) -> dict[str, Any]:
        head = np.full(self._config.head_shape, 20, dtype=np.uint8)
        wrist = np.full(self._config.wrist_shape, 30, dtype=np.uint8)
        left_rgb = np.full(self._config.tactile_rgb_shape, 60, dtype=np.uint8)
        right_rgb = np.full(self._config.tactile_rgb_shape, 70, dtype=np.uint8)
        left_depth = np.full(
            self._config.tactile_depth_shape,
            self.scenario.initial_left_depth_mm,
            dtype=np.float32,
        )
        right_depth = np.full(
            self._config.tactile_depth_shape,
            self.scenario.initial_right_depth_mm,
            dtype=np.float32,
        )
        if self.scenario.malformed_field == "head_rgb":
            head = head[:10, :10]
        if self.scenario.malformed_field == "left_rgb":
            left_rgb = left_rgb[..., 0]
        if self.scenario.malformed_field == "right_depth":
            right_depth = right_depth[..., None]
        if self.scenario.nan_field == "head_rgb":
            head = head.astype(np.float32)
            head[0, 0, 0] = np.nan
        if self.scenario.nan_field == "right_depth":
            right_depth[0, 0] = np.nan
        canonical_names = self._config.canonical_joint_names
        canonical_joint = self._canonical_joint9()
        if canonical_joint.shape == (9,):
            values_by_name = dict(zip(canonical_names, canonical_joint))
            joint = np.asarray(
                [values_by_name[name] for name in self.live_joint_names],
                dtype=np.float32,
            )
        else:
            joint = canonical_joint
        embodiment: dict[str, Any] = {"joint": self._wrap(joint)}
        if self._config.action_spec == EE8_ACTION_SPEC:
            ee = np.asarray(
                [self._state, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            embodiment["ee"] = self._wrap(ee)
        raw: dict[str, Any] = {
            "step": self._native_step,
            "observation": {
                self._config.aliases.head_camera: {"rgb": self._wrap(head)},
                self._config.aliases.wrist_camera: {"rgb": self._wrap(wrist)},
            },
            "tactile": {
                self._config.aliases.left_tactile: {
                    self._config.aliases.tactile_payload: self._wrap(left_rgb),
                    "depth": self._wrap(left_depth),
                },
                self._config.aliases.right_tactile: {
                    self._config.aliases.tactile_payload: self._wrap(right_rgb),
                    "depth": self._wrap(right_depth),
                },
            },
            "embodiment": embodiment,
        }
        if self.scenario.missing_field == "head":
            del raw["observation"][self._config.aliases.head_camera]
        elif self.scenario.missing_field == "left_tactile":
            del raw["tactile"][self._config.aliases.left_tactile]
        elif self.scenario.missing_field == "right_tactile":
            del raw["tactile"][self._config.aliases.right_tactile]
        elif self.scenario.missing_field == "joint":
            del raw["embodiment"]["joint"]
        return raw


def make_fake_runtime(
    config: UniVTACBackendConfig,
    scenario: Optional[FakeUpstreamScenario] = None,
    live_joint_names: Optional[Tuple[str, ...]] = None,
    construction_seed: int = 11,
) -> Tuple[UniVTACTaskRuntime, FakeUpstreamTask]:
    """Construct one fresh task/runtime pair for a single benchmark reset."""

    active_scenario = FakeUpstreamScenario() if scenario is None else scenario
    names = (
        tuple(reversed(config.canonical_joint_names))
        if live_joint_names is None
        else tuple(live_joint_names)
    )
    task = FakeUpstreamTask(config, names, active_scenario)

    def encode_action(row: Array) -> Array:
        return np.ascontiguousarray(row, dtype=np.float32).copy()

    runtime = UniVTACTaskRuntime(
        task=task,
        handshake=config.expected_handshake(names),
        construction_seed=construction_seed,
        encode_action=encode_action,
        prepare_reset=lambda: None,
        close_runtime=task.close,
        capture_state=task.capture_state,
        restore_state=task.restore_state,
        snapshot_state_sha256=task.snapshot_state_sha256,
    )
    return runtime, task
