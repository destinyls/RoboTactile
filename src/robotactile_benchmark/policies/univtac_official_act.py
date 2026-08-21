"""Closed-loop adapter for the official UniVTAC ``policy_last.ckpt`` family.

This module is deliberately separate from :mod:`act`: an official UniVTAC ACT
artifact must never be presented as the qualified StrictACT ``policy_best``
runtime.  It contains no torch or upstream imports.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Protocol, Tuple, cast

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    ActionPlan,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, ObservationRecord
from robotactile_benchmark.policies.act import preprocess_camera, preprocess_tactile


class OfficialACTProfile(str, Enum):
    """Artifact profiles published in the shared UniVTAC checkpoint tree."""

    UNIVTAC = "univtac"
    VISION_ONLY = "vision_only"


@dataclass(frozen=True)
class _ProfileSpec:
    config_name: str
    config_sha256: str
    camera_names: Tuple[str, ...]
    tactile_names: Tuple[str, ...]


_PROFILE_SPECS: Mapping[OfficialACTProfile, _ProfileSpec] = MappingProxyType(
    {
        OfficialACTProfile.UNIVTAC: _ProfileSpec(
            "train_config.yml",
            "acdab30e50fa7280918804c4a196f75a6db6e854a3c7a86a0ddd84791c533397",
            ("cam_high",),
            ("tac_left", "tac_right"),
        ),
        OfficialACTProfile.VISION_ONLY: _ProfileSpec(
            "train_config_vision.yml",
            "427c54337b56b0958606de4a235b6e13527877f75c21c2939045b5a6b1375cc2",
            ("cam_high",),
            (),
        ),
    }
)
_TASK_IDS = frozenset(
    {
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    }
)


class OfficialUniVTACACTRuntime(Protocol):
    """Minimal upstream ``ACT`` inference boundary used by this adapter."""

    def reset(self) -> None: ...

    def get_action(self, observation: Mapping[str, object]) -> Array: ...


RuntimeInputTransform = Callable[[Mapping[str, Array]], Mapping[str, object]]


class _PolicyState(str, Enum):
    NEW = "new"
    READY = "ready"
    AWAITING_COMMIT = "awaiting_commit"
    ABORTED = "aborted"
    CLOSED = "closed"


def _identity_transform(value: Mapping[str, Array]) -> Mapping[str, object]:
    return value


def _runtime_args(
    profile: OfficialACTProfile,
    task_id: str,
    encoder_path: Path,
    device_name: str,
) -> dict[str, object]:
    tactile_names = (
        ["tac_left", "tac_right"] if profile is OfficialACTProfile.UNIVTAC else []
    )
    return {
        "state_dim": 8,
        "kl_weight": 10.0,
        "chunk_size": 50,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "temporal_agg": True,
        "device": device_name,
        "ckpt_dir": "",
        "policy_class": "ACT",
        "task_name": task_id,
        "seed": 0,
        "num_epochs": 1,
        "num_steps": 4000,
        "batch_size": 64,
        "save_freq": 1000,
        "position_embedding": "sine",
        "lr_vision_backbone": 0.00001,
        "weight_decay": 0.0001,
        "lr": 0.00001,
        "masks": False,
        "dilation": False,
        "backbone": "resnet18",
        "nheads": 8,
        "enc_layers": 4,
        "dec_layers": 7,
        "pre_norm": False,
        "dropout": 0.1,
        "camera_names": ["cam_high"],
        "tactile_names": tactile_names,
        "lr_tactile_backbone": 0.00001 if tactile_names else 0.0,
        "tactile_masks": False,
        "tactile_backbone": "resnet18",
        "tactile_ckpt": str(encoder_path),
        "tactile_dilation": False,
    }


def _construct_pinned_upstream_runtime(
    *,
    source_path: Path,
    source_sha256: str,
    profile: OfficialACTProfile,
    task_id: str,
    encoder_path: Path,
    device_name: str,
    torch_module: Any,
) -> Any:
    """Import upstream ``ACT`` directly without importing its outer ``Policy``."""

    occupied = tuple(sys.modules)
    if any(name == "detr" or name.startswith("detr.") for name in occupied):
        raise RuntimeError("official ACT cannot shadow an existing detr module")
    if any(name == "util" or name.startswith("util.") for name in occupied):
        raise RuntimeError("official ACT cannot shadow an existing util module")
    before = set(sys.modules)
    source_name = f"_robotactile_official_act_{source_sha256[:12]}"
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(source_path.parent), str(source_path.parent / "detr")]
        spec = importlib.util.spec_from_file_location(source_name, source_path)
        if spec is None or spec.loader is None or spec.origin is None:
            raise ImportError("official ACT source has no filesystem loader")
        if Path(spec.origin).resolve(strict=True) != source_path.resolve(strict=True):
            raise ValueError("official ACT import origin mismatch")
        module = importlib.util.module_from_spec(spec)
        sys.modules[source_name] = module
        spec.loader.exec_module(module)
        runtime_class = getattr(module, "ACT", None)
        if (
            runtime_class is None
            or getattr(runtime_class, "__module__", None) != source_name
        ):
            raise ImportError("pinned official ACT class is unavailable")
        cuda_device = getattr(getattr(torch_module, "cuda", None), "device", None)
        if not callable(cuda_device):
            raise RuntimeError("official ACT CUDA device context is unavailable")
        args = _runtime_args(profile, task_id, encoder_path, device_name)
        with cuda_device(device_name):
            return runtime_class(args)
    finally:
        sys.path[:] = old_path
        for name in tuple(set(sys.modules) - before):
            if (
                name == source_name
                or name == "detr"
                or name.startswith("detr.")
                or name == "util"
                or name.startswith("util.")
            ):
                sys.modules.pop(name, None)


def _release_runtime(runtime: Optional[Any], torch_module: Any) -> None:
    if runtime is not None:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
    empty_cache = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


class _OwnedRuntime:
    def __init__(self, runtime: Any, torch_module: Any) -> None:
        self._runtime: Optional[Any] = runtime
        self._torch = torch_module

    def reset(self) -> None:
        if self._runtime is None:
            raise RuntimeError("official ACT runtime is closed")
        self._runtime.reset()

    def get_action(self, observation: Mapping[str, object]) -> Array:
        if self._runtime is None:
            raise RuntimeError("official ACT runtime is closed")
        return cast(Array, self._runtime.get_action(observation))

    def close(self) -> None:
        if self._runtime is None:
            return
        runtime, self._runtime = self._runtime, None
        _release_runtime(runtime, self._torch)


def _torch_transform(torch_module: Any) -> RuntimeInputTransform:
    def transform(value: Mapping[str, Array]) -> Mapping[str, object]:
        result: dict[str, object] = {"qpos": value["qpos"]}
        for key in value.keys() - {"qpos"}:
            result[key] = torch_module.from_numpy(np.ascontiguousarray(value[key]))
        return result

    return transform


class OfficialUniVTACACTPolicy:
    """One-step qpos8 policy around the official upstream ``ACT`` runtime."""

    def __init__(
        self,
        identity: PolicyIdentity,
        runtime: OfficialUniVTACACTRuntime,
        *,
        artifact_task: str,
        profile: OfficialACTProfile,
        input_transform: RuntimeInputTransform = _identity_transform,
    ) -> None:
        if type(identity) is not PolicyIdentity:
            raise TypeError("identity must be an exact PolicyIdentity")
        if identity.action_spec != ACTION_SPEC:
            raise ValueError("official ACT requires qpos8_next_step")
        normalized_profile = (
            profile
            if isinstance(profile, OfficialACTProfile)
            else OfficialACTProfile(profile)
        )
        tactile = normalized_profile is OfficialACTProfile.UNIVTAC
        if (
            identity.consumes_tactile is not tactile
            or identity.supports_structural_absence is tactile
        ):
            raise ValueError("official ACT profile capabilities mismatch identity")
        if not isinstance(artifact_task, str) or not artifact_task:
            raise ValueError("official ACT artifact task must be non-empty")
        if not callable(input_transform):
            raise TypeError("input_transform must be callable")
        self.identity = identity
        self.profile = normalized_profile
        self._runtime = runtime
        self._artifact_task = artifact_task
        self._input_transform = input_transform
        self._state = _PolicyState.NEW
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending_plan: Optional[ActionPlan] = None

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._state is _PolicyState.CLOSED:
            raise RuntimeError("closed official ACT policy cannot reset")
        if type(context) is not PolicyEpisodeContext:
            raise TypeError("context must be an exact PolicyEpisodeContext")
        if context.action_spec != self.identity.action_spec:
            raise ValueError("context action spec does not match policy identity")
        if context.task != self._artifact_task:
            raise ValueError("context task does not match official ACT artifact task")
        self._runtime.reset()
        self._context = context
        self._pending_plan = None
        self._state = _PolicyState.READY

    def _validate_observation(self, observation: ObservationRecord) -> None:
        if type(observation) is not ObservationRecord:
            raise TypeError("official ACT infer accepts only exact ObservationRecord")
        if self._state is not _PolicyState.READY or self._context is None:
            raise RuntimeError("official ACT infer requires reset and completed commit")
        if (
            observation.episode_id != self._context.episode_id
            or observation.task != self._context.task
            or observation.seed != self._context.initial_seed
        ):
            raise ValueError("official ACT observation identity mismatch")
        if observation.proprio.shape != (8,):
            raise ValueError("official ACT proprio must have exact shape (8,)")
        if observation.proprio.dtype != np.float32:
            raise TypeError("official ACT proprio must use exact float32")
        if not np.isfinite(observation.proprio).all():
            raise ValueError("official ACT proprio must be finite")

    def _encode(self, observation: ObservationRecord) -> Mapping[str, Array]:
        self._validate_observation(observation)
        if "top" not in observation.vision:
            raise ValueError("official ACT observation requires vision['top']")
        encoded: dict[str, Array] = {
            "cam_high": preprocess_camera(observation.vision["top"]),
            "qpos": np.ascontiguousarray(observation.proprio.copy()),
        }
        if self.profile is OfficialACTProfile.VISION_ONLY:
            return encoded
        for slot_id, key in (("left", "tac_left"), ("right", "tac_right")):
            sensor = observation.sensor(slot_id)
            if not sensor.payload_present or sensor.payload is None:
                raise ValueError("official UniVTAC ACT requires both tactile payloads")
            encoded[key] = preprocess_tactile(sensor.payload, slot_id)
        return encoded

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        encoded = self._encode(observation)
        runtime_input = self._input_transform(encoded)
        self._state = _PolicyState.ABORTED
        action = self._runtime.get_action(runtime_input)
        if not isinstance(action, np.ndarray):
            raise TypeError("official ACT output must be a numpy array")
        if action.dtype != np.float32:
            raise TypeError("official ACT output must use exact float32")
        if action.shape != (1, 8):
            raise ValueError("official ACT output must have exact shape (1, 8)")
        if not np.isfinite(action).all():
            raise ValueError("official ACT output must be finite")
        plan = ActionPlan(ACTION_SPEC, observation.step_index, action)
        self._pending_plan = plan
        self._state = _PolicyState.AWAITING_COMMIT
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        if (
            self._state is not _PolicyState.AWAITING_COMMIT
            or self._pending_plan is None
        ):
            raise RuntimeError("official ACT commit requires one pending inference")
        if type(execution) is not PolicyExecution:
            raise TypeError("execution must be an exact PolicyExecution")
        if execution.action_plan_sha256 != self._pending_plan.sha256:
            raise ValueError("execution does not match the pending action plan")
        expected = self._pending_plan.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("executed actions do not match the pending action plan")
        self._pending_plan = None
        self._state = _PolicyState.READY

    def abort(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("abort reason code must be non-empty")
        if self._state is _PolicyState.CLOSED:
            return
        self._pending_plan = None
        self._state = _PolicyState.ABORTED

    def close(self) -> None:
        if self._state is _PolicyState.CLOSED:
            return
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()
        self._pending_plan = None
        self._context = None
        self._state = _PolicyState.CLOSED


__all__ = ["OfficialACTProfile", "OfficialUniVTACACTPolicy"]
