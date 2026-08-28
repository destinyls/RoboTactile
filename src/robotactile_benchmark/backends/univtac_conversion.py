"""Strict conversion from live UniVTAC observations to benchmark records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.adapters.univtac import (
    ContactPhaseState,
    UniVTACRecordBuilder,
)
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    UniVTACRuntimeHandshake,
)
from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    build_evaluation_record,
    canonical_hash,
    freeze_array,
)


class UniVTACConversionError(ValueError):
    """Stable fail-closed error at the raw UniVTAC boundary."""

    def __init__(self, message: str, *, code: str = "conversion_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ConvertedUniVTACObservation:
    """One dense benchmark record plus its native simulator witnesses."""

    record: EvaluationRecord
    native_step_id: int
    simulator_state_sha256: str
    joint_reorder_witness_sha256: str
    canonical_joint9: Array
    model_visible_qpos8: Array
    left_phase_state: ContactPhaseState
    right_phase_state: ContactPhaseState

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "canonical_joint9", freeze_array(self.canonical_joint9, np.float32)
        )
        object.__setattr__(
            self,
            "model_visible_qpos8",
            freeze_array(self.model_visible_qpos8, np.float32),
        )

    def phase_state_mapping(self) -> Dict[str, ContactPhaseState]:
        """Return a fresh explicit phase-state mapping for the next frame."""

        return {
            "left": self.left_phase_state,
            "right": self.right_phase_state,
        }


def cuda_like_to_numpy(value: Any, field_name: str) -> Array:
    """Convert CUDA-like values through the exact audited method sequence."""

    if isinstance(value, np.ndarray):
        return cast(Array, value)
    current = value
    for method_name in ("detach", "cpu", "contiguous", "numpy"):
        method = getattr(current, method_name, None)
        if not callable(method):
            raise UniVTACConversionError(
                f"{field_name} lacks CUDA conversion method {method_name}"
            )
        current = method()
    if not isinstance(current, np.ndarray):
        raise UniVTACConversionError(
            f"{field_name} CUDA conversion did not produce a numpy array"
        )
    return cast(Array, current)


def _joint_names(
    live_joint_names: Sequence[str], canonical_joint_names: Sequence[str]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    live = tuple(live_joint_names)
    canonical = tuple(canonical_joint_names)
    if (
        len(live) != 9
        or len(set(live)) != 9
        or len(canonical) != 9
        or len(set(canonical)) != 9
        or set(live) != set(canonical)
    ):
        raise UniVTACConversionError(
            "live joint names must uniquely match the canonical joint9 set"
        )
    return live, canonical


def reorder_joint9(
    raw_joint: Array,
    live_joint_names: Sequence[str],
    canonical_joint_names: Sequence[str],
) -> Array:
    """Reorder the complete measured joint9 state by canonical joint names."""

    live, canonical = _joint_names(live_joint_names, canonical_joint_names)
    if not isinstance(raw_joint, np.ndarray):
        raise UniVTACConversionError("raw joint9 must be a numpy array")
    if raw_joint.dtype != np.float32 or raw_joint.shape != (9,):
        raise UniVTACConversionError("raw joint9 must have dtype float32 and shape [9]")
    if not np.isfinite(raw_joint).all():
        raise UniVTACConversionError("raw joint9 must be finite")
    indices = {name: index for index, name in enumerate(live)}
    reordered = np.asarray(
        [raw_joint[indices[name]] for name in canonical], dtype=np.float32
    )
    return freeze_array(reordered, dtype=np.float32)


def joint9_to_qpos8(canonical_joint9: Array) -> Array:
    """Project measured joint9 to the upstream model-visible qpos8 state.

    UniVTAC commands both Franka finger joints through one shared scalar, while
    its ``get_gripper_qpos`` observation reads ``panda_finger_joint1``.  The two
    measured finger joints remain distinct physics evidence and may differ
    under contact, so qpos8 follows that pinned upstream observation rule.
    """

    if (
        not isinstance(canonical_joint9, np.ndarray)
        or canonical_joint9.dtype != np.float32
        or canonical_joint9.shape != (9,)
        or not np.isfinite(canonical_joint9).all()
    ):
        raise UniVTACConversionError(
            "canonical joint9 must be finite float32 with shape [9]"
        )
    return freeze_array(
        np.concatenate((canonical_joint9[:7], canonical_joint9[-2:-1])),
        dtype=np.float32,
    )


def validate_action_batch(actions: Array, config: UniVTACBackendConfig) -> Array:
    """Validate a complete registered 8D batch before simulator side effects."""

    if not isinstance(actions, np.ndarray):
        raise UniVTACConversionError(
            "actions must be a numpy array", code="action_type"
        )
    if actions.dtype != np.float32:
        raise UniVTACConversionError(
            "actions must have exact dtype float32", code="action_dtype"
        )
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 8:
        raise UniVTACConversionError(
            "actions must have non-empty shape [K,8]", code="action_shape"
        )
    if not np.isfinite(actions).all():
        raise UniVTACConversionError("actions must be finite", code="action_nonfinite")
    lower = np.asarray(config.action_lower_bounds, dtype=np.float32)
    upper = np.asarray(config.action_upper_bounds, dtype=np.float32)
    if np.any(actions < lower) or np.any(actions > upper):
        raise UniVTACConversionError(
            "actions violate frozen action bounds", code="action_bounds"
        )
    if config.action_spec == EE8_ACTION_SPEC:
        quaternion_norm = np.linalg.norm(actions[:, 3:7], axis=1)
        if not np.allclose(quaternion_norm, 1.0, atol=1e-4, rtol=0.0):
            raise UniVTACConversionError(
                "ee8 quaternion must be unit length", code="action_quaternion"
            )
    return freeze_array(actions, dtype=np.float32)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UniVTACConversionError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _field(value: Mapping[str, Any], name: str, parent: str) -> Any:
    try:
        return value[name]
    except KeyError as error:
        raise UniVTACConversionError(
            f"required UniVTAC field is missing: {parent}/{name}",
            code="missing_required_field",
        ) from error


def _typed_array(
    value: Any,
    name: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> Array:
    array = cuda_like_to_numpy(value, name)
    if array.shape != expected_shape:
        raise UniVTACConversionError(
            f"{name} shape mismatch: expected {expected_shape}, got {array.shape}",
            code="field_shape",
        )
    if array.dtype != expected_dtype:
        raise UniVTACConversionError(
            f"{name} dtype mismatch: expected {expected_dtype}, got {array.dtype}",
            code="field_dtype",
        )
    if not np.isfinite(array).all():
        raise UniVTACConversionError(f"{name} must be finite", code="field_nonfinite")
    return cast(Array, np.ascontiguousarray(array))


def _tactile_rgb_array(
    value: Any,
    name: str,
    expected_shape: Tuple[int, ...],
) -> Array:
    """Bridge pinned TacEx RGB buffers to an owned canonical uint8 copy."""

    array = cuda_like_to_numpy(value, name)
    if array.shape != expected_shape:
        raise UniVTACConversionError(
            f"{name} shape mismatch: expected {expected_shape}, got {array.shape}",
            code="field_shape",
        )
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise UniVTACConversionError(
            f"{name} dtype mismatch: expected uint8 or float32, got {array.dtype}",
            code="field_dtype",
        )
    if not np.isfinite(array).all():
        raise UniVTACConversionError(f"{name} must be finite", code="field_nonfinite")
    if array.dtype == np.dtype(np.float32):
        if np.any(array < 0.0) or np.any(array > 255.0):
            raise UniVTACConversionError(
                f"{name} float32 values must be in [0,255]",
                code="field_range",
            )
        if not np.equal(array, np.trunc(array)).all():
            raise UniVTACConversionError(
                f"{name} float32 values must be exact integers",
                code="field_fractional",
            )
    return cast(Array, np.array(array, dtype=np.uint8, order="C", copy=True))


def _raw_leaves(
    raw: Mapping[str, Any], config: UniVTACBackendConfig
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Optional[Array]]:
    observation = _mapping(_field(raw, "observation", "root"), "observation")
    tactile = _mapping(_field(raw, "tactile", "root"), "tactile")
    embodiment = _mapping(_field(raw, "embodiment", "root"), "embodiment")
    head = _mapping(
        _field(observation, config.aliases.head_camera, "observation"),
        "head camera",
    )
    wrist = _mapping(
        _field(observation, config.aliases.wrist_camera, "observation"),
        "wrist camera",
    )
    left = _mapping(
        _field(tactile, config.aliases.left_tactile, "tactile"),
        "left tactile",
    )
    right = _mapping(
        _field(tactile, config.aliases.right_tactile, "tactile"),
        "right tactile",
    )
    uint8 = np.dtype(np.uint8)
    float32 = np.dtype(np.float32)
    raw_ee = None
    if config.action_spec == EE8_ACTION_SPEC:
        raw_ee = _typed_array(
            _field(embodiment, "ee", "embodiment"),
            "ee7",
            (7,),
            float32,
        )
    return (
        _typed_array(_field(head, "rgb", "head"), "head RGB", config.head_shape, uint8),
        _typed_array(
            _field(wrist, "rgb", "wrist"), "wrist RGB", config.wrist_shape, uint8
        ),
        _tactile_rgb_array(
            _field(left, config.aliases.tactile_payload, config.aliases.left_tactile),
            "left tactile RGB",
            config.tactile_rgb_shape,
        ),
        _typed_array(
            _field(left, "depth", config.aliases.left_tactile),
            "left tactile depth",
            config.tactile_depth_shape,
            float32,
        ),
        _tactile_rgb_array(
            _field(right, config.aliases.tactile_payload, config.aliases.right_tactile),
            "right tactile RGB",
            config.tactile_rgb_shape,
        ),
        _typed_array(
            _field(right, "depth", config.aliases.right_tactile),
            "right tactile depth",
            config.tactile_depth_shape,
            float32,
        ),
        _typed_array(
            _field(embodiment, "joint", "embodiment"),
            "joint9",
            (9,),
            float32,
        ),
        raw_ee,
    )


def convert_raw_observation(
    raw: Mapping[str, Any],
    *,
    config: UniVTACBackendConfig,
    handshake: UniVTACRuntimeHandshake,
    phase_states: Mapping[str, ContactPhaseState],
    episode_id: str,
    task_id: str,
    initial_seed: int,
    benchmark_step: int,
) -> ConvertedUniVTACObservation:
    """Convert one native frame while keeping native and dense steps distinct."""

    config.validate_handshake(handshake)
    if task_id != config.task.task_id:
        raise UniVTACConversionError("task ID does not match backend config")
    native_step = _field(raw, "step", "root")
    if (
        isinstance(native_step, (bool, np.bool_))
        or not isinstance(native_step, Integral)
        or int(native_step) < 0
    ):
        raise UniVTACConversionError("native step must be a non-negative integer")
    if isinstance(benchmark_step, bool) or benchmark_step < 0:
        raise UniVTACConversionError("benchmark step must be a non-negative integer")
    (
        head,
        wrist,
        left_rgb,
        left_depth,
        right_rgb,
        right_depth,
        raw_joint,
        raw_ee,
    ) = _raw_leaves(raw, config)
    joint = reorder_joint9(
        raw_joint,
        handshake.live_joint_names,
        config.canonical_joint_names,
    )
    qpos8 = joint9_to_qpos8(joint)
    proprio8 = qpos8
    if raw_ee is not None:
        proprio8 = freeze_array(
            np.concatenate((raw_ee, joint[-2:-1])), dtype=np.float32
        )
    builder_raw = {
        "step": benchmark_step,
        "observation": {
            config.aliases.head_camera: {"rgb": head},
            config.aliases.wrist_camera: {"rgb": wrist},
        },
        "tactile": {
            config.aliases.left_tactile: {
                config.aliases.tactile_payload: left_rgb,
                "depth": left_depth,
            },
            config.aliases.right_tactile: {
                config.aliases.tactile_payload: right_rgb,
                "depth": right_depth,
            },
        },
        "embodiment": {"joint": joint},
    }
    record, next_states = UniVTACRecordBuilder(
        config.aliases, config.phase_tracker
    ).build_with_phase_states(
        builder_raw,
        phase_states=phase_states,
        episode_id=episode_id,
        task=task_id,
        seed=initial_seed,
        step_index=benchmark_step,
    )
    record = build_evaluation_record(
        replace(record.observation, proprio=proprio8),
        record.provenance,
    )
    joint_witness = canonical_hash(
        {
            "live_joint_names": handshake.live_joint_names,
            "canonical_joint_names": config.canonical_joint_names,
            "canonical_joint9": joint,
            "model_visible_qpos8": qpos8,
            "shared_gripper_qpos": float(joint[-2]),
        }
    )
    state_hash = canonical_hash(
        {
            "native_step": int(native_step),
            "head_rgb": head,
            "wrist_rgb": wrist,
            "left_rgb_marker": left_rgb,
            "left_depth": left_depth,
            "right_rgb_marker": right_rgb,
            "right_depth": right_depth,
            "canonical_joint9": joint,
            "model_visible_proprio8": proprio8,
        }
    )
    return ConvertedUniVTACObservation(
        record=record,
        native_step_id=int(native_step),
        simulator_state_sha256=state_hash,
        joint_reorder_witness_sha256=joint_witness,
        canonical_joint9=joint,
        model_visible_qpos8=qpos8,
        left_phase_state=next_states["left"],
        right_phase_state=next_states["right"],
    )
