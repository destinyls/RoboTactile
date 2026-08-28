"""Pure numerical contracts for N0-TWAM temporal diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Optional, Tuple

import numpy as np

from robotactile_benchmark.contracts import Array, freeze_array

ChunkPhase = Literal["cold", "warm"]
TemporalModality = Literal["video", "tactile", "action"]
State20Encoder = Callable[[object], object]

NATIVE_ACTION_CHANNELS = 20
NATIVE_ACTION_FRAMES = 2
NATIVE_ACTION_HORIZON = 12
FIRST_WARM_FRAME_COUNT = 2
_TEMPORAL_MODALITIES: Tuple[TemporalModality, ...] = (
    "video",
    "tactile",
    "action",
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _ee8_rows(value: object, name: str) -> Array:
    result = np.asarray(value)
    if (
        result.dtype != np.float32
        or result.ndim != 2
        or result.shape[1] != 8
        or result.shape[0] < 1
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be finite float32 [H,8]")
    quaternion_norms = np.linalg.norm(result[:, 3:7], axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(f"{name} quaternion rows must have unit norm")
    return result


def build_expert_native_action(
    expert_ee8_history: object,
    *,
    frame_start_indices: Sequence[int],
    ee8_to_state20: State20Encoder,
) -> Array:
    """Pack two expert EE8 horizons into native ``[20,2,12]`` order.

    ``frame_start_indices`` selects two frame-major 12-step horizons from a
    continuous expert trajectory. The state encoder is injected explicitly so
    this diagnostic module never imports or initializes the N0 runtime.
    """

    expert = _ee8_rows(expert_ee8_history, "expert EE8 history")
    starts = tuple(
        _integer(value, "frame start index") for value in frame_start_indices
    )
    if len(starts) != NATIVE_ACTION_FRAMES or starts[0] >= starts[1]:
        raise ValueError("frame_start_indices must contain two increasing indices")
    if any(start + NATIVE_ACTION_HORIZON > expert.shape[0] for start in starts):
        raise ValueError("expert history does not cover both 12-step horizons")

    native = np.empty(
        (NATIVE_ACTION_CHANNELS, NATIVE_ACTION_FRAMES, NATIVE_ACTION_HORIZON),
        dtype=np.float32,
    )
    for frame_index, start in enumerate(starts):
        for slot_index in range(NATIVE_ACTION_HORIZON):
            state = np.asarray(ee8_to_state20(expert[start + slot_index]))
            if (
                state.dtype != np.float32
                or state.shape != (NATIVE_ACTION_CHANNELS,)
                or not np.isfinite(state).all()
            ):
                raise ValueError("ee8_to_state20 must return finite float32 [20]")
            native[:, frame_index, slot_index] = state
    return freeze_array(native, np.float32)


def build_cold_expert_native_action(
    expert_ee8_history: object,
    *,
    anchor_index: int,
    ee8_to_state20: State20Encoder,
) -> Array:
    """Build training-consistent cold conditioning plus one expert horizon.

    Native frame 0 repeats the absolute anchor state. The server converts frame
    1 to deltas relative to frame 0's final slot, then zeros normalized frame 0
    exactly as ``pi05_condition_first_frame_zero`` did during training.
    """

    expert = _ee8_rows(expert_ee8_history, "expert EE8 history")
    anchor = _integer(anchor_index, "anchor_index")
    if anchor + NATIVE_ACTION_HORIZON > expert.shape[0]:
        raise ValueError("expert history does not cover the cold 12-step horizon")
    anchor_state = np.asarray(ee8_to_state20(expert[anchor]))
    if (
        anchor_state.dtype != np.float32
        or anchor_state.shape != (NATIVE_ACTION_CHANNELS,)
        or not np.isfinite(anchor_state).all()
    ):
        raise ValueError("ee8_to_state20 must return finite float32 [20]")
    native = np.empty(
        (NATIVE_ACTION_CHANNELS, NATIVE_ACTION_FRAMES, NATIVE_ACTION_HORIZON),
        dtype=np.float32,
    )
    native[:, 0, :] = anchor_state[:, None]
    for slot_index in range(NATIVE_ACTION_HORIZON):
        target_state = np.asarray(ee8_to_state20(expert[anchor + slot_index]))
        if (
            target_state.dtype != np.float32
            or target_state.shape != (NATIVE_ACTION_CHANNELS,)
            or not np.isfinite(target_state).all()
        ):
            raise ValueError("ee8_to_state20 must return finite float32 [20]")
        native[:, 1, slot_index] = target_state
    return freeze_array(native, np.float32)


def select_warm_keyframe_indices(
    *,
    anchor_index: int,
    target_index: int,
    keyframe_stride: int = 3,
    keyframe_count: int = 4,
) -> Tuple[int, ...]:
    """Select the four causal commit frames between cold anchor and warm target."""

    anchor = _integer(anchor_index, "anchor_index")
    target = _integer(target_index, "target_index")
    stride = _integer(keyframe_stride, "keyframe_stride", 1)
    count = _integer(keyframe_count, "keyframe_count", 1)
    if target != anchor + stride * count:
        raise ValueError(
            "target_index must equal anchor_index + keyframe_stride * keyframe_count"
        )
    return tuple(anchor + stride * offset for offset in range(1, count + 1))


@dataclass(frozen=True)
class ChunkHorizonSlice:
    """Executable frame-major horizon slice for one N0 chunk phase."""

    phase: ChunkPhase
    native_frame_start: int
    native_frame_stop: int
    slots_per_frame: int = NATIVE_ACTION_HORIZON

    def __post_init__(self) -> None:
        expected = (1, 2) if self.phase == "cold" else (0, 2)
        if self.phase not in ("cold", "warm"):
            raise ValueError("chunk phase must be 'cold' or 'warm'")
        if (self.native_frame_start, self.native_frame_stop) != expected:
            raise ValueError("native frame slice disagrees with chunk phase")
        if self.slots_per_frame != NATIVE_ACTION_HORIZON:
            raise ValueError("N0 native horizon must contain 12 slots per frame")

    @property
    def flat_start(self) -> int:
        return self.native_frame_start * self.slots_per_frame

    @property
    def flat_stop(self) -> int:
        return self.native_frame_stop * self.slots_per_frame

    @property
    def emitted_horizon_count(self) -> int:
        return self.flat_stop - self.flat_start

    def select(self, full_chunk_ee8: object) -> Array:
        """Validate and select from a full frame-major ``[24,8]`` decode."""

        full = _ee8_rows(full_chunk_ee8, "full chunk EE8")
        expected = NATIVE_ACTION_FRAMES * self.slots_per_frame
        if full.shape != (expected, 8):
            raise ValueError(f"full chunk EE8 must have shape ({expected}, 8)")
        return freeze_array(full[self.flat_start : self.flat_stop], np.float32)


def chunk_horizon_slice(phase: ChunkPhase) -> ChunkHorizonSlice:
    """Return cold-skip or warm-full executable horizon semantics."""

    if phase == "cold":
        return ChunkHorizonSlice("cold", 1, 2)
    if phase == "warm":
        return ChunkHorizonSlice("warm", 0, 2)
    raise ValueError("chunk phase must be 'cold' or 'warm'")


def validate_emitted_horizons(value: object, phase: ChunkPhase) -> Array:
    """Validate the already-decoded EE8 rows emitted by cold or warm serving."""

    result = _ee8_rows(value, "emitted EE8 horizons")
    expected = chunk_horizon_slice(phase).emitted_horizon_count
    if result.shape != (expected, 8):
        raise ValueError(
            f"{phase} emitted EE8 horizons must have shape ({expected}, 8)"
        )
    return freeze_array(result, np.float32)


@dataclass(frozen=True)
class PerHorizonActionError:
    """EE8 error at one absolute horizon index."""

    horizon_index: int
    translation_l2_m: float
    rotation_geodesic_deg: float
    gripper_abs: float


@dataclass(frozen=True)
class HorizonActionErrorAggregate:
    """Mean and worst-case errors over the selected horizons."""

    horizon_count: int
    translation_mean_l2_m: float
    translation_max_l2_m: float
    rotation_mean_geodesic_deg: float
    rotation_max_geodesic_deg: float
    gripper_mean_abs: float
    gripper_max_abs: float


@dataclass(frozen=True)
class HorizonActionMetrics:
    """Per-horizon EE8 errors and their aggregate."""

    horizons: Tuple[PerHorizonActionError, ...]
    aggregate: HorizonActionErrorAggregate


def horizon_action_metrics(
    predicted: object,
    target: object,
    *,
    horizon_start: int = 0,
    horizon_stop: Optional[int] = None,
) -> HorizonActionMetrics:
    """Measure translation, quaternion geodesic, and gripper errors per horizon."""

    prediction = _ee8_rows(predicted, "predicted EE8 horizons")
    reference = _ee8_rows(target, "target EE8 horizons")
    if prediction.shape != reference.shape:
        raise ValueError("predicted and target EE8 shapes must match")
    start = _integer(horizon_start, "horizon_start")
    stop = (
        prediction.shape[0]
        if horizon_stop is None
        else _integer(horizon_stop, "horizon_stop", 1)
    )
    if not start < stop <= prediction.shape[0]:
        raise ValueError("metric horizon is outside the EE8 chunk")

    selected_prediction = prediction[start:stop]
    selected_reference = reference[start:stop]
    translation = np.linalg.norm(
        selected_prediction[:, :3] - selected_reference[:, :3], axis=1
    )
    quaternion_dots = np.abs(
        np.sum(selected_prediction[:, 3:7] * selected_reference[:, 3:7], axis=1)
    )
    rotation = np.degrees(2.0 * np.arccos(np.clip(quaternion_dots, 0.0, 1.0)))
    gripper = np.abs(selected_prediction[:, 7] - selected_reference[:, 7])
    rows = tuple(
        PerHorizonActionError(
            horizon_index=start + index,
            translation_l2_m=float(translation[index]),
            rotation_geodesic_deg=float(rotation[index]),
            gripper_abs=float(gripper[index]),
        )
        for index in range(stop - start)
    )
    return HorizonActionMetrics(
        horizons=rows,
        aggregate=HorizonActionErrorAggregate(
            horizon_count=len(rows),
            translation_mean_l2_m=float(np.mean(translation)),
            translation_max_l2_m=float(np.max(translation)),
            rotation_mean_geodesic_deg=float(np.mean(rotation)),
            rotation_max_geodesic_deg=float(np.max(rotation)),
            gripper_mean_abs=float(np.mean(gripper)),
            gripper_max_abs=float(np.max(gripper)),
        ),
    )


def causal_phase_id(modality: TemporalModality, chunk_index: int) -> int:
    """Map video/tactile to even phases and action to the following odd phase."""

    if modality not in _TEMPORAL_MODALITIES:
        raise ValueError("modality must be video, tactile, or action")
    chunk = _integer(chunk_index, "chunk_index")
    return 2 * chunk + (1 if modality == "action" else 0)


@dataclass(frozen=True)
class CausalMaskTruth:
    """Training and effective serving attention for one query-to-key direction."""

    query_modality: TemporalModality
    key_modality: TemporalModality
    query_phase_id: int
    key_phase_id: int
    training_clean_to_clean: bool
    training_noisy_to_clean: bool
    training_noisy_to_noisy: bool
    serve_query_to_clean_cache: bool
    serve_query_to_current_noisy: bool


def causal_phase_truth_table(chunk_index: int = 0) -> Tuple[CausalMaskTruth, ...]:
    """Return the same-chunk causal direction table used to audit train/serve."""

    chunk = _integer(chunk_index, "chunk_index")
    rows = []
    for query_modality in _TEMPORAL_MODALITIES:
        query_phase = causal_phase_id(query_modality, chunk)
        for key_modality in _TEMPORAL_MODALITIES:
            key_phase = causal_phase_id(key_modality, chunk)
            rows.append(
                CausalMaskTruth(
                    query_modality=query_modality,
                    key_modality=key_modality,
                    query_phase_id=query_phase,
                    key_phase_id=key_phase,
                    training_clean_to_clean=key_phase <= query_phase,
                    training_noisy_to_clean=key_phase < query_phase,
                    training_noisy_to_noisy=key_phase == query_phase,
                    serve_query_to_clean_cache=key_phase < query_phase,
                    serve_query_to_current_noisy=key_phase == query_phase,
                )
            )
    return tuple(rows)


@dataclass(frozen=True)
class FirstWarmTemporalInvariant:
    """Frame-axis parity report for the first post-commit warm inference."""

    expected_frames: int
    video_frames: int
    action_frames: int
    tactile_frames: int
    violation_codes: Tuple[str, ...]
    detail: str

    @property
    def passed(self) -> bool:
        return not self.violation_codes


def check_first_warm_tactile_temporal_invariant(
    *, video_frames: int, action_frames: int, tactile_frames: int
) -> FirstWarmTemporalInvariant:
    """Require video/action/tactile to retain the same two-frame warm axis."""

    video = _integer(video_frames, "video_frames", 1)
    action = _integer(action_frames, "action_frames", 1)
    tactile = _integer(tactile_frames, "tactile_frames", 1)
    codes: list[str] = []
    for modality, count in (
        ("VIDEO", video),
        ("ACTION", action),
        ("TACTILE", tactile),
    ):
        if count != FIRST_WARM_FRAME_COUNT:
            codes.append(f"FIRST_WARM_{modality}_FRAME_COUNT")
    if len({video, action, tactile}) != 1:
        codes.append("FIRST_WARM_MODALITY_FRAME_MISMATCH")
    detail = (
        f"first-warm frame counts: video F={video}, action F={action}, "
        f"tactile F={tactile}; expected all F={FIRST_WARM_FRAME_COUNT}"
    )
    return FirstWarmTemporalInvariant(
        expected_frames=FIRST_WARM_FRAME_COUNT,
        video_frames=video,
        action_frames=action,
        tactile_frames=tactile,
        violation_codes=tuple(codes),
        detail=detail,
    )


__all__ = [
    "CausalMaskTruth",
    "ChunkHorizonSlice",
    "FirstWarmTemporalInvariant",
    "HorizonActionErrorAggregate",
    "HorizonActionMetrics",
    "PerHorizonActionError",
    "build_cold_expert_native_action",
    "build_expert_native_action",
    "causal_phase_id",
    "causal_phase_truth_table",
    "check_first_warm_tactile_temporal_invariant",
    "chunk_horizon_slice",
    "horizon_action_metrics",
    "select_warm_keyframe_indices",
    "validate_emitted_horizons",
]
