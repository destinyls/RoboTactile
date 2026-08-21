"""Pure-Python UniVTAC raw-observation adapter without Isaac imports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import numpy as np

from robotactile_benchmark.constants import LOGICAL_STEP_SECONDS
from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
    ObservationRecord,
    SensorObservation,
    SensorProvenance,
    array_sha256,
    build_evaluation_record,
)


@dataclass(frozen=True)
class UniVTACAliasManifest:
    """Explicit raw field aliases; no runtime guessing is permitted."""

    head_camera: str
    wrist_camera: str
    left_tactile: str
    right_tactile: str
    left_physical_source_id: str
    right_physical_source_id: str
    sensor_type: str
    far_plane_mm: float
    calibration_id: str
    calibration_config_sha256: str
    tactile_payload: str = "rgb_marker"

    def __post_init__(self) -> None:
        for field_name in (
            "head_camera",
            "wrist_camera",
            "left_tactile",
            "right_tactile",
            "left_physical_source_id",
            "right_physical_source_id",
            "sensor_type",
            "calibration_id",
            "calibration_config_sha256",
            "tactile_payload",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if not np.isfinite(self.far_plane_mm) or self.far_plane_mm <= 0.0:
            raise ValueError("far_plane_mm must be finite and positive")
        if self.head_camera == self.wrist_camera:
            raise ValueError("UniVTAC camera aliases must be distinct")
        if self.left_tactile == self.right_tactile:
            raise ValueError("UniVTAC tactile aliases must be distinct")
        if self.left_physical_source_id == self.right_physical_source_id:
            raise ValueError("UniVTAC physical tactile sources must be distinct")
        if len(self.calibration_config_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.calibration_config_sha256
        ):
            raise ValueError("calibration_config_sha256 must be a lowercase SHA256")


@dataclass(frozen=True)
class ContactPhaseState:
    """Explicit per-sensor state for depth-hysteresis phase derivation."""

    in_contact: bool


@dataclass(frozen=True)
class DepthPhaseTracker:
    """Convert indentation into evaluator-private phases with hysteresis."""

    on_threshold_mm: float
    off_threshold_mm: float

    def __post_init__(self) -> None:
        if self.off_threshold_mm < 0.0 or self.on_threshold_mm <= self.off_threshold_mm:
            raise ValueError("phase thresholds require 0 <= off < on")

    def update(
        self, indentation_mm: float, state: ContactPhaseState
    ) -> Tuple[ContactPhase, ContactPhaseState]:
        if state.in_contact:
            if indentation_mm <= self.off_threshold_mm:
                return ContactPhase.RELEASE, ContactPhaseState(False)
            return ContactPhase.SUSTAINED, state
        if indentation_mm >= self.on_threshold_mm:
            return ContactPhase.ONSET, ContactPhaseState(True)
        return ContactPhase.FREE, state


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise KeyError(f"required UniVTAC field is missing: {key}") from exc


class UniVTACRecordBuilder:
    """Build benchmark records at the raw `rgb_marker` boundary."""

    def __init__(
        self,
        aliases: UniVTACAliasManifest,
        phase_tracker: DepthPhaseTracker,
    ) -> None:
        self._aliases = aliases
        self._phase_tracker = phase_tracker

    @staticmethod
    def _payload(value: Any) -> Array:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("UniVTAC rgb_marker must be HWC with three channels")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("UniVTAC RGB must have a numeric dtype")
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
            raise ValueError("UniVTAC RGB must be finite and in [0,255]")
        if np.issubdtype(array.dtype, np.floating):
            array = np.rint(array)
        return cast(Array, np.asarray(array, dtype=np.uint8))

    def build_stateless_single_frame(
        self,
        raw: Mapping[str, Any],
        episode_id: str,
        task: str,
        seed: int,
        step_index: int,
    ) -> EvaluationRecord:
        """Canonicalize one isolated frame without retaining phase state."""

        record, _ = self.build_with_phase_states(
            raw,
            phase_states={
                "left": ContactPhaseState(False),
                "right": ContactPhaseState(False),
            },
            episode_id=episode_id,
            task=task,
            seed=seed,
            step_index=step_index,
        )
        return record

    def build_with_phase_states(
        self,
        raw: Mapping[str, Any],
        phase_states: Mapping[str, ContactPhaseState],
        episode_id: str,
        task: str,
        seed: int,
        step_index: int,
    ) -> Tuple[EvaluationRecord, Dict[str, ContactPhaseState]]:
        """Canonicalize one frame and return explicit next phase states."""

        if set(phase_states) != {"left", "right"}:
            raise ValueError("phase state must exactly cover left and right")

        raw_step = _require(raw, "step")
        if (
            isinstance(raw_step, (bool, np.bool_))
            or not isinstance(raw_step, (int, np.integer))
            or int(raw_step) < 0
        ):
            raise ValueError("UniVTAC raw step must be a non-negative integer")
        if int(raw_step) != step_index:
            raise ValueError("caller step_index must equal the UniVTAC raw step")

        cameras = _require(raw, "observation")
        tactile = _require(raw, "tactile")
        embodiment = _require(raw, "embodiment")
        time_s = step_index * LOGICAL_STEP_SECONDS
        vision = {
            "top": self._payload(
                _require(_require(cameras, self._aliases.head_camera), "rgb")
            ),
            "wrist_l": self._payload(
                _require(_require(cameras, self._aliases.wrist_camera), "rgb")
            ),
        }
        sensors = []
        provenance = []
        next_phase_states: Dict[str, ContactPhaseState] = {}
        for slot_id, alias, physical_source_id in (
            (
                "left",
                self._aliases.left_tactile,
                self._aliases.left_physical_source_id,
            ),
            (
                "right",
                self._aliases.right_tactile,
                self._aliases.right_physical_source_id,
            ),
        ):
            raw_sensor = _require(tactile, alias)
            payload = self._payload(_require(raw_sensor, self._aliases.tactile_payload))
            depth = np.asarray(_require(raw_sensor, "depth"), dtype=np.float32)
            if depth.ndim != 2 or not np.isfinite(depth).all():
                raise ValueError("UniVTAC tactile depth must be finite HW")
            indentation = max(0.0, self._aliases.far_plane_mm - float(depth.min()))
            phase, next_state = self._phase_tracker.update(
                indentation, phase_states[slot_id]
            )
            next_phase_states[slot_id] = next_state
            sensors.append(
                SensorObservation(
                    slot_id=slot_id,
                    payload=payload,
                    payload_present=True,
                    declared_validity=True,
                    delivery_index=step_index,
                    delivery_time_s=time_s,
                    visible_source_time_s=None,
                    frame_id=f"{alias}_rgb_marker_raw",
                    calibration_id=self._aliases.calibration_id,
                )
            )
            provenance.append(
                SensorProvenance(
                    slot_id=slot_id,
                    physical_source_id=physical_source_id,
                    source_index=int(raw_step),
                    source_time_s=time_s,
                    payload_sha256=array_sha256(payload),
                    calibration_sha256=self._aliases.calibration_config_sha256,
                    phase=phase,
                )
            )
        joint = np.asarray(_require(embodiment, "joint"), dtype=np.float32)
        if joint.shape != (9,) or not np.isfinite(joint).all():
            raise ValueError("UniVTAC joint observation must be finite shape [9]")
        observation = ObservationRecord(
            episode_id=episode_id,
            task=task,
            seed=seed,
            step_index=step_index,
            tactile=tuple(sensors),
            vision=vision,
            proprio=joint,
        )
        return build_evaluation_record(
            observation, tuple(provenance)
        ), next_phase_states
