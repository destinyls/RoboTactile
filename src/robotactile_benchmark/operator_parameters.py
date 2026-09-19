"""Versioned, explicit parameters for every atomic operator instance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

from robotactile_benchmark.constants import (
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPTICAL_CONTACT_STRESS_REGISTRY_ID,
    OPTICAL_DECISION_STRESS_REGISTRY_ID,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    OPTICAL_MARKER_REGISTRY_IDS,
    OPTICAL_MARKER_STRESS_REGISTRY_ID,
    SENSOR_FULLFRAME_STRESS_REGISTRY_ID,
    SEVERITY_REGISTRY_ID,
    THREE_DOSE_STRESS_REGISTRY_IDS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import canonical_hash, freeze_value, thaw_value
from robotactile_benchmark.severity import severity_value

PARAMETERIZATION_VERSION = "canonical_operator_instance_v2"


def _seed_offset(operator_id: str, seed: int, axis: str) -> float:
    """Return a deterministic normalized template offset in [-0.03, 0.03]."""

    digest = canonical_hash(
        {"operator_id": operator_id, "operator_seed": seed, "axis": axis}
    )
    unit = int(digest[:8], 16) / float(0xFFFFFFFF)
    return round((unit - 0.5) * 0.06, 8)


def _a2_offsets(
    existing: Mapping[str, Any], window_length: int, expected_count: int
) -> Tuple[int, ...]:
    raw_offsets = existing.get("erased_offsets")
    if raw_offsets is None:
        if expected_count == 1:
            return (0,)
        return tuple(
            int(round(position * (window_length - 1) / (expected_count - 1)))
            for position in range(expected_count)
        )
    if not isinstance(raw_offsets, (list, tuple)) or any(
        isinstance(offset, bool) or not isinstance(offset, int)
        for offset in raw_offsets
    ):
        raise ValueError("A2 erased_offsets must be an integer sequence")
    offsets = tuple(raw_offsets)
    if len(offsets) != expected_count:
        raise ValueError("A2 erased_offsets count must match its severity dose")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("A2 erased_offsets must be sorted and unique")
    if any(offset < 0 or offset >= window_length for offset in offsets):
        raise ValueError("A2 erased_offsets must lie inside the fault window")
    return offsets


def _base_parameters(
    operator_id: str,
    severity_level: int,
    operator_seed: int,
    start_index: int,
    stop_index: int,
    sensor_slots: Tuple[str, ...],
    existing: Mapping[str, Any],
    severity_registry: str,
) -> Dict[str, Any]:
    dose = severity_value(operator_id, severity_level, registry_id=severity_registry)
    if severity_registry in THREE_DOSE_STRESS_REGISTRY_IDS and sensor_slots != (
        "left",
        "right",
    ):
        raise ValueError("three-dose stress profiles require both sensor slots")
    window_length = stop_index - start_index
    parameters: Dict[str, Any] = {
        "parameterization_version": PARAMETERIZATION_VERSION,
        "template_seed": operator_seed,
    }
    temporal_schedule = existing.get("temporal_schedule")
    full_episode = temporal_schedule == "full_episode_v1"
    window_to_end = temporal_schedule == "window_to_end_v1"
    if "temporal_schedule" in existing:
        if not (full_episode or window_to_end):
            raise ValueError("unsupported temporal_schedule")
        temporal_operators = {
            "T1_fixed_source_delay",
            "T2_held_last_freeze",
            "T3_inter_sensor_skew",
        }
        if full_episode and (
            severity_registry not in OPTICAL_MARKER_REGISTRY_IDS
            or operator_id not in temporal_operators
        ):
            raise ValueError("temporal_schedule requires an optical temporal operator")
        if window_to_end and (
            severity_registry
            not in OPTICAL_MARKER_REGISTRY_IDS
            | {SEVERITY_REGISTRY_ID, DIAGNOSTIC_STRESS_MAX_REGISTRY_ID}
            or operator_id not in temporal_operators
        ):
            raise ValueError("window_to_end_v1 requires a temporal operator")
        if full_episode and start_index != 0:
            raise ValueError("full_episode_v1 temporal_schedule requires start_index=0")
        if window_to_end and start_index == 0:
            raise ValueError(
                "window_to_end_v1 temporal_schedule requires start_index>0"
            )
        parameters["temporal_schedule"] = temporal_schedule
    if operator_requires_rest_reference(
        operator_id, severity_registry=severity_registry
    ):
        rest_sha = existing.get("rest_reference_sha256")
        if not isinstance(rest_sha, str) or len(rest_sha) != 64:
            raise ValueError("rest_reference_sha256 must be a lowercase SHA256")
        if any(character not in "0123456789abcdef" for character in rest_sha):
            raise ValueError("rest_reference_sha256 must be a lowercase SHA256")
        parameters["rest_reference_sha256"] = rest_sha

    if operator_id == "A1_stream_absence":
        count = max(1, min(window_length, int(round(window_length * float(dose)))))
        parameters["affected_offsets"] = list(range(count))
    elif operator_id == "A2_frame_erasure":
        if "a2_end_policy" in existing:
            if existing["a2_end_policy"] != "episode_censored_v1":
                raise ValueError("unsupported a2_end_policy")
            parameters["a2_end_policy"] = "episode_censored_v1"
        count = max(1, min(window_length, int(round(window_length * float(dose)))))
        if severity_registry in {
            OPTICAL_DECISION_STRESS_REGISTRY_ID,
            OPTICAL_MARKER_STRESS_REGISTRY_ID,
            OPTICAL_MARKER_EXTREME_REGISTRY_ID,
        }:
            if window_length < 2:
                raise ValueError(
                    "optical stress A2 needs two observations for intermittency"
                )
            count = min(count, window_length - 1)
        parameters["erased_offsets"] = list(_a2_offsets(existing, window_length, count))
    elif operator_id == "F1_global_response_drift":
        if severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID:
            parameters.update(
                target_gain=0.0,
                temporal_path="immediate_step",
                response_domain="absolute_black_frame",
                ablation_id="tactile_null_black_frame_v1",
            )
        else:
            parameters.update(target_gain=float(dose), temporal_path="linear_ramp")
    elif operator_id == "F2_spatial_sensitivity_loss":
        parameters.update(
            retained_gain=float(dose),
            center_xy=[
                round(0.5 + _seed_offset(operator_id, operator_seed, "x"), 8),
                round(0.5 + _seed_offset(operator_id, operator_seed, "y"), 8),
            ],
            sigma_fraction=0.22,
            minimum_sigma_px=2.0,
        )
    elif operator_id == "F3_persistent_surface_artifact":
        scalar = int(dose)
        parameters.update(
            scar_slope=0.45,
            scar_intercept_fraction=round(
                0.18 + _seed_offset(operator_id, operator_seed, "intercept"), 8
            ),
            scar_half_width_px=max(1.0, scalar * 0.6),
            rgb_delta=[0.035 * scalar, -0.018 * scalar, 0.010 * scalar],
        )
    elif operator_id == "F4_local_nonresponsive_patch":
        scalar = int(dose)
        parameters.update(
            center_xy=[
                round(0.5 + _seed_offset(operator_id, operator_seed, "x"), 8),
                round(0.5 + _seed_offset(operator_id, operator_seed, "y"), 8),
            ],
            radius_fraction=0.10 + scalar * 0.025,
            minimum_radius_px=2,
            fill_mode="rest_reference",
        )
    elif operator_id == "F5_contact_shape_distortion":
        parameters.update(
            warp_domain="current_rgb",
            field="compact_quartic_horizontal",
            center_xy=[
                round(0.5 + _seed_offset(operator_id, operator_seed, "x"), 8),
                round(0.5 + _seed_offset(operator_id, operator_seed, "y"), 8),
            ],
            support_radius_fraction=0.30,
            displacement_px=int(dose),
            axis="horizontal",
            direction="positive",
            interpolation="bilinear_float64",
            boundary="identity",
            activation_phases=["contact_onset", "sustained_contact"],
        )
    elif operator_id == "F6_history_residual_imprint":
        parameters.update(history_mix=float(dose), memory_beta=0.72)
    elif operator_id == "F7_high_load_saturation":
        parameters.update(
            synthesis_domain="current_rgb_same_frame",
            anchor_filter="uniform_box",
            anchor_radius_fraction=0.06,
            minimum_anchor_radius_px=2,
            padding="reflect_no_edge_repeat",
            accumulation="float64",
            response_norm="l2_rgb_detail",
            response_knee=float(dose) * 0.35,
            plateau_width_ratio=0.35,
            transfer="identity_then_rational_plateau",
            activation_phases=["contact_onset", "sustained_contact"],
        )
    elif operator_id == "T1_fixed_source_delay":
        lag = int(dose)
        source_map = [
            max(0, index - lag) if full_episode or window_to_end else index - lag
            for index in range(start_index, stop_index)
        ]
        if min(source_map) < 0:
            raise ValueError("T1 fault window must start after delay warm-up")
        parameters.update(lag_frames=lag, source_index_map=source_map)
        if window_to_end:
            parameters.update(startup_policy="hold_first", steady_state_start_index=lag)
    elif operator_id == "T2_held_last_freeze":
        duration = (
            window_length
            if window_to_end or severity_registry == DIAGNOSTIC_STRESS_MAX_REGISTRY_ID
            else min(window_length, int(dose))
        )
        held_index = max(0, start_index - 1)
        parameters.update(
            hold_duration_frames=duration,
            held_source_index=held_index,
            source_index_map=[held_index] * duration
            + list(range(start_index + duration, stop_index)),
        )
        if window_to_end:
            parameters["hold_policy"] = "until_window_end"
    elif operator_id == "T3_inter_sensor_skew":
        delayed_slot = sensor_slots[-1]
        skew = int(dose)
        per_slot_source_map = {
            slot_id: [
                max(0, index - skew) if slot_id == delayed_slot else index
                for index in range(start_index, stop_index)
            ]
            for slot_id in sensor_slots
        }
        parameters.update(
            delayed_slot=delayed_slot,
            skew_frames=skew,
            source_index_map=per_slot_source_map,
        )
        if window_to_end:
            parameters.update(
                startup_policy="hold_first", steady_state_start_index=skew
            )
    elif operator_id == "C1_sensor_identity_misrouting":
        count = max(1, min(window_length, int(round(window_length * float(dose)))))
        parameters.update(
            affected_offsets=list(range(count)),
            route_map={"left": "right", "right": "left"},
        )
    elif operator_id == "C2_frame_misregistration":
        if existing.get("realization") != "registered_pixels":
            raise ValueError("C2 realization must be 'registered_pixels'")
        shift = int(dose)
        parameters.update(
            realization="registered_pixels",
            translation_xy_px=[shift, 0],
            interpolation="integer_copy",
            fill_mode="rest_reference",
        )
    if severity_registry in OPTICAL_MARKER_REGISTRY_IDS:
        from robotactile_benchmark.optical.parameters import extend_parameters

        parameters = extend_parameters(
            operator_id,
            severity_level,
            start_index,
            stop_index,
            sensor_slots,
            existing,
            parameters,
            stress=severity_registry == OPTICAL_MARKER_STRESS_REGISTRY_ID,
            extreme=severity_registry == OPTICAL_MARKER_EXTREME_REGISTRY_ID,
            decision_stress=severity_registry == OPTICAL_DECISION_STRESS_REGISTRY_ID,
            contact_stress=severity_registry == OPTICAL_CONTACT_STRESS_REGISTRY_ID,
            fullframe_stress=severity_registry == SENSOR_FULLFRAME_STRESS_REGISTRY_ID,
        )
    return parameters


def materialize_operator_parameters(
    operator_id: str,
    severity_level: int,
    operator_seed: int,
    start_index: int,
    stop_index: int,
    sensor_slots: Tuple[str, ...],
    supplied: Mapping[str, Any],
    severity_registry: str = SEVERITY_REGISTRY_ID,
) -> Dict[str, Any]:
    """Fill and validate the complete versioned operator-instance contract."""

    frozen = freeze_value(supplied)
    existing = dict(thaw_value(frozen))
    expected = _base_parameters(
        operator_id,
        severity_level,
        operator_seed,
        start_index,
        stop_index,
        sensor_slots,
        existing,
        severity_registry,
    )
    descriptor = {**expected, "operator_id": operator_id}
    expected["instance_descriptor_sha256"] = canonical_hash(descriptor)
    unknown = set(existing) - set(expected)
    if unknown:
        raise ValueError(
            f"parameters contain unregistered fields for {operator_id}: "
            f"{sorted(unknown)}"
        )
    for key, value in existing.items():
        if key == "erased_offsets":
            continue
        if value != expected[key]:
            raise ValueError(
                f"parameter {key} does not match {operator_id} severity/seed contract"
            )
    return expected


def static_parameter_view(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove window-dependent schedules for faulted pairing."""

    schedule_keys = {"affected_offsets", "erased_offsets", "source_index_map"}
    return {
        key: thaw_value(value)
        for key, value in parameters.items()
        if key not in schedule_keys and key != "instance_descriptor_sha256"
    }


def rematerialization_inputs(
    operator_id: str,
    parameters: Mapping[str, Any],
    preserve_erasure_schedule: bool = False,
) -> Dict[str, Any]:
    """Return only caller-supplied inputs needed to build a fresh instance."""

    inputs: Dict[str, Any] = {}
    if "spatial_calibration" in parameters:
        inputs["spatial_calibration"] = thaw_value(parameters["spatial_calibration"])
    if "a2_end_policy" in parameters:
        inputs["a2_end_policy"] = parameters["a2_end_policy"]
    if "temporal_schedule" in parameters:
        inputs["temporal_schedule"] = parameters["temporal_schedule"]
    if "sample_period_s" in parameters:
        inputs["sample_period_s"] = parameters["sample_period_s"]
    if "rest_reference_sha256" in parameters:
        inputs["rest_reference_sha256"] = parameters["rest_reference_sha256"]
    if operator_id == "C2_frame_misregistration":
        inputs["realization"] = parameters["realization"]
    if operator_id == "A2_frame_erasure" and preserve_erasure_schedule:
        inputs["erased_offsets"] = thaw_value(parameters["erased_offsets"])
    return inputs
