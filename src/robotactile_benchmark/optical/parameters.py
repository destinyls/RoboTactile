"""Frozen engineering settings, distinct from legacy paper severity paths."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def extend_parameters(
    operator_id: str,
    level: int,
    start: int,
    stop: int,
    slots: tuple[str, ...],
    supplied: Mapping[str, Any],
    base: dict[str, Any],
    *,
    stress: bool = False,
    extreme: bool = False,
    decision_stress: bool = False,
    contact_stress: bool = False,
    fullframe_stress: bool = False,
) -> dict[str, Any]:
    """Freeze optical settings; observation cadence is explicitly caller supplied."""
    if sum((stress, extreme, decision_stress, contact_stress, fullframe_stress)) > 1:
        raise ValueError("optical stress and extreme profiles are mutually exclusive")
    period = supplied.get("sample_period_s")
    if isinstance(period, bool) or not isinstance(period, (float, int)):
        raise ValueError("optical_marker_v1 requires explicit sample_period_s")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("sample_period_s must be finite and positive")
    result = dict(base)
    result.update(
        parameterization_version="optical_operator_instance_v1",
        sample_period_s=float(period),
    )
    if operator_id.startswith("F"):
        result["optical"] = {
            "marker_max_channel": 65,
            "marker_guard_px": 2,
            "inpaint_radius_fraction": 0.035,
            "minimum_inpaint_radius_px": 6,
            "evidence_tier": "G1",
        }
    if operator_id == "F1_global_response_drift":
        strength = (0.06, 0.12, 0.22, 0.34, 0.48)[level - 1]
        result.update(
            target_gain_rgb=[
                1.0 - strength,
                1.0 - 0.55 * strength,
                1.0 + 0.15 * strength,
            ],
            target_offset_rgb=[0.0, 0.01 * level, -0.006 * level],
        )
    elif operator_id == "F2_spatial_sensitivity_loss":
        result.update(
            center_xy=[0.5, 0.5],
            core_radius_fraction=0.16,
            support_radius_fraction=0.28,
        )
    elif operator_id == "F3_persistent_surface_artifact":
        result.update(scar_half_width_px=1.5 + 0.9 * level)
    elif operator_id == "F4_local_nonresponsive_patch":
        result.update(center_xy=[0.74, 0.50], edge_feather_fraction=0.02)
    elif operator_id == "F5_contact_shape_distortion":
        # The optical profile scales with native resolution, not the legacy px dose.
        result.pop("displacement_px")
        result.update(displacement_fraction=(0.01, 0.02, 0.03, 0.045, 0.06)[level - 1])
    elif operator_id == "F6_history_residual_imprint":
        tau = (0.15, 0.30, 0.60, 1.20, 2.40)[level - 1]
        result.update(
            recovery_tau_s=tau,
            buildup_tau_s=0.03,
            memory_beta=math.exp(-float(period) / tau),
        )
    elif operator_id == "F7_high_load_saturation":
        for legacy_anchor in (
            "anchor_filter",
            "anchor_radius_fraction",
            "minimum_anchor_radius_px",
            "padding",
            "accumulation",
        ):
            result.pop(legacy_anchor)
        result.update(
            synthesis_domain="marker_free_rest_residual",
            response_norm="l2_rgb_optical_residual",
            response_knee=(0.20, 0.15, 0.10, 0.065, 0.035)[level - 1],
        )
    elif operator_id in {
        "T1_fixed_source_delay",
        "T2_held_last_freeze",
        "T3_inter_sensor_skew",
    }:
        seconds = 1.20 if stress else (0.05, 0.10, 0.20, 0.40, 0.80)[level - 1]
        if extreme:
            seconds = 1.50
        frames = max(1, int(math.ceil(seconds / float(period) - 1e-9)))
        if decision_stress:
            frames = 48
            seconds = frames * float(period)
            result.update(
                reference_action_chunk_steps=24,
                requested_source_delay_frames=frames,
            )
        if contact_stress:
            frames = (180, 240, 360)[(level - 1) // 2]
            seconds = frames * float(period)
            result["requested_source_delay_frames"] = frames
        result["requested_duration_s"] = seconds
        temporal_schedule = supplied.get("temporal_schedule")
        full_episode = temporal_schedule == "full_episode_v1"
        window_to_end = temporal_schedule == "window_to_end_v1"
        if operator_id == "T1_fixed_source_delay":
            if start < frames and not (full_episode or window_to_end):
                raise ValueError(
                    "T1 requires source-time warm-up before its fault window"
                )
            result.update(
                lag_frames=frames,
                source_index_map=[
                    max(0, i - frames) if full_episode or window_to_end else i - frames
                    for i in range(start, stop)
                ],
            )
            if full_episode or window_to_end:
                result.update(
                    startup_policy="hold_first",
                    steady_state_start_index=frames,
                )
        elif operator_id == "T2_held_last_freeze":
            duration = (
                stop - start
                if full_episode or window_to_end
                else min(stop - start, frames)
            )
            result.update(
                hold_duration_frames=duration,
                source_index_map=[max(0, start - 1)] * duration
                + list(range(start + duration, stop)),
            )
            if full_episode or window_to_end:
                result.update(
                    hold_policy=(
                        "until_episode_end" if full_episode else "until_window_end"
                    ),
                    effective_hold_duration_s=duration * float(period),
                )
        else:
            if len(slots) != 2:
                raise ValueError("T3 requires two sensor streams")
            result.update(
                skew_frames=frames,
                source_index_map={
                    slot: [
                        max(0, i - frames) if slot == slots[-1] else i
                        for i in range(start, stop)
                    ]
                    for slot in slots
                },
            )
            if window_to_end or (contact_stress and full_episode):
                result.update(
                    startup_policy="hold_first",
                    steady_state_start_index=frames,
                )
    elif operator_id == "C2_frame_misregistration":
        result.update(translation_xy_px=[6 * level, 0])
    if stress:
        overrides: dict[str, dict[str, Any]] = {
            "F1_global_response_drift": {
                "target_gain_rgb": [0.12, 0.35, 0.60],
                "target_offset_rgb": [0.0, 0.05, -0.05],
            },
            "F2_spatial_sensitivity_loss": {
                "core_radius_fraction": 0.30,
                "support_radius_fraction": 0.42,
            },
            "F3_persistent_surface_artifact": {
                "scar_half_width_px": 14.0,
                "rgb_delta": [0.35, -0.18, 0.10],
            },
            "F4_local_nonresponsive_patch": {
                "radius_fraction": 0.34,
                "edge_feather_fraction": 0.03,
            },
            "F5_contact_shape_distortion": {
                "support_radius_fraction": 0.42,
                "displacement_fraction": 0.12,
            },
            "F6_history_residual_imprint": {
                "recovery_tau_s": 6.0,
                "buildup_tau_s": 0.02,
                "memory_beta": math.exp(-float(period) / 6.0),
            },
            "F7_high_load_saturation": {"response_knee": 0.008},
            "C2_frame_misregistration": {"translation_xy_px": [60, 0]},
        }
        result.update(overrides.get(operator_id, {}))
    if extreme or decision_stress:
        extreme_overrides: dict[str, dict[str, Any]] = {
            "F1_global_response_drift": {
                "target_gain_rgb": [0.04, 0.12, 0.30],
                "target_offset_rgb": [0.0, 0.01, -0.03],
            },
            "F2_spatial_sensitivity_loss": {
                "core_radius_fraction": 0.36,
                "support_radius_fraction": 0.48,
            },
            "F3_persistent_surface_artifact": {
                "scar_half_width_px": 20.0,
                "rgb_delta": [0.48, -0.28, 0.18],
            },
            "F4_local_nonresponsive_patch": {
                "radius_fraction": 0.42,
                "edge_feather_fraction": 0.04,
            },
            "F5_contact_shape_distortion": {
                "support_radius_fraction": 0.48,
                "displacement_fraction": 0.20,
            },
            "F6_history_residual_imprint": {
                "recovery_tau_s": 10.0,
                "buildup_tau_s": 0.015,
                "memory_beta": math.exp(-float(period) / 10.0),
            },
            "F7_high_load_saturation": {"response_knee": 0.002},
            "C2_frame_misregistration": {"translation_xy_px": [96, 0]},
        }
        result.update(extreme_overrides.get(operator_id, {}))
    if decision_stress and operator_id == "F1_global_response_drift":
        result.update(temporal_path="ramp_then_hold_v1", rise_duration_frames=6)
    if contact_stress or fullframe_stress:
        result["diagnostic_profile"] = (
            "optical_contact_stress_v2"
            if contact_stress
            else "sensor_fullframe_stress_v1"
        )
        result["stress_parameter_version"] = "contact_stress_parameters_v2"
    if contact_stress and operator_id == "F1_global_response_drift":
        result.update(
            temporal_path="ramp_then_hold_v1",
            rise_duration_frames=6,
            target_gain_rgb=(
                [0.50, 0.60, 0.70],
                [0.20, 0.30, 0.45],
                [0.04, 0.12, 0.30],
            )[(level - 1) // 2],
            target_offset_rgb=[0.0, 0.01, -0.03],
        )
    if contact_stress and operator_id == "F3_persistent_surface_artifact":
        from robotactile_benchmark.optical.stress_templates import (
            validate_spatial_calibration,
        )

        result.update(
            scar_template_version="ranked_band_v1",
            target_contact_coverage=(0.30, 0.60, 0.85)[(level - 1) // 2],
            rgb_delta=[0.48, -0.28, 0.18],
            calibration_status="uncalibrated_geometric_template",
        )
        if "spatial_calibration" in supplied:
            result["spatial_calibration"] = validate_spatial_calibration(
                supplied["spatial_calibration"]
            )
            result["calibration_status"] = "frozen_clean_response_template"
    if fullframe_stress:
        result.update(
            marker_policy="occlude_including_markers_v1",
            fullframe_mask_version="ranked_center_square_v1",
            occluded_area_fraction=(0.30, 0.60, 0.90)[(level - 1) // 2],
            center_xy=[0.5, 0.5],
            fill_rgb=[0, 0, 0],
            fill_mode="constant_rgb_diagnostic",
        )
    return result
