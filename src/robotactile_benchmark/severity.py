"""Registered paper and explicitly non-paper diagnostic severity doses."""

from typing import Any, Dict, Tuple

from robotactile_benchmark.constants import (
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPTICAL_CONTACT_STRESS_REGISTRY_ID,
    OPTICAL_DECISION_STRESS_REGISTRY_ID,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    OPTICAL_MARKER_REGISTRY_ID,
    OPTICAL_MARKER_STRESS_REGISTRY_ID,
    SENSOR_FULLFRAME_STRESS_REGISTRY_ID,
    SEVERITY_REGISTRY_ID,
)

SEVERITY_PATHS: Dict[str, Tuple[Any, ...]] = {
    "A1_stream_absence": (0.05, 0.10, 0.20, 0.40, 0.80),
    "A2_frame_erasure": (0.05, 0.10, 0.20, 0.40, 0.80),
    "F1_global_response_drift": (0.95, 0.88, 0.78, 0.65, 0.50),
    "F2_spatial_sensitivity_loss": (0.90, 0.75, 0.55, 0.35, 0.15),
    "F3_persistent_surface_artifact": (1, 2, 3, 4, 5),
    "F4_local_nonresponsive_patch": (1, 2, 3, 4, 5),
    "F5_contact_shape_distortion": (1, 2, 3, 4, 5),
    "F6_history_residual_imprint": (0.12, 0.22, 0.35, 0.50, 0.68),
    "F7_high_load_saturation": (0.90, 0.72, 0.55, 0.40, 0.28),
    "T1_fixed_source_delay": (1, 2, 4, 8, 16),
    "T2_held_last_freeze": (2, 4, 8, 16, 32),
    "T3_inter_sensor_skew": (1, 2, 3, 4, 5),
    "C1_sensor_identity_misrouting": (0.05, 0.10, 0.20, 0.40, 0.80),
    "C2_frame_misregistration": (1, 2, 3, 4, 5),
}

# Same fourteen IDs, separate engineering profile. T* primary doses are seconds;
# the immutable instance additionally stores the achieved integer source map.
OPTICAL_MARKER_PATHS: Dict[str, Tuple[Any, ...]] = {
    **SEVERITY_PATHS,
    "F2_spatial_sensitivity_loss": (0.90, 0.75, 0.55, 0.30, 0.10),
    "F1_global_response_drift": (0.94, 0.88, 0.78, 0.66, 0.52),
    "F7_high_load_saturation": (0.20, 0.15, 0.10, 0.065, 0.035),
    "T1_fixed_source_delay": (0.05, 0.10, 0.20, 0.40, 0.80),
    "T2_held_last_freeze": (0.05, 0.10, 0.20, 0.40, 0.80),
    "T3_inter_sensor_skew": (0.05, 0.10, 0.20, 0.40, 0.80),
    "C2_frame_misregistration": (6, 12, 18, 24, 30),
}

# Deliberately outside the paper S1--S5 calibration.  Each value is the
# strongest implementation-valid dose used by the destructive stress test.
DIAGNOSTIC_STRESS_MAX_VALUES: Dict[str, Any] = {
    "A1_stream_absence": 1.0,
    "A2_frame_erasure": 1.0,
    "F1_global_response_drift": 0.0,
    "F2_spatial_sensitivity_loss": 0.0,
    "F3_persistent_surface_artifact": 12,
    "F4_local_nonresponsive_patch": 12,
    "F5_contact_shape_distortion": 20,
    "F6_history_residual_imprint": 0.95,
    "F7_high_load_saturation": 0.05,
    "T1_fixed_source_delay": 16,
    "T2_held_last_freeze": 1.0,
    "T3_inter_sensor_skew": 16,
    "C1_sensor_identity_misrouting": 1.0,
    "C2_frame_misregistration": 30,
}

# This is a diagnostic model-reliance ablation, not a paper severity path.
# The only valid operator realization is immediate, full-window replacement of
# both tactile streams by dtype- and shape-preserving all-zero RGB frames.
DIAGNOSTIC_TACTILE_NULL_VALUES: Dict[str, Any] = {
    "F1_global_response_drift": 0.0,
}

DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_VALUES: Dict[str, Any] = {
    "A1_stream_absence": 1.0,
}

# Explicit observation-level stress doses, outside the standard severity path.
OPTICAL_MARKER_STRESS_VALUES: Dict[str, Any] = {
    "A1_stream_absence": 1.0,
    "A2_frame_erasure": 0.90,
    "F1_global_response_drift": 0.12,
    "F2_spatial_sensitivity_loss": 0.02,
    "F3_persistent_surface_artifact": 10,
    "F4_local_nonresponsive_patch": 0.34,
    "F5_contact_shape_distortion": 0.12,
    "F6_history_residual_imprint": 0.90,
    "F7_high_load_saturation": 0.008,
    "T1_fixed_source_delay": 1.20,
    "T2_held_last_freeze": 1.20,
    "T3_inter_sensor_skew": 1.20,
    "C1_sensor_identity_misrouting": 1.0,
    "C2_frame_misregistration": 60,
}


OPTICAL_MARKER_EXTREME_VALUES: Dict[str, Any] = {
    "A1_stream_absence": 1.0,
    "A2_frame_erasure": 0.98,
    "F1_global_response_drift": 0.04,
    "F2_spatial_sensitivity_loss": 0.005,
    "F3_persistent_surface_artifact": 14,
    "F4_local_nonresponsive_patch": 0.42,
    "F5_contact_shape_distortion": 0.20,
    "F6_history_residual_imprint": 0.97,
    "F7_high_load_saturation": 0.002,
    "T1_fixed_source_delay": 1.50,
    "T2_held_last_freeze": 1.50,
    "T3_inter_sensor_skew": 1.50,
    "C1_sensor_identity_misrouting": 1.0,
    "C2_frame_misregistration": 96,
}


OPTICAL_DECISION_STRESS_VALUES: Dict[str, Any] = {
    **OPTICAL_MARKER_EXTREME_VALUES,
    "T1_fixed_source_delay": 48,
    "T2_held_last_freeze": 48,
    "T3_inter_sensor_skew": 48,
}


THREE_DOSE_STRESS_PATHS = {
    OPTICAL_CONTACT_STRESS_REGISTRY_ID: {
        "F1_global_response_drift": (0.50, 0.20, 0.04),
        "F3_persistent_surface_artifact": (0.30, 0.60, 0.85),
        "T1_fixed_source_delay": (180, 240, 360),
        "T3_inter_sensor_skew": (180, 240, 360),
    },
    SENSOR_FULLFRAME_STRESS_REGISTRY_ID: {
        "F4_local_nonresponsive_patch": (0.30, 0.60, 0.90),
    },
}


def severity_value(
    operator_id: str,
    level: int,
    *,
    registry_id: str = SEVERITY_REGISTRY_ID,
) -> Any:
    """Look up one registered native dose."""

    if registry_id in THREE_DOSE_STRESS_PATHS:
        if isinstance(level, bool) or level not in (1, 3, 5):
            raise ValueError("three-dose stress registry requires level 1, 3, or 5")
        if operator_id not in THREE_DOSE_STRESS_PATHS[registry_id]:
            raise ValueError("operator unsupported by three-dose stress registry")
        return THREE_DOSE_STRESS_PATHS[registry_id][operator_id][(level - 1) // 2]
    diagnostic_values = {
        OPTICAL_DECISION_STRESS_REGISTRY_ID: OPTICAL_DECISION_STRESS_VALUES,
        OPTICAL_MARKER_EXTREME_REGISTRY_ID: OPTICAL_MARKER_EXTREME_VALUES,
        OPTICAL_MARKER_STRESS_REGISTRY_ID: OPTICAL_MARKER_STRESS_VALUES,
        DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID: (
            DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_VALUES
        ),
        DIAGNOSTIC_STRESS_MAX_REGISTRY_ID: DIAGNOSTIC_STRESS_MAX_VALUES,
        DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID: DIAGNOSTIC_TACTILE_NULL_VALUES,
    }
    if registry_id in diagnostic_values:
        if level != 5:
            raise ValueError("diagnostic stress registry only defines level 5")
        try:
            return diagnostic_values[registry_id][operator_id]
        except KeyError as exc:
            raise KeyError(f"unknown severity path: {operator_id}") from exc
    if registry_id not in {SEVERITY_REGISTRY_ID, OPTICAL_MARKER_REGISTRY_ID}:
        raise ValueError(f"unsupported severity registry: {registry_id}")
    try:
        paths = (
            OPTICAL_MARKER_PATHS
            if registry_id == OPTICAL_MARKER_REGISTRY_ID
            else SEVERITY_PATHS
        )
        path = paths[operator_id]
    except KeyError as exc:
        raise KeyError(f"unknown severity path: {operator_id}") from exc
    if not 1 <= level <= len(path):
        raise ValueError("severity level must be in [1, 5]")
    return path[level - 1]
