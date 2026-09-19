"""Stable paper-to-code identifiers."""

from typing import FrozenSet, Tuple

CORE_OPERATOR_IDS: FrozenSet[str] = frozenset(
    {
        "A1_stream_absence",
        "A2_frame_erasure",
        "F1_global_response_drift",
        "F2_spatial_sensitivity_loss",
        "F3_persistent_surface_artifact",
        "F4_local_nonresponsive_patch",
        "F5_contact_shape_distortion",
        "F6_history_residual_imprint",
        "F7_high_load_saturation",
        "T1_fixed_source_delay",
        "T2_held_last_freeze",
        "T3_inter_sensor_skew",
        "C1_sensor_identity_misrouting",
        "C2_frame_misregistration",
    }
)

REST_REFERENCE_OPERATOR_IDS: FrozenSet[str] = frozenset(
    {
        "F1_global_response_drift",
        "F2_spatial_sensitivity_loss",
        "F3_persistent_surface_artifact",
        "F4_local_nonresponsive_patch",
        "F6_history_residual_imprint",
        "C2_frame_misregistration",
    }
)

SENSOR_SLOTS: Tuple[str, str] = ("left", "right")
LOGICAL_STEP_SECONDS: float = 1.0 / 120.0
FAULT_MANIFEST_SEMANTIC_VERSION: str = "2.0"
OPERATOR_IMPLEMENTATION_VERSION: str = "0.4.0"
SEVERITY_REGISTRY_ID: str = "provisional_engineering_v2"
OPTICAL_MARKER_REGISTRY_ID: str = "optical_marker_v1"
OPTICAL_MARKER_STRESS_REGISTRY_ID: str = "optical_marker_stress_v1"
OPTICAL_MARKER_EXTREME_REGISTRY_ID: str = "optical_marker_extreme_v1"
OPTICAL_DECISION_STRESS_REGISTRY_ID: str = "optical_decision_stress_v1"
OPTICAL_CONTACT_STRESS_REGISTRY_ID: str = "optical_contact_stress_v2"
SENSOR_FULLFRAME_STRESS_REGISTRY_ID: str = "sensor_fullframe_stress_v1"
THREE_DOSE_STRESS_REGISTRY_IDS: FrozenSet[str] = frozenset(
    {
        OPTICAL_CONTACT_STRESS_REGISTRY_ID,
        SENSOR_FULLFRAME_STRESS_REGISTRY_ID,
    }
)
OPTICAL_MARKER_REGISTRY_IDS: FrozenSet[str] = frozenset(
    {
        OPTICAL_MARKER_REGISTRY_ID,
        OPTICAL_MARKER_STRESS_REGISTRY_ID,
        OPTICAL_MARKER_EXTREME_REGISTRY_ID,
        OPTICAL_DECISION_STRESS_REGISTRY_ID,
        *THREE_DOSE_STRESS_REGISTRY_IDS,
    }
)
DIAGNOSTIC_STRESS_MAX_REGISTRY_ID: str = "diagnostic_stress_max_v1"
DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID: str = "diagnostic_tactile_null_black_v1"
DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID: str = (
    "diagnostic_observed_tactile_absence_v1"
)
DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS: FrozenSet[str] = frozenset(
    {
        OPTICAL_DECISION_STRESS_REGISTRY_ID,
        OPTICAL_MARKER_STRESS_REGISTRY_ID,
        OPTICAL_MARKER_EXTREME_REGISTRY_ID,
        DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
        DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
        DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    }
)
SUPPORTED_SEVERITY_REGISTRY_IDS: FrozenSet[str] = frozenset(
    {
        SEVERITY_REGISTRY_ID,
        OPTICAL_MARKER_REGISTRY_ID,
        *DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS,
        *THREE_DOSE_STRESS_REGISTRY_IDS,
    }
)


def operator_requires_rest_reference(
    operator_id: str,
    *,
    severity_registry: str = SEVERITY_REGISTRY_ID,
) -> bool:
    """Return whether one registered realization consumes a rest artifact."""

    if severity_registry == SENSOR_FULLFRAME_STRESS_REGISTRY_ID:
        return False
    if severity_registry in OPTICAL_MARKER_REGISTRY_IDS:
        return (
            operator_id in REST_REFERENCE_OPERATOR_IDS
            or operator_id == "F7_high_load_saturation"
        )
    return operator_id in REST_REFERENCE_OPERATOR_IDS and not (
        operator_id == "F1_global_response_drift"
        and severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID
    )
