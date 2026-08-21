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
