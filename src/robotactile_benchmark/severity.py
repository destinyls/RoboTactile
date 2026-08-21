"""Provisional, within-operator monotone engineering severity paths."""

from typing import Any, Dict, Tuple

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


def severity_value(operator_id: str, level: int) -> Any:
    """Look up one registered native dose."""

    try:
        path = SEVERITY_PATHS[operator_id]
    except KeyError as exc:
        raise KeyError(f"unknown severity path: {operator_id}") from exc
    if not 1 <= level <= len(path):
        raise ValueError("severity level must be in [1, 5]")
    return path[level - 1]
