"""Contract tests for the explicitly non-paper destructive stress profile."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators.fidelity_transforms import (
    compact_horizontal_warp,
)
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.severity import (
    DIAGNOSTIC_STRESS_MAX_VALUES,
    severity_value,
)

_REST_SHA256 = "a" * 64


def _parameters(operator_id: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    if operator_id.startswith("F") and operator_id not in {
        "F5_contact_shape_distortion",
        "F7_high_load_saturation",
    }:
        parameters["rest_reference_sha256"] = _REST_SHA256
    if operator_id == "C2_frame_misregistration":
        parameters.update(
            rest_reference_sha256=_REST_SHA256,
            realization="registered_pixels",
        )
    return parameters


@pytest.mark.parametrize("operator_id", sorted(CORE_OPERATOR_IDS))
def test_stress_max_materializes_registered_extreme(operator_id: str) -> None:
    manifest = FaultManifest(
        operator_id=operator_id,
        severity_level=5,
        operator_seed=20260829,
        start_index=16,
        stop_index=300,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters=_parameters(operator_id),
        severity_registry=DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    )

    assert manifest.severity_registry == DIAGNOSTIC_STRESS_MAX_REGISTRY_ID
    assert FaultManifest.from_dict(manifest.to_dict()) == manifest
    assert (
        severity_value(
            operator_id,
            5,
            registry_id=DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
        )
        == DIAGNOSTIC_STRESS_MAX_VALUES[operator_id]
    )


def test_stress_max_registry_is_level_five_only() -> None:
    with pytest.raises(ValueError, match="requires level 5"):
        FaultManifest(
            operator_id="C2_frame_misregistration",
            severity_level=4,
            operator_seed=1,
            start_index=16,
            stop_index=300,
            sensor_slots=("left", "right"),
            observability=Observability.BLIND,
            parameters=_parameters("C2_frame_misregistration"),
            severity_registry=DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
        )


def test_stress_max_resource_matches_runtime_and_f5_is_geometry_valid() -> None:
    resource = load_severity_registry(DIAGNOSTIC_STRESS_MAX_REGISTRY_ID)
    assert resource["calibration_status"].startswith("destructive")
    assert {
        operator_id: entry["value"] for operator_id, entry in resource["paths"].items()
    } == DIAGNOSTIC_STRESS_MAX_VALUES

    height, width = 240, 320
    xx = np.arange(width, dtype=np.uint16)[None, :]
    payload = np.repeat(xx, height, axis=0)
    payload = np.repeat(payload[..., None], 3, axis=2).astype(np.uint8)
    warped = compact_horizontal_warp(
        payload,
        center_xy=(0.5, 0.5),
        support_radius_fraction=0.30,
        displacement_px=20,
    )
    assert warped.shape == payload.shape
    assert np.any(warped != payload)


def test_stress_max_t2_freezes_the_complete_registered_window() -> None:
    manifest = FaultManifest(
        operator_id="T2_held_last_freeze",
        severity_level=5,
        operator_seed=7,
        start_index=16,
        stop_index=600,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={},
        severity_registry=DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    )

    assert manifest.parameters["hold_duration_frames"] == 584
    assert len(manifest.parameters["source_index_map"]) == 584
    assert set(manifest.parameters["source_index_map"]) == {15}
