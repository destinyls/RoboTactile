"""Opt-in stress contracts and shared delivery, without empirical claims."""

import math

import numpy as np
import pytest
from test_optical14_profile import manifest_for, optical_fixture

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    OPTICAL_MARKER_REGISTRY_ID,
    OPTICAL_MARKER_STRESS_REGISTRY_ID,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operator_parameters import materialize_operator_parameters
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.severity import OPTICAL_MARKER_STRESS_VALUES, severity_value
from robotactile_benchmark.streaming.session import StreamingFaultSession


@pytest.mark.parametrize("op", sorted(CORE_OPERATOR_IDS))
def test_stress_roundtrip_delivery_and_standard_isolation(op):
    clean, rest = optical_fixture(period=0.1)
    standard = manifest_for(op, rest, level=5, period=0.1)
    standard_dict = standard.to_dict()
    before = apply_fault(clean, standard, rest)
    supplied = {"sample_period_s": 0.1}
    if "rest_reference_sha256" in standard.parameters:
        supplied["rest_reference_sha256"] = rest.sha256
    if op.startswith("C2"):
        supplied["realization"] = "registered_pixels"
    stress = FaultManifest(
        op,
        5,
        23,
        12,
        26,
        standard.sensor_slots,
        Observability.BLIND,
        supplied,
        severity_registry=OPTICAL_MARKER_STRESS_REGISTRY_ID,
    )
    assert FaultManifest.from_dict(stress.to_dict()).sha256 == stress.sha256
    assert stress.reparameterized().sha256 == stress.sha256
    result = apply_fault(clean, stress, rest)
    assert result.validation.passed, result.validation
    session = StreamingFaultSession(stress, rest)
    assert [r.delivered_record_sha256 for r in result.records] == [
        session.deliver_one(r).delivered_record_sha256 for r in clean
    ]
    assert standard.to_dict() == standard_dict
    assert apply_fault(clean, standard, rest).trace_sha256 == before.trace_sha256
    with pytest.raises(ValueError, match="level 5"):
        stress.reparameterized(severity_level=4)


@pytest.mark.parametrize("op", sorted(CORE_OPERATOR_IDS))
def test_canonical_stress_native_dose_and_schedules(op):
    p = materialize_operator_parameters(
        op,
        5,
        23,
        90,
        177,
        ("left", "right"),
        {
            "sample_period_s": 1 / 60,
            **(
                {"rest_reference_sha256": "a" * 64}
                if op
                in {
                    "F1_global_response_drift",
                    "F2_spatial_sensitivity_loss",
                    "F3_persistent_surface_artifact",
                    "F4_local_nonresponsive_patch",
                    "F6_history_residual_imprint",
                    "F7_high_load_saturation",
                    "C2_frame_misregistration",
                }
                else {}
            ),
            **({"realization": "registered_pixels"} if op.startswith("C2") else {}),
        },
        severity_registry=OPTICAL_MARKER_STRESS_REGISTRY_ID,
    )
    dose = severity_value(op, 5, registry_id=OPTICAL_MARKER_STRESS_REGISTRY_ID)
    if op.startswith(("A1", "C1")):
        assert len(p["affected_offsets"]) == 87
    elif op.startswith("A2"):
        assert len(p["erased_offsets"]) == round(87 * dose) < 87
    elif op.startswith("F1"):
        assert p["target_gain"] == dose == p["target_gain_rgb"][0] == 0.12
    elif op.startswith("F2"):
        assert p["retained_gain"] == dose == 0.02
    elif op.startswith("F3"):
        assert p["scar_half_width_px"] == 14
        assert p["rgb_delta"] == [0.35, -0.18, 0.10]
    elif op.startswith("F4"):
        assert p["radius_fraction"] == dose == 0.34
    elif op.startswith("F5"):
        assert p["displacement_fraction"] == dose == 0.12
        assert 1 - 8 / (3 * math.sqrt(3)) * dose / p["support_radius_fraction"] > 0
    elif op.startswith("F6"):
        assert p["history_mix"] == dose == 0.90
        assert p["recovery_tau_s"] == 6
    elif op.startswith("F7"):
        assert p["response_knee"] == dose == 0.008
    elif op.startswith("T"):
        assert p["requested_duration_s"] == dose == 1.2
        if op.startswith("T1"):
            assert p["lag_frames"] == 72
            assert p["source_index_map"] == list(range(18, 105))
        elif op.startswith("T2"):
            assert p["hold_duration_frames"] == 72
            assert p["source_index_map"] == [89] * 72 + list(range(162, 177))
        else:
            assert p["skew_frames"] == 72
            assert p["source_index_map"]["right"] == list(range(18, 105))
    else:
        assert p["translation_xy_px"] == [dose, 0] == [60, 0]


def test_standard_registered_doses_remain_unchanged():
    assert (
        severity_value(
            "F2_spatial_sensitivity_loss", 5, registry_id=OPTICAL_MARKER_REGISTRY_ID
        )
        == 0.10
    )
    assert (
        severity_value(
            "T1_fixed_source_delay", 5, registry_id=OPTICAL_MARKER_REGISTRY_ID
        )
        == 0.80
    )
    assert np.isclose(
        severity_value(
            "F7_high_load_saturation", 5, registry_id=OPTICAL_MARKER_REGISTRY_ID
        ),
        0.035,
    )


def test_short_stress_erasure_window_remains_intermittent():
    p = materialize_operator_parameters(
        "A2_frame_erasure",
        5,
        23,
        90,
        94,
        ("right",),
        {"sample_period_s": 1 / 60},
        severity_registry=OPTICAL_MARKER_STRESS_REGISTRY_ID,
    )
    assert len(p["erased_offsets"]) == 3


def test_stress_resource_matches_runtime_doses_and_claim_boundary():
    resource = load_severity_registry(OPTICAL_MARKER_STRESS_REGISTRY_ID)
    assert resource["registry_id"] == OPTICAL_MARKER_STRESS_REGISTRY_ID
    assert resource["level"] == 5
    assert resource["cross_operator_comparable"] is False
    assert "not physically calibrated" in resource["calibration_status"]
    assert {
        op: path["value"] for op, path in resource["paths"].items()
    } == OPTICAL_MARKER_STRESS_VALUES
