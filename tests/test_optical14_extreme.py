"""Engineering extreme invariants and pre-change optical compatibility receipts."""

import math

import pytest
from test_optical14_profile import manifest_for, optical_fixture

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operator_parameters import rematerialization_inputs
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.severity import OPTICAL_MARKER_EXTREME_VALUES
from robotactile_benchmark.streaming.session import StreamingFaultSession


def build_manifest(op, rest, registry=OPTICAL_MARKER_EXTREME_REGISTRY_ID, period=0.1):
    original = manifest_for(op, rest, level=5, period=period)
    data = original.to_dict()
    data["severity_registry"] = registry
    data["parameters"] = rematerialization_inputs(op, original.parameters)
    return FaultManifest.from_dict(data)


@pytest.mark.parametrize("op", sorted(CORE_OPERATOR_IDS))
def test_extreme_stream_batch_and_canonical_roundtrip(op):
    clean, rest = optical_fixture(0.1)
    manifest = build_manifest(op, rest)
    assert FaultManifest.from_dict(manifest.to_dict()).sha256 == manifest.sha256
    assert manifest.reparameterized().sha256 == manifest.sha256
    result = apply_fault(clean, manifest, rest)
    assert result.validation.passed, result.validation
    session = StreamingFaultSession(manifest, rest)
    assert [r.delivered_record_sha256 for r in result.records] == [
        session.deliver_one(r).delivered_record_sha256 for r in clean
    ]
    with pytest.raises(ValueError, match="level 5"):
        manifest.reparameterized(severity_level=4)


@pytest.mark.parametrize(
    "registry,expected",
    [
        (
            "optical_marker_v1",
            "05bc4fcc2e352eada2497e41b22351ada9f3785701954d7a88bffec6ef3b1cb6",
        ),
        (
            "optical_marker_stress_v1",
            "12ebca932996a2e3751a3c49a5a59f82edffbeafc48070a1f815bd4e39709bc4",
        ),
    ],
)
def test_old_profiles_match_pre_extreme_manifest_and_output_hashes(registry, expected):
    clean, rest = optical_fixture(0.1)
    hashes = []
    for op in sorted(CORE_OPERATOR_IDS):
        manifest = build_manifest(op, rest, registry)
        hashes.append(
            [manifest.sha256, apply_fault(clean, manifest, rest).trace_sha256]
        )
    assert canonical_hash(hashes) == expected


def test_extreme_resource_and_native_parameters():
    resource = load_severity_registry(OPTICAL_MARKER_EXTREME_REGISTRY_ID)
    assert {
        op: p["value"] for op, p in resource["paths"].items()
    } == OPTICAL_MARKER_EXTREME_VALUES
    assert "not physically calibrated" in resource["calibration_status"]
    assert resource["cross_operator_comparable"] is False
    _, rest = optical_fixture(0.1)
    parameters = {
        op[:2]: build_manifest(op, rest).parameters for op in CORE_OPERATOR_IDS
    }
    assert parameters["F1"]["target_gain_rgb"] == (0.04, 0.12, 0.30)
    assert parameters["F1"]["target_offset_rgb"] == (0.0, 0.01, -0.03)
    assert parameters["F2"]["retained_gain"] == 0.005
    assert parameters["F2"]["core_radius_fraction"] == 0.36
    assert parameters["F2"]["support_radius_fraction"] == 0.48
    assert parameters["F3"]["scar_half_width_px"] == 20
    assert parameters["F3"]["rgb_delta"] == (0.48, -0.28, 0.18)
    assert parameters["F4"]["radius_fraction"] == 0.42
    assert parameters["F4"]["edge_feather_fraction"] == 0.04
    assert parameters["F5"]["displacement_fraction"] == 0.20
    assert parameters["F5"]["support_radius_fraction"] == 0.48
    assert 1 - 8 / (3 * math.sqrt(3)) * 0.20 / 0.48 > 0
    assert parameters["F6"]["history_mix"] == 0.97
    assert parameters["F6"]["recovery_tau_s"] == 10
    assert parameters["F6"]["buildup_tau_s"] == 0.015
    assert parameters["F7"]["response_knee"] == 0.002
    assert parameters["C2"]["translation_xy_px"] == (96, 0)
    for key in ("A1", "C1"):
        assert len(parameters[key]["affected_offsets"]) == 10


@pytest.mark.parametrize(
    "op", ["T1_fixed_source_delay", "T2_held_last_freeze", "T3_inter_sensor_skew"]
)
def test_extreme_actual_cadence_and_source_map(op):
    _, rest = optical_fixture(0.1)
    initial = build_manifest(op, rest)
    data = initial.to_dict()
    data.update(start_index=90, stop_index=190)
    data["parameters"] = rematerialization_inputs(op, initial.parameters)
    data["parameters"]["sample_period_s"] = 1 / 60
    p = FaultManifest.from_dict(data).parameters
    assert p["requested_duration_s"] == 1.5
    if op.startswith("T1"):
        assert p["lag_frames"] == 90
        assert p["source_index_map"] == tuple(range(100))
    elif op.startswith("T2"):
        assert p["hold_duration_frames"] == 90
        assert p["source_index_map"] == (89,) * 90 + tuple(range(180, 190))
    else:
        assert p["skew_frames"] == 90
        assert p["source_index_map"]["right"] == tuple(range(100))


@pytest.mark.parametrize("length", [2, 4, 10, 100])
def test_extreme_a2_retains_recovery_frame(length):
    _, rest = optical_fixture(0.1)
    m = build_manifest("A2_frame_erasure", rest)
    m = m.reparameterized(stop_index=m.start_index + length)
    assert len(m.parameters["erased_offsets"]) == min(
        length - 1, max(1, round(length * 0.98))
    )
