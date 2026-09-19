"""Behavioral invariants; synthetic fixtures never represent empirical evidence."""

from dataclasses import replace

import numpy as np
import pytest

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    OPTICAL_MARKER_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import (
    ContactPhase,
    array_sha256,
    build_evaluation_record,
)
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.optical.fields import optical_field, spatial_weight
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.streaming.session import StreamingFaultSession
from robotactile_benchmark.validators import validate_delivery


def optical_fixture(period=0.05):
    records = make_synthetic_episode(28)
    yy, xx = np.mgrid[:72, :96]
    base = np.full((72, 96, 3), [95, 130, 155], dtype=np.uint8)
    lattice = ((xx % 12 - 5) ** 2 + (yy % 12 - 5) ** 2) <= 2**2
    base[lattice] = 10
    rest = replace(
        make_synthetic_rest_references(),
        payloads={slot: base.copy() for slot in ("left", "right")},
    )
    result = []
    for i, record in enumerate(records):
        phase = (
            ContactPhase.ONSET
            if i == 8
            else ContactPhase.SUSTAINED
            if 8 < i < 17
            else ContactPhase.RELEASE
            if 17 <= i < 24
            else ContactPhase.FREE
        )
        sensors, provenance = [], []
        for slot in ("left", "right"):
            rgb = base.copy()
            if 8 <= i < 17:
                # Contact moves markers: the old rest-dot locations must not return.
                rgb[:] = [95, 130, 155]
                bump = np.exp(-((xx - 52) ** 2 + (yy - 36) ** 2) / 240)
                rgb = np.clip(
                    rgb.astype(float) + bump[..., None] * [90, 45, -50], 0, 255
                ).astype(np.uint8)
                shifted = (((xx - 3) % 12 - 5) ** 2 + (yy % 12 - 5) ** 2) <= 2**2
                rgb[shifted] = 10
            source = replace(
                record.provenance_for(slot),
                source_time_s=i * period,
                payload_sha256=array_sha256(rgb),
                phase=phase,
            )
            sensors.append(
                replace(
                    record.observation.sensor(slot),
                    payload=rgb,
                    delivery_time_s=i * period,
                )
            )
            provenance.append(source)
        result.append(
            build_evaluation_record(
                replace(record.observation, tactile=tuple(sensors)), tuple(provenance)
            )
        )
    return tuple(result), rest


def manifest_for(op, rest, level=3, period=0.05):
    parameters = {"sample_period_s": period}
    if operator_requires_rest_reference(
        op, severity_registry=OPTICAL_MARKER_REGISTRY_ID
    ):
        parameters["rest_reference_sha256"] = rest.sha256
    if op.startswith("C2"):
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        op,
        level,
        23,
        16,
        26,
        ("left", "right") if op.startswith(("C1", "T3")) else ("right",),
        Observability.BLIND,
        parameters,
        severity_registry=OPTICAL_MARKER_REGISTRY_ID,
    )


@pytest.mark.parametrize("op", sorted(CORE_OPERATOR_IDS))
def test_all14_stream_batch_provenance_and_roundtrip(op):
    clean, rest = optical_fixture()
    manifest = manifest_for(op, rest)
    result = apply_fault(clean, manifest, rest)
    assert result.validation.passed, result.validation
    session = StreamingFaultSession(manifest, rest)
    assert [r.delivered_record_sha256 for r in result.records] == [
        session.deliver_one(r).delivered_record_sha256 for r in clean
    ]
    assert FaultManifest.from_dict(manifest.to_dict()).sha256 == manifest.sha256
    assert manifest.reparameterized().sha256 == manifest.sha256
    assert (
        manifest.reparameterized(severity_level=4).parameters["sample_period_s"] == 0.05
    )


@pytest.mark.parametrize(
    "op",
    [
        "F2_spatial_sensitivity_loss",
        "F4_local_nonresponsive_patch",
        "F6_history_residual_imprint",
        "F7_high_load_saturation",
    ],
)
def test_current_markers_and_no_rest_lattice(op):
    clean, rest = optical_fixture()
    manifest = manifest_for(op, rest)
    result = apply_fault(clean, manifest, rest)
    current = clean[16].observation.sensor("right").payload
    delivered = result.records[16].observation.sensor("right").payload
    _, markers = optical_field(current, manifest.parameters["optical"])
    assert np.array_equal(current[markers], delivered[markers])
    original_rest_dots = np.max(rest.payload_for("right"), axis=2) < 20
    old_only = original_rest_dots & ~markers
    assert not np.any(np.max(delivered[old_only], axis=1) < 40)


def test_optical_manifest_describes_actual_resolution_and_response_domains():
    _, rest = optical_fixture()
    warp = manifest_for("F5_contact_shape_distortion", rest).parameters
    saturation = manifest_for("F7_high_load_saturation", rest).parameters
    assert "displacement_px" not in warp
    assert warp["displacement_fraction"] == 0.03
    assert saturation["synthesis_domain"] == "marker_free_rest_residual"
    assert saturation["response_norm"] == "l2_rgb_optical_residual"
    assert "anchor_filter" not in saturation


@pytest.mark.parametrize(
    "op", ["F2_spatial_sensitivity_loss", "F4_local_nonresponsive_patch"]
)
def test_locality_and_severity(op):
    clean, rest = optical_fixture()
    amplitudes = []
    current = clean[16].observation.sensor("right").payload
    for level in range(1, 6):
        manifest = manifest_for(op, rest, level)
        result = apply_fault(clean, manifest, rest)
        delivered = result.records[16].observation.sensor("right").payload
        support = (
            spatial_weight(
                current.shape, manifest.parameters, dead_patch=op.startswith("F4")
            )
            > 0
        )
        assert np.array_equal(current[~support], delivered[~support])
        amplitudes.append(float(np.abs(current.astype(float) - delivered).sum()))
    assert all(a < b for a, b in zip(amplitudes, amplitudes[1:]))


def test_history_is_causal_and_uses_elapsed_time():
    clean, rest = optical_fixture()
    manifest = manifest_for("F6_history_residual_imprint", rest)
    result = apply_fault(clean, manifest, rest)
    prefix = StreamingFaultSession(manifest, rest)
    assert [prefix.deliver_one(r).delivered_record_sha256 for r in clean[:20]] == [
        r.delivered_record_sha256 for r in result.records[:20]
    ]
    # Fixed physical half-life is independent of the declared observation cadence.
    tau = float(manifest.parameters["recovery_tau_s"])
    assert np.exp(-0.05 / tau) ** 20 == pytest.approx(np.exp(-0.01 / tau) ** 100)
    altered = replace(
        clean[18],
        provenance=tuple(replace(p, source_time_s=0.1) for p in clean[18].provenance),
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        session = StreamingFaultSession(manifest, rest)
        for r in (*clean[:18], altered):
            session.deliver_one(r)


def test_timing_mismatch_rejected_and_no_future_frames():
    clean, rest = optical_fixture()
    manifest = manifest_for("T1_fixed_source_delay", rest, period=0.025)
    with pytest.raises(ValueError, match="cadence"):
        apply_fault(clean, manifest, rest)


def test_corrupted_marker_fails_independent_validation():
    clean, rest = optical_fixture()
    manifest = manifest_for("F2_spatial_sensitivity_loss", rest)
    result = list(apply_fault(clean, manifest, rest).records)
    record = result[16]
    sensor = record.observation.sensor("right")
    rgb = sensor.payload.copy()
    rgb[5, 8] = 255
    sensors = tuple(
        replace(s, payload=rgb) if s.slot_id == "right" else s
        for s in record.observation.tactile
    )
    provenance = tuple(
        replace(p, payload_sha256=array_sha256(rgb)) if p.slot_id == "right" else p
        for p in record.provenance
    )
    result[16] = build_evaluation_record(
        replace(record.observation, tactile=sensors), provenance
    )
    report = validate_delivery(clean, result, manifest, rest)
    assert not report.passed
    assert "MARKER_IDENTITY_CHANGED" in report.failure_codes


@pytest.mark.parametrize(
    ("op", "case", "code"),
    [
        ("F2_spatial_sensitivity_loss", "outside", "NO_LOCAL_CONTACT_RESPONSE"),
        ("F4_local_nonresponsive_patch", "outside", "NO_LOCAL_CONTACT_RESPONSE"),
        ("F5_contact_shape_distortion", "free", "NO_CONTACT_SAMPLE"),
        ("F6_history_residual_imprint", "empty_history", "NO_RESIDUAL_WITNESS"),
        ("F7_high_load_saturation", "below_knee", "NO_ABOVE_KNEE_SAMPLE"),
    ],
)
def test_equation_agreement_without_fault_signature_is_not_success(op, case, code):
    clean, rest = optical_fixture()
    controls = []
    for record in clean:
        sensors, provenance = [], []
        for slot in ("left", "right"):
            rgb = rest.payload_for(slot).copy()
            if case == "outside":
                rgb[:8, :8, 0] = 180
            elif case == "below_knee":
                rgb[..., 0] += 1
            source = record.provenance_for(slot)
            if case == "free":
                source = replace(source, phase=ContactPhase.FREE)
            sensors.append(replace(record.observation.sensor(slot), payload=rgb))
            provenance.append(replace(source, payload_sha256=array_sha256(rgb)))
        controls.append(
            build_evaluation_record(
                replace(record.observation, tactile=tuple(sensors)), tuple(provenance)
            )
        )
    result = apply_fault(controls, manifest_for(op, rest), rest)
    assert not result.validation.passed
    assert code in result.validation.failure_codes
