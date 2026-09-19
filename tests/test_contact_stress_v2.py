"""Synthetic engineering contracts, never measured task success evidence."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_optical14_profile import optical_fixture

from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.optical.stress_templates import (
    calibrate_contact_scars,
    ranked_spatial_mask,
)
from robotactile_benchmark.optical.validation import measure_optical
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.streaming.session import StreamingFaultSession

CONTACT = "optical_contact_stress_v2"
FULL = "sensor_fullframe_stress_v1"


def manifest(op, rest, level=5, registry=CONTACT, extra=None):
    parameters = {"sample_period_s": 1 / 120, **(extra or {})}
    if operator_requires_rest_reference(op, severity_registry=registry):
        parameters["rest_reference_sha256"] = rest.sha256
    if op.startswith("T"):
        parameters["temporal_schedule"] = "window_to_end_v1"
    return FaultManifest(
        op,
        level,
        23,
        5,
        28,
        ("left", "right"),
        Observability.BLIND,
        parameters,
        severity_registry=registry,
    )


@pytest.mark.parametrize("level", [1, 3, 5])
@pytest.mark.parametrize(
    "op,registry",
    [
        ("F1_global_response_drift", CONTACT),
        ("F3_persistent_surface_artifact", CONTACT),
        ("T1_fixed_source_delay", CONTACT),
        ("T3_inter_sensor_skew", CONTACT),
        ("F4_local_nonresponsive_patch", FULL),
    ],
)
def test_registry_roundtrip_schema_delivery_and_streaming(level, op, registry):
    clean, rest = optical_fixture(1 / 120)
    m = manifest(op, rest, level, registry)
    assert m.reparameterized().sha256 == m.sha256
    assert load_severity_registry(registry)["levels"] == [1, 3, 5]
    batch = apply_fault(clean, m, None if registry == FULL else rest)
    assert batch.validation.passed, batch.validation
    session = StreamingFaultSession(m, None if registry == FULL else rest)
    streamed = tuple(session.deliver_one(r) for r in clean)
    for i, actual in enumerate(streamed):
        for slot in m.sensor_slots:
            expected = batch.records[i].observation.sensor(slot).payload
            assert np.array_equal(actual.observation.sensor(slot).payload, expected)


def test_f1_six_frames_reaches_all_three_targets():
    clean, rest = optical_fixture(1 / 120)
    for level, gain in zip(
        (1, 3, 5), ([0.5, 0.6, 0.7], [0.2, 0.3, 0.45], [0.04, 0.12, 0.30])
    ):
        m = manifest("F1_global_response_drift", rest, level)
        session = StreamingFaultSession(m, rest)
        delivered = [session.deliver_one(r) for r in clean]
        for i in (10, 12, 25):
            expected = np.rint(
                np.clip(
                    clean[i].observation.sensor("right").payload / 255 * gain
                    + [0, 0.01, -0.03],
                    0,
                    1,
                )
                * 255
            ).astype(np.uint8)
            assert np.allclose(
                delivered[i].observation.sensor("right").payload,
                expected,
                atol=1,
                rtol=0,
            )


def test_delays_are_source_frames_and_causal():
    _, rest = optical_fixture()
    for level, frames in zip((1, 3, 5), (180, 240, 360)):
        for op in ("T1_fixed_source_delay", "T3_inter_sensor_skew"):
            m = manifest(op, rest, level).reparameterized(stop_index=500)
            assert m.parameters["requested_duration_s"] == frames / 120
            source = m.parameters["source_index_map"]
            if op.startswith("T3"):
                assert source["left"] == tuple(range(5, 500))
                source = source["right"]
            assert source == tuple(max(0, i - frames) for i in range(5, 500))


def test_fullframe_masks_nested_and_markers_are_occluded():
    clean, rest = optical_fixture(1 / 120)
    previous = np.zeros((72, 96), dtype=bool)
    for level, fraction in zip((1, 3, 5), (0.3, 0.6, 0.9)):
        m = manifest("F4_local_nonresponsive_patch", rest, level, FULL)
        mask = ranked_spatial_mask((72, 96), (0.5, 0.5), fraction)
        assert mask.sum() == round(fraction * 72 * 96)
        assert np.all(mask[previous])
        previous = mask
        session = StreamingFaultSession(m, None)
        rows = [session.deliver_one(r) for r in clean]
        source = clean[6].observation.sensor("left").payload
        actual = rows[6].observation.sensor("left").payload
        assert np.all(actual[mask] == 0)
        assert np.array_equal(actual[~mask], source[~mask])
        assert np.any(mask & np.all(source == 10, axis=-1))


def test_calibration_template_is_frozen_and_bound():
    clean, rest = optical_fixture(1 / 120)
    weights = np.zeros((72, 96))
    weights[20:40, 10:80] = 1
    calibration = calibrate_contact_scars({"left": weights, "right": weights}, "a" * 64)
    m = manifest(
        "F3_persistent_surface_artifact",
        rest,
        extra={"spatial_calibration": calibration},
    )
    assert m.reparameterized().sha256 == m.sha256
    assert apply_fault(clean, m, rest).validation.passed
    with pytest.raises(ValueError):
        manifest("F1_global_response_drift", rest, extra={"rise_duration_frames": 1})
    with pytest.raises(ValueError):
        manifest(
            "F1_global_response_drift", rest, extra={"spatial_calibration": calibration}
        )
    for level in (2, 4):
        with pytest.raises(ValueError):
            manifest("F1_global_response_drift", rest, level)
    for op in ("A1_stream_absence", "A2_frame_erasure", "F2_spatial_sensitivity_loss"):
        with pytest.raises(ValueError):
            manifest(op, rest)


def test_json_schema_matches_new_registries():
    validate = pytest.importorskip("jsonschema").validate
    _, rest = optical_fixture()
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas/fault_manifest.schema.json").read_text()
    )
    for level in (1, 3, 5):
        for op, registry in (
            ("F1_global_response_drift", CONTACT),
            ("F3_persistent_surface_artifact", CONTACT),
            ("T1_fixed_source_delay", CONTACT),
            ("T3_inter_sensor_skew", CONTACT),
            ("F4_local_nonresponsive_patch", FULL),
        ):
            validate(manifest(op, rest, level, registry).to_dict(), schema)


def test_fullframe_validator_rejects_marker_restoration():
    clean, rest = optical_fixture(1 / 120)
    m = manifest("F4_local_nonresponsive_patch", rest, registry=FULL)
    result = apply_fault(clean, m).records
    rows = list(result)
    row = rows[10]
    sensors = list(row.observation.tactile)
    source = clean[10].observation.sensor(sensors[0].slot_id).payload
    corrupted = sensors[0].payload.copy()
    marker = np.all(source == 10, axis=-1)
    corrupted[marker] = source[marker]
    sensors[0] = replace(sensors[0], payload=corrupted)
    rows[10] = replace(
        row, observation=replace(row.observation, tactile=tuple(sensors))
    )
    assert not measure_optical(clean, rows, m, None)[0]


def test_calibration_rejects_non_nested_and_unknown_parameters():
    _, rest = optical_fixture()
    weights = np.ones((12, 16))
    calibration = calibrate_contact_scars({"left": weights, "right": weights}, "a" * 64)
    calibration["slots"]["left"]["area_fractions"] = [0.8, 0.4, 0.9]
    with pytest.raises(ValueError, match="nested"):
        manifest(
            "F3_persistent_surface_artifact",
            rest,
            extra={"spatial_calibration": calibration},
        )
    m = manifest("F4_local_nonresponsive_patch", rest, registry=FULL)
    encoded = m.to_dict()
    encoded["parameters"]["marker_policy"] = "preserve_current"
    with pytest.raises(ValueError, match="does not match"):
        FaultManifest.from_dict(encoded)
