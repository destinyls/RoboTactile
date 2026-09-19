"""Decision-time dosing must reach the policy without changing old profiles."""

import numpy as np
import pytest
from test_optical14_profile import optical_fixture

from robotactile_benchmark.constants import (
    OPTICAL_DECISION_STRESS_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.streaming.session import StreamingFaultSession


def decision_manifest(op, rest, period=0.1, stop=28):
    parameters = {"sample_period_s": period}
    if op.startswith("T"):
        parameters["temporal_schedule"] = "window_to_end_v1"
    if operator_requires_rest_reference(
        op, severity_registry=OPTICAL_DECISION_STRESS_REGISTRY_ID
    ):
        parameters["rest_reference_sha256"] = rest.sha256
    return FaultManifest(
        op,
        5,
        23,
        5,
        stop,
        ("left", "right"),
        Observability.BLIND,
        parameters,
        severity_registry=OPTICAL_DECISION_STRESS_REGISTRY_ID,
    )


def test_f1_plateau_is_independent_of_episode_horizon():
    clean, rest = optical_fixture()
    a = decision_manifest("F1_global_response_drift", rest, 0.05)
    b = a.reparameterized(stop_index=301)
    short, long = StreamingFaultSession(a, rest), StreamingFaultSession(b, rest)
    outputs = []
    for r in clean:
        x, y = short.deliver_one(r), long.deliver_one(r)
        assert np.array_equal(
            x.observation.sensor("right").payload, y.observation.sensor("right").payload
        )
        outputs.append(y)
    assert np.array_equal(
        outputs[4].observation.sensor("right").payload,
        clean[4].observation.sensor("right").payload,
    )
    for i in (10, 12, 25):
        rgb = clean[i].observation.sensor("right").payload.astype(float) / 255
        expected = np.rint(
            np.clip(rgb * [0.04, 0.12, 0.30] + [0, 0.01, -0.03], 0, 1) * 255
        ).astype(np.uint8)
        assert np.array_equal(outputs[i].observation.sensor("right").payload, expected)
    assert apply_fault(clean, a, rest).validation.passed
    assert a.reparameterized().sha256 == a.sha256


@pytest.mark.parametrize("period", [1 / 120, 1 / 60, 0.1])
@pytest.mark.parametrize(
    "op", ["T1_fixed_source_delay", "T2_held_last_freeze", "T3_inter_sensor_skew"]
)
def test_temporal_dose_uses_observation_steps_not_a_fictitious_clock(period, op):
    _, rest = optical_fixture(period)
    initial = decision_manifest(op, rest, period)
    from robotactile_benchmark.operator_parameters import rematerialization_inputs

    data = initial.to_dict()
    data.update(start_index=5, stop_index=105)
    data["parameters"] = rematerialization_inputs(op, initial.parameters)
    data["parameters"]["temporal_schedule"] = "window_to_end_v1"
    m = FaultManifest.from_dict(data)
    p = m.parameters
    assert p["requested_duration_s"] == 48 * period
    if op.startswith("T1"):
        assert p["source_index_map"][0] == 0
        assert p["source_index_map"][48] == 5
    elif op.startswith("T2"):
        assert p["source_index_map"] == (4,) * 100
    else:
        assert p["source_index_map"]["right"][48] == 5
        assert p["source_index_map"]["left"][48] == 53
    assert m.reparameterized().sha256 == m.sha256


def test_decision_stress_resource_is_explicitly_diagnostic():
    r = load_severity_registry(OPTICAL_DECISION_STRESS_REGISTRY_ID)
    assert r["cross_operator_comparable"] is False
    assert "not physically calibrated" in r["calibration_status"]
    assert r["paths"]["T3_inter_sensor_skew"]["value"] == 48
