"""Controlled stimuli verify production behavior, not empirical sensor fidelity."""

import json

from robotactile_benchmark.optical.probes import run_analytical_probes


def test_analytical_probes_are_eligible_json_serializable_and_pass():
    result = run_analytical_probes()
    json.dumps(result, allow_nan=False)
    assert result["F6"]["passed"], result["F6"]["runs"]
    assert result["F7"]["passed"], result["F7"]["checks"]
    assert result["F6"]["same_current_rest_different_history"]
    for probe in result.values():
        assert probe["empirical_episode_count"] == 0
        assert "NOT empirical" in probe["stimulus"]
    for run in result["F6"]["runs"]:
        assert run["checks"]["eligible"]
        assert run["times_s"][-1] == 12.0
        assert run["measured"]["t90_s"] > run["measured"]["t50_s"] > 0


def test_no_fault_delivery_cannot_pass_or_invent_eligible_history(monkeypatch):
    monkeypatch.setattr(
        "robotactile_benchmark.optical.probes.OpticalDelivery.deliver",
        lambda self, record: record,
    )
    result = run_analytical_probes()
    assert not result["F6"]["passed"]
    assert not result["F6"]["runs"][0]["checks"]["eligible"]
    assert not result["F7"]["passed"]
