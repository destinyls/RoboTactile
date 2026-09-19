"""Explicit endpoint censoring preserves observed A2 validation and legacy rules."""

from __future__ import annotations

import pytest

from robotactile_benchmark.closed_loop.delivery import OnlineFaultSession
from robotactile_benchmark.closed_loop.validation import validate_online_delivery
from robotactile_benchmark.contracts import EvaluationRecord
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operator_parameters import rematerialization_inputs
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.validators import ValidationReport


def _manifest(policy: str | None = "episode_censored_v1") -> FaultManifest:
    return FaultManifest(
        operator_id="A2_frame_erasure",
        severity_level=5,
        operator_seed=23,
        start_index=0,
        stop_index=5,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={} if policy is None else {"a2_end_policy": policy},
    )


def _online(
    manifest: FaultManifest, length: int
) -> tuple[
    tuple[EvaluationRecord, ...], tuple[EvaluationRecord, ...], ValidationReport
]:
    clean = make_synthetic_episode(length=max(10, length))[:length]
    session = OnlineFaultSession(manifest)
    delivered = tuple(session.deliver(record) for record in clean)
    report = session.finalize().validation
    assert report is not None
    return clean, delivered, report


def test_policy_is_hash_bound_and_survives_rematerialization() -> None:
    strict = _manifest(None)
    censored = _manifest()
    assert "a2_end_policy" not in strict.parameters
    assert strict.sha256 != censored.sha256
    assert (
        strict.parameters["instance_descriptor_sha256"]
        != censored.parameters["instance_descriptor_sha256"]
    )
    assert rematerialization_inputs(censored.operator_id, censored.parameters) == {
        "a2_end_policy": "episode_censored_v1"
    }
    assert FaultManifest.from_dict(censored.to_dict()) == censored


@pytest.mark.parametrize("policy", ["strict", "unknown", ""])
def test_unregistered_endpoint_policies_are_rejected(policy: str) -> None:
    with pytest.raises(ValueError, match="a2_end_policy"):
        _manifest(policy)


def test_endpoint_policy_is_a2_only() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        FaultManifest(
            operator_id="A1_stream_absence",
            severity_level=3,
            operator_seed=23,
            start_index=0,
            stop_index=5,
            sensor_slots=("left",),
            observability=Observability.DECLARED,
            parameters={"a2_end_policy": "episode_censored_v1"},
        )


def test_terminal_erasure_is_censored_but_default_remains_invalid() -> None:
    clean, _, report = _online(_manifest(), 5)
    assert report.passed
    assert report.metrics["a2_resume_status"] == "right_censored"
    assert report.metrics["a2_right_censored"] is True
    assert report.metrics["a2_resume_observed"] is False
    assert apply_fault(clean, _manifest()).validation == report
    _, _, strict_report = _online(_manifest(None), 5)
    assert "A2_RESUME_MISSING" in strict_report.failure_codes
    with pytest.raises(ValueError, match="later clean payload"):
        apply_fault(clean, _manifest(None))


def test_observed_resume_and_future_planned_erasure_are_distinct() -> None:
    _, _, completed = _online(_manifest(), 6)
    _, _, truncated = _online(_manifest(), 3)
    for report in (completed, truncated):
        assert report.passed
        assert report.metrics["a2_resume_status"] == "observed_resume"
        assert report.metrics["a2_resume_observed"] is True
        assert report.metrics["a2_right_censored"] is False


def test_censoring_never_excuses_interior_missing_payload() -> None:
    manifest = _manifest()
    clean, delivered, _ = _online(manifest, 5)
    bad_manifest = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=5,
        operator_seed=23,
        start_index=1,
        stop_index=2,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )
    corrupted = list(delivered)
    corrupted[1] = apply_fault(clean, bad_manifest).records[1]
    report = validate_online_delivery(clean, corrupted, manifest)
    assert not report.passed
    assert report.failure_codes
    assert report.metrics["a2_resume_status"] == "invalid_delivery"
    assert report.metrics["a2_resume_observed"] is False


def test_offline_replay_still_rejects_window_exceeding_episode() -> None:
    with pytest.raises(ValueError, match="window exceeds"):
        apply_fault(make_synthetic_episode(length=10)[:3], _manifest())
