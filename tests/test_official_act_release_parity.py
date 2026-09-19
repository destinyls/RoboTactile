"""Release-parity checks for the official UniVTAC ACT execution surface."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_official_act_live_cli import _request

from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.execution.official_act import (
    execute_official_act_live_run,
    execute_official_act_paired_live_runs,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition


def _backend(loaded: object) -> DeterministicFakeBackend:
    return DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        success_predicate_id=loaded.run_spec.success_predicate_id,  # type: ignore[attr-defined]
    )


def _policy_loader(identity: object, _request: object) -> DeterministicFakePolicy:
    return DeterministicFakePolicy(identity)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "profile",
    (LiveCaptureProfile.METRICS_ONLY, LiveCaptureProfile.PREVIEW),
)
def test_single_act_run_supports_compact_capture(
    tmp_path: Path,
    profile: LiveCaptureProfile,
) -> None:
    request = _request(tmp_path / profile.value, Condition.CLEAN)
    artifact = execute_official_act_live_run(
        request,
        artifact_root=tmp_path / "checkpoints",
        stats_sha256="e" * 64,
        encoder_sha256="f" * 64,
        backend_factory=_backend,
        policy_loader=_policy_loader,
        capture_profile=profile,
    )

    assert artifact.capture_profile is profile
    assert artifact.root_receipt.result_sha256 == artifact.evidence.result.sha256


def test_paired_act_publishes_verified_artifacts_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = _request(tmp_path / "clean", Condition.CLEAN)
    fault = FaultManifest(
        operator_id="C1_sensor_identity_misrouting",
        severity_level=3,
        operator_seed=31,
        start_index=0,
        stop_index=2,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={},
    )
    fault_path = tmp_path / "fault.json"
    fault_path.write_text(json.dumps(fault.to_dict()), encoding="utf-8")
    requests = (
        clean,
        replace(
            clean,
            condition=Condition.FAULTED,
            output_dir=tmp_path / "faulted-artifact",
            fault_manifest_path=fault_path,
        ),
        _request(tmp_path / "no_touch", Condition.NO_TOUCH),
    )
    trials = tuple(SimpleNamespace(identifier=index) for index in range(3))
    results = tuple(SimpleNamespace(sha256=str(index) * 64) for index in range(3))
    paired = SimpleNamespace(
        executions=tuple(
            SimpleNamespace(
                loaded=SimpleNamespace(trial=trial),
                evidence=SimpleNamespace(result=result),
            )
            for trial, result in zip(trials, results)
        )
    )
    artifacts = tuple(
        SimpleNamespace(
            trial=trial,
            root_receipt=SimpleNamespace(result_sha256=result.sha256),
            capture_profile=LiveCaptureProfile.PREVIEW,
        )
        for trial, result in zip(trials, results)
    )
    events: list[str] = []
    loaded_stats: list[str] = []
    loaded_policies: list[DeterministicFakePolicy] = []

    def policy_loader(
        identity: object, load_request: object
    ) -> DeterministicFakePolicy:
        loaded_stats.append(load_request.manifest.stats_sha256)  # type: ignore[attr-defined]
        policy = DeterministicFakePolicy(identity)  # type: ignore[arg-type]
        loaded_policies.append(policy)
        return policy

    def execute(*_args: object, **kwargs: object) -> object:
        policy_factory = kwargs["policy_factory"]
        assert callable(policy_factory)
        for request in requests:
            policy_factory(load_live_univtac_run(request)).close()
        publisher = kwargs["pre_close_publisher"]
        assert callable(publisher)
        publisher(paired)
        events.append("session_closed")
        return paired

    monkeypatch.setattr(
        "robotactile_benchmark.execution.official_act.execute_paired_live_univtac_runs",
        execute,
    )
    monkeypatch.setattr(
        "robotactile_benchmark.execution.official_act.load_live_univtac_artifact",
        lambda path: artifacts[
            tuple(request.output_dir for request in requests).index(Path(path))
        ],
    )

    published = []
    result = execute_official_act_paired_live_runs(
        requests,
        artifact_root=tmp_path / "checkpoints",
        stats_sha256={
            OfficialACTProfile.UNIVTAC: "e" * 64,
            OfficialACTProfile.VISION_ONLY: "d" * 64,
        },
        encoder_sha256="f" * 64,
        policy_loader=policy_loader,
        capture_profile=LiveCaptureProfile.PREVIEW,
        pre_close_publisher=lambda value: (
            events.append("receipt_published"),
            published.append(value),
        ),
    )

    assert events == ["receipt_published", "session_closed"]
    assert loaded_stats == ["e" * 64, "d" * 64]
    assert len(loaded_policies) == 2
    assert [policy.close_count for policy in loaded_policies] == [1, 1]
    assert published == [result]
    assert result.artifacts == artifacts
