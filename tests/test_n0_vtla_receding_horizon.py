"""50-to-8 opt-in contracts and real runner/adapter integration with test doubles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_n0_vtla_integration import FakeClient, _context, _identity, _observation
from test_tactile_availability_groups import binding

from robotactile_benchmark.closed_loop.contracts import BackendSignal, PolicyExecution
from robotactile_benchmark.closed_loop.fakes import DeterministicFakeBackend
from robotactile_benchmark.closed_loop.runner import run_closed_loop_trial_with_evidence
from robotactile_benchmark.contracts import Array, array_sha256, build_evaluation_record
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_values import (
    live_request_identity,
    run_content_sha256_from_identity,
    validate_live_request_identity,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.policies.n0_vtla import OfficialN0VTLAPolicy
from robotactile_benchmark.policies.n0_vtla_execution import RECEDING_HORIZON_50X8
from robotactile_benchmark.policies.tactile_availability import TactileAvailabilityMode
from scripts.retrained_evaluation import group
from scripts.retrained_evaluation.a2_same_process import A2, TASK, prepare


def shelf_binding(root: Path) -> dict[str, Any]:
    value = binding(root)
    value["tasks"] = {
        TASK: {"prompt": "Reorient a bottle upright and place it on a shelf"}
    }
    return value


def prefix_request(root: Path) -> LiveUniVTACRunRequest:
    value = shelf_binding(root)
    value["evaluation"]["n0_vtla_execution_profile"] = RECEDING_HORIZON_50X8
    return group.build_clean(value, TASK, root / "group", 0)


def test_prepare_preserves_fault_and_separates_protocol_identities(
    tmp_path: Path,
) -> None:
    source = tmp_path / "binding.json"
    group.write_json(source, shelf_binding(tmp_path))
    original = source.read_bytes()
    prepare(source, tmp_path / "old", 0)
    prepare(source, tmp_path / "new", 0, RECEDING_HORIZON_50X8)
    old = load_live_univtac_request(tmp_path / "old/prepared/requests/clean.json")
    new = load_live_univtac_request(tmp_path / "new/prepared/requests/clean.json")
    assert old.execute_action_steps == 50 and new.execute_action_steps == 8
    assert new.max_observation_steps == old.max_observation_steps == 301
    assert new.max_control_cycles == 38
    assert old.success_profile_id == new.success_profile_id
    assert (
        old.base_system_id != new.base_system_id
        and old.config_sha256 != new.config_sha256
    )
    assert old.checkpoint_sha256 == new.checkpoint_sha256
    assert "n0_vtla_execution_profile" not in live_univtac_request_to_dict(old)
    assert "n0_vtla_execution_profile" not in live_request_identity(
        load_live_univtac_run(old)
    )
    relative = f"prepared/faults/optical_marker_extreme_v1/{A2}.json"
    assert (tmp_path / "old" / relative).read_bytes() == (
        tmp_path / "new" / relative
    ).read_bytes()
    fault = json.loads((tmp_path / "new" / relative).read_text())
    assert set(range(0, 300, 8)) <= set(fault["parameters"]["erased_offsets"])
    assert source.read_bytes() == original
    plan = json.loads((tmp_path / "new/plan.json").read_text())
    assert plan["conditions"] == ["clean", A2] and plan["planned_episodes"] == 2
    assert plan["prediction_horizon"] == 50 and plan["execute_action_steps"] == 8
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "new", 0, RECEDING_HORIZON_50X8)


@pytest.mark.parametrize(
    "changes",
    [
        {"task_id": "lift_bottle"},
        {"n0_vtla_execution_profile": "unknown"},
        {"n0_vtla_execution_profile": None},
        {"retrained_control_hz": 60},
        {"execute_action_steps": 50},
        {"execute_action_steps": 7},
        *(
            {"policy_kind": kind}
            for kind in LivePolicyKind
            if kind is not LivePolicyKind.N0_VTLA
        ),
    ],
)
def test_profile_cannot_leak_or_silently_change_stride(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        replace(prefix_request(tmp_path), **changes)


@pytest.mark.parametrize(
    "model,steps", [("n0_vtla", 50), ("ftp1_policy", 1), ("dream_tac", 20)]
)
def test_other_default_execution_contracts_unchanged(
    tmp_path: Path, model: str, steps: int
) -> None:
    value = shelf_binding(tmp_path)
    value["model"] = model
    request = group.build_clean(value, TASK, tmp_path / "group", 0)
    assert request.execute_action_steps == steps
    assert "n0_vtla_execution_profile" not in live_univtac_request_to_dict(request)
    value["evaluation"]["n0_vtla_execution_profile"] = RECEDING_HORIZON_50X8
    if model != "n0_vtla":
        with pytest.raises(ValueError, match="another model"):
            group.build_clean(value, TASK, tmp_path / "other", 0)


def test_request_schema_and_identity_roundtrip(tmp_path: Path) -> None:
    request = prefix_request(tmp_path)
    document = live_univtac_request_to_dict(request)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "schemas/live_univtac_request.schema.json"
        ).read_text()
    )
    assert schema["properties"]["n0_vtla_execution_profile"] == {
        "const": RECEDING_HORIZON_50X8
    }
    branch = next(
        item
        for item in schema["allOf"]
        if item.get("if") == {"required": ["n0_vtla_execution_profile"]}
    )
    assert branch["then"]["properties"] == {
        "policy_kind": {"const": "n0_vtla"},
        "task_id": {"const": TASK},
        "execute_action_steps": {"const": 8},
        "retrained_control_hz": {"const": 10},
    }
    assert any(
        item.get("if", {}).get("not") == {"required": ["n0_vtla_execution_profile"]}
        and item.get("then", {}).get("properties", {}).get("execute_action_steps")
        == {"const": 50}
        for item in schema["allOf"]
    )
    path = tmp_path / "request.json"
    group.write_json(path, document)
    assert load_live_univtac_request(path) == request
    loaded = load_live_univtac_run(request)
    identity = live_request_identity(loaded)
    assert run_content_sha256_from_identity(identity) == loaded.content_sha256
    assert (
        validate_live_request_identity(
            identity, loaded.trial, loaded.run_spec, None, None
        )
        == identity
    )
    for index, changes in enumerate(
        [
            {"n0_vtla_execution_profile": None},
            {"execute_action_steps": 50},
            {"policy_kind": "dream_tac"},
        ]
    ):
        changed = {**document, **changes}
        bad_path = tmp_path / f"invalid-{index}.json"
        group.write_json(bad_path, changed)
        with pytest.raises(ValueError):
            load_live_univtac_request(bad_path)
    for field in ["n0_vtla_execution_profile", "retrained_control_hz"]:
        broken = {k: v for k, v in identity.items() if k != field}
        with pytest.raises(ValueError):
            validate_live_request_identity(
                broken, loaded.trial, loaded.run_spec, None, None
            )


class RecordingClient:
    """Return 50 distinct actions per query so stale-tail execution is detectable."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.plans: list[Array] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def infer(self, observation: Mapping[str, object]) -> Array:
        self.requests.append(dict(observation))
        actions = np.zeros((50, 32), dtype=np.float32)
        actions[:, :7] = (
            np.arange(50, dtype=np.float32)[:, None] + len(self.requests) * 50
        ) * 0.0001
        actions[8:, 7] = 0.04
        self.plans.append(actions.copy())
        return actions

    def close(self) -> None:
        pass


@pytest.mark.parametrize("faulted", [False, True])
def test_runner_38_fresh_queries_terminal_four_and_artifact_reload(
    tmp_path: Path, faulted: bool
) -> None:
    source = tmp_path / "binding.json"
    group.write_json(source, shelf_binding(tmp_path))
    prepare(source, tmp_path / "prepared", 0, RECEDING_HORIZON_50X8)
    name = f"optical_marker_extreme_v1/{A2}" if faulted else "clean"
    request = load_live_univtac_request(
        tmp_path / f"prepared/prepared/requests/{name}.json"
    )
    loaded = load_live_univtac_run(request)
    records = []
    for record in make_synthetic_episode(301):
        tactile, provenance = [], []
        for sensor, origin in zip(record.observation.tactile, record.provenance):
            payload = np.full(
                (6, 8, 3), record.observation.step_index % 255, dtype=np.uint8
            )
            tactile.append(replace(sensor, payload=payload))
            provenance.append(replace(origin, payload_sha256=array_sha256(payload)))
        observation = replace(
            record.observation,
            tactile=tuple(tactile),
            proprio=record.observation.proprio[:8],
        )
        records.append(build_evaluation_record(observation, tuple(provenance)))
    backend = DeterministicFakeBackend(
        records, success_predicate_id=loaded.run_spec.success_predicate_id
    )
    backend.action_spec = loaded.trial.action_spec
    client = RecordingClient()
    policy = OfficialN0VTLAPolicy(
        loaded.policy_identity,
        lambda: client,
        task_id=TASK,
        training_prompt=loaded.run_spec.prompt,
        tactile_availability_mode=TactileAvailabilityMode.NATIVE_MISSING,
        execution_profile=request.n0_vtla_execution_profile,
    )
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        policy,
        fault_manifest=loaded.fault_manifest,
    )
    assert len(client.requests) == backend.execute_count == 38
    assert client.reset_count == backend.reset_count == 1
    for query, step in enumerate(range(0, 300, 8)):
        message = client.requests[query]
        assert np.array_equal(message["state"], backend.observation_trace[step].proprio)
        assert np.array_equal(
            message["observation/image"], backend.observation_trace[step].vision["top"]
        )
        assert ("observation/left_tactile" not in message) is faulted
        assert ("observation/right_tactile" not in message) is faulted
    artifact = tmp_path / "artifact"
    write_live_univtac_artifact(
        artifact, loaded, evidence, capture_profile=LiveCaptureProfile.PAPER_FULL
    )
    reloaded = load_live_univtac_artifact(artifact)
    assert reloaded.evidence.result == evidence.result
    terminal = json.loads((artifact / "terminal_result.json").read_text())
    assert terminal["validation_passed"] and terminal["score_eligible"]
    assert (
        terminal["terminal_status"] == "timeout"
        and terminal["observation_count"] == 301
    )
    actions = json.loads((artifact / "action_trace.json").read_text())["entries"]
    assert [entry["source_step_index"] for entry in actions] == list(range(0, 300, 8))
    assert [entry["executed_actions"]["shape"][0] for entry in actions] == [8] * 37 + [
        4
    ]
    for index, entry in enumerate(actions):
        actual = np.load(
            artifact / entry["executed_actions"]["path"], allow_pickle=False
        )
        assert np.array_equal(actual, client.plans[index][: len(actual), :8])


@pytest.mark.parametrize(
    "count,signal",
    [
        (4, BackendSignal.RUNNING),
        (9, BackendSignal.TIMEOUT),
        (50, BackendSignal.RUNNING),
    ],
)
def test_commit_rejects_wrong_prefix(
    tmp_path: Path, count: int, signal: BackendSignal
) -> None:
    policy = OfficialN0VTLAPolicy(
        _identity(),
        FakeClient,
        task_id=TASK,
        training_prompt="shelf",
        execution_profile=RECEDING_HORIZON_50X8,
    )
    policy.reset(replace(_context(), task=TASK, instruction="shelf"))
    obs = replace(_observation(), task=TASK)
    plan = policy.infer(obs)
    with pytest.raises(RuntimeError):
        policy.commit(
            PolicyExecution(
                action_plan_sha256=plan.sha256,
                executed_actions=plan.actions[:count],
                delivered_observations=tuple(obs for _ in range(count)),
                terminal_signal=signal,
            )
        )
