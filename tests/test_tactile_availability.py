"""Missing input routing, zero-fill controls and immutable artifact identity."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_n0_vtla_integration import FakeClient, _context, _identity, _observation
from test_retrained_group_options import binding

from robotactile_benchmark.closed_loop.contracts import BackendSignal, PolicyExecution
from robotactile_benchmark.execution.live_artifacts_values import (
    live_request_identity,
    run_content_sha256_from_identity,
    validate_live_request_identity,
)
from robotactile_benchmark.execution.live_univtac import _make_live_policy
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.paired_live_univtac import _validate_paired_group
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.policies.n0_vtla import OfficialN0VTLAPolicy
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode as Mode,
)
from robotactile_benchmark.policies.tactile_availability import (
    ZeroFillTactilePolicy,
    zero_fill_observation,
)
from robotactile_benchmark.trials import Condition
from scripts.retrained_evaluation import group


def missing_observation(slots: tuple[str, ...] = ("left", "right")):
    observation = _observation()
    for slot in slots:
        observation = observation.replace_sensor(
            observation.sensor(slot).without_payload(False)
        )
    return observation


@pytest.mark.parametrize("slots", [("left",), ("right",), ("left", "right")])
def test_native_omits_only_missing_keys(slots: tuple[str, ...]) -> None:
    client = FakeClient()
    policy = OfficialN0VTLAPolicy(
        replace(_identity(), supports_structural_absence=True),
        lambda: client,
        tactile_availability_mode=Mode.NATIVE_MISSING,
    )
    policy.reset(_context())
    policy.infer(missing_observation(slots))
    message = client.requests[-1]
    assert message["robotactile_tactile_protocol"] == "native_missing_v1"
    for slot in ("left", "right"):
        assert (f"observation/{slot}_tactile" in message) == (slot not in slots)
    np.testing.assert_array_equal(
        message["observation/image"], _observation().vision["top"]
    )


def test_zero_fills_only_absence_and_preserves_fault_record() -> None:
    original = missing_observation(("left",))
    delivered = zero_fill_observation(original, (6, 8, 3))
    assert original.sensor("left").payload is None
    assert delivered.sensor("left").payload_present
    assert np.count_nonzero(delivered.sensor("left").payload) == 0
    assert delivered.sensor("left").visible_source_time_s is None
    np.testing.assert_array_equal(
        delivered.sensor("right").payload, original.sensor("right").payload
    )
    np.testing.assert_array_equal(delivered.proprio, original.proprio)
    with pytest.raises(ValueError, match="shape"):
        zero_fill_observation(original, (12, 16, 3))


def test_vtla_zero_mode_has_explicit_wire_label() -> None:
    client = FakeClient()
    policy = OfficialN0VTLAPolicy(
        replace(_identity(), supports_structural_absence=True),
        lambda: client,
        tactile_availability_mode=Mode.ZERO_FILL,
        tactile_zero_shape=(6, 8, 3),
    )
    policy.reset(_context())
    policy.infer(missing_observation())
    message = client.requests[-1]
    assert message["robotactile_tactile_protocol"] == "zero_fill_v1"
    for slot in ("left", "right"):
        assert np.count_nonzero(message[f"observation/{slot}_tactile"]) == 0


def test_default_still_rejects_missing_payload() -> None:
    client = FakeClient()
    policy = OfficialN0VTLAPolicy(_identity(), lambda: client)
    policy.reset(_context())
    with pytest.raises(ValueError, match="both tactile"):
        policy.infer(missing_observation())
    assert client.requests == []


def test_zero_wrapper_adapts_infer_and_commit() -> None:
    client = FakeClient()
    inner = OfficialN0VTLAPolicy(_identity(), lambda: client)
    policy = ZeroFillTactilePolicy(
        replace(_identity(), supports_structural_absence=True), inner, (6, 8, 3)
    )
    policy.reset(_context())
    obs = missing_observation()
    plan = policy.infer(obs)
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=(obs,) * 50,
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    assert obs.sensor("left").payload is None
    assert "robotactile_tactile_protocol" not in client.requests[-1]
    policy.close()
    assert client.closed


@pytest.mark.parametrize("mode", list(Mode))
def test_request_and_artifact_roundtrip(tmp_path: Path, mode: Mode) -> None:
    request = group.build_clean(binding(tmp_path), "lift_can", tmp_path / "out", 0)
    request = replace(
        request,
        tactile_availability_mode=mode,
        tactile_zero_shape=(6, 8, 3) if mode is Mode.ZERO_FILL else None,
    )
    raw = live_univtac_request_to_dict(request)
    assert ("tactile_availability_mode" in raw) == (mode is not Mode.REQUIRED)
    path = tmp_path / "request.json"
    group.write_json(path, raw)
    assert load_live_univtac_request(path) == request
    loaded = load_live_univtac_run(request)
    assert loaded.policy_identity.supports_structural_absence == (
        mode is not Mode.REQUIRED
    )
    identity = live_request_identity(loaded)
    validate_live_request_identity(identity, loaded.trial, loaded.run_spec, None, None)
    assert run_content_sha256_from_identity(identity) == loaded.content_sha256
    if mode is not Mode.REQUIRED:
        default = load_live_univtac_run(
            replace(
                request,
                tactile_availability_mode=Mode.REQUIRED,
                tactile_zero_shape=None,
            )
        )
        assert loaded.content_sha256 != default.content_sha256


@pytest.mark.parametrize("model", ["n0_twam", "ftp1_policy", "dream_tac"])
def test_zero_factory_preserves_outer_identity(tmp_path: Path, model: str) -> None:
    # The factory-level wrapper is shared; the fake inner policy avoids all GPUs.
    request = group.build_clean(binding(tmp_path), "lift_can", tmp_path / "out", 0)
    from robotactile_benchmark.execution.contracts import LivePolicyKind

    kind = LivePolicyKind.N0 if model == "n0_twam" else LivePolicyKind(model)
    updates = {
        "policy_kind": kind,
        "execute_action_steps": {"n0_twam": 24, "ftp1_policy": 1, "dream_tac": 20}[
            model
        ],
        "tactile_availability_mode": Mode.ZERO_FILL,
        "tactile_zero_shape": (6, 8, 3),
    }
    if model == "n0_twam":
        updates.update(
            retrained_prompt=None,
            retrained_control_hz=None,
            retrained_tactile_payload=None,
        )
    loaded = load_live_univtac_run(replace(request, **updates))
    identities = []

    class Inner:
        def __init__(self, identity):
            self.identity = identity

    def factory(inner):
        identities.append(inner.policy_identity)
        return Inner(inner.policy_identity)

    policy = _make_live_policy(loaded, factory, None)
    assert isinstance(policy, ZeroFillTactilePolicy)
    assert policy.identity == loaded.policy_identity
    assert not identities[0].supports_structural_absence


def test_pair_rejects_mixed_availability_protocols(tmp_path: Path) -> None:
    source = binding(tmp_path)
    source["evaluation"] = {
        "operators": ["A1_stream_absence"],
        "severity_registries": ["optical_marker_extreme_v1"],
        "tactile_availability_mode": "native_missing_v1",
    }
    plan = group.prepare_group(source, "lift_can", tmp_path / "group", 0)
    clean = load_live_univtac_request(plan.parent / "requests/clean.json")
    faulted = replace(
        clean,
        condition=Condition.FAULTED,
        fault_manifest_path=plan.parent
        / "faults/optical_marker_extreme_v1/A1_stream_absence.json",
        tactile_availability_mode=Mode.ZERO_FILL,
        tactile_zero_shape=(6, 8, 3),
    )
    with pytest.raises(ValueError, match="tactile_availability_mode"):
        _validate_paired_group(
            (load_live_univtac_run(clean), load_live_univtac_run(faulted))
        )
