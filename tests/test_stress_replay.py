"""CPU engineering checks; no synthetic result is a model SR claim."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robotactile_benchmark.contracts import build_evaluation_record
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.integrations.n0_twam.input_probe import (
    install_n0_input_probe,
)
from robotactile_benchmark.n0_fault_campaign.stress_replay import run_fixed_input_replay
from robotactile_benchmark.n0_fault_campaign.stress_replay_evidence import (
    summarize_replay_probes,
)


def records(length=61):
    clean, fault = [], []
    for record in make_synthetic_episode(length):
        obs = replace(
            record.observation,
            task="lift_bottle",
            proprio=np.array([0, 0, 0, 1, 0, 0, 0, 0.02], dtype=np.float32),
        )
        clean.append(build_evaluation_record(obs, record.provenance))
        changed = replace(
            obs,
            tactile=tuple(
                replace(s, payload=np.zeros_like(s.payload)) for s in obs.tactile
            ),
        )
        # Replay engine consumes observations; operator validation is a separate gate.
        fault.append(SimpleNamespace(observation=changed))
    return clean, fault


def server_type():
    class Server:
        def __init__(self):
            self.calls = []
            self.closed = False
            self.frame_st_id = 0
            self.job_config = SimpleNamespace(
                local_tactile_mode="current", tactile_global_zero=False
            )

        def _build_tactile_tensor(self, obs):
            frames = (
                obs["tactile"] if isinstance(obs["tactile"], list) else [obs["tactile"]]
            )
            return np.stack(
                [np.stack(list(frame.values())) for frame in frames]
            ).astype(np.float32)

        def _encode_tactile_obs(self, obs):
            tensor = self._build_tactile_tensor(obs)
            self.mean = float(tensor.mean())
            return {
                "tactile_global_latent": np.array([self.mean], np.float32),
                "tactile_local_latent": np.array([self.mean * 2], np.float32),
            }

        def infer(self, obs):
            self.calls.append(copy.deepcopy(obs))
            if obs.get("reset"):
                self.frame_st_id = 0
                self.mean = 0
                return {}
            if obs.get("compute_kv_cache"):
                self._encode_tactile_obs(obs)
                self.frame_st_id += 2
                return {}
            if self.frame_st_id == 0:
                self._encode_tactile_obs(obs)
            action = np.zeros((20, 2, 12), np.float32)
            action[0] = self.mean / 1000
            action[3] = 1
            action[7] = 1
            action[9] = 0.02
            return {"action": action}

        def close(self):
            self.closed = True

    return Server


def run(factory, *, clean=None, fault=None, steps=(12, 36, 60)):
    if clean is None:
        clean, fault = records()
    return run_fixed_input_replay(
        clean_records=clean,
        fault_records={"fullframe-s5": fault},
        rpc_factory=factory,
        seed=5,
        source_sha256="a" * 64,
        protocol_sha256="b" * 64,
        target_steps=steps,
    )


def test_independent_reset_full_history_and_no_fault_action_feedback():
    instances = []

    def factory():
        server = server_type()()
        instances.append(server)
        return server

    report = run(factory)
    assert len(instances) == 3 and all(s.closed for s in instances)
    for server in instances:
        assert server.calls[0]["reset"] and server.calls[0]["seed"] == 5
        assert len(server.calls) == 8  # reset, four infers, three commits
        assert [
            len(c["tactile"]) for c in server.calls if c.get("compute_kv_cache")
        ] == [4, 8, 8]
    for branch in instances[1:]:
        for reference, actual in zip(instances[0].calls, branch.calls):
            if actual.get("compute_kv_cache"):
                assert np.array_equal(actual["state"], reference["state"])
                assert actual["current_state"] == reference["current_state"]
    assert all(
        c["clean_repeat"]["translation_max_mm"] == 0 for c in report["comparisons"]
    )
    assert all(
        c["clean_fault"]["translation_max_mm"] > 0 for c in report["comparisons"]
    )
    assert report["actions_executed"] is report["success_rate_claimed"] is False


def test_zero_noise_effect_is_kept_without_rejection():
    clean, _ = records()
    report = run(server_type(), clean=clean, fault=clean)
    assert all(
        c["clean_fault"]["translation_max_mm"] == 0 for c in report["comparisons"]
    )


@pytest.mark.parametrize("field", ["vision", "proprio", "step"])
def test_input_isolation_or_missing_prefix_fails_before_model(field):
    clean, fault = records()
    obs = fault[6].observation
    if field == "vision":
        obs = replace(obs, vision={k: np.zeros_like(v) for k, v in obs.vision.items()})
    elif field == "proprio":
        obs = replace(obs, proprio=obs.proprio + 0.1)
    else:
        obs = replace(obs, step_index=99)
    fault[6] = SimpleNamespace(observation=obs)
    with pytest.raises(ValueError, match="RGB/proprio"):
        run(lambda: pytest.fail("no model should start"), clean=clean, fault=fault)


@pytest.mark.parametrize("steps", [(0,), (13,), (36, 12), (12, 12), (84,)])
def test_targets_are_bounded_and_predeclared(steps):
    with pytest.raises(ValueError, match="targets"):
        run(server_type(), steps=steps)


def test_failure_closes_session_without_retry():
    server = server_type()()
    original = server.infer

    def infer(obs):
        if not obs.get("reset"):
            raise RuntimeError("diagnostic inference failed")
        return original(obs)

    server.infer = infer
    with pytest.raises(RuntimeError):
        run(lambda: server)
    assert server.closed


def test_probe_arrays_join_actual_branches_and_rng_missing_stays_unverified(
    tmp_path, monkeypatch
):
    from robotactile_benchmark.integrations.n0_twam import input_probe

    monkeypatch.setattr(
        input_probe,
        "_rng_fingerprint",
        lambda: {
            "rng_after_reset_sha256": None,
            "rng_after_reset_domains": {},
            "rng_after_reset_missing": ["torch_unavailable"],
        },
    )
    server = server_type()
    install_n0_input_probe(tmp_path, 8, server_type=server, capture_arrays=True)
    run(server)
    report = summarize_replay_probes(
        tmp_path,
        source_sha256="a" * 64,
        protocol_sha256="b" * 64,
        conditions=["fullframe-s5"],
    )
    assert report["complete_encode_coverage"]
    assert not report["rng_after_reset_equal_and_complete"]
    assert all(d["identical"] for d in report["differences"] if d["branch"] == "repeat")
    assert any(
        d["relative_l2"] > 0 for d in report["differences"] if d["branch"] == "fault"
    )
    # Even shape-consistent tampering is rejected by the persisted file hash.
    trace = next(tmp_path.glob("*/*.json"))
    item = json.loads(trace.read_text())
    path = (
        trace.parent / item["tactile_tensor_outputs"][0]["array_file"]["relative_path"]
    )
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize_replay_probes(
            tmp_path,
            source_sha256="a" * 64,
            protocol_sha256="b" * 64,
            conditions=["fullframe-s5"],
        )
