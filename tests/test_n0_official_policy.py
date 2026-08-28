from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import ObservationRecord
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import (
    OfficialN0Policy,
    ee8_to_state20,
    n0_training_prompt,
)
from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    OfficialN0CommitTransform,
)


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="official-n0-test",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _record(step: int) -> ObservationRecord:
    source = make_synthetic_episode(length=40)[step].observation
    return replace(
        source,
        task="pull_out_key",
        seed=17,
        proprio=np.asarray((0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02), np.float32),
    )


def _native() -> np.ndarray:
    action = np.zeros((20, 2, 12), dtype=np.float32)
    action[0] = 0.1
    action[1] = 0.2
    action[2] = 0.3
    action[3] = 1.0
    action[7] = 1.0
    action[9] = 0.02
    return action


class FakeClient:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, int]] = []
        self.infer_calls: list[object] = []
        self.commit_counts: list[int] = []
        self.commit_payloads: list[dict[str, object]] = []
        self.last_native: np.ndarray | None = None
        self.closed = False

    def reset(self, *, prompt: str, seed: int) -> None:
        self.reset_calls.append((prompt, seed))

    def infer(self, observation: object) -> np.ndarray:
        self.infer_calls.append(observation)
        self.last_native = _native()
        return self.last_native

    def commit(self, **payload: object) -> None:
        self.commit_counts.append(len(payload["video_keyframes"]))  # type: ignore[arg-type]
        self.commit_payloads.append(payload)

    def discard_terminal(self) -> None:
        raise AssertionError("test executes complete running chunks")

    def close(self) -> None:
        self.closed = True


def test_official_policy_uses_cold_skip_then_full_chunk_and_grounding() -> None:
    client = FakeClient()
    policy = OfficialN0Policy(
        _identity(),
        lambda: client,
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
    )
    prompt = n0_training_prompt("pull_out_key")
    policy.reset(
        PolicyEpisodeContext(
            episode_id=_record(0).episode_id,
            task="pull_out_key",
            initial_seed=17,
            exogenous_seed=29,
            instruction=prompt,
            action_spec=EE8_ACTION_SPEC,
        )
    )

    first = policy.infer(_record(0))
    assert first.actions.shape == (12, 8)
    assert np.allclose(first.actions[:, 3], 1.0)
    policy.commit(
        PolicyExecution(
            action_plan_sha256=first.sha256,
            executed_actions=first.actions,
            delivered_observations=tuple(_record(index) for index in range(1, 13)),
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    second = policy.infer(_record(12))
    assert second.actions.shape == (24, 8)
    policy.commit(
        PolicyExecution(
            action_plan_sha256=second.sha256,
            executed_actions=second.actions,
            delivered_observations=tuple(_record(index) for index in range(13, 37)),
            terminal_signal=BackendSignal.RUNNING,
        )
    )

    assert client.reset_calls == [(prompt, 29)]
    assert client.commit_counts == [4, 8]
    policy.close()
    assert client.closed is True


def test_grasp_classify_executes_raw_native_action_without_transform() -> None:
    client = FakeClient()
    policy = OfficialN0Policy(
        _identity(),
        lambda: client,
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
    )
    proprio = np.array(_record(0).proprio, dtype=np.float32, copy=True)
    proprio[7] = np.float32(0.0054)
    record = replace(_record(0), task="grasp_classify", proprio=proprio)
    policy.reset(
        PolicyEpisodeContext(
            episode_id=record.episode_id,
            task=record.task,
            initial_seed=record.seed,
            exogenous_seed=29,
            instruction=n0_training_prompt(record.task),
            action_spec=EE8_ACTION_SPEC,
        )
    )

    plan = policy.infer(record)

    assert plan.actions.shape == (12, 8)
    assert np.all(plan.actions[:, 7] == np.float32(0.02))
    assert client.last_native is not None
    assert np.all(client.last_native[9] == np.float32(0.02))
    assert np.allclose(
        plan.actions[:, :7],
        np.asarray((0.1, 0.2, 0.3, 1, 0, 0, 0), dtype=np.float32),
    )
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(
                replace(_record(index), task="grasp_classify") for index in range(1, 13)
            ),
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    committed = client.commit_payloads[0]["native_action"]
    inferred = client.commit_payloads[0]["inferred_action"]
    assert isinstance(committed, np.ndarray)
    assert isinstance(inferred, np.ndarray)
    assert np.all(inferred[9] == np.float32(0.02))
    assert np.array_equal(committed, inferred)
    assert (
        client.commit_payloads[0]["action_transform"]
        is OfficialN0CommitTransform.IDENTITY
    )


def test_grasp_identity_passes_real_client_gate_and_commits_raw_kv_state() -> None:
    class RPC:
        def __init__(self) -> None:
            self.requests: list[Mapping[str, object]] = []

        def get_server_metadata(self) -> Mapping[str, object]:
            return {}

        def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]:
            self.requests.append(observation)
            if observation.get("reset") is True or observation.get("compute_kv_cache"):
                return {}
            return {"action": _native()}

        def close(self) -> None:
            return None

    rpc = RPC()
    policy = OfficialN0Policy(
        _identity(),
        lambda: OfficialN0Client(rpc),
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
    )
    proprio = np.array(_record(0).proprio, dtype=np.float32, copy=True)
    proprio[7] = np.float32(0.0054)
    record = replace(_record(0), task="grasp_classify", proprio=proprio)
    policy.reset(
        PolicyEpisodeContext(
            episode_id=record.episode_id,
            task=record.task,
            initial_seed=record.seed,
            exogenous_seed=29,
            instruction=n0_training_prompt(record.task),
            action_spec=EE8_ACTION_SPEC,
        )
    )

    plan = policy.infer(record)
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(
                replace(_record(index), task="grasp_classify") for index in range(1, 13)
            ),
            terminal_signal=BackendSignal.RUNNING,
        )
    )

    kv_request = rpc.requests[-1]
    committed = kv_request["state"]
    assert kv_request["compute_kv_cache"] is True
    assert isinstance(committed, np.ndarray)
    assert np.array_equal(committed, _native())


def _channel_coded_record(step: int = 0) -> ObservationRecord:
    source = _record(step)
    vision = {}
    for key, offset in (("top", 0), ("wrist_l", 10)):
        image = np.empty_like(source.vision[key])
        image[..., 0] = 1 + offset
        image[..., 1] = 2 + offset
        image[..., 2] = 3 + offset
        vision[key] = image
    result = replace(source, vision=vision)
    for slot_id, offset in (("left", 20), ("right", 30)):
        sensor = result.sensor(slot_id)
        assert sensor.payload is not None
        payload = np.empty_like(sensor.payload)
        payload[..., 0] = 1 + offset
        payload[..., 1] = 2 + offset
        payload[..., 2] = 3 + offset
        result = result.replace_sensor(replace(sensor, payload=payload))
    return result


def _inference_images(
    client: FakeClient,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = client.infer_calls[-1]
    assert isinstance(payload, dict)
    vision = payload["obs"]
    tactile = payload["tactile"]
    assert isinstance(vision, dict)
    assert isinstance(tactile, dict)
    return vision, tactile


def test_live_profile_matches_checkpoint_jpeg_domain_without_mutating_record() -> None:
    client = FakeClient()
    policy = OfficialN0Policy(
        _identity(),
        lambda: client,
        input_profile=N0_LIVE_UNIVTAC_INPUT_PROFILE,
    )
    record = _channel_coded_record()
    prompt = n0_training_prompt(record.task)
    policy.reset(
        PolicyEpisodeContext(
            episode_id=record.episode_id,
            task=record.task,
            initial_seed=record.seed,
            exogenous_seed=29,
            instruction=prompt,
            action_spec=EE8_ACTION_SPEC,
        )
    )

    plan = policy.infer(record)

    vision, tactile = _inference_images(client)
    assert np.array_equal(vision["observation.images.top"][0, 0], (3, 2, 1))
    assert np.array_equal(vision["observation.images.wrist_l"][0, 0], (13, 12, 11))
    assert np.array_equal(tactile["observation.images.tactile_a"][0, 0], (23, 22, 21))
    assert np.array_equal(tactile["observation.images.tactile_b"][0, 0], (33, 32, 31))
    assert np.array_equal(record.vision["top"][0, 0], (1, 2, 3))
    assert policy.input_profile.channel_transform == "reverse_rgb"
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(
                _channel_coded_record(index) for index in range(1, 13)
            ),
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    grounding = client.commit_payloads[0]["video_keyframes"]
    assert isinstance(grounding, tuple)
    first_grounding = grounding[0]
    assert np.array_equal(
        first_grounding["observation.images.top"][0, 0],
        (3, 2, 1),
    )


def test_recorded_profile_preserves_checkpoint_pil_channels() -> None:
    client = FakeClient()
    policy = OfficialN0Policy(
        _identity(),
        lambda: client,
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
    )
    record = _channel_coded_record()
    policy.reset(
        PolicyEpisodeContext(
            episode_id=record.episode_id,
            task=record.task,
            initial_seed=record.seed,
            exogenous_seed=29,
            instruction=n0_training_prompt(record.task),
            action_spec=EE8_ACTION_SPEC,
        )
    )

    policy.infer(record)

    vision, tactile = _inference_images(client)
    assert np.array_equal(vision["observation.images.top"][0, 0], (1, 2, 3))
    assert np.array_equal(tactile["observation.images.tactile_a"][0, 0], (21, 22, 23))
    assert policy.input_profile.channel_transform == "identity"


def test_ee8_state_encoding_matches_identity_rot6d_layout() -> None:
    state = ee8_to_state20(
        np.asarray((0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02), np.float32)
    )
    assert state.shape == (20,)
    assert np.allclose(
        state[:10],
        (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02),
    )
