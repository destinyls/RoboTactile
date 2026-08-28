from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    OfficialN0ClientState,
    OfficialN0CommitTransform,
)

_PULL_PROMPT = "Untwist and extract a key from a lock"
_GRASP_PROMPT = "Grasp an object and classify it by tactile texture"


def _state(*, qpos_m: float = 0.0) -> np.ndarray:
    state = np.zeros(20, dtype=np.float32)
    state[9] = np.float32(qpos_m)
    return state


def _observation(prompt: str, *, qpos_m: float = 0.0) -> dict[str, object]:
    state = _state(qpos_m=qpos_m)
    return {
        "obs": {},
        "tactile": {},
        "current_state": state.tolist(),
        "prompt": prompt,
    }


class FakeRPC:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, object]] = []
        self.closed = False

    def get_server_metadata(self) -> Mapping[str, object]:
        return {}

    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]:
        self.requests.append(observation)
        if observation.get("reset") is True or observation.get("compute_kv_cache"):
            return {}
        return {"action": np.zeros((20, 2, 12), dtype=np.float32)}

    def close(self) -> None:
        self.closed = True


def test_official_client_runs_reset_infer_commit_without_retry() -> None:
    rpc = FakeRPC()
    client = OfficialN0Client(rpc)
    client.reset(prompt=_PULL_PROMPT, seed=17)
    action = client.infer(_observation(_PULL_PROMPT))
    keyframes = tuple(
        {"observation.images.top": np.zeros((2, 2, 3), np.uint8)} for _ in range(4)
    )
    tactile = tuple(
        {"observation.images.tactile_a": np.zeros((2, 2, 3), np.uint8)}
        for _ in range(4)
    )
    client.commit(
        video_keyframes=keyframes,
        tactile_keyframes=tactile,
        inferred_action=action,
        native_action=action,
        action_transform=OfficialN0CommitTransform.IDENTITY,
        current_state=_state(),
        prompt=_PULL_PROMPT,
    )

    assert client.state is OfficialN0ClientState.READY
    assert len(rpc.requests) == 3
    assert rpc.requests[-1]["compute_kv_cache"] is True
    assert np.array_equal(rpc.requests[-1]["state"], action)
    client.close()
    assert rpc.closed is True


def test_official_client_enters_indeterminate_state_after_commit_failure() -> None:
    class FailingRPC(FakeRPC):
        def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]:
            if observation.get("compute_kv_cache"):
                raise TimeoutError("lost commit acknowledgement")
            return super().infer(observation)

    client = OfficialN0Client(FailingRPC())
    client.reset(prompt="prompt", seed=1)
    action = client.infer(_observation("prompt"))
    frames = tuple({} for _ in range(4))
    with pytest.raises(TimeoutError):
        client.commit(
            video_keyframes=frames,
            tactile_keyframes=frames,
            inferred_action=action,
            native_action=action,
            action_transform=OfficialN0CommitTransform.IDENTITY,
            current_state=_state(),
            prompt="prompt",
        )
    assert client.state is OfficialN0ClientState.INDETERMINATE
    with pytest.raises(RuntimeError, match="ready"):
        client.infer(_observation("prompt"))


def test_official_client_commits_grasp_action_by_identity() -> None:
    rpc = FakeRPC()
    client = OfficialN0Client(rpc)
    prompt = _GRASP_PROMPT
    client.reset(prompt=prompt, seed=1)
    inferred = client.infer(_observation(prompt, qpos_m=0.0054))
    frames = tuple({} for _ in range(4))

    client.commit(
        video_keyframes=frames,
        tactile_keyframes=frames,
        inferred_action=inferred,
        native_action=inferred,
        action_transform=OfficialN0CommitTransform.IDENTITY,
        current_state=_state(qpos_m=0.0054),
        prompt=prompt,
    )

    assert np.array_equal(rpc.requests[-1]["state"], inferred)
    assert client.state is OfficialN0ClientState.READY


def test_official_client_rejects_any_grasp_action_change() -> None:
    rpc = FakeRPC()
    client = OfficialN0Client(rpc)
    prompt = _GRASP_PROMPT
    client.reset(prompt=prompt, seed=1)
    inferred = client.infer(_observation(prompt, qpos_m=0.0054))
    committed = np.array(inferred, copy=True)
    committed[9] = np.float32(0.0054)
    committed[0, 0, 0] = np.float32(1.0)
    frames = tuple({} for _ in range(4))

    with pytest.raises(ValueError, match="identity commit"):
        client.commit(
            video_keyframes=frames,
            tactile_keyframes=frames,
            inferred_action=inferred,
            native_action=committed,
            action_transform=OfficialN0CommitTransform.IDENTITY,
            current_state=_state(qpos_m=0.0054),
            prompt=prompt,
        )


def test_official_client_exposes_only_identity_commit_transform() -> None:
    assert tuple(OfficialN0CommitTransform) == (OfficialN0CommitTransform.IDENTITY,)
