"""Thin stateful client for the official N0-TWAM websocket protocol."""

from __future__ import annotations

import importlib
import socket
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, cast

import numpy as np

from robotactile_benchmark.contracts import Array, freeze_array

NATIVE_ACTION_SHAPE = (20, 2, 12)


class OfficialN0CommitTransform(str, Enum):
    """Strictly enumerated transforms allowed between inference and KV commit."""

    IDENTITY = "identity_v1"


class OfficialN0RPC(Protocol):
    """Dependency-light surface of the upstream websocket client."""

    def get_server_metadata(self) -> Mapping[str, object]: ...

    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class OfficialN0ClientState(str, Enum):
    CONNECTED = "connected"
    READY = "ready"
    AWAITING_COMMIT = "awaiting_commit"
    INDETERMINATE = "indeterminate"
    CLOSED = "closed"


def _response(value: object, operation: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"official N0 {operation} response must be a mapping")
    return cast(Mapping[str, object], value)


def _timing_only(value: object, operation: str) -> None:
    response = _response(value, operation)
    if set(response) - {"server_timing"}:
        raise RuntimeError(f"official N0 {operation} returned unexpected fields")


class OfficialN0Client:
    """One official reset/infer/grounding session with no automatic retries."""

    def __init__(self, rpc: OfficialN0RPC) -> None:
        if not isinstance(rpc.get_server_metadata(), Mapping):
            raise TypeError("official N0 server metadata must be a mapping")
        self._rpc = rpc
        self.state = OfficialN0ClientState.CONNECTED
        self._reset_prompt: Optional[str] = None
        self._pending_action: Optional[Array] = None
        self._pending_current_state: Optional[Array] = None

    def reset(self, *, prompt: str, seed: int) -> None:
        if self.state not in {
            OfficialN0ClientState.CONNECTED,
            OfficialN0ClientState.INDETERMINATE,
        }:
            raise RuntimeError("official N0 reset requires a fresh or failed session")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("official N0 prompt must be non-empty")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("official N0 seed must be a non-negative integer")
        try:
            _timing_only(
                self._rpc.infer({"reset": True, "prompt": prompt, "seed": seed}),
                "reset",
            )
        except Exception:
            self.state = OfficialN0ClientState.INDETERMINATE
            raise
        self._reset_prompt = prompt
        self._pending_action = None
        self._pending_current_state = None
        self.state = OfficialN0ClientState.READY

    def infer(self, observation: Mapping[str, object]) -> Array:
        if self.state is not OfficialN0ClientState.READY:
            raise RuntimeError("official N0 infer requires ready state")
        if observation.get("prompt") != self._reset_prompt:
            raise ValueError("official N0 infer prompt does not match reset")
        current_state = np.asarray(observation.get("current_state"), dtype=np.float32)
        if current_state.shape != (20,) or not np.isfinite(current_state).all():
            raise ValueError(
                "official N0 infer current_state must be finite shape (20,)"
            )
        try:
            response = _response(self._rpc.infer(observation), "infer")
            if set(response) - {"action", "server_timing"} or "action" not in response:
                raise RuntimeError("official N0 infer response fields mismatch")
            action = np.asarray(response["action"])
            if action.dtype != np.float32:
                action = action.astype(np.float32)
            if action.shape != NATIVE_ACTION_SHAPE or not np.isfinite(action).all():
                raise ValueError(
                    f"official N0 action must be finite float32 {NATIVE_ACTION_SHAPE}"
                )
        except Exception:
            self.state = OfficialN0ClientState.INDETERMINATE
            raise
        self._pending_action = freeze_array(action, np.float32)
        self._pending_current_state = freeze_array(current_state, np.float32)
        self.state = OfficialN0ClientState.AWAITING_COMMIT
        return self._pending_action

    def commit(
        self,
        *,
        video_keyframes: Sequence[Mapping[str, Array]],
        tactile_keyframes: Sequence[Mapping[str, Array]],
        inferred_action: Array,
        native_action: Array,
        action_transform: OfficialN0CommitTransform,
        current_state: Array,
        prompt: str,
    ) -> None:
        if (
            self.state is not OfficialN0ClientState.AWAITING_COMMIT
            or self._pending_action is None
            or self._pending_current_state is None
        ):
            raise RuntimeError("official N0 commit requires one pending inference")
        inferred = np.asarray(inferred_action)
        if (
            inferred.dtype != np.float32
            or inferred.shape != NATIVE_ACTION_SHAPE
            or not np.isfinite(inferred).all()
            or not np.array_equal(inferred, self._pending_action)
        ):
            raise ValueError(
                "official N0 inferred action does not match pending inference"
            )
        committed = np.asarray(native_action)
        if (
            committed.dtype != np.float32
            or committed.shape != NATIVE_ACTION_SHAPE
            or not np.isfinite(committed).all()
        ):
            raise ValueError(
                f"official N0 commit action must be finite float32 {NATIVE_ACTION_SHAPE}"
            )
        if prompt != self._reset_prompt:
            raise ValueError("official N0 commit prompt does not match reset")
        if action_transform is not OfficialN0CommitTransform.IDENTITY:
            raise ValueError("official N0 commit requires identity")
        if not np.array_equal(committed, inferred):
            raise ValueError("official N0 identity commit action does not match infer")
        state = np.asarray(current_state)
        if (
            state.dtype != np.float32
            or state.shape != (20,)
            or not np.array_equal(state, self._pending_current_state)
        ):
            raise ValueError("official N0 current_state must be float32 shape (20,)")
        if len(video_keyframes) not in {4, 8}:
            raise ValueError(
                "official N0 commit requires four or eight video keyframes"
            )
        if len(tactile_keyframes) != len(video_keyframes):
            raise ValueError("official N0 video/tactile keyframe counts must match")
        payload: dict[str, object] = {
            "obs": tuple(video_keyframes),
            "tactile": tuple(tactile_keyframes),
            "state": committed,
            "current_state": state.tolist(),
            "compute_kv_cache": True,
            "imagine": False,
            "prompt": prompt,
        }
        try:
            _timing_only(self._rpc.infer(payload), "commit")
        except Exception:
            self.state = OfficialN0ClientState.INDETERMINATE
            raise
        self._pending_action = None
        self._pending_current_state = None
        self.state = OfficialN0ClientState.READY

    def discard_terminal(self) -> None:
        if self.state is not OfficialN0ClientState.AWAITING_COMMIT:
            raise RuntimeError("official N0 terminal discard requires pending infer")
        self._pending_action = None
        self._pending_current_state = None
        self.state = OfficialN0ClientState.INDETERMINATE

    def close(self) -> None:
        if self.state is OfficialN0ClientState.CLOSED:
            return
        self._rpc.close()
        self._reset_prompt = None
        self._pending_action = None
        self._pending_current_state = None
        self.state = OfficialN0ClientState.CLOSED


class _UpstreamRPC:
    """Normalize the upstream client, whose close method is not public."""

    def __init__(self, client: object) -> None:
        self._client = client

    def get_server_metadata(self) -> Mapping[str, object]:
        return cast(
            Mapping[str, object],
            cast(Any, self._client).get_server_metadata(),
        )

    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]:
        return cast(
            Mapping[str, object],
            cast(Any, self._client).infer(dict(observation)),
        )

    def close(self) -> None:
        websocket = getattr(self._client, "_ws", None)
        close = getattr(websocket, "close", None)
        if callable(close):
            close()


def load_official_n0_rpc(
    *, source_root: Path, host: str, port: int, api_key: Optional[str] = None
) -> OfficialN0RPC:
    """Load only the pinned upstream numpy websocket client from its checkout."""

    root = Path(source_root).resolve(strict=True)
    utils_root = root / "n0_twam/utils"
    client_file = utils_root / "Simple_Remote_Infer/deploy/websocket_client_policy.py"
    if utils_root.is_symlink() or not client_file.is_file():
        raise ValueError("official N0 websocket client source is unavailable")
    if str(utils_root) not in sys.path:
        sys.path.append(str(utils_root))
    module = importlib.import_module(
        "Simple_Remote_Infer.deploy.websocket_client_policy"
    )
    module_file = Path(cast(str, module.__file__)).resolve(strict=True)
    if module_file != client_file.resolve(strict=True):
        raise RuntimeError(
            "official N0 websocket client imported from another checkout"
        )
    client_type = vars(module).get("WebsocketClientPolicy")
    if not callable(client_type):
        raise TypeError("official N0 WebsocketClientPolicy is unavailable")
    if not isinstance(host, str) or not host or host.strip() != host:
        raise ValueError("official N0 host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("official N0 port must be in [1,65535]")
    with socket.create_connection((host, port), timeout=10.0):
        pass
    return _UpstreamRPC(client_type(host=host, port=port, api_key=api_key))


__all__ = [
    "NATIVE_ACTION_SHAPE",
    "OfficialN0CommitTransform",
    "OfficialN0Client",
    "OfficialN0ClientState",
    "OfficialN0RPC",
    "load_official_n0_rpc",
]
