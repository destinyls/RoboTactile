"""Dependency-light client for Dream-Tac's official Franka HTTP server."""

from __future__ import annotations

import base64
import json
import socket
import struct
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Mapping
from typing import Optional, cast

import numpy as np

from robotactile_benchmark.contracts import Array, freeze_array


class DreamTacTransportError(RuntimeError):
    """The pinned upstream server is unavailable or violates its wire contract."""


def _endpoint(value: str) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise ValueError("Dream-Tac endpoint must be a non-empty URL")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Dream-Tac endpoint must be an HTTP(S) origin URL")
    return value.rstrip("/")


def _rgb_png_base64(value: object, name: str) -> str:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be exact uint8 HWC RGB")
    height, width, _ = value.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    scanlines = b"".join(b"\x00" + value[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


class OfficialDreamTacClient:
    """Synchronous client for upstream `/info` and `/infer` endpoints."""

    def __init__(self, endpoint: str, *, timeout_s: float = 120.0) -> None:
        self._endpoint = _endpoint(endpoint)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a real number")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        self._timeout_s = float(timeout_s)
        self._closed = False

    def _request(
        self, path: str, *, payload: Optional[Mapping[str, object]] = None
    ) -> Mapping[str, object]:
        if self._closed:
            raise DreamTacTransportError("closed Dream-Tac client cannot send")
        body = None
        method = "GET"
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self._endpoint}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            raise DreamTacTransportError("Dream-Tac HTTP request failed") from error
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DreamTacTransportError(
                "Dream-Tac response is not valid JSON"
            ) from error
        if not isinstance(value, Mapping):
            raise DreamTacTransportError("Dream-Tac response must be a mapping")
        return cast(Mapping[str, object], value)

    def reset(
        self, *, experiment_config: str, action_horizon: int, use_tactile: bool
    ) -> None:
        """Validate the stateless upstream service handshake at episode reset."""

        info = self._request("/info")
        expected = {
            "config": experiment_config,
            "chunk_size": action_horizon,
            "use_tactile": use_tactile,
        }
        for name, value in expected.items():
            if info.get(name) != value:
                raise DreamTacTransportError(
                    f"Dream-Tac /info {name} does not match the artifact contract"
                )
        if info.get("service_name") != "Cosmos Policy Franka API":
            raise DreamTacTransportError("Dream-Tac /info service identity mismatch")

    def infer(self, observation: Mapping[str, object]) -> Array:
        expected = {
            "cam_front",
            "cam_high",
            "instruction",
            "state",
            "tactile_left",
            "tactile_right",
            "tactile_self_attn_gate",
        }
        if set(observation) != expected:
            raise ValueError("Dream-Tac observation fields mismatch")
        state = observation["state"]
        if (
            not isinstance(state, np.ndarray)
            or state.dtype != np.float32
            or state.shape != (6,)
        ):
            raise ValueError("Dream-Tac state must be exact float32 shape (6,)")
        instruction = observation["instruction"]
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("Dream-Tac instruction must be non-empty")
        tactile_gate = observation["tactile_self_attn_gate"]
        if (
            isinstance(tactile_gate, bool)
            or not isinstance(tactile_gate, (int, float))
            or not np.isfinite(tactile_gate)
            or not 0.15 <= float(tactile_gate) <= 1.0
        ):
            raise ValueError(
                "Dream-Tac tactile_self_attn_gate must be finite in [0.15, 1.0]"
            )
        payload: dict[str, object] = {
            "images": {
                "cam_front": _rgb_png_base64(observation["cam_front"], "cam_front"),
                "cam_high": _rgb_png_base64(observation["cam_high"], "cam_high"),
                "tactile_left": _rgb_png_base64(
                    observation["tactile_left"], "tactile_left"
                ),
                "tactile_right": _rgb_png_base64(
                    observation["tactile_right"], "tactile_right"
                ),
            },
            "instruction": instruction,
            "state": state.tolist(),
            "tactile_self_attn_gate": float(tactile_gate),
        }
        response = self._request("/infer", payload=payload)
        if response.get("success") is not True:
            message = response.get("error", "unknown server error")
            raise DreamTacTransportError(
                f"Dream-Tac server rejected request: {message}"
            )
        try:
            actions = np.asarray(response["actions"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise DreamTacTransportError(
                "Dream-Tac response actions are missing or invalid"
            ) from error
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise DreamTacTransportError(
                "Dream-Tac actions must be a finite matrix with width 7"
            )
        return freeze_array(actions, np.float32)

    def close(self) -> None:
        self._closed = True


__all__ = ["DreamTacTransportError", "OfficialDreamTacClient"]
