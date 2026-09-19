"""Dependency-light client for the official N0-VTLA ZMQ/msgpack server."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from robotactile_benchmark.contracts import Array, freeze_array


class N0VTLATransportError(RuntimeError):
    """The official server is unavailable or violates its wire contract."""


def _wire_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    return value


class OfficialN0VTLAClient:
    """Synchronous REQ client; pyzmq and msgpack remain optional dependencies."""

    def __init__(self, endpoint: str, *, timeout_ms: int = 120_000) -> None:
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise ValueError("N0-VTLA endpoint must be a tcp:// address")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
            raise TypeError("timeout_ms must be an integer")
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        try:
            self._zmq: Any = importlib.import_module("zmq")
            self._msgpack: Any = importlib.import_module("msgpack")
        except ImportError as error:
            raise N0VTLATransportError(
                "install RoboTactile with the n0-vtla extra (pyzmq, msgpack)"
            ) from error
        self._context: Any = self._zmq.Context()
        self._socket: Any = self._context.socket(self._zmq.REQ)
        self._socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._closed = False

    def _round_trip(self, request: Mapping[str, object]) -> Mapping[str, object]:
        if self._closed:
            raise N0VTLATransportError("closed N0-VTLA client cannot send")
        try:
            payload = self._msgpack.packb(_wire_value(request), use_bin_type=True)
            self._socket.send(payload)
            response = self._msgpack.unpackb(
                self._socket.recv(), raw=False, strict_map_key=False
            )
        except Exception as error:
            raise N0VTLATransportError("N0-VTLA ZMQ request failed") from error
        if not isinstance(response, Mapping):
            raise N0VTLATransportError("N0-VTLA response must be a mapping")
        if response.get("status") != "ok":
            message = response.get("message", "unknown server error")
            raise N0VTLATransportError(f"N0-VTLA server rejected request: {message}")
        return response

    def reset(self) -> None:
        self._round_trip({"cmd": "reset"})

    def infer(self, observation: Mapping[str, object]) -> Array:
        response = self._round_trip({"cmd": "predict", **dict(observation)})
        try:
            actions = np.asarray(response["actions"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise N0VTLATransportError(
                "N0-VTLA response actions are missing or invalid"
            ) from error
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise N0VTLATransportError("N0-VTLA actions must be a finite matrix")
        return freeze_array(actions, np.float32)

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close(linger=0)
        self._context.term()
        self._closed = True


__all__ = ["N0VTLATransportError", "OfficialN0VTLAClient"]
