"""Dependency-light ZMQ/msgpack client for the official FTP-1 worker."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from typing import Any, Final, Optional, cast

import numpy as np

from robotactile_benchmark.contracts import Array, freeze_array
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    ACTION_DIM,
    ACTION_HORIZON,
    CHUNK_FIRST_N,
    CHUNK_INDEX_OFFSET,
    COLOR_CONTRACT,
    TASK_RELEASES,
    TEMPORAL_ENSEMBLE_K,
)

WIRE_PROTOCOL_VERSION: Final = "robotactile-ftp1-zmq-v2"
POLICY_RANDOMNESS_CONTRACT: Final = "episode_exogenous_seed_v1"
OUTPUT_ACTION_DIM: Final = 8
MAX_POLICY_SEED: Final = (1 << 63) - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ARRAY_FIELDS = frozenset({"__ndarray__", "data", "dtype", "shape"})
_METADATA_FIELDS = frozenset(
    {
        "action_horizon",
        "action_rep",
        "camera_keys",
        "checkpoint_sha256",
        "chunk_first_n",
        "chunk_index_offset",
        "color_contract",
        "model",
        "model_action_dim",
        "output_action_dim",
        "prompt",
        "protocol_version",
        "randomness_contract",
        "serve_bundle_sha256",
        "source_commit",
        "task_id",
        "temporal_ensemble_k",
        "use_tactile",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {"left", "prompt", "qpos8", "right", "task", "top", "wrist"}
)


class FTP1PolicyTransportError(RuntimeError):
    """The FTP-1 worker is unavailable or violates its wire contract."""


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    selected = _strict_string(value, name)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return selected


def encode_wire_value(value: object) -> object:
    """Encode arrays as zero-copy-friendly msgpack binary descriptors."""

    if isinstance(value, np.ndarray):
        if value.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
            raise TypeError("FTP-1 wire arrays must use uint8 or float32")
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "data": array.tobytes(order="C"),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("FTP-1 wire mapping keys must be strings")
        return {str(key): encode_wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_wire_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, np.generic):
        return encode_wire_value(value.item())
    raise TypeError(f"unsupported FTP-1 wire value: {type(value).__name__}")


def decode_wire_value(value: object) -> object:
    """Decode and validate msgpack ndarray descriptors recursively."""

    if isinstance(value, Mapping):
        if value.get("__ndarray__") is True:
            if set(value) != _ARRAY_FIELDS:
                raise ValueError("FTP-1 ndarray descriptor fields mismatch")
            dtype_name = value["dtype"]
            if dtype_name not in ("uint8", "float32"):
                raise ValueError("FTP-1 ndarray descriptor dtype is unsupported")
            shape = value["shape"]
            if not isinstance(shape, list) or any(
                type(item) is not int or item < 0 for item in shape
            ):
                raise ValueError("FTP-1 ndarray descriptor shape is invalid")
            data = value["data"]
            if not isinstance(data, bytes):
                raise ValueError("FTP-1 ndarray descriptor data must be bytes")
            dtype: np.dtype[Any] = np.dtype(cast(str, dtype_name))
            expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(data) != expected_size:
                raise ValueError("FTP-1 ndarray descriptor byte length mismatch")
            array = np.frombuffer(data, dtype=dtype).reshape(tuple(shape)).copy()
            return freeze_array(array, dtype)
        if "__ndarray__" in value:
            raise ValueError("FTP-1 ndarray marker must be true")
        if any(not isinstance(key, str) for key in value):
            raise ValueError("FTP-1 wire response mapping keys must be strings")
        return {str(key): decode_wire_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_wire_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise ValueError(f"unsupported FTP-1 response value: {type(value).__name__}")


class OfficialFTP1PolicyClient:
    """Synchronous, content-bound client for one task-specific FTP-1 worker."""

    def __init__(
        self,
        endpoint: str,
        *,
        expected_metadata: Optional[Mapping[str, object]] = None,
        timeout_ms: int = 120_000,
        retrained_task: Optional[tuple[str, str, bool]] = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise ValueError("FTP-1 endpoint must be a tcp:// address")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
            raise TypeError("timeout_ms must be an integer")
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        if retrained_task is not None:
            if expected_metadata is None:
                raise ValueError("retrained FTP-1 requires frozen expected metadata")
            from robotactile_benchmark.backends.univtac_contracts import (
                build_univtac_backend_config,
            )

            build_univtac_backend_config(retrained_task[0])
            _strict_string(retrained_task[1], "retrained prompt")
            if type(retrained_task[2]) is not bool:
                raise ValueError("retrained wrist route must be boolean")
        self._retrained_task = retrained_task
        try:
            self._zmq: Any = importlib.import_module("zmq")
            self._msgpack: Any = importlib.import_module("msgpack")
        except ImportError as error:
            raise FTP1PolicyTransportError(
                "install RoboTactile with the ftp1-policy transport dependencies"
            ) from error
        if expected_metadata is not None:
            self._validate_metadata_shape(expected_metadata, retrained_task)
            self._expected_metadata: Optional[dict[str, object]] = dict(
                expected_metadata
            )
        else:
            self._expected_metadata = None
        self._context: Any = self._zmq.Context()
        self._socket: Any = self._context.socket(self._zmq.REQ)
        self._socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._reset = False
        self._closed = False

    def _round_trip(
        self, request: Mapping[str, object], *, response_fields: frozenset[str]
    ) -> Mapping[str, object]:
        if self._closed:
            raise FTP1PolicyTransportError("closed FTP-1 client cannot send")
        try:
            payload = self._msgpack.packb(encode_wire_value(request), use_bin_type=True)
            self._socket.send(payload)
            raw = self._msgpack.unpackb(
                self._socket.recv(), raw=False, strict_map_key=False
            )
            decoded = decode_wire_value(raw)
        except Exception as error:
            if isinstance(error, FTP1PolicyTransportError):
                raise
            raise FTP1PolicyTransportError("FTP-1 ZMQ request failed") from error
        if not isinstance(decoded, Mapping) or set(decoded) != response_fields:
            raise FTP1PolicyTransportError("FTP-1 response fields mismatch")
        if decoded.get("status") != "ok":
            raise FTP1PolicyTransportError("FTP-1 worker rejected request")
        return decoded

    def _validate_metadata(self, value: object) -> Mapping[str, object]:
        self._validate_metadata_shape(value, self._retrained_task)
        selected = cast(Mapping[str, object], value)
        if self._expected_metadata is None:
            self._expected_metadata = dict(selected)
        if dict(selected) != self._expected_metadata:
            raise FTP1PolicyTransportError("FTP-1 metadata identity mismatch")
        return selected

    @staticmethod
    def _validate_metadata_shape(
        value: object, retrained_task: Optional[tuple[str, str, bool]] = None
    ) -> None:
        if not isinstance(value, Mapping) or set(value) != _METADATA_FIELDS:
            raise FTP1PolicyTransportError("FTP-1 metadata fields mismatch")
        fixed = {
            "action_horizon": ACTION_HORIZON,
            "action_rep": "absolute",
            "chunk_first_n": CHUNK_FIRST_N,
            "chunk_index_offset": CHUNK_INDEX_OFFSET,
            "color_contract": COLOR_CONTRACT,
            "model": "ftp1_policy",
            "model_action_dim": ACTION_DIM,
            "output_action_dim": OUTPUT_ACTION_DIM,
            "protocol_version": WIRE_PROTOCOL_VERSION,
            "randomness_contract": POLICY_RANDOMNESS_CONTRACT,
            "temporal_ensemble_k": TEMPORAL_ENSEMBLE_K,
            "use_tactile": True,
        }
        if any(value.get(name) != expected for name, expected in fixed.items()):
            raise FTP1PolicyTransportError("FTP-1 metadata contract mismatch")
        task_id = value.get("task_id")
        if retrained_task is None:
            if task_id not in TASK_RELEASES:
                raise FTP1PolicyTransportError("FTP-1 metadata task is unsupported")
            release = TASK_RELEASES[cast(str, task_id)]
            expected_prompt, use_wrist = release.prompt, release.camera_route == "all"
        else:
            expected_task, expected_prompt, use_wrist = retrained_task
            if task_id != expected_task:
                raise FTP1PolicyTransportError("FTP-1 retrained task identity mismatch")
        if value.get("prompt") != expected_prompt:
            raise FTP1PolicyTransportError("FTP-1 metadata prompt mismatch")
        expected_cameras = ["camera_ego_rgb_0"]
        if use_wrist:
            expected_cameras.append("right_wrist_camera_rgb_0")
        if value.get("camera_keys") != expected_cameras:
            raise FTP1PolicyTransportError("FTP-1 metadata camera route mismatch")
        if _COMMIT.fullmatch(str(value.get("source_commit"))) is None:
            raise FTP1PolicyTransportError("FTP-1 metadata source commit is invalid")
        for name in ("checkpoint_sha256", "serve_bundle_sha256"):
            if _SHA256.fullmatch(str(value.get(name))) is None:
                raise FTP1PolicyTransportError(f"FTP-1 metadata {name} is invalid")

    def health(self) -> Mapping[str, object]:
        response = self._round_trip(
            {"cmd": "health"}, response_fields=frozenset({"status", "metadata"})
        )
        return self._validate_metadata(response["metadata"])

    def reset(self, *, task_id: str, prompt: str, seed: int) -> Mapping[str, object]:
        if self._retrained_task is not None:
            valid_task = (task_id, prompt) == self._retrained_task[:2]
        else:
            valid_task = (
                task_id in TASK_RELEASES and prompt == TASK_RELEASES[task_id].prompt
            )
        if not valid_task:
            raise ValueError("FTP-1 reset task/prompt identity mismatch")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= MAX_POLICY_SEED
        ):
            raise ValueError("FTP-1 seed must be in [0, 2**63 - 1]")
        if self._expected_metadata is not None and (
            task_id != self._expected_metadata["task_id"]
            or prompt != self._expected_metadata["prompt"]
        ):
            raise ValueError("FTP-1 reset disagrees with expected worker identity")
        response = self._round_trip(
            {
                "cmd": "reset",
                "prompt": prompt,
                "seed": seed,
                "task_id": task_id,
            },
            response_fields=frozenset({"status", "metadata", "seed"}),
        )
        if type(response["seed"]) is not int or response["seed"] != seed:
            raise FTP1PolicyTransportError("FTP-1 reset seed acknowledgement mismatch")
        metadata = self._validate_metadata(response["metadata"])
        self._reset = True
        return metadata

    def infer(self, observation: Mapping[str, object]) -> Array:
        if not self._reset:
            raise FTP1PolicyTransportError("FTP-1 predict requires reset")
        if not isinstance(observation, Mapping):
            raise TypeError("FTP-1 observation must be a mapping")
        keys = set(observation)
        if keys not in (
            _OBSERVATION_FIELDS,
            _OBSERVATION_FIELDS - {"wrist"},
        ):
            raise ValueError("FTP-1 observation fields mismatch")
        self._validate_observation(observation)
        expected_metadata = self._expected_metadata
        if expected_metadata is None:
            raise FTP1PolicyTransportError("FTP-1 observation requires worker identity")
        if (
            observation["task"] != expected_metadata["task_id"]
            or observation["prompt"] != expected_metadata["prompt"]
        ):
            raise ValueError("FTP-1 observation task/prompt identity mismatch")
        request = {
            "cmd": "predict",
            "prompt": observation["prompt"],
            "task_id": observation["task"],
            **{
                name: item
                for name, item in observation.items()
                if name not in {"task", "prompt"}
            },
        }
        response = self._round_trip(
            request,
            response_fields=frozenset({"status", "action", "metadata"}),
        )
        self._validate_metadata(response["metadata"])
        action = response["action"]
        if not isinstance(action, np.ndarray):
            raise FTP1PolicyTransportError("FTP-1 action must be an ndarray")
        if action.dtype != np.float32 or action.shape not in ((8,), (1, 8)):
            raise FTP1PolicyTransportError(
                "FTP-1 action must be float32 shape (8,) or (1, 8)"
            )
        normalized = action.reshape(8)
        if not np.isfinite(normalized).all():
            raise FTP1PolicyTransportError("FTP-1 action must be finite")
        return freeze_array(normalized, np.float32)

    def _validate_observation(self, value: Mapping[str, object]) -> None:
        expected_images = ("top", "left", "right")
        for name in expected_images:
            image = value[name]
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
            ):
                raise ValueError(f"FTP-1 {name} must be uint8 HWC with 3 channels")
        wrist = value.get("wrist")
        if wrist is not None and (
            not isinstance(wrist, np.ndarray)
            or wrist.dtype != np.uint8
            or wrist.ndim != 3
            or wrist.shape[-1] != 3
        ):
            raise ValueError("FTP-1 wrist must be uint8 HWC with 3 channels")
        if self._expected_metadata is None:
            raise FTP1PolicyTransportError("FTP-1 observation requires worker identity")
        camera_keys = self._expected_metadata["camera_keys"]
        if ("wrist" in value) != (
            camera_keys == ["camera_ego_rgb_0", "right_wrist_camera_rgb_0"]
        ):
            raise ValueError("FTP-1 wrist presence disagrees with camera route")
        qpos8 = value["qpos8"]
        if (
            not isinstance(qpos8, np.ndarray)
            or qpos8.dtype != np.float32
            or qpos8.shape != (8,)
            or not np.isfinite(qpos8).all()
        ):
            raise ValueError("FTP-1 qpos8 must be finite float32 shape (8,)")

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close(linger=0)
        self._context.term()
        self._closed = True
        self._reset = False


__all__ = [
    "FTP1PolicyTransportError",
    "MAX_POLICY_SEED",
    "POLICY_RANDOMNESS_CONTRACT",
    "OfficialFTP1PolicyClient",
    "OUTPUT_ACTION_DIM",
    "WIRE_PROTOCOL_VERSION",
    "decode_wire_value",
    "encode_wire_value",
]
