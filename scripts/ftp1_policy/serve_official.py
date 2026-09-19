#!/usr/bin/env python3
"""Serve the pinned FTP-1 UniVTAC policy over a binary ZMQ contract."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

# FTP-1 inference is PyTorch based. Keeping JAX on CPU prevents it from
# competing for the policy GPU when the upstream package imports JAX helpers.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

Array: TypeAlias = npt.NDArray[Any]
FloatArray: TypeAlias = npt.NDArray[np.float32]
ImageArray: TypeAlias = npt.NDArray[np.uint8]

PROTOCOL_VERSION = "robotactile-ftp1-zmq-v2"
POLICY_RANDOMNESS_CONTRACT = "episode_exogenous_seed_v1"
PINNED_SOURCE_COMMIT = "89fa681d6c014cce28300946b7526db808e0b1c1"
MODEL_ACTION_DIM = 120
ACTION_HORIZON = 32
CHUNK_INDEX_OFFSET = 1
CHUNK_FIRST_N = 20
TEMPORAL_ENSEMBLE_K = 0.01
GRIPPER_INDEX = 44
TACTILE_KEY = "right_tactile_gripper"
TACTILE_SENSOR = "GelSightMini"
_NDARRAY_MARKER = "__ndarray__"
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_ARRAY_BYTES = 512 * 1024 * 1024
_MAX_POLICY_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class TaskContract:
    prompt: str
    camera_keys: tuple[str, ...]


TASK_CONTRACTS: Mapping[str, TaskContract] = MappingProxyType(
    {
        "insert_hole": TaskContract("insert the stick to the hole.", ("top",)),
        "insert_tube": TaskContract(
            "insert the tube to the fixed slot.", ("top", "wrist")
        ),
        "lift_can": TaskContract(
            "grasp the can and lifts it vertically without slippage.",
            ("top", "wrist"),
        ),
        "lift_bottle": TaskContract(
            "grasp the bottle and lift it vertically, keeping its final base "
            "within 5 cm of the wall.",
            ("top",),
        ),
        "pull_out_key": TaskContract("pull out the key.", ("top",)),
        "put_bottle_in_shelf": TaskContract(
            "grasp the bottle, then position it into the shelf cavity.",
            ("top",),
        ),
    }
)


class FTP1Wrapper(Protocol):
    model_config: object

    def get_state_dim(self) -> int: ...

    def get_action_dim(self) -> int: ...

    def get_action_horizon(self) -> int: ...

    def infer(
        self,
        *,
        images: Mapping[str, Array],
        state: Array,
        prompt: str,
        tactiles: Mapping[str, Array],
        tactile_function_areas: Mapping[str, list[int]],
        tactile_sensors: Mapping[str, str],
    ) -> Array: ...


def _require_sha256(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in _HEX_DIGITS for char in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _policy_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    if not 0 <= value <= _MAX_POLICY_SEED:
        raise ValueError("seed must be in [0, 2**63 - 1]")
    return value


def _seed_policy_randomness(seed: int) -> None:
    selected = _policy_seed(seed)
    random.seed(selected)
    np.random.seed(selected % (1 << 32))
    torch: Any = importlib.import_module("torch")
    manual_seed = getattr(torch, "manual_seed", None)
    cuda = getattr(torch, "cuda", None)
    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if not callable(manual_seed) or not callable(manual_seed_all):
        raise RuntimeError("FTP-1 runtime does not expose deterministic torch seeding")
    manual_seed(selected)
    manual_seed_all(selected)


def encode_ndarray(value: Array) -> dict[str, object]:
    """Encode an exact uint8/float32 ndarray without expanding it to lists."""

    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise TypeError("wire arrays must have dtype uint8 or float32")
    contiguous = np.ascontiguousarray(array)
    return {
        _NDARRAY_MARKER: True,
        "dtype": contiguous.dtype.name,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(order="C"),
    }


def decode_ndarray(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...] | None = None,
) -> Array:
    """Decode and freeze a binary ndarray descriptor with strict bounds."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a binary ndarray descriptor")
    if set(value) != {_NDARRAY_MARKER, "dtype", "shape", "data"}:
        raise ValueError(f"{name} ndarray descriptor fields are invalid")
    if value.get(_NDARRAY_MARKER) is not True:
        raise ValueError(f"{name} ndarray marker is invalid")
    expected_dtype = np.dtype(dtype)
    if value.get("dtype") != expected_dtype.name:
        raise ValueError(f"{name} must have dtype {expected_dtype.name}")
    raw_shape = value.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
        raise ValueError(f"{name} shape must be a non-empty sequence")
    dimensions: list[int] = []
    for dimension in raw_shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(f"{name} shape dimensions must be integers")
        if dimension < 1:
            raise ValueError(f"{name} shape dimensions must be positive")
        dimensions.append(dimension)
    decoded_shape = tuple(dimensions)
    if shape is not None and decoded_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    data = value.get("data")
    if not isinstance(data, bytes):
        raise TypeError(f"{name} data must be msgpack binary")
    element_count = 1
    for dimension in decoded_shape:
        element_count *= dimension
    expected_size = element_count * expected_dtype.itemsize
    if expected_size > _MAX_ARRAY_BYTES:
        raise ValueError(f"{name} binary payload exceeds the wire size limit")
    if len(data) != expected_size:
        raise ValueError(f"{name} binary payload length is invalid")
    result = np.frombuffer(data, dtype=expected_dtype).reshape(decoded_shape)
    result.setflags(write=False)
    return cast(Array, result)


def _image_from_request(request: Mapping[str, object], name: str) -> ImageArray:
    value = decode_ndarray(request.get(name), name=name, dtype=np.dtype(np.uint8))
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC with three channels")
    return cast(ImageArray, value)


def _resize_upstream_image(image: ImageArray) -> ImageArray:
    if image.shape[:2] != (224, 224):
        cv2: Any = importlib.import_module("cv2")

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    # Pinned eval_ftp1.py forwards channels unchanged despite a contradictory
    # file-header comment. This executable-code contract is parity-critical.
    return cast(ImageArray, np.ascontiguousarray(image, dtype=np.uint8))


class FTP1PolicyEngine:
    """Stateful single-episode FTP-1 policy with upstream temporal ensembling."""

    def __init__(
        self,
        wrapper: FTP1Wrapper,
        *,
        task_id: str,
        source_commit: str,
        checkpoint_sha256: str,
        serve_bundle_sha256: str,
        task_contract: TaskContract | None = None,
    ) -> None:
        try:
            contract = (
                TASK_CONTRACTS[task_id] if task_contract is None else task_contract
            )
        except KeyError as error:
            raise ValueError(f"unsupported UniVTAC task: {task_id}") from error
        if wrapper.get_state_dim() != MODEL_ACTION_DIM:
            raise ValueError("FTP-1 state dimension must be exactly 120")
        if wrapper.get_action_dim() != MODEL_ACTION_DIM:
            raise ValueError("FTP-1 action dimension must be exactly 120")
        if wrapper.get_action_horizon() != ACTION_HORIZON:
            raise ValueError("FTP-1 action horizon must be exactly 32")
        if not bool(getattr(wrapper.model_config, "use_tactile_input", False)):
            raise ValueError("FTP-1 robustness serving requires tactile input")
        self._wrapper = wrapper
        self._task_id = task_id
        self._contract = contract
        official_camera_keys = ["camera_ego_rgb_0"]
        if "wrist" in contract.camera_keys:
            official_camera_keys.append("right_wrist_camera_rgb_0")
        self._metadata: Mapping[str, object] = MappingProxyType(
            {
                "protocol_version": PROTOCOL_VERSION,
                "randomness_contract": POLICY_RANDOMNESS_CONTRACT,
                "model": "ftp1_policy",
                "source_commit": source_commit,
                "checkpoint_sha256": _require_sha256(
                    checkpoint_sha256, "checkpoint_sha256"
                ),
                "serve_bundle_sha256": _require_sha256(
                    serve_bundle_sha256, "serve_bundle_sha256"
                ),
                "task_id": task_id,
                "prompt": contract.prompt,
                "camera_keys": tuple(official_camera_keys),
                "action_rep": "absolute",
                "output_action_dim": 8,
                "model_action_dim": MODEL_ACTION_DIM,
                "action_horizon": ACTION_HORIZON,
                "chunk_index_offset": CHUNK_INDEX_OFFSET,
                "chunk_first_n": CHUNK_FIRST_N,
                "temporal_ensemble_k": TEMPORAL_ENSEMBLE_K,
                "use_tactile": True,
                "color_contract": "upstream_passthrough_v1",
            }
        )
        self.reset()

    @property
    def metadata(self) -> dict[str, object]:
        payload = dict(self._metadata)
        camera_keys = self._metadata["camera_keys"]
        if not isinstance(camera_keys, tuple):
            raise RuntimeError("FTP-1 camera metadata is not immutable")
        payload["camera_keys"] = list(camera_keys)
        return payload

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            _seed_policy_randomness(seed)
        self._history: list[tuple[FloatArray, int, FloatArray]] = []
        self._execution_step = 0

    def _validate_identity(self, request: Mapping[str, object]) -> None:
        task_id = request.get("task_id", request.get("task"))
        if task_id != self._task_id:
            raise ValueError("request task_id does not match the fixed FTP-1 server")
        prompt = request.get("prompt")
        if prompt != self._contract.prompt:
            raise ValueError("request prompt does not match the official task prompt")

    def predict(self, request: Mapping[str, object]) -> FloatArray:
        self._validate_identity(request)
        top = _image_from_request(request, "top")
        images = {"camera_ego_rgb_0": _resize_upstream_image(top)}
        if "wrist" in self._contract.camera_keys:
            wrist = _image_from_request(request, "wrist")
            images["right_wrist_camera_rgb_0"] = _resize_upstream_image(wrist)
        left = _resize_upstream_image(_image_from_request(request, "left"))
        right = _resize_upstream_image(_image_from_request(request, "right"))
        qpos8 = decode_ndarray(
            request.get("qpos8"),
            name="qpos8",
            dtype=np.dtype(np.float32),
            shape=(8,),
        )
        if not np.isfinite(qpos8).all():
            raise ValueError("qpos8 must contain only finite values")

        state = np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32)
        state[0, 9:16] = qpos8[:7]
        state[0, GRIPPER_INDEX] = qpos8[7]
        tactile = np.stack((left, right), axis=0)[None].astype(np.float32)
        raw = cast(
            FloatArray,
            np.asarray(
                self._wrapper.infer(
                    images=images,
                    state=state,
                    prompt=self._contract.prompt,
                    tactiles={TACTILE_KEY: tactile},
                    tactile_function_areas={TACTILE_KEY: [0, 1]},
                    tactile_sensors={TACTILE_KEY: TACTILE_SENSOR},
                ),
            ),
        )
        if raw.dtype != np.float32:
            raise ValueError("FTP-1 wrapper output must have exact dtype float32")
        if raw.shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
            raise ValueError("FTP-1 wrapper output must have exact shape (32, 120)")
        if not np.isfinite(raw).all():
            raise ValueError("FTP-1 wrapper output must be finite")

        execution_step = self._execution_step
        self._history.append((raw, execution_step, np.array(qpos8, copy=True)))
        valid: list[FloatArray] = []
        retained: list[tuple[FloatArray, int, FloatArray]] = []
        for chunk, inference_step, base_qpos8 in self._history:
            prediction_index = execution_step - inference_step + CHUNK_INDEX_OFFSET
            if not (
                CHUNK_INDEX_OFFSET
                <= prediction_index
                < CHUNK_INDEX_OFFSET + CHUNK_FIRST_N
            ):
                continue
            retained.append((chunk, inference_step, base_qpos8))
            model_action = chunk[prediction_index]
            absolute = np.empty(8, dtype=np.float32)
            absolute[:7] = base_qpos8[:7] + model_action[9:16]
            absolute[7] = model_action[GRIPPER_INDEX]
            valid.append(absolute)
        self._history = retained
        if not valid:
            raise RuntimeError("FTP-1 temporal ensemble has no executable action")
        candidates = np.stack(valid)
        weights = np.exp(-TEMPORAL_ENSEMBLE_K * np.arange(len(valid), dtype=np.float32))
        weights /= weights.sum()
        action = np.asarray(
            (candidates * weights[:, None]).sum(axis=0), dtype=np.float32
        )
        action.setflags(write=False)
        self._execution_step += 1
        return cast(FloatArray, action)


class FTP1RequestHandler:
    def __init__(self, engine: FTP1PolicyEngine) -> None:
        self._engine = engine
        self._episode_ready = False

    def dispatch(self, request: object) -> dict[str, object]:
        if not isinstance(request, Mapping):
            raise TypeError("FTP-1 request must be a mapping")
        command = request.get("cmd")
        if command == "health":
            return {"status": "ok", "metadata": self._engine.metadata}
        if command == "reset":
            self._engine._validate_identity(request)
            seed = _policy_seed(request.get("seed"))
            self._episode_ready = False
            self._engine.reset(seed)
            self._episode_ready = True
            return {
                "status": "ok",
                "metadata": self._engine.metadata,
                "seed": seed,
            }
        if command == "predict":
            if not self._episode_ready:
                raise RuntimeError("FTP-1 predict requires reset")
            action = self._engine.predict(request)
            return {
                "status": "ok",
                "action": encode_ndarray(action),
                "metadata": self._engine.metadata,
            }
        raise ValueError(f"unsupported FTP-1 command: {command!r}")


def _verify_source(source_root: Path, expected_commit: str) -> None:
    wrapper_path = source_root / "src/openpi/policies/ftp1_inference_wrapper.py"
    if not wrapper_path.is_file():
        raise ValueError(f"FTP-1 wrapper is absent: {wrapper_path}")
    actual = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise ValueError(
            f"FTP-1 source commit mismatch: expected {expected_commit}, got {actual}"
        )


def _load_wrapper(args: argparse.Namespace) -> FTP1Wrapper:
    _verify_source(args.source_root, args.expected_source_commit)
    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    if train_config.get("action_joint_rep") != "mix":
        raise ValueError("FTP-1 checkpoint train_config action_joint_rep must be mix")
    sys.path.insert(0, str(args.source_root / "src"))
    sys.path.insert(0, str(args.source_root))
    from openpi.policies.ftp1_inference_wrapper import (  # type: ignore[import-not-found]
        FTP1InferenceWrapper,
    )

    return cast(
        FTP1Wrapper,
        FTP1InferenceWrapper(
            checkpoint_dir=str(args.checkpoint_dir),
            domain_name=args.domain_name,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", "--endpoint", dest="bind", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--serve-bundle-sha256", required=True)
    parser.add_argument("--domain-name", required=True)
    parser.add_argument("--task", choices=tuple(TASK_CONTRACTS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--expected-source-commit", default=PINNED_SOURCE_COMMIT)
    return parser


def _serve(bind: str, handler: FTP1RequestHandler) -> None:
    if not bind.startswith("tcp://"):
        raise ValueError("FTP-1 bind endpoint must use tcp://")
    msgpack: Any = importlib.import_module("msgpack")
    zmq: Any = importlib.import_module("zmq")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind)
    logging.info("FTP-1 server ready at %s", bind)
    try:
        while True:
            try:
                request = msgpack.unpackb(
                    socket.recv(), raw=False, strict_map_key=False
                )
                response = handler.dispatch(request)
            except Exception as error:
                logging.exception("FTP-1 request failed")
                response = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            socket.send(msgpack.packb(response, use_bin_type=True))
    except KeyboardInterrupt:
        logging.info("FTP-1 server interrupted")
    finally:
        socket.close(linger=0)
        context.term()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if args.expected_source_commit != PINNED_SOURCE_COMMIT:
        raise ValueError("FTP-1 server requires the pinned reviewed source commit")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    wrapper = _load_wrapper(args)
    engine = FTP1PolicyEngine(
        wrapper,
        task_id=args.task,
        source_commit=args.expected_source_commit,
        checkpoint_sha256=args.checkpoint_sha256,
        serve_bundle_sha256=args.serve_bundle_sha256,
    )
    _serve(args.bind, FTP1RequestHandler(engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
