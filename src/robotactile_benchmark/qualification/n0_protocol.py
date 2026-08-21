"""Cross-package CPU fake qualification for the frozen N0 two-phase wire."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, Tuple, cast

import numpy as np

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.policies.n0 import N0Policy
from robotactile_benchmark.qualification.absence import (
    EffectProbe,
    run_structural_absence_checks,
)
from robotactile_benchmark.transport.n0_client import N0Client, N0ClientState
from robotactile_benchmark.transport.n0_contracts import N0GroundingFrame

_IMAGE_SHAPE = (3, 4, 3)
_SOURCE_COMMIT = "9036c130409f8cf5494b12489fea339f7213b9d6"


class _FakeEngine:
    def __init__(self) -> None:
        self.cache_position = 0
        self.commit_count = 0
        self.last_payload: Mapping[str, object] | None = None

    def reset(self, *, prompt: str, task: str, seed: int, hard: bool) -> None:
        del prompt, task, seed, hard
        self.cache_position = 0

    def infer(self, observation: Mapping[str, object]) -> Array:
        if set(observation) != {
            "vision",
            "tactile",
            "proprio",
            "observation_digest",
        }:
            raise RuntimeError("N0 fake engine received a non-canonical observation")
        return np.zeros((8, 2, 4), dtype=np.float32)

    def commit(self, binding: object, grounding_payload: object) -> None:
        del binding
        if not isinstance(grounding_payload, Mapping):
            raise TypeError("N0 grounding payload must be a mapping")
        self.commit_count += 1
        self.last_payload = grounding_payload
        self.cache_position += 1

    def abort(self, reason_code: str) -> None:
        del reason_code
        self.cache_position = 0


class _GatewayTransport:
    def __init__(self, gateway: object, engine: _FakeEngine) -> None:
        self._gateway = gateway
        self._engine = engine
        self.operations: list[str] = []
        self.handshake_document: Mapping[str, object] | None = None
        self.prepare_engine_commit_count = -1
        self._connected = False

    def connect(self) -> Mapping[str, object]:
        handshake = cast(Any, self._gateway).acquire("policy-qualification")
        if not isinstance(handshake, Mapping):
            raise TypeError("N0 gateway handshake must be a mapping")
        self.handshake_document = handshake
        self._connected = True
        return handshake

    def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
        operation = request["op"]
        if type(operation) is not str:
            raise TypeError("N0 request operation must be a string")
        self.operations.append(operation)
        response = cast(Any, self._gateway).handle("policy-qualification", request)
        if operation == "prepare_commit":
            self.prepare_engine_commit_count = self._engine.commit_count
        if not isinstance(response, Mapping):
            raise TypeError("N0 gateway acknowledgement must be a mapping")
        return response

    def close(self) -> None:
        if self._connected:
            cast(Any, self._gateway).release("policy-qualification")
            self._connected = False


def _load_gateway_types(workspace_root: Path) -> tuple[Any, Any]:
    package_root = (workspace_root / "N0-TWAM").resolve(strict=True)
    sys.path.insert(0, str(package_root))
    try:
        protocol = importlib.import_module("n0_twam.transport.robotactile_protocol")
        gateway = importlib.import_module("n0_twam.transport.robotactile_gateway")
    finally:
        sys.path.pop(0)
    expected_protocol = package_root / "n0_twam/transport/robotactile_protocol.py"
    expected_gateway = package_root / "n0_twam/transport/robotactile_gateway.py"
    protocol_file = protocol.__file__
    gateway_file = gateway.__file__
    if (
        protocol_file is None
        or gateway_file is None
        or Path(protocol_file).resolve(strict=True) != expected_protocol
        or Path(gateway_file).resolve(strict=True) != expected_gateway
    ):
        raise RuntimeError("N0 gateway modules do not come from the reviewed source")
    return protocol.ServerIdentity, gateway.RoboTactileGateway


def _observation() -> dict[str, object]:
    unsigned: dict[str, object] = {
        "vision": {
            "top": np.full(_IMAGE_SHAPE, 10, dtype=np.uint8),
            "wrist_l": np.full(_IMAGE_SHAPE, 20, dtype=np.uint8),
        },
        "tactile": {
            "tactile_a": np.full(_IMAGE_SHAPE, 30, dtype=np.uint8),
            "tactile_b": np.full(_IMAGE_SHAPE, 40, dtype=np.uint8),
        },
        "proprio": np.zeros(8, dtype=np.float32),
    }
    return {**unsigned, "observation_digest": canonical_hash(unsigned)}


def _frames() -> tuple[N0GroundingFrame, ...]:
    return tuple(
        N0GroundingFrame(
            step_index=step,
            top=np.full(_IMAGE_SHAPE, 10 + step, dtype=np.uint8),
            wrist_l=np.full(_IMAGE_SHAPE, 20 + step, dtype=np.uint8),
            tactile_a=np.full(_IMAGE_SHAPE, 30 + step, dtype=np.uint8),
            tactile_b=np.full(_IMAGE_SHAPE, 40 + step, dtype=np.uint8),
            proprio=np.full(8, step / 100.0, dtype=np.float32),
        )
        for step in range(1, 9)
    )


def _absence_factory(identity: PolicyIdentity) -> Tuple[N0Policy, EffectProbe]:
    transport_effects = [0]

    def forbidden_client() -> NoReturn:
        transport_effects[0] += 1
        raise AssertionError("N0 client factory crossed structural preflight")

    policy = N0Policy(identity, cast(Any, forbidden_client))

    def probe() -> Tuple[int, int]:
        return 0, transport_effects[0]

    return policy, probe


def run_n0_protocol(
    identity: PolicyIdentity, workspace_root: Path, task_id: str
) -> tuple[Mapping[str, object], Mapping[str, Any], str]:
    """Use the actual benchmark client and reviewed gateway with a fake engine."""

    server_identity_type, gateway_type = _load_gateway_types(workspace_root)
    normalizer_sha = canonical_hash("cpu-fake-normalizer-not-loaded")
    bundle_sha = canonical_hash("cpu-fake-serve-bundle-not-loaded")
    prompt_sha = canonical_hash("cpu-fake-prompt-manifest")
    server_identity = server_identity_type(
        source_commit=_SOURCE_COMMIT,
        checkpoint_sha256=identity.checkpoint_sha256,
        config_sha256=identity.config_sha256,
        normalizer_sha256=normalizer_sha,
        serve_bundle_sha256=bundle_sha,
        prompt_manifest_sha256=prompt_sha,
        action_lower_bounds=(-2.0,) * 8,
        action_upper_bounds=(2.0,) * 8,
        camera_shapes=(_IMAGE_SHAPE, _IMAGE_SHAPE),
        tactile_shapes=(_IMAGE_SHAPE, _IMAGE_SHAPE),
        server_epoch="cpu-fake-policy-qualification-v1",
    )
    engine = _FakeEngine()
    transport = _GatewayTransport(gateway_type(server_identity, engine), engine)
    client = N0Client(
        transport,
        expected_source_commit=_SOURCE_COMMIT,
        expected_checkpoint_sha256=identity.checkpoint_sha256,
        expected_config_sha256=identity.config_sha256,
        expected_normalizer_sha256=normalizer_sha,
        expected_serve_bundle_sha256=bundle_sha,
        expected_prompt_manifest_sha256=prompt_sha,
    )
    client.handshake()
    client.reset("cpu-policy-qualification", "insert HDMI", task_id, 1001)
    native = client.infer(0, _observation())
    executed = np.transpose(native, (1, 2, 0)).reshape(8, 8).copy()
    frames = _frames()
    client.commit(
        step_index=0,
        native_action=native,
        executed_actions=executed,
        grounding_frames=frames,
    )
    if client.state is not N0ClientState.READY or engine.last_payload is None:
        raise RuntimeError("N0 cross-package protocol did not complete")
    if len(cast(list[object], engine.last_payload["obs"])) != 8:
        raise RuntimeError("N0 gateway did not reconstruct eight grounding frames")
    result = {
        "protocol_status": "passed",
        "request_operations": tuple(transport.operations),
        "prepare_engine_commit_count": transport.prepare_engine_commit_count,
        "final_engine_commit_count": engine.commit_count,
        "final_cache_position": engine.cache_position,
        "client_state": client.state.value,
        "handshake_sha256": canonical_hash(transport.handshake_document),
        "executed_actions_sha256": canonical_hash(executed),
        "grounding_frames_sha256": canonical_hash(
            tuple(frame.to_wire() for frame in frames)
        ),
        "grounding_frame_count": len(frames),
        "fake_gateway_executed": True,
    }
    client.close()
    absence = run_structural_absence_checks(
        identity, lambda: _absence_factory(identity)
    )
    return result, absence, "artifact_unavailable"
