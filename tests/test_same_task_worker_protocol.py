"""Unit contracts for same_task_worker_v1 framed IPC."""

from __future__ import annotations

import io
import struct
from dataclasses import FrozenInstanceError

import pytest

from robotactile_benchmark.execution.same_task_worker_protocol import (
    MAX_FRAME_BYTES,
    SAME_TASK_WORKER_CONTRACT,
    SAME_TASK_WORKER_SEMANTIC_VERSION,
    SameTaskWorkerReadyIdentity,
    SameTaskWorkerRequest,
    SameTaskWorkerResponse,
    compute_request_id,
    read_frame,
    write_frame,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class PartialReader(io.BytesIO):
    """Return at most two bytes to exercise exact-read framing."""

    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 2) if size >= 0 else 2)


def _request(**overrides: object) -> SameTaskWorkerRequest:
    identity: dict[str, object] = {
        "campaign_manifest_sha256": SHA_A,
        "task_id": "insert_hole",
        "ordinal": 4,
        "attempt_id": "20260827T084444.995916Z",
        "request_file_sha256": SHA_B,
    }
    identity.update({key: value for key, value in overrides.items() if key in identity})
    request_id = compute_request_id(**identity)  # type: ignore[arg-type]
    values: dict[str, object] = {
        "worker_contract": SAME_TASK_WORKER_CONTRACT,
        "semantic_version": SAME_TASK_WORKER_SEMANTIC_VERSION,
        "request_id": request_id,
        **identity,
        "trial_manifest_sha256": SHA_C,
    }
    values.update(overrides)
    return SameTaskWorkerRequest(**values)  # type: ignore[arg-type]


def test_frame_round_trip_and_partial_reads() -> None:
    request = _request()
    stream = io.BytesIO()

    write_frame(stream, request.to_document())
    document = read_frame(PartialReader(stream.getvalue()))

    assert SameTaskWorkerRequest.from_document(document) == request


@pytest.mark.parametrize(
    "wire",
    [
        b"",
        b"\x00\x00",
        struct.pack(">I", 8) + b"{}",
    ],
)
def test_frame_rejects_clean_or_partial_eof(wire: bytes) -> None:
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(wire))


def test_frame_rejects_zero_oversize_and_noncanonical_payload() -> None:
    for size in (0, MAX_FRAME_BYTES + 1):
        with pytest.raises(ValueError, match="payload size"):
            read_frame(io.BytesIO(struct.pack(">I", size)))

    payload = b'{"task_id": "insert_hole"}'
    with pytest.raises(ValueError, match="canonical"):
        read_frame(io.BytesIO(struct.pack(">I", len(payload)) + payload))


def test_frame_rejects_duplicate_keys_and_non_object() -> None:
    duplicate = b'{"a":1,"a":2}'
    with pytest.raises(ValueError, match="duplicate"):
        read_frame(io.BytesIO(struct.pack(">I", len(duplicate)) + duplicate))

    sequence = b"[]"
    with pytest.raises(TypeError, match="object"):
        read_frame(io.BytesIO(struct.pack(">I", len(sequence)) + sequence))


def test_request_rejects_identity_hash_and_field_mismatches() -> None:
    with pytest.raises(ValueError, match="request_id"):
        _request(request_id=SHA_D)
    with pytest.raises(ValueError, match="task_id"):
        _request(task_id="bad task")
    with pytest.raises(ValueError, match="ordinal"):
        _request(ordinal=-1)

    document = _request().to_document()
    document["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        SameTaskWorkerRequest.from_document(document)


def test_ready_identity_is_frozen_and_strict() -> None:
    ready = SameTaskWorkerReadyIdentity(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        session_id="session-1",
        task_id="lift_can",
        process_id=123,
        campaign_manifest_sha256=SHA_A,
        source_binding_sha256=SHA_B,
        integration_config_sha256=SHA_C,
        action_execution_contract="robotactile_n0_training_60hz_ee_v1",
    )

    assert SameTaskWorkerReadyIdentity.from_document(ready.to_document()) == ready
    with pytest.raises(FrozenInstanceError):
        ready.task_id = "insert_hole"  # type: ignore[misc]


def test_response_requires_terminal_evidence_and_matches_request() -> None:
    request = _request()
    response = SameTaskWorkerResponse(
        worker_contract=SAME_TASK_WORKER_CONTRACT,
        semantic_version=SAME_TASK_WORKER_SEMANTIC_VERSION,
        request_id=request.request_id,
        task_id=request.task_id,
        ordinal=request.ordinal,
        attempt_id=request.attempt_id,
        status="completed",
        return_code=0,
        artifact_root_sha256=SHA_D,
        exception_code=None,
    )

    response.validate_against(request)
    assert SameTaskWorkerResponse.from_document(response.to_document()) == response
    mismatched = SameTaskWorkerResponse(
        **{**response.to_document(), "ordinal": request.ordinal + 1}  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        mismatched.validate_against(request)
    with pytest.raises(ValueError, match="artifact digest"):
        SameTaskWorkerResponse(
            **{**response.to_document(), "artifact_root_sha256": None}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exception_code"):
        SameTaskWorkerResponse(
            **{
                **response.to_document(),
                "status": "failed",
                "return_code": 2,
                "artifact_root_sha256": None,
            }  # type: ignore[arg-type]
        )
