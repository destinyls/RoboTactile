"""Deterministic NPY storage and fail-closed array descriptor loading."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robotactile_benchmark.closed_loop.artifact_contracts import (
    MAX_MEMBER_BYTES,
    ArtifactValidationError,
    require_sha256,
    validate_member_path,
)
from robotactile_benchmark.contracts import Array, array_sha256, freeze_array

_ARRAY_DESCRIPTOR_FIELDS = frozenset(
    {"path", "file_sha256", "array_sha256", "dtype", "shape"}
)
_SAFE_DTYPES = frozenset({"uint8", "float32"})
_MAX_ARRAY_ELEMENTS = 4_000_000


def file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_npy(value: Array) -> bytes:
    """Encode a C-order array with a fixed NPY v1 header."""

    array = np.ascontiguousarray(value)
    if str(array.dtype) not in _SAFE_DTYPES or array.dtype.hasobject:
        raise ArtifactValidationError(
            "array dtype is outside the bounded bundle profile"
        )
    stream = io.BytesIO()
    np.lib.format.write_array(  # type: ignore[no-untyped-call]
        stream, array, version=(1, 0), allow_pickle=False
    )
    return stream.getvalue()


@dataclass
class ArrayFileStore:
    """Deduplicating builder for content-addressed array payload members."""

    files: dict[str, bytes] = field(default_factory=dict)

    def add(self, value: Array) -> dict[str, object]:
        array = freeze_array(value)
        raw = encode_npy(array)
        file_hash = file_sha256(raw)
        path = f"arrays/{file_hash}.npy"
        if len(raw) > MAX_MEMBER_BYTES:
            raise ArtifactValidationError("encoded array exceeds the member size cap")
        prior = self.files.setdefault(path, raw)
        if prior != raw:
            raise ArtifactValidationError("array content address collision")
        digest = array_sha256(array)
        if digest is None:
            raise ArtifactValidationError("array content hash cannot be null")
        return {
            "path": path,
            "file_sha256": file_hash,
            "array_sha256": digest,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }


def _descriptor(value: object) -> tuple[str, str, str, str, tuple[int, ...]]:
    if not isinstance(value, dict) or set(value) != _ARRAY_DESCRIPTOR_FIELDS:
        raise ArtifactValidationError("array descriptor fields mismatch")
    path = validate_member_path(value["path"])
    file_hash = require_sha256(value["file_sha256"], "array file sha256")
    content_hash = require_sha256(value["array_sha256"], "array content sha256")
    dtype = value["dtype"]
    shape_value = value["shape"]
    if not isinstance(dtype, str) or dtype not in _SAFE_DTYPES:
        raise ArtifactValidationError("array descriptor dtype is not allowed")
    if (
        not isinstance(shape_value, list)
        or not 1 <= len(shape_value) <= 4
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in shape_value
        )
    ):
        raise ArtifactValidationError("array descriptor shape is invalid")
    shape = tuple(shape_value)
    if math.prod(shape) > _MAX_ARRAY_ELEMENTS:
        raise ArtifactValidationError("array descriptor exceeds the element cap")
    if path != f"arrays/{file_hash}.npy":
        raise ArtifactValidationError("array path is not its file content address")
    return path, file_hash, content_hash, dtype, shape


def load_array(
    value: object,
    files: Mapping[str, bytes],
    referenced_paths: set[str],
) -> Array:
    """Load one exact NPY payload after header, size, and hash verification."""

    path, expected_file_hash, expected_array_hash, dtype, shape = _descriptor(value)
    if path not in files:
        raise ArtifactValidationError("array descriptor references an absent member")
    referenced_paths.add(path)
    raw = files[path]
    if not raw or len(raw) > MAX_MEMBER_BYTES:
        raise ArtifactValidationError("array member size is outside bounds")
    if file_sha256(raw) != expected_file_hash:
        raise ArtifactValidationError("array file content hash mismatch")
    header_shape, fortran_order, header_dtype, payload_offset = _read_npy_header(raw)
    if header_dtype.hasobject or str(header_dtype) not in _SAFE_DTYPES:
        raise ArtifactValidationError("object or unsupported NPY dtype is forbidden")
    if fortran_order:
        raise ArtifactValidationError("Fortran-order NPY payloads are forbidden")
    if header_shape != shape or str(header_dtype) != dtype:
        raise ArtifactValidationError("NPY header disagrees with its descriptor")
    expected_size = payload_offset + math.prod(shape) * header_dtype.itemsize
    if expected_size != len(raw):
        raise ArtifactValidationError("NPY payload is truncated or has trailing bytes")
    try:
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ArtifactValidationError("NPY payload cannot be decoded safely") from error
    if not loaded.flags.c_contiguous or not np.isfinite(loaded).all():
        raise ArtifactValidationError("array must be C-contiguous and finite")
    digest = array_sha256(loaded)
    if digest != expected_array_hash:
        raise ArtifactValidationError("array content hash mismatch")
    if encode_npy(loaded) != raw:
        raise ArtifactValidationError("NPY bytes are not the canonical v1 encoding")
    return freeze_array(loaded)


def _read_npy_header(raw: bytes) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)  # type: ignore[no-untyped-call]
        if version != (1, 0):
            raise ArtifactValidationError("unsupported NPY format version")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(  # type: ignore[no-untyped-call]
            stream
        )
    except (ValueError, EOFError) as error:
        raise ArtifactValidationError("invalid NPY header") from error
    if not 1 <= len(shape) <= 4 or any(
        isinstance(item, bool) or item < 1 for item in shape
    ):
        raise ArtifactValidationError("NPY shape is outside bounds")
    if math.prod(shape) > _MAX_ARRAY_ELEMENTS:
        raise ArtifactValidationError("NPY array exceeds the element cap")
    return tuple(int(item) for item in shape), bool(fortran_order), dtype, stream.tell()
