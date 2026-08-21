"""Canonical JSON/NPY and descriptor-safe I/O for large live traces."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.contracts import Array, array_sha256
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_MAX_BUNDLE_BYTES,
    LIVE_MAX_MEMBER_BYTES,
    LIVE_MAX_MEMBER_COUNT,
    LiveArtifactValidationError,
    require_live_sha256,
    validate_live_member_path,
)
from robotactile_benchmark.execution.live_artifacts_fs import (
    LiveBundleSnapshot,
    read_live_member,
)

_SAFE_DTYPES = frozenset({"uint8", "float32"})
_ARRAY_FIELDS = frozenset(
    {"path", "file_sha256", "array_sha256", "dtype", "shape", "size_bytes"}
)


def canonical_live_json_bytes(value: object) -> bytes:
    """Return deterministic strict JSON with one trailing newline."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise LiveArtifactValidationError("value is outside canonical JSON") from error
    return (payload + "\n").encode("utf-8")


def strict_live_json_bytes(raw: bytes, name: str) -> object:
    """Reject duplicate keys, non-finite constants, BOM/CRLF, and noncanonical JSON."""

    if not raw:
        raise LiveArtifactValidationError(f"{name} is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LiveArtifactValidationError(f"{name} is not UTF-8") from error
    if text.startswith("\ufeff") or "\r" in text:
        raise LiveArtifactValidationError(f"{name} contains BOM or CRLF")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LiveArtifactValidationError(f"{name} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise LiveArtifactValidationError(f"{name} has non-finite constant {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise LiveArtifactValidationError(f"{name} is invalid JSON") from error
    if canonical_live_json_bytes(value) != raw:
        raise LiveArtifactValidationError(f"{name} is not canonical JSON")
    return value


def live_sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode_live_npy(value: Array) -> bytes:
    """Encode one safe C-order array using the canonical NPY v1 header."""

    array = np.ascontiguousarray(value)
    _validate_array_value(array)
    stream = io.BytesIO()
    np.lib.format.write_array(  # type: ignore[no-untyped-call]
        stream, array, version=(1, 0), allow_pickle=False
    )
    raw = stream.getvalue()
    if len(raw) > LIVE_MAX_MEMBER_BYTES:
        raise LiveArtifactValidationError("encoded live array exceeds member cap")
    return raw


def _validate_array_value(array: np.ndarray[Any, Any]) -> None:
    if array.dtype.hasobject or str(array.dtype) not in _SAFE_DTYPES:
        raise LiveArtifactValidationError("live array dtype is not allowed")
    if not 1 <= array.ndim <= 4 or any(int(item) < 1 for item in array.shape):
        raise LiveArtifactValidationError("live array shape is outside bounds")
    if array.nbytes + 256 > LIVE_MAX_MEMBER_BYTES:
        raise LiveArtifactValidationError("live array exceeds preallocation cap")
    if array.dtype == np.float32 and not np.isfinite(array).all():
        raise LiveArtifactValidationError("live float array must be finite")


def estimate_unique_live_array_bytes(values: Iterable[Array]) -> tuple[int, int]:
    """Preflight unique array count/bytes without allocating the full artifact."""

    seen: set[str] = set()
    estimated = 0
    for value in values:
        array = np.ascontiguousarray(value)
        _validate_array_value(array)
        digest = array_sha256(array)
        if digest is None or digest in seen:
            continue
        seen.add(digest)
        estimated += array.nbytes + 256
        if estimated > LIVE_MAX_BUNDLE_BYTES or len(seen) > LIVE_MAX_MEMBER_COUNT:
            raise LiveArtifactValidationError("live arrays exceed preallocation limits")
    return len(seen), estimated


@dataclass
class LiveArrayWriter:
    """Write deduplicated content-addressed NPY members directly into staging."""

    root: Path
    descriptors: dict[str, dict[str, object]] = field(default_factory=dict)
    total_bytes: int = 0

    def add(self, value: Array) -> dict[str, object]:
        array = np.ascontiguousarray(value)
        digest = array_sha256(array)
        if digest is None:
            raise LiveArtifactValidationError("live array hash cannot be null")
        prior = self.descriptors.get(digest)
        if prior is not None:
            return dict(prior)
        raw = encode_live_npy(array)
        file_hash = live_sha256_bytes(raw)
        path = f"arrays/{file_hash}.npy"
        descriptor: dict[str, object] = {
            "path": path,
            "file_sha256": file_hash,
            "array_sha256": digest,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "size_bytes": len(raw),
        }
        self.total_bytes += len(raw)
        if self.total_bytes > LIVE_MAX_BUNDLE_BYTES:
            raise LiveArtifactValidationError("live array bytes exceed bundle cap")
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != raw:
                raise LiveArtifactValidationError("live array address collision")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
        self.descriptors[digest] = descriptor
        return dict(descriptor)


def load_live_array(
    value: object,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> Array:
    """Validate descriptor/header/bytes before allocating a typed array."""

    if not isinstance(value, dict) or set(value) != _ARRAY_FIELDS:
        raise LiveArtifactValidationError("live array descriptor fields mismatch")
    path = validate_live_member_path(value["path"])
    file_hash = require_live_sha256(value["file_sha256"], "array file")
    array_hash = require_live_sha256(value["array_sha256"], "array content")
    dtype = value["dtype"]
    shape_value = value["shape"]
    size = value["size_bytes"]
    if not isinstance(dtype, str) or dtype not in _SAFE_DTYPES:
        raise LiveArtifactValidationError("live array descriptor dtype is forbidden")
    if (
        not isinstance(shape_value, list)
        or not 1 <= len(shape_value) <= 4
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in shape_value
        )
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 1 <= size <= LIVE_MAX_MEMBER_BYTES
    ):
        raise LiveArtifactValidationError("live array descriptor shape/size is invalid")
    shape = tuple(shape_value)
    itemsize = np.dtype(dtype).itemsize
    if math.prod(shape) * itemsize + 128 > size:
        raise LiveArtifactValidationError(
            "live descriptor cannot fit its declared bytes"
        )
    if path != f"arrays/{file_hash}.npy":
        raise LiveArtifactValidationError("live array path is not content-addressed")
    metadata = snapshot.files.get(path)
    if metadata is None or metadata.size_bytes != size:
        raise LiveArtifactValidationError("live array descriptor size mismatch")
    raw = read_live_member(snapshot, path, LIVE_MAX_MEMBER_BYTES)
    if live_sha256_bytes(raw) != file_hash:
        raise LiveArtifactValidationError("live array file hash mismatch")
    header_shape, header_dtype, payload_offset = _read_live_npy_header(raw)
    if header_shape != shape or str(header_dtype) != dtype:
        raise LiveArtifactValidationError("live NPY header disagrees with descriptor")
    if payload_offset + math.prod(shape) * itemsize != len(raw):
        raise LiveArtifactValidationError("live NPY payload has wrong byte length")
    try:
        loaded = np.asarray(np.load(io.BytesIO(raw), allow_pickle=False))
    except (OSError, ValueError, EOFError) as error:
        raise LiveArtifactValidationError(
            "live NPY payload cannot be decoded"
        ) from error
    if not loaded.flags.c_contiguous or array_sha256(loaded) != array_hash:
        raise LiveArtifactValidationError("live array content hash mismatch")
    if loaded.dtype == np.float32 and not np.isfinite(loaded).all():
        raise LiveArtifactValidationError("loaded live array is non-finite")
    if encode_live_npy(loaded) != raw:
        raise LiveArtifactValidationError("live NPY is not canonical v1")
    referenced_paths.add(path)
    return loaded


def _read_live_npy_header(raw: bytes) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)  # type: ignore[no-untyped-call]
        if version != (1, 0):
            raise LiveArtifactValidationError("unsupported live NPY version")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)  # type: ignore[no-untyped-call]
    except (ValueError, EOFError) as error:
        raise LiveArtifactValidationError("invalid live NPY header") from error
    if fortran or dtype.hasobject or str(dtype) not in _SAFE_DTYPES:
        raise LiveArtifactValidationError("unsafe live NPY header")
    if not 1 <= len(shape) <= 4 or any(int(item) < 1 for item in shape):
        raise LiveArtifactValidationError("live NPY shape is outside bounds")
    return tuple(int(item) for item in shape), dtype, stream.tell()
