"""Small dependency-local JSON identity helpers for Dream-Tac training."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def receipt_sha256(payload: Mapping[str, object], *, field: str) -> str:
    unsigned = dict(payload)
    claimed = unsigned.pop(field, None)
    actual = canonical_json_sha256(unsigned)
    if claimed != actual:
        raise ValueError(f"receipt digest is invalid: {field}")
    return actual


def signed_payload(payload: Mapping[str, object], *, field: str) -> dict[str, object]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"digest field already exists: {field}")
    result[field] = canonical_json_sha256(result)
    return result


def _atomic_json_no_clobber(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_or_verify_json(path: Path, payload: object) -> None:
    if path.exists():
        if load_json_object(path) != payload:
            raise FileExistsError(f"existing artifact differs: {path}")
        return
    try:
        _atomic_json_no_clobber(path, payload)
    except FileExistsError:
        if load_json_object(path) != payload:
            raise FileExistsError(f"existing artifact differs: {path}") from None


__all__ = [
    "canonical_json_sha256",
    "load_json_object",
    "receipt_sha256",
    "signed_payload",
    "write_or_verify_json",
]
