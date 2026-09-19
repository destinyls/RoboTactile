"""Deterministic no-clobber JSON receipts for HCU probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

WriteDisposition = Literal["created", "verified_existing"]


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def signed_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    if "receipt_sha256" in result:
        raise ValueError("receipt payload already contains receipt_sha256")
    result["receipt_sha256"] = canonical_json_sha256(result)
    return result


def verify_receipt(payload: Mapping[str, object]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise ValueError("HCU probe receipt SHA256 is invalid")


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"HCU probe receipt must be a JSON object: {path}")
    return payload


def write_or_verify_receipt(
    path: Path, payload: Mapping[str, object]
) -> WriteDisposition:
    """Create one receipt atomically, or verify an identical existing receipt."""

    verify_receipt(payload)
    if path.exists():
        existing = _load_json_object(path)
        verify_receipt(existing)
        if existing != dict(payload):
            raise FileExistsError(f"refusing to overwrite different receipt: {path}")
        return "verified_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
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
    except FileExistsError:
        existing = _load_json_object(path)
        verify_receipt(existing)
        if existing != dict(payload):
            raise FileExistsError(
                f"refusing to overwrite different receipt: {path}"
            ) from None
        return "verified_existing"
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return "created"


__all__ = [
    "WriteDisposition",
    "canonical_json_sha256",
    "signed_receipt",
    "verify_receipt",
    "write_or_verify_receipt",
]
