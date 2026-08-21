"""Canonical atomic no-clobber policy qualification persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from robotactile_benchmark.qualification.receipt import (
    PolicyQualificationError,
    PolicyQualificationReceipt,
)


def policy_qualification_receipt_bytes(
    receipt: PolicyQualificationReceipt,
) -> bytes:
    return (
        json.dumps(
            receipt.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _verify_existing(path: Path, payload: bytes) -> bool:
    if path.is_symlink() or not path.is_file():
        raise PolicyQualificationError(
            "policy qualification output already exists but is not a regular file"
        )
    try:
        existing = path.read_bytes()
    except OSError as error:
        raise PolicyQualificationError(
            "policy qualification output already exists and is unreadable"
        ) from error
    if existing == payload:
        return False
    raise PolicyQualificationError(
        "policy qualification output already exists with different bytes"
    )


def write_policy_qualification_receipt(
    path: Path, receipt: PolicyQualificationReceipt
) -> bool:
    """Publish once atomically; identical content is an idempotent no-op."""

    if path.is_symlink():
        raise PolicyQualificationError(
            "policy qualification output already exists as a symlink"
        )
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = policy_qualification_receipt_bytes(receipt)
    if target.exists():
        return _verify_existing(target, payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return _verify_existing(target, payload)
        return True
    finally:
        temporary.unlink(missing_ok=True)
