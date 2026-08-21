"""Atomic no-clobber persistence for UniVTAC qualification receipts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from robotactile_benchmark.backends.qualification_receipt import (
    UniVTACQualificationError,
    UniVTACQualificationReceipt,
)


def qualification_receipt_bytes(receipt: UniVTACQualificationReceipt) -> bytes:
    """Encode the only canonical on-disk representation for a receipt."""

    document = receipt.to_dict()
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _verify_existing(path: Path, payload: bytes) -> bool:
    if path.is_symlink() or not path.is_file():
        raise UniVTACQualificationError(
            "qualification output already exists but is not a regular file"
        )
    try:
        existing = path.read_bytes()
    except OSError as error:
        raise UniVTACQualificationError(
            "qualification output already exists and is unreadable"
        ) from error
    if existing == payload:
        return False
    raise UniVTACQualificationError(
        "qualification output already exists with different or invalid bytes"
    )


def write_qualification_receipt(
    path: Path, receipt: UniVTACQualificationReceipt
) -> bool:
    """Publish one complete receipt atomically without replacing any path."""

    if path.is_symlink():
        raise UniVTACQualificationError(
            "qualification output already exists as a symlink"
        )
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = qualification_receipt_bytes(receipt)
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
