"""No-clobber persistence for paired live execution receipts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution.paired_live_univtac import (
    PairedLiveUniVTACExecutionResult,
)


@dataclass(frozen=True)
class PairedExecutionReceiptWrite:
    path: Path
    file_sha256: str
    group_content_sha256: str


def write_paired_execution_receipt(
    path: Path,
    result: PairedLiveUniVTACExecutionResult,
) -> PairedExecutionReceiptWrite:
    """Atomically publish one canonical receipt without overwriting evidence."""

    if type(result) is not PairedLiveUniVTACExecutionResult:
        raise TypeError("result must be an exact PairedLiveUniVTACExecutionResult")
    output = Path(path).absolute()
    parent = output.parent
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError("paired receipt parent must be a real directory")
    parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(result.to_dict())
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file() or output.read_bytes() != raw:
            raise FileExistsError("paired execution receipt already exists")
        return PairedExecutionReceiptWrite(
            path=output,
            file_sha256=hashlib.sha256(raw).hexdigest(),
            group_content_sha256=result.group_content_sha256,
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if output.is_symlink() or output.read_bytes() != raw:
        raise RuntimeError("published paired execution receipt changed")
    return PairedExecutionReceiptWrite(
        path=output,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        group_content_sha256=result.group_content_sha256,
    )
