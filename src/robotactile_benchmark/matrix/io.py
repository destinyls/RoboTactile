"""Strict, atomic, no-clobber persistence for matrix orchestration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from robotactile_benchmark.matrix.contracts import MatrixCellSpec
from robotactile_benchmark.matrix.manifest import MatrixManifest
from robotactile_benchmark.matrix.results import MatrixCellReceipt
from robotactile_benchmark.matrix.states import MatrixCellState
from robotactile_benchmark.matrix.summary import MatrixSummary

MATRIX_MANIFEST_PATH = "matrix_manifest.json"
MATRIX_SUMMARY_PATH = "matrix_summary.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_SUMMARY_BYTES = 4 * 1024 * 1024
_MAX_RECEIPT_BYTES = 128 * 1024
_T = TypeVar("_T")


class MatrixResumeError(ValueError):
    """Raised when persisted matrix state cannot be strictly and safely reused."""


def canonical_matrix_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("matrix value is outside canonical JSON") from error
    return (text + "\n").encode("utf-8")


def prepare_matrix_output(output: Path, manifest: MatrixManifest) -> None:
    """Create or strictly verify the immutable matrix request boundary."""

    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise MatrixResumeError("matrix output must be a real directory")
    output.mkdir(parents=True, exist_ok=True)
    cells = output / "cells"
    if cells.is_symlink() or (cells.exists() and not cells.is_dir()):
        raise MatrixResumeError("matrix cells path must be a real directory")
    cells.mkdir(exist_ok=True)
    path = output / MATRIX_MANIFEST_PATH
    if path.exists() or path.is_symlink():
        loaded = _load_typed(
            path, _MAX_MANIFEST_BYTES, MatrixManifest.from_dict, "matrix manifest"
        )
        if loaded != manifest or loaded.sha256 != manifest.sha256:
            raise MatrixResumeError("existing matrix manifest does not match request")
    else:
        _publish_no_clobber(path, canonical_matrix_json_bytes(manifest.to_dict()))
    _verify_matrix_output(output, manifest)


def load_matrix_manifest_file(path: Path) -> MatrixManifest:
    """Strict-load one standalone canonical matrix manifest."""

    return _load_typed(
        Path(path),
        _MAX_MANIFEST_BYTES,
        MatrixManifest.from_dict,
        "matrix manifest",
    )


def _verify_matrix_output(output: Path, manifest: MatrixManifest) -> None:
    if output.is_symlink() or not output.is_dir():
        raise MatrixResumeError("matrix output must be an existing real directory")
    cells = output / "cells"
    if cells.is_symlink() or not cells.is_dir():
        raise MatrixResumeError("matrix cells path must be an existing real directory")
    loaded = _load_typed(
        output / MATRIX_MANIFEST_PATH,
        _MAX_MANIFEST_BYTES,
        MatrixManifest.from_dict,
        "matrix manifest",
    )
    if loaded != manifest or loaded.sha256 != manifest.sha256:
        raise MatrixResumeError("existing matrix manifest does not match request")
    _reject_unknown_cell_members(cells, manifest)


def cell_receipt_path(output: Path, cell: MatrixCellSpec) -> Path:
    return Path(output) / "cells" / f"{cell.sha256}.json"


def load_cell_receipt(output: Path, cell: MatrixCellSpec) -> MatrixCellReceipt:
    path = cell_receipt_path(output, cell)
    try:
        receipt = _load_typed(
            path,
            _MAX_RECEIPT_BYTES,
            MatrixCellReceipt.from_dict,
            "verified cell receipt",
        )
        _validate_receipt_cell(receipt, cell)
        return receipt
    except MatrixResumeError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise MatrixResumeError("existing verified cell cannot be reused") from error


def write_cell_receipt(
    output: Path, cell: MatrixCellSpec, receipt: MatrixCellReceipt
) -> None:
    _validate_receipt_cell(receipt, cell)
    path = cell_receipt_path(output, cell)
    raw = canonical_matrix_json_bytes(receipt.to_dict())
    if path.exists() or path.is_symlink():
        existing = load_cell_receipt(output, cell)
        if (
            existing != receipt
            or canonical_matrix_json_bytes(existing.to_dict()) != raw
        ):
            raise MatrixResumeError("existing verified cell cannot be replaced")
        return
    _publish_no_clobber(path, raw)


def write_matrix_summary(output: Path, summary: MatrixSummary) -> None:
    """Publish a final summary atomically; an existing different byte is fatal."""

    path = Path(output) / MATRIX_SUMMARY_PATH
    raw = canonical_matrix_json_bytes(summary.to_dict())
    if path.exists() or path.is_symlink():
        existing = _load_typed(
            path, _MAX_SUMMARY_BYTES, MatrixSummary.from_dict, "matrix summary"
        )
        if (
            existing != summary
            or canonical_matrix_json_bytes(existing.to_dict()) != raw
        ):
            raise MatrixResumeError("existing matrix summary cannot be clobbered")
        return
    _publish_no_clobber(path, raw)


def load_matrix_summary(output: Path, manifest: MatrixManifest) -> MatrixSummary:
    """Strict-load the manifest, every cell receipt, and their final summary."""

    output = Path(output)
    _verify_matrix_output(output, manifest)
    summary = _load_typed(
        output / MATRIX_SUMMARY_PATH,
        _MAX_SUMMARY_BYTES,
        MatrixSummary.from_dict,
        "matrix summary",
    )
    if (
        summary.matrix_id != manifest.matrix_id
        or summary.kind is not manifest.kind
        or summary.manifest_sha256 != manifest.sha256
        or summary.pair_key != manifest.pair_key
        or len(summary.cells) != len(manifest.cells)
    ):
        raise MatrixResumeError("matrix summary does not match its manifest")
    expected_states = tuple(
        MatrixCellState.from_receipt(cell, load_cell_receipt(output, cell))
        for cell in manifest.cells
    )
    if summary.cells != expected_states:
        raise MatrixResumeError("matrix summary does not match verified cell receipts")
    return summary


def _load_typed(
    path: Path,
    maximum_bytes: int,
    factory: Callable[[object], _T],
    label: str,
) -> _T:
    try:
        raw = _read_regular_file(path, maximum_bytes)
        value = _strict_json(raw, label)
        typed = factory(value)
        if canonical_matrix_json_bytes(value) != raw:
            raise MatrixResumeError(f"{label} is not canonical JSON")
        return typed
    except MatrixResumeError:
        raise
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        raise MatrixResumeError(f"{label} failed strict validation") from error


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise MatrixResumeError(f"matrix member is not a regular file: {path.name}")
    before = path.stat()
    if not 0 < before.st_size <= maximum_bytes:
        raise MatrixResumeError(f"matrix member size is invalid: {path.name}")
    raw = path.read_bytes()
    after = path.stat()
    stable_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stable_identity or len(raw) != before.st_size:
        raise MatrixResumeError(f"matrix member changed during read: {path.name}")
    return raw


def _strict_json(raw: bytes, label: str) -> object:
    if not raw or raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise MatrixResumeError(f"{label} has noncanonical encoding")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MatrixResumeError(f"{label} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MatrixResumeError(f"{label} has non-finite constant {value}")

    try:
        return cast(
            object,
            json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=pairs_hook,
                parse_constant=reject_constant,
            ),
        )
    except json.JSONDecodeError as error:
        raise MatrixResumeError(f"{label} is invalid JSON") from error


def _publish_no_clobber(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                raise MatrixResumeError(
                    f"matrix member cannot be clobbered: {path.name}"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _validate_receipt_cell(receipt: MatrixCellReceipt, cell: MatrixCellSpec) -> None:
    if (
        receipt.cell_sha256 != cell.sha256
        or receipt.trial_manifest_sha256 != cell.trial.sha256
        or receipt.task != cell.task
        or receipt.pair_key != cell.pair_key
        or receipt.condition is not cell.trial.condition
        or receipt.operator_id != cell.operator_id
        or receipt.severity_level != cell.severity_level
    ):
        raise MatrixResumeError(
            "existing verified cell does not match its specification"
        )


def _reject_unknown_cell_members(cells: Path, manifest: MatrixManifest) -> None:
    allowed = {f"{cell.sha256}.json" for cell in manifest.cells}
    for member in cells.iterdir():
        if member.is_symlink() or not member.is_file() or member.name not in allowed:
            raise MatrixResumeError("matrix cells directory contains an unknown member")
