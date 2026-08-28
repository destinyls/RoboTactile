"""Strict canonical persistence for clean campaign contracts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, cast

from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignManifest,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)

if TYPE_CHECKING:
    from robotactile_benchmark.clean_baseline.aggregation import (
        CleanBaselineSummary,
    )

CLEAN_MAX_JSON_BYTES = 128 * 1024 * 1024


def read_canonical_json_file(
    path: Path, name: str
) -> tuple[Mapping[str, object], bytes]:
    """Read one stable canonical non-symlink JSON object."""

    target = Path(path).absolute()
    if target.is_symlink() or not target.is_file():
        raise CleanCampaignError(f"{name} must be a regular non-symlink file")
    try:
        before = target.stat()
        if before.st_size < 1 or before.st_size > CLEAN_MAX_JSON_BYTES:
            raise CleanCampaignError(f"{name} size is outside bounds")
        raw = target.read_bytes()
        after = target.stat()
    except OSError as error:
        raise CleanCampaignError(f"{name} cannot be read") from error
    if (
        len(raw) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise CleanCampaignError(f"{name} changed while being read")
    try:
        value = strict_json_bytes(raw, name)
    except (TypeError, ValueError) as error:
        raise CleanCampaignError(f"{name} is not strict canonical JSON") from error
    if not isinstance(value, Mapping):
        raise CleanCampaignError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], value), raw


def write_canonical_no_clobber(path: Path, document: object) -> bool:
    """Publish canonical JSON once; identical bytes are an idempotent no-op."""

    target = Path(path).absolute()
    payload = canonical_json_bytes(document)
    _reject_existing_symlink_components(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_components(target.parent)
    if target.is_symlink():
        raise CleanCampaignError("clean campaign output cannot be a symlink")
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


def _verify_existing(path: Path, payload: bytes) -> bool:
    if path.is_symlink() or not path.is_file():
        raise CleanCampaignError(
            "clean campaign output exists but is not a regular file"
        )
    try:
        existing = path.read_bytes()
    except OSError as error:
        raise CleanCampaignError("clean campaign output cannot be read") from error
    if existing != payload:
        raise FileExistsError("refusing to replace a different clean campaign output")
    return False


def _reject_existing_symlink_components(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists() and candidate.is_symlink():
            raise CleanCampaignError(
                "clean campaign output path cannot traverse a symlink"
            )


def write_clean_campaign_manifest(path: Path, manifest: CleanCampaignManifest) -> bool:
    if type(manifest) is not CleanCampaignManifest:
        raise TypeError("manifest must be an exact CleanCampaignManifest")
    return write_canonical_no_clobber(path, manifest.to_dict())


def load_clean_campaign_manifest(path: Path) -> CleanCampaignManifest:
    document, _ = read_canonical_json_file(path, "clean campaign manifest")
    return CleanCampaignManifest.from_dict(document)


def write_clean_baseline_summary(path: Path, summary: CleanBaselineSummary) -> bool:
    from robotactile_benchmark.clean_baseline.aggregation import CleanBaselineSummary

    if type(summary) is not CleanBaselineSummary:
        raise TypeError("summary must be an exact CleanBaselineSummary")
    return write_canonical_no_clobber(path, summary.to_dict())


def load_clean_baseline_summary(path: Path) -> CleanBaselineSummary:
    from robotactile_benchmark.clean_baseline.aggregation import CleanBaselineSummary

    document, _ = read_canonical_json_file(path, "clean baseline summary")
    return CleanBaselineSummary.from_dict(document)


__all__ = [
    "CLEAN_MAX_JSON_BYTES",
    "load_clean_baseline_summary",
    "load_clean_campaign_manifest",
    "read_canonical_json_file",
    "write_canonical_no_clobber",
    "write_clean_baseline_summary",
    "write_clean_campaign_manifest",
]
