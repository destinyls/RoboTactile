"""Strict external repository pins for RoboTactile integrations."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
    strict_json_bytes,
)

_SCHEMA_VERSION = "robotactile-integrations-lock-v2"
_LOCK_PATH = "integrations/integrations.lock.json"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_DIRECTORY = re.compile(r"^[A-Za-z0-9._-]+$")
_EXPECTED_IDS = ("act_runtime", "curobo", "isaaclab", "n0_twam", "univtac")
_ALLOWED_FIELDS = frozenset(
    {
        "artifact_schema",
        "commit_sha",
        "install_script",
        "integration_id",
        "license_spdx",
        "release_ready",
        "repository_url",
        "source_directory",
        "source_inventory_required",
    }
)


class IntegrationProvenanceError(ValueError):
    """An external source pin or checkout violates the release contract."""


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise IntegrationProvenanceError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ExternalIntegrationPin:
    """One immutable external source and license boundary."""

    integration_id: str
    repository_url: str
    commit_sha: str
    license_spdx: str
    install_script: str
    artifact_schema: str
    source_directory: str
    source_inventory_required: bool
    release_ready: bool

    def __post_init__(self) -> None:
        if self.integration_id not in _EXPECTED_IDS:
            raise IntegrationProvenanceError("unknown external integration pin")
        if not _SHA40.fullmatch(self.commit_sha):
            raise IntegrationProvenanceError("commit_sha must be a lowercase Git SHA")
        if not self.repository_url.startswith("https://github.com/"):
            raise IntegrationProvenanceError("repository_url must use GitHub HTTPS")
        if _SOURCE_DIRECTORY.fullmatch(self.source_directory) is None:
            raise IntegrationProvenanceError("source_directory is unsafe")
        install = Path(self.install_script)
        if (
            install.is_absolute()
            or ".." in install.parts
            or install.suffix != ".sh"
            or install.parts[0] not in {"integrations", "scripts"}
        ):
            raise IntegrationProvenanceError("install_script is unsafe")
        if self.source_inventory_required is not True:
            raise IntegrationProvenanceError("full source inventory is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_schema": self.artifact_schema,
            "commit_sha": self.commit_sha,
            "install_script": self.install_script,
            "integration_id": self.integration_id,
            "license_spdx": self.license_spdx,
            "release_ready": self.release_ready,
            "repository_url": self.repository_url,
            "source_directory": self.source_directory,
            "source_inventory_required": self.source_inventory_required,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExternalIntegrationPin":
        if not isinstance(value, Mapping) or set(value) != _ALLOWED_FIELDS:
            raise IntegrationProvenanceError("external pin fields mismatch")
        for field in (
            "artifact_schema",
            "commit_sha",
            "install_script",
            "integration_id",
            "license_spdx",
            "repository_url",
            "source_directory",
        ):
            _string(value[field], field)
        if (
            type(value["release_ready"]) is not bool
            or type(value["source_inventory_required"]) is not bool
        ):
            raise IntegrationProvenanceError("external pin flags must be booleans")
        return cls(**cast(dict[str, Any], dict(value)))


@dataclass(frozen=True)
class IntegrationLock:
    """Exact ordered inventory of external RoboTactile dependencies."""

    schema_version: str
    entries: tuple[ExternalIntegrationPin, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise IntegrationProvenanceError("integration lock version mismatch")
        if tuple(entry.integration_id for entry in self.entries) != _EXPECTED_IDS:
            raise IntegrationProvenanceError("integration lock inventory mismatch")

    def by_id(self, integration_id: str) -> ExternalIntegrationPin:
        matches = tuple(
            entry for entry in self.entries if entry.integration_id == integration_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown external integration: {integration_id}")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ExternalCheckoutReceipt:
    """Read-only result of verifying one clean external checkout."""

    integration_id: str
    repository_url: str
    commit_sha: str
    lock_sha256: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_lock(path: Optional[Path]) -> bytes:
    if path is None:
        packaged = resources.files("robotactile_benchmark").joinpath(_LOCK_PATH)
        if packaged.is_file():
            return packaged.read_bytes()
        path = _project_root() / _LOCK_PATH
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise IntegrationProvenanceError("integration lock must be a regular file")
    before = selected.stat()
    if not 0 < before.st_size <= 64 * 1024:
        raise IntegrationProvenanceError("integration lock size is invalid")
    raw = selected.read_bytes()
    after = selected.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise IntegrationProvenanceError("integration lock changed during read")
    return raw


def load_integration_lock(path: Optional[Path] = None) -> IntegrationLock:
    """Load the canonical lock while rejecting coordinated metadata drift."""

    raw = _read_lock(path)
    try:
        value = strict_json_bytes(raw, "integrations.lock.json")
    except (TypeError, ValueError) as error:
        raise IntegrationProvenanceError("integration lock is not canonical") from error
    if not isinstance(value, Mapping) or set(value) != {"entries", "schema_version"}:
        raise IntegrationProvenanceError("integration lock fields mismatch")
    entries = value["entries"]
    if not isinstance(entries, list):
        raise IntegrationProvenanceError("integration lock entries must be a list")
    lock = IntegrationLock(
        schema_version=_string(value["schema_version"], "schema_version"),
        entries=tuple(ExternalIntegrationPin.from_dict(entry) for entry in entries),
    )
    if canonical_json_bytes(lock.to_dict()) != raw:
        raise IntegrationProvenanceError(
            "integration lock typed value is not canonical"
        )
    return lock


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", os.fspath(root), *args),
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegrationProvenanceError(
            "external checkout Git query failed"
        ) from error
    return result.stdout.strip()


def _normalize_url(value: str) -> str:
    return value.removesuffix(".git").rstrip("/")


def verify_external_checkout(
    pin: ExternalIntegrationPin, root: Path
) -> ExternalCheckoutReceipt:
    """Verify exact origin, commit, and clean state without importing upstream."""

    selected = Path(root)
    if selected.is_symlink() or not selected.is_dir():
        raise IntegrationProvenanceError("external checkout must be a real directory")
    resolved = selected.resolve(strict=True)
    if resolved != selected.absolute():
        raise IntegrationProvenanceError("external checkout path contains a symlink")
    commit = _git(resolved, "rev-parse", "HEAD")
    origin = _git(resolved, "remote", "get-url", "origin")
    dirty = _git(resolved, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != pin.commit_sha:
        raise IntegrationProvenanceError("external checkout commit mismatch")
    if _normalize_url(origin) != _normalize_url(pin.repository_url):
        raise IntegrationProvenanceError("external checkout origin mismatch")
    if dirty:
        raise IntegrationProvenanceError("external checkout is dirty")
    lock_sha256 = sha256_bytes(canonical_json_bytes(pin.to_dict()))
    return ExternalCheckoutReceipt(
        integration_id=pin.integration_id,
        repository_url=pin.repository_url,
        commit_sha=pin.commit_sha,
        lock_sha256=lock_sha256,
    )


__all__ = [
    "ExternalCheckoutReceipt",
    "ExternalIntegrationPin",
    "IntegrationLock",
    "IntegrationProvenanceError",
    "load_integration_lock",
    "verify_external_checkout",
]
