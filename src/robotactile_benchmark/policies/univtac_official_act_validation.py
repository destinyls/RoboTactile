"""Non-allocating validation for official UniVTAC ACT artifacts."""

from __future__ import annotations

import hashlib
import pickle
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

import numpy as np

from robotactile_benchmark.contracts import Array
from robotactile_benchmark.policies.univtac_official_act_manifest import (
    OfficialUniVTACACTArtifactManifest,
)

_STAT_KEYS = frozenset({"qpos_mean", "qpos_std", "action_mean", "action_std"})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"official ACT {name} must be a non-symlink regular file")
    return path.resolve(strict=True)


def _git_head(root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("official ACT upstream tracked tree is dirty")
    return head


def _verify_manifest_files(manifest: OfficialUniVTACACTArtifactManifest) -> None:
    if _git_head(manifest.upstream_root) != manifest.upstream_commit:
        raise ValueError("official ACT upstream HEAD mismatch")
    files = (
        (manifest.checkpoint_path, manifest.checkpoint_sha256, "checkpoint"),
        (manifest.stats_path, manifest.stats_sha256, "stats"),
        (manifest.encoder_path, manifest.encoder_sha256, "encoder"),
        (manifest.source_path, manifest.source_sha256, "source"),
        (manifest.config_path, manifest.config_sha256, "config"),
    )
    for path, expected, name in files:
        resolved = _regular_file(path, name)
        if _file_sha256(resolved) != expected:
            raise ValueError(f"official ACT {name} SHA256 mismatch")


class _RestrictedStatsUnpickler(pickle.Unpickler):
    """Allow only numpy array reconstruction after a trusted SHA256 match."""

    _ALLOWED = frozenset(
        {
            ("numpy", "dtype"),
            ("numpy", "ndarray"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"forbidden stats global: {module}.{name}")
        return super().find_class(module, name)


def _read_restricted_stats(handle: BinaryIO) -> object:
    return _RestrictedStatsUnpickler(handle).load()


def _load_stats(path: Path) -> Mapping[str, Array]:
    """Load hash-authorized numpy data; pickle remains a bounded trust boundary."""

    with path.open("rb") as handle:
        raw = _read_restricted_stats(handle)
    if not isinstance(raw, Mapping) or not _STAT_KEYS.issubset(raw):
        raise ValueError("official ACT stats must contain four normalization arrays")
    result: dict[str, Array] = {}
    for name in _STAT_KEYS:
        value = raw[name]
        if type(value) is not np.ndarray:
            raise TypeError(f"official ACT stat {name} must be an exact numpy array")
        if value.dtype != np.float32:
            raise TypeError(f"official ACT stat {name} must use exact float32")
        if value.shape != (8,):
            raise ValueError(f"official ACT stat {name} must have exact shape (8,)")
        if not np.isfinite(value).all():
            raise ValueError(f"official ACT stat {name} must be finite")
        if name.endswith("_std") and not np.all(value > 0.0):
            raise ValueError(f"official ACT stat {name} must be strictly positive")
        frozen = np.ascontiguousarray(value.copy())
        frozen.setflags(write=False)
        result[name] = frozen
    return MappingProxyType(result)


def validate_official_univtac_act_artifact(
    manifest: OfficialUniVTACACTArtifactManifest,
) -> Mapping[str, str]:
    """Validate a complete official ACT artifact without importing Torch."""

    if type(manifest) is not OfficialUniVTACACTArtifactManifest:
        raise TypeError("manifest must be an exact official ACT manifest")
    _verify_manifest_files(manifest)
    _load_stats(manifest.stats_path)
    return MappingProxyType(
        {
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "config_sha256": manifest.config_sha256,
            "encoder_sha256": manifest.encoder_sha256,
            "profile": manifest.profile.value,
            "source_sha256": manifest.source_sha256,
            "stats_sha256": manifest.stats_sha256,
            "task_id": manifest.task_id,
            "upstream_commit": manifest.upstream_commit,
        }
    )


__all__ = ["validate_official_univtac_act_artifact"]
