"""Pinned live loader for official UniVTAC ACT ``policy_last`` artifacts.

The loader is intentionally independent of ``load_strict_act_policy``.  The two
artifact families use different filenames, source contracts, and evidence
levels, so accepting one through the other's loader would be an identity bug.
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Optional

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import UPSTREAM_COMMIT
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.contracts import Array
from robotactile_benchmark.policies.univtac_official_act import (
    _PROFILE_SPECS,
    _TASK_IDS,
    OfficialACTProfile,
    OfficialUniVTACACTPolicy,
    _construct_pinned_upstream_runtime,
    _OwnedRuntime,
    _release_runtime,
    _torch_transform,
)

_ACT_SOURCE_RELATIVE = Path("policy/ACT/act_policy.py")
_ACT_SOURCE_SHA256 = "9e622713ebefa199dc0e80ca8ceb40db61d65162c2d5d753f4776e516d3b3cb1"
_STAT_KEYS = frozenset({"qpos_mean", "qpos_std", "action_mean", "action_std"})


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _absolute(value: Path, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    return value.absolute()


@dataclass(frozen=True)
class OfficialUniVTACACTArtifactManifest:
    """Path and digest authority for one task/profile official ACT artifact."""

    task_id: str
    profile: OfficialACTProfile
    artifact_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    stats_path: Path
    stats_sha256: str
    encoder_path: Path
    encoder_sha256: str
    upstream_root: Path
    upstream_commit: str
    source_path: Path
    source_sha256: str
    config_path: Path
    config_sha256: str
    semantic_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.semantic_version != "1.0":
            raise ValueError("official ACT manifest semantic version mismatch")
        if self.task_id not in _TASK_IDS:
            raise ValueError("official ACT manifest task is not in the frozen registry")
        profile = (
            self.profile
            if isinstance(self.profile, OfficialACTProfile)
            else OfficialACTProfile(self.profile)
        )
        object.__setattr__(self, "profile", profile)
        for name in (
            "artifact_root",
            "checkpoint_path",
            "stats_path",
            "encoder_path",
            "upstream_root",
            "source_path",
            "config_path",
        ):
            object.__setattr__(self, name, _absolute(getattr(self, name), name))
        for name in (
            "checkpoint_sha256",
            "stats_sha256",
            "encoder_sha256",
            "source_sha256",
            "config_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.upstream_commit != UPSTREAM_COMMIT:
            raise ValueError("official ACT upstream commit mismatch")
        spec = _PROFILE_SPECS[profile]
        expected_profile = self.artifact_root / self.task_id / profile.value
        expected = {
            "checkpoint_path": expected_profile / "policy_last.ckpt",
            "stats_path": expected_profile / "dataset_stats.pkl",
            "encoder_path": self.artifact_root / "encoder.pth",
            "source_path": self.upstream_root / _ACT_SOURCE_RELATIVE,
            "config_path": self.upstream_root / "policy/ACT" / spec.config_name,
        }
        for name, path in expected.items():
            if getattr(self, name) != path:
                raise ValueError(f"official ACT {name} does not match frozen layout")
        if self.source_sha256 != _ACT_SOURCE_SHA256:
            raise ValueError("official ACT source SHA256 mismatch")
        if self.config_sha256 != spec.config_sha256:
            raise ValueError("official ACT config/profile SHA256 mismatch")

    @classmethod
    def for_shared_root(
        cls,
        *,
        task_id: str,
        profile: OfficialACTProfile,
        artifact_root: Path,
        upstream_root: Path,
        checkpoint_sha256: str,
        stats_sha256: str,
        encoder_sha256: str,
    ) -> OfficialUniVTACACTArtifactManifest:
        normalized = (
            profile
            if isinstance(profile, OfficialACTProfile)
            else OfficialACTProfile(profile)
        )
        root = _absolute(artifact_root, "artifact_root")
        source_root = _absolute(upstream_root, "upstream_root")
        profile_root = root / task_id / normalized.value
        spec = _PROFILE_SPECS[normalized]
        return cls(
            task_id=task_id,
            profile=normalized,
            artifact_root=root,
            checkpoint_path=profile_root / "policy_last.ckpt",
            checkpoint_sha256=checkpoint_sha256,
            stats_path=profile_root / "dataset_stats.pkl",
            stats_sha256=stats_sha256,
            encoder_path=root / "encoder.pth",
            encoder_sha256=encoder_sha256,
            upstream_root=source_root,
            upstream_commit=UPSTREAM_COMMIT,
            source_path=source_root / _ACT_SOURCE_RELATIVE,
            source_sha256=_ACT_SOURCE_SHA256,
            config_path=source_root / "policy/ACT" / spec.config_name,
            config_sha256=spec.config_sha256,
        )


@dataclass(frozen=True)
class OfficialUniVTACACTLoadRequest:
    """Explicit opt-in to allocate the official upstream runtime."""

    manifest: OfficialUniVTACACTArtifactManifest
    task_id: str
    profile: OfficialACTProfile
    device_name: str
    live: bool

    def __post_init__(self) -> None:
        if type(self.manifest) is not OfficialUniVTACACTArtifactManifest:
            raise TypeError("manifest must be an exact official ACT manifest")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("official ACT request task must be non-empty")
        profile = (
            self.profile
            if isinstance(self.profile, OfficialACTProfile)
            else OfficialACTProfile(self.profile)
        )
        object.__setattr__(self, "profile", profile)
        if not isinstance(self.device_name, str) or not self.device_name.startswith(
            "cuda"
        ):
            raise ValueError(
                "official upstream ACT live loading requires a CUDA device"
            )
        if type(self.live) is not bool:
            raise TypeError("live must be bool")


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


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _construct_upstream_runtime(
    manifest: OfficialUniVTACACTArtifactManifest,
    device_name: str,
    torch_module: Any,
) -> Any:
    return _construct_pinned_upstream_runtime(
        source_path=manifest.source_path,
        source_sha256=manifest.source_sha256,
        profile=manifest.profile,
        task_id=manifest.task_id,
        encoder_path=manifest.encoder_path,
        device_name=device_name,
        torch_module=torch_module,
    )


def _validate_identity(
    identity: PolicyIdentity, request: OfficialUniVTACACTLoadRequest
) -> None:
    manifest = request.manifest
    if request.live is not True:
        raise ValueError("official ACT loading requires an explicit live request")
    if request.task_id != manifest.task_id:
        raise ValueError("official ACT request/manifest task mismatch")
    if request.profile is not manifest.profile:
        raise ValueError("official ACT request/manifest profile mismatch")
    if identity.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("official ACT checkpoint identity mismatch")
    if identity.config_sha256 != manifest.config_sha256:
        raise ValueError("official ACT config identity mismatch")
    tactile = manifest.profile is OfficialACTProfile.UNIVTAC
    if (
        identity.action_spec != ACTION_SPEC
        or identity.consumes_tactile is not tactile
        or identity.supports_structural_absence is tactile
    ):
        raise ValueError("official ACT identity/profile capabilities mismatch")


def load_official_univtac_act_policy(
    identity: PolicyIdentity,
    request: OfficialUniVTACACTLoadRequest,
) -> OfficialUniVTACACTPolicy:
    """Load a hash-authorized official ACT only after explicit live opt-in."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if type(request) is not OfficialUniVTACACTLoadRequest:
        raise TypeError("request must be an exact official ACT load request")
    _validate_identity(identity, request)
    manifest = request.manifest
    _verify_manifest_files(manifest)
    stats = _load_stats(manifest.stats_path)
    torch_module = _import_torch()
    runtime: Optional[Any] = None
    try:
        state = torch_module.load(
            manifest.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, Mapping):
            raise TypeError("official ACT checkpoint must contain a state mapping")
        runtime = _construct_upstream_runtime(
            manifest, request.device_name, torch_module
        )
        model = getattr(runtime, "policy", None)
        load_state_dict = getattr(model, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError("official ACT runtime lacks a policy state boundary")
        load_state_dict(state, strict=True)
        runtime.stats = stats
        owned = _OwnedRuntime(runtime, torch_module)
        runtime = None
        return OfficialUniVTACACTPolicy(
            identity,
            owned,
            artifact_task=manifest.task_id,
            profile=manifest.profile,
            input_transform=_torch_transform(torch_module),
        )
    except Exception:
        _release_runtime(runtime, torch_module)
        raise


__all__ = [
    "OfficialUniVTACACTArtifactManifest",
    "OfficialUniVTACACTLoadRequest",
    "load_official_univtac_act_policy",
    "validate_official_univtac_act_artifact",
]
