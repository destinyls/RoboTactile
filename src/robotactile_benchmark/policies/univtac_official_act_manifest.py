"""Frozen manifest and allocation request for official UniVTAC ACT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.backends.univtac_contracts import UPSTREAM_COMMIT
from robotactile_benchmark.policies.univtac_official_act import (
    _PROFILE_SPECS,
    _TASK_IDS,
    OfficialACTProfile,
)

_ACT_SOURCE_RELATIVE = Path("policy/ACT/act_policy.py")
_ACT_SOURCE_SHA256 = "9e622713ebefa199dc0e80ca8ceb40db61d65162c2d5d753f4776e516d3b3cb1"


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


__all__ = ["OfficialUniVTACACTArtifactManifest", "OfficialUniVTACACTLoadRequest"]
