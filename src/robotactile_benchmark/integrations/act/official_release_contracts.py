"""Frozen contracts for the official UniVTAC ACT artifact release."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Optional, cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)

OFFICIAL_ACT_REPOSITORY_ID = "byml/UniVTAC"
OFFICIAL_ACT_REPOSITORY_TYPE = "dataset"
OFFICIAL_ACT_REVISION = "172331dbbce95bc04c3e59b22f32dc72ba5561ae"
OFFICIAL_ACT_LOCK_SCHEMA = "robotactile-act-artifacts-lock-v1"
OFFICIAL_ACT_LOCK_FAMILY = "official_univtac_act_hf_release"
OFFICIAL_ACT_TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
OFFICIAL_ACT_PROFILES = ("univtac", "vision_only")
RUNTIME_EVIDENCE = "artifact_installation_only_no_model_or_simulator_execution_v1"
REFERENCE_EVIDENCE = "upstream_reference_only_no_local_execution_v1"
PLAN_SCHEMA = "robotactile-official-act-install-plan-v1"
RECEIPT_SCHEMA = "robotactile-official-act-install-receipt-v1"

_LOCK_RESOURCE = "integrations/act_artifacts.lock.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_POLICY_DATA = (
    (
        "grasp_classify",
        "univtac",
        "0220023a804f418cf247e84f76d7f4b9280220d605c25b4521987af84ef17576",
    ),
    (
        "grasp_classify",
        "vision_only",
        "20c0dab589b4a04581578be5363668fe599ea5186dcfba16bdb856aeee82d5e8",
    ),
    (
        "insert_HDMI",
        "univtac",
        "05893cadc6700904b3fed144ed234e7adf4e057a1b37d6474209f078ab03427f",
    ),
    (
        "insert_HDMI",
        "vision_only",
        "857cb6c8601f099e47f1a558bfce22d5fb03f40d0e5ef024116e51c1baa6f100",
    ),
    (
        "insert_hole",
        "univtac",
        "f0110759fe9a7d2a09fca033aa9b32045732ee25ba721a94b93c4b8305379c7d",
    ),
    (
        "insert_hole",
        "vision_only",
        "40a48d7551c0534df0e66ef3ed84944ac48bda0ee22d21079773b66bf94bb1a5",
    ),
    (
        "insert_tube",
        "univtac",
        "6260aa3f5e4e89089099e89912dddda7ddd2eb1979efd6186835826f1feddb39",
    ),
    (
        "insert_tube",
        "vision_only",
        "4c3e48e91ed1bf5d6a988b07e29ed00ebcf0aab81dd566fc92631b203c53e462",
    ),
    (
        "lift_bottle",
        "univtac",
        "a6d5d8c0513357fdbafa1cc658cde03770d434a0dd08993697203e89b0f5d14a",
    ),
    (
        "lift_bottle",
        "vision_only",
        "1493dd1cf0ebfb7b4f7312b1778ead1bdd16cf6837288ef89dd6e60eee71a398",
    ),
    (
        "lift_can",
        "univtac",
        "524eec391e404ddeca28e753eab31a5e3b17f633f8dff0310266726461e9358c",
    ),
    (
        "lift_can",
        "vision_only",
        "7a7bded7a36451f72cdc44fdcc1e32297b51eecdf74f47d5a2ad136d62871aca",
    ),
    (
        "pull_out_key",
        "univtac",
        "2973ec8eb78beba9b1a636783921935470be051a675538d63b4fe2c5fd416404",
    ),
    (
        "pull_out_key",
        "vision_only",
        "e05ab741c7abc041f433dd07a3e37afde4a30559bd503825c46067979acb572e",
    ),
    (
        "put_bottle_in_shelf",
        "univtac",
        "f46fdc26bc52ee0e5580c86c09291ee32eb50a5b9b56f555f35b25e40bdd2ce0",
    ),
    (
        "put_bottle_in_shelf",
        "vision_only",
        "9ffa99425d4e653682acdf7a38d6f50eaa825efb074d26156615ec66908d6487",
    ),
)
_STATS_DATA = (
    (
        "grasp_classify",
        1845,
        "3bd8d4df70e1129d944462445470dc9d7706471423a5e6df9998afa641a7a335",
    ),
    (
        "insert_HDMI",
        4213,
        "8d08577d488442c3833c1d21ed9d6b6f1602af336043714c05d734f73de31114",
    ),
    (
        "insert_hole",
        5909,
        "4f28a439f82ab8bfc0d112e75d915c08b415b6aa27e7ebda18945e2d4a3d0cc5",
    ),
    (
        "insert_tube",
        6581,
        "a1eee654c1e643e49e3bda25f78f0d5f4543fa16da7c816b008de5a0d27952ea",
    ),
    (
        "lift_bottle",
        10390,
        "d0839807c19dadac83188ad8f451d29e1b6b8a9c4212ccf002859f57b3f1af36",
    ),
    (
        "lift_can",
        6101,
        "408e1cda38c2fdc8bdb40797f79de921f55ee2b9790ba5ecabbc42e527336c7d",
    ),
    (
        "pull_out_key",
        6005,
        "cb8e34b7101e1a9bf2c0a09c64d2f8038ba0a276014e22822ffbe8e7364a73b3",
    ),
    (
        "put_bottle_in_shelf",
        13142,
        "32e8fa1ba0f764e4e37c952895e5e7693e81f7f4d7cf94f1e439e5331ae3e13d",
    ),
)


class OfficialACTReleaseError(ValueError):
    """The frozen release contract or a downloaded file is invalid."""


@dataclass(frozen=True)
class OfficialACTInstallResult:
    downloaded_file_count: int
    published_destination_count: int
    receipt_path: Path
    receipt_sha256: str
    reused_destination_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "downloaded_file_count": self.downloaded_file_count,
            "published_destination_count": self.published_destination_count,
            "receipt_path": str(self.receipt_path),
            "receipt_sha256": self.receipt_sha256,
            "reused_destination_count": self.reused_destination_count,
        }


def safe_relative(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise OfficialACTReleaseError(f"{name} must be a safe POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise OfficialACTReleaseError(f"{name} must be a safe POSIX path")
    return value


def valid_sha256(value: str) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def official_download_url(remote_path: str) -> str:
    safe_relative(remote_path, "download remote path")
    quoted = urllib.parse.quote(remote_path, safe="/")
    return (
        "https://huggingface.co/datasets/byml/UniVTAC/resolve/"
        f"{OFFICIAL_ACT_REVISION}/{quoted}?download=true"
    )


@dataclass(frozen=True)
class OfficialACTReleaseFile:
    kind: str
    profile: Optional[str]
    remote_path: str
    sha256: str
    size_bytes: int
    task_id: Optional[str]

    def __post_init__(self) -> None:
        safe_relative(self.remote_path, "remote_path")
        if self.kind not in {"encoder", "policy", "stats"}:
            raise OfficialACTReleaseError("unknown official ACT artifact kind")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise OfficialACTReleaseError("artifact size must be a positive integer")
        if not valid_sha256(self.sha256):
            raise OfficialACTReleaseError("artifact SHA256 must be lowercase hex")
        expected: str
        if self.kind == "encoder":
            if self.task_id is not None or self.profile is not None:
                raise OfficialACTReleaseError("encoder must be shared")
            expected = "checkpoints/encoder.pth"
        elif self.kind == "policy":
            if (
                self.task_id not in OFFICIAL_ACT_TASKS
                or self.profile not in OFFICIAL_ACT_PROFILES
            ):
                raise OfficialACTReleaseError("policy task/profile is unsupported")
            expected = f"checkpoints/{self.task_id}/{self.profile}/policy_last.ckpt"
        else:
            if self.task_id not in OFFICIAL_ACT_TASKS or self.profile is not None:
                raise OfficialACTReleaseError("stats must be shared by task profiles")
            expected = f"checkpoints/{self.task_id}/univtac/dataset_stats.pkl"
        if self.remote_path != expected:
            raise OfficialACTReleaseError(
                "artifact remote path disagrees with semantics"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "profile": self.profile,
            "remote_path": self.remote_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OfficialACTReleaseFile":
        fields = {"kind", "profile", "remote_path", "sha256", "size_bytes", "task_id"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise OfficialACTReleaseError("ACT artifact lock file fields mismatch")
        return cls(**cast(dict[str, Any], dict(value)))


def _builtin_files() -> tuple[OfficialACTReleaseFile, ...]:
    files = [
        OfficialACTReleaseFile(
            "encoder",
            None,
            "checkpoints/encoder.pth",
            "28097c91a1a1d051f65266fc7fdd84420dc184da9dcfc2006aa40af144cf15bf",
            165482707,
            None,
        )
    ]
    files.extend(
        OfficialACTReleaseFile(
            "policy",
            profile,
            f"checkpoints/{task}/{profile}/policy_last.ckpt",
            sha256,
            382746500,
            task,
        )
        for task, profile, sha256 in _POLICY_DATA
    )
    files.extend(
        OfficialACTReleaseFile(
            "stats",
            None,
            f"checkpoints/{task}/univtac/dataset_stats.pkl",
            sha256,
            size,
            task,
        )
        for task, size, sha256 in _STATS_DATA
    )
    return tuple(sorted(files, key=lambda item: item.remote_path))


_BUILTIN_FILES = _builtin_files()


@dataclass(frozen=True)
class OfficialACTReleaseLock:
    artifact_family: str
    files: tuple[OfficialACTReleaseFile, ...]
    repository_id: str
    repository_type: str
    revision: str
    schema_version: str
    semantic_version: str

    def __post_init__(self) -> None:
        identity = (
            self.artifact_family,
            self.repository_id,
            self.repository_type,
            self.revision,
            self.schema_version,
            self.semantic_version,
        )
        expected = (
            OFFICIAL_ACT_LOCK_FAMILY,
            OFFICIAL_ACT_REPOSITORY_ID,
            OFFICIAL_ACT_REPOSITORY_TYPE,
            OFFICIAL_ACT_REVISION,
            OFFICIAL_ACT_LOCK_SCHEMA,
            "1.0",
        )
        if identity != expected:
            raise OfficialACTReleaseError("official ACT release identity mismatch")
        if self.files != _BUILTIN_FILES:
            raise OfficialACTReleaseError("official ACT release inventory mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_family": self.artifact_family,
            "files": [item.to_dict() for item in self.files],
            "repository_id": self.repository_id,
            "repository_type": self.repository_type,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "semantic_version": self.semantic_version,
        }


def builtin_official_act_release_lock() -> OfficialACTReleaseLock:
    return OfficialACTReleaseLock(
        OFFICIAL_ACT_LOCK_FAMILY,
        _BUILTIN_FILES,
        OFFICIAL_ACT_REPOSITORY_ID,
        OFFICIAL_ACT_REPOSITORY_TYPE,
        OFFICIAL_ACT_REVISION,
        OFFICIAL_ACT_LOCK_SCHEMA,
        "1.0",
    )


def _read_lock(path: Optional[Path]) -> bytes:
    selected: Optional[Path] = None
    if path is None:
        packaged = resources.files("robotactile_benchmark").joinpath(_LOCK_RESOURCE)
        if packaged.is_file():
            return packaged.read_bytes()
        repository = Path(__file__).resolve().parents[4] / _LOCK_RESOURCE
        if repository.is_file():
            selected = repository
    else:
        selected = Path(path)
    if selected is None:
        return canonical_json_bytes(builtin_official_act_release_lock().to_dict())
    if selected.is_symlink() or not selected.is_file():
        raise OfficialACTReleaseError("ACT artifact lock must be a regular file")
    before = selected.stat()
    if not 0 < before.st_size <= 64 * 1024:
        raise OfficialACTReleaseError("ACT artifact lock size is invalid")
    raw = selected.read_bytes()
    after = selected.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise OfficialACTReleaseError("ACT artifact lock changed during read")
    return raw


def load_official_act_release_lock(
    path: Optional[Path] = None,
) -> OfficialACTReleaseLock:
    raw = _read_lock(path)
    try:
        value = strict_json_bytes(raw, "act_artifacts.lock.json")
    except (TypeError, ValueError) as error:
        raise OfficialACTReleaseError("ACT artifact lock is not canonical") from error
    fields = {
        "artifact_family",
        "files",
        "repository_id",
        "repository_type",
        "revision",
        "schema_version",
        "semantic_version",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OfficialACTReleaseError("ACT artifact lock fields mismatch")
    raw_files = value["files"]
    if not isinstance(raw_files, list):
        raise OfficialACTReleaseError("ACT artifact lock files must be a list")
    lock = OfficialACTReleaseLock(
        artifact_family=cast(str, value["artifact_family"]),
        files=tuple(OfficialACTReleaseFile.from_dict(item) for item in raw_files),
        repository_id=cast(str, value["repository_id"]),
        repository_type=cast(str, value["repository_type"]),
        revision=cast(str, value["revision"]),
        schema_version=cast(str, value["schema_version"]),
        semantic_version=cast(str, value["semantic_version"]),
    )
    if canonical_json_bytes(lock.to_dict()) != raw:
        raise OfficialACTReleaseError("ACT artifact lock typed value is not canonical")
    return lock


__all__ = [
    "OFFICIAL_ACT_LOCK_FAMILY",
    "OFFICIAL_ACT_PROFILES",
    "OFFICIAL_ACT_REPOSITORY_ID",
    "OFFICIAL_ACT_REPOSITORY_TYPE",
    "OFFICIAL_ACT_REVISION",
    "OFFICIAL_ACT_TASKS",
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "REFERENCE_EVIDENCE",
    "RUNTIME_EVIDENCE",
    "OfficialACTInstallResult",
    "OfficialACTReleaseError",
    "OfficialACTReleaseFile",
    "OfficialACTReleaseLock",
    "builtin_official_act_release_lock",
    "load_official_act_release_lock",
    "official_download_url",
    "safe_relative",
    "valid_sha256",
]
