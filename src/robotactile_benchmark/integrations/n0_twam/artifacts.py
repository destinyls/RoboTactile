"""External N0-TWAM model-bundle identity contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

_SCHEMA_VERSION = "robotactile-n0-artifact-v1"
_EXTERNAL_COMMIT = "9036c130409f8cf5494b12489fea339f7213b9d6"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "bundle_root",
        "checkpoint_sha256",
        "config_sha256",
        "external_commit",
        "normalizer_sha256",
        "schema_version",
    }
)


@dataclass(frozen=True)
class N0TWAMArtifactManifest:
    """Paths stay external while every model resource is content addressed."""

    bundle_root: Path
    checkpoint_sha256: str
    config_sha256: str
    normalizer_sha256: str
    external_commit: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_root, Path) or not self.bundle_root.is_absolute():
            raise ValueError("bundle_root must be an absolute pathlib.Path")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("N0-TWAM artifact schema version mismatch")
        if self.external_commit != _EXTERNAL_COMMIT:
            raise ValueError("N0-TWAM external commit mismatch")
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "normalizer_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256")

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_root": str(self.bundle_root),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "external_commit": self.external_commit,
            "normalizer_sha256": self.normalizer_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "N0TWAMArtifactManifest":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("N0-TWAM artifact manifest fields mismatch")
        document = cast(dict[str, object], dict(value))
        root = document.pop("bundle_root")
        if not isinstance(root, str):
            raise TypeError("bundle_root must be a string")
        return cls(bundle_root=Path(root), **cast(dict[str, str], document))


__all__ = ["N0TWAMArtifactManifest"]
