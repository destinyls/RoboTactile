"""Read-only planning for legacy split-root deployment contents."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.deployment.layout import DeploymentLayout

_PREFIXES = (
    (Path("src/UniVTAC"), Path("sources/UniVTAC")),
    (Path("src/WorldArena"), Path("sources/WorldArena")),
    (Path("src/N0-TWAM"), Path("sources/N0-TWAM")),
    (Path("src/IsaacLab"), Path("sources/IsaacLab")),
    (Path("src/curobo"), Path("sources/curobo")),
    (Path("artifacts/checkpoints"), Path("artifacts/models/act")),
    (Path("artifacts/live_univtac"), Path("artifacts/live-univtac")),
    (Path("requests/live_univtac"), Path("requests/four-condition")),
    (Path("artifacts/rest-references"), Path("artifacts/rest-references")),
)


@dataclass(frozen=True)
class MigrationPlanEntry:
    source: Path
    destination: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "destination": str(self.destination),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source": str(self.source),
        }


@dataclass(frozen=True)
class DeploymentMigrationPlan:
    legacy_root: Path
    deployment_root: Path
    entries: tuple[MigrationPlanEntry, ...]
    unmapped_file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "deployment_root": str(self.deployment_root),
            "entries": [entry.to_dict() for entry in self.entries],
            "evidence_level": "read_only_migration_plan_v1",
            "legacy_root": str(self.legacy_root),
            "proposed_file_count": len(self.entries),
            "source_deleted": False,
            "unmapped_file_count": self.unmapped_file_count,
            "writes_performed": False,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _destination(relative: Path, layout: DeploymentLayout) -> Path | None:
    for old, new in _PREFIXES:
        if relative == old or old in relative.parents:
            suffix = relative.relative_to(old)
            return layout.root / new / suffix
    return None


def plan_legacy_migration(
    legacy_root: Path, layout: DeploymentLayout
) -> DeploymentMigrationPlan:
    """Hash recognized files without creating, copying, or deleting anything."""

    selected = Path(legacy_root).expanduser().absolute()
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError("legacy root must be a real directory")
    if selected == layout.root or selected in layout.root.parents:
        raise ValueError("legacy root cannot contain the deployment root")
    if layout.root in selected.parents:
        raise ValueError("legacy root cannot be inside the deployment root")
    entries: list[MigrationPlanEntry] = []
    unmapped = 0
    for directory, directory_names, file_names in os.walk(selected):
        directory_names.sort()
        file_names.sort()
        root = Path(directory)
        for name in file_names:
            source = root / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"legacy file must be regular: {source}")
            destination = _destination(source.relative_to(selected), layout)
            if destination is None:
                unmapped += 1
                continue
            entries.append(
                MigrationPlanEntry(
                    source=source,
                    destination=destination,
                    size_bytes=source.stat().st_size,
                    sha256=_sha256_file(source),
                )
            )
    return DeploymentMigrationPlan(selected, layout.root, tuple(entries), unmapped)


__all__ = [
    "DeploymentMigrationPlan",
    "MigrationPlanEntry",
    "plan_legacy_migration",
]
