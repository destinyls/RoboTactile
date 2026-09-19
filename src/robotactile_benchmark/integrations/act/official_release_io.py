"""Streaming, no-clobber IO for official UniVTAC ACT artifacts."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.integrations.act.official_release_contracts import (
    OFFICIAL_ACT_LOCK_FAMILY,
    OFFICIAL_ACT_REPOSITORY_ID,
    OFFICIAL_ACT_REPOSITORY_TYPE,
    OFFICIAL_ACT_REVISION,
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    REFERENCE_EVIDENCE,
    RUNTIME_EVIDENCE,
    OfficialACTInstallResult,
    OfficialACTReleaseError,
    official_download_url,
    safe_relative,
    valid_sha256,
)

_USER_AGENT = "RoboTactile/0.6 official-ACT-artifact-installer"


@dataclass(frozen=True)
class PlannedOfficialACTFile:
    destinations: tuple[str, ...]
    evidence_level: str
    kind: str
    remote_path: str
    sha256: str
    size_bytes: int
    source_url: str

    def __post_init__(self) -> None:
        safe_relative(self.remote_path, "planned remote path")
        if not self.destinations or len(set(self.destinations)) != len(
            self.destinations
        ):
            raise OfficialACTReleaseError("planned destinations must be unique")
        for destination in self.destinations:
            safe_relative(destination, "planned destination")
        if self.source_url != official_download_url(self.remote_path):
            raise OfficialACTReleaseError("planned source URL is not hash-pinned")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise OfficialACTReleaseError("planned size is invalid")
        if not valid_sha256(self.sha256):
            raise OfficialACTReleaseError("planned SHA256 is invalid")
        if self.evidence_level not in {RUNTIME_EVIDENCE, REFERENCE_EVIDENCE}:
            raise OfficialACTReleaseError("planned evidence level is invalid")
        runtime_kinds = {"encoder", "policy", "stats"}
        reference_kinds = {"reference_log", "reference_metadata"}
        if (
            self.evidence_level == RUNTIME_EVIDENCE and self.kind not in runtime_kinds
        ) or (
            self.evidence_level == REFERENCE_EVIDENCE
            and self.kind not in reference_kinds
        ):
            raise OfficialACTReleaseError("planned kind/evidence boundary mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "destinations": list(self.destinations),
            "evidence_level": self.evidence_level,
            "kind": self.kind,
            "remote_path": self.remote_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class OfficialACTInstallPlan:
    artifact_root: Path
    files: tuple[PlannedOfficialACTFile, ...]
    include_reference: bool
    lock_sha256: str
    profiles: tuple[str, ...]
    tasks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_root": str(self.artifact_root),
            "evidence_level": RUNTIME_EVIDENCE,
            "files": [item.to_dict() for item in self.files],
            "include_reference": self.include_reference,
            "lock_sha256": self.lock_sha256,
            "profiles": list(self.profiles),
            "reference_evidence_level": (
                REFERENCE_EVIDENCE if self.include_reference else None
            ),
            "repository_id": OFFICIAL_ACT_REPOSITORY_ID,
            "repository_type": OFFICIAL_ACT_REPOSITORY_TYPE,
            "revision": OFFICIAL_ACT_REVISION,
            "schema_version": PLAN_SCHEMA,
            "tasks": list(self.tasks),
            "total_download_size_bytes": sum(item.size_bytes for item in self.files),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_existing(path: Path, item: PlannedOfficialACTFile) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        raise FileExistsError(f"refusing non-regular ACT artifact: {path}")
    if path.stat().st_size != item.size_bytes or _sha256_file(path) != item.sha256:
        raise FileExistsError(f"refusing to replace different ACT artifact: {path}")
    return True


def _prepare_real_parent(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts[:-1]:
        current /= part
        if current.is_symlink():
            raise OfficialACTReleaseError("artifact destination parent is a symlink")
        if current.exists():
            if not current.is_dir():
                raise OfficialACTReleaseError(
                    "artifact destination parent is not a directory"
                )
        else:
            try:
                current.mkdir()
            except FileExistsError:
                if current.is_symlink() or not current.is_dir():
                    raise OfficialACTReleaseError(
                        "artifact destination parent changed during creation"
                    ) from None
        if current.resolve() != current.absolute():
            raise OfficialACTReleaseError("artifact destination parent escapes root")
    return root / PurePosixPath(relative)


def _prepare_artifact_root(root: Path) -> None:
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        if current.is_symlink():
            raise OfficialACTReleaseError("artifact root ancestor is a symlink")
        if current.exists():
            if not current.is_dir():
                raise OfficialACTReleaseError(
                    "artifact root ancestor is not a directory"
                )
        else:
            current.mkdir()
    if root.resolve() != root.absolute():
        raise OfficialACTReleaseError("artifact root must be a real directory")


def _stream_download(
    item: PlannedOfficialACTFile, target: Path, timeout_s: float
) -> None:
    request = urllib.request.Request(
        item.source_url,
        headers={"Accept-Encoding": "identity", "User-Agent": _USER_AGENT},
        method="GET",
    )
    digest = hashlib.sha256()
    size = 0
    try:
        response = urllib.request.urlopen(request, timeout=timeout_s)
        with response, target.open("xb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > item.size_bytes:
                    raise OfficialACTReleaseError("download exceeds frozen size")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if size != item.size_bytes or digest.hexdigest() != item.sha256:
        target.unlink(missing_ok=True)
        raise OfficialACTReleaseError("download does not match frozen size/SHA256")


def _publish_link(source: Path, target: Path, item: PlannedOfficialACTFile) -> bool:
    if _validated_existing(target, item):
        return False
    try:
        os.link(source, target)
    except FileExistsError:
        if not _validated_existing(target, item):
            raise
        return False
    return True


def _receipt(plan: OfficialACTInstallPlan) -> dict[str, object]:
    return {
        "artifact_family": OFFICIAL_ACT_LOCK_FAMILY,
        "evidence_level": RUNTIME_EVIDENCE,
        "install_plan": plan.to_dict(),
        "lock_sha256": plan.lock_sha256,
        "plan_sha256": plan.sha256,
        "reference_evidence_level": (
            REFERENCE_EVIDENCE if plan.include_reference else None
        ),
        "repository_id": OFFICIAL_ACT_REPOSITORY_ID,
        "repository_type": OFFICIAL_ACT_REPOSITORY_TYPE,
        "revision": OFFICIAL_ACT_REVISION,
        "schema_version": RECEIPT_SCHEMA,
    }


def _publish_receipt(path: Path, payload: bytes) -> None:
    if (
        path.parent.is_symlink()
        or not path.parent.is_dir()
        or path.parent.resolve() != path.parent.absolute()
    ):
        raise OfficialACTReleaseError("ACT install receipt parent is unsafe")
    if path.is_symlink():
        raise FileExistsError("ACT install receipt cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different ACT install receipt")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("concurrent ACT install receipt disagrees") from None
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_receipt(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise FileExistsError("ACT install receipt cannot be a symlink")
    if path.exists() and (not path.is_file() or path.read_bytes() != payload):
        raise FileExistsError("refusing to replace a different ACT install receipt")


def _receipt_target(
    root: Path, plan: OfficialACTInstallPlan, selected: Optional[Path]
) -> Path:
    if selected is None:
        relative = f"official_release_receipts/install-{plan.sha256[:16]}.json"
    else:
        candidate = Path(selected).expanduser().absolute()
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise OfficialACTReleaseError(
                "explicit receipt must remain inside artifact root"
            ) from error
    safe_relative(relative, "receipt path")
    return _prepare_real_parent(root, relative)


def _install_validated_plan(
    plan: OfficialACTInstallPlan,
    *,
    timeout_s: float = 120.0,
    receipt_path: Optional[Path] = None,
) -> OfficialACTInstallResult:
    """Install a plan already rebuilt from the frozen release lock."""

    if type(plan) is not OfficialACTInstallPlan:
        raise TypeError("plan must be an exact OfficialACTInstallPlan")
    if (
        not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be positive and finite")
    root = plan.artifact_root
    _prepare_artifact_root(root)
    receipt = _receipt_target(root, plan, receipt_path)
    payload = canonical_json_bytes(_receipt(plan))
    _preflight_receipt(receipt, payload)
    targets: dict[str, Path] = {}
    for item in plan.files:
        for relative in item.destinations:
            if relative in targets:
                raise OfficialACTReleaseError("duplicate planned destination")
            target = _prepare_real_parent(root, relative)
            targets[relative] = target
            _validated_existing(target, item)
    downloaded = published = reused = 0
    with tempfile.TemporaryDirectory(prefix=".official-act-release-", dir=root) as name:
        staging = Path(name)
        for index, item in enumerate(plan.files):
            existing = next(
                (
                    targets[path]
                    for path in item.destinations
                    if _validated_existing(targets[path], item)
                ),
                None,
            )
            source = existing
            if source is None:
                source = staging / f"{index:03d}.download"
                _stream_download(item, source, float(timeout_s))
                downloaded += 1
            for relative in item.destinations:
                if _publish_link(source, targets[relative], item):
                    published += 1
                else:
                    reused += 1
    _publish_receipt(receipt, payload)
    return OfficialACTInstallResult(
        downloaded,
        published,
        receipt,
        hashlib.sha256(payload).hexdigest(),
        reused,
    )


def _validate_frozen_plan(plan: OfficialACTInstallPlan) -> None:
    from robotactile_benchmark.integrations.act.official_release import (
        build_official_act_install_plan,
    )
    from robotactile_benchmark.integrations.act.official_release_contracts import (
        load_official_act_release_lock,
    )

    lock = load_official_act_release_lock()
    expected = build_official_act_install_plan(
        plan.artifact_root,
        lock=lock,
        tasks=plan.tasks,
        profiles=plan.profiles,
        include_reference=plan.include_reference,
    )
    if canonical_json_bytes(plan.to_dict()) != canonical_json_bytes(expected.to_dict()):
        raise OfficialACTReleaseError("install plan is not frozen-lock-derived")


def install_official_act_release(
    plan: OfficialACTInstallPlan,
    *,
    timeout_s: float = 120.0,
    receipt_path: Optional[Path] = None,
) -> OfficialACTInstallResult:
    """Validate against the frozen lock, then install without clobbering."""

    if type(plan) is not OfficialACTInstallPlan:
        raise TypeError("plan must be an exact OfficialACTInstallPlan")
    _validate_frozen_plan(plan)
    return _install_validated_plan(
        plan,
        timeout_s=timeout_s,
        receipt_path=receipt_path,
    )


__all__ = [
    "OfficialACTInstallPlan",
    "OfficialACTInstallResult",
    "PlannedOfficialACTFile",
    "install_official_act_release",
]
