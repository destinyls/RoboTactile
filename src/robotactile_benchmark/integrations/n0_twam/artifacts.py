"""Content-addressed official N0-TWAM UniVTAC serving artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock

SCHEMA_VERSION = "robotactile-n0-official-artifact-v3"
BASE_REPOSITORY = "NeoteAI/n0-twam-base"
BASE_REVISION = "dafcb053902cc7a51310780fdfdb71ff8e79c0be"
CHECKPOINT_REPOSITORY = "NeoteAI/n0-twam-univtac-delta"
CHECKPOINT_REVISION = "7694e63707a8c9e69e1a1242c4ed74ee39b7bb51"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST_CACHE_ENV = "ROBOTACTILE_N0_DIGEST_CACHE_DIR"
_DIGEST_CACHE_SCHEMA = "robotactile-n0-digest-cache-v1"
_SERVE_TASK_IDS = {
    "grasp_classify": "univtac_grasp_classify_hdf5_current",
    "insert_HDMI": "univtac_insert_HDMI_rot6d_current",
    "insert_hole": "univtac_insert_hole_rot6d_current",
    "insert_tube": "univtac_insert_tube_rot6d_current",
    "lift_bottle": "univtac_lift_bottle_rot6d_current",
    "lift_can": "univtac_lift_can_rot6d_current",
    "pull_out_key": "univtac_pull_out_key_rot6d_current",
    "put_bottle_in_shelf": "univtac_put_bottle_in_shelf_rot6d_current",
}
_TASK_IDS = frozenset(_SERVE_TASK_IDS)
_FIELDS = frozenset(
    {
        "action_mode",
        "base_repository",
        "base_revision",
        "base_root",
        "bundle_root",
        "checkpoint_path",
        "checkpoint_repository",
        "checkpoint_revision",
        "checkpoint_root",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "external_commit",
        "normalizer_path",
        "normalizer_sha256",
        "prompt_manifest_path",
        "prompt_manifest_sha256",
        "schema_version",
        "serve_bundle_manifest_path",
        "serve_bundle_sha256",
        "serve_bundle_root",
        "serve_info_path",
        "serve_info_sha256",
        "serve_pool_root",
        "serve_tasks_path",
        "serve_tasks_sha256",
        "serve_task_id",
        "task_id",
        "train_meta_path",
        "train_meta_sha256",
    }
)


def serve_task_id(task_id: str) -> str:
    """Map a benchmark task to the released checkpoint's training repo ID."""

    try:
        return _SERVE_TASK_IDS[task_id]
    except KeyError as error:
        raise ValueError(f"unsupported official N0 UniVTAC task: {task_id}") from error


def _absolute(value: Path, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path")
    return value.absolute()


def _inside(root: Path, path: Path, name: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(f"{name} must be below bundle_root")


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    """Return content-relevant identity stable across distributed filesystems."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _digest_cache_root() -> Path | None:
    raw = os.environ.get(_DIGEST_CACHE_ENV)
    if raw is None:
        return None
    if not raw or "\n" in raw:
        raise ValueError(f"{_DIGEST_CACHE_ENV} must be a non-empty absolute path")
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError(f"{_DIGEST_CACHE_ENV} must be an absolute path")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{_DIGEST_CACHE_ENV} must be a real directory")
    metadata = root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(f"{_DIGEST_CACHE_ENV} must be private and owned by this user")
    return root


def _digest_cache_receipt_path(root: Path, absolute_path: str) -> Path:
    key = hashlib.sha256(absolute_path.encode("utf-8")).hexdigest()
    return root / f"{key}.json"


@contextmanager
def _digest_cache_lock(root: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root / ".lock", flags, 0o600)
    except OSError as error:
        raise ValueError("N0-TWAM digest cache lock is unavailable") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _digest_cache_document(
    absolute_path: str,
    identity: tuple[int, int, int, int],
    digest: str,
) -> dict[str, object]:
    return {
        "absolute_path": absolute_path,
        "identity": {
            "st_dev": identity[0],
            "st_ino": identity[1],
            "st_size": identity[2],
            "st_mtime_ns": identity[3],
        },
        "schema_version": _DIGEST_CACHE_SCHEMA,
        "sha256": digest,
    }


def _load_digest_cache_receipt(
    path: Path,
    absolute_path: str,
    identity: tuple[int, int, int, int],
) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = strict_json_bytes(path.read_bytes(), "N0 digest cache receipt")
    except (OSError, ValueError):
        return None
    if not isinstance(document, Mapping) or set(document) != {
        "absolute_path",
        "identity",
        "schema_version",
        "sha256",
    }:
        return None
    recorded_identity = document["identity"]
    if not isinstance(recorded_identity, Mapping) or set(recorded_identity) != {
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
    }:
        return None
    identity_values = tuple(
        recorded_identity[name]
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    )
    digest = document["sha256"]
    if (
        document["schema_version"] != _DIGEST_CACHE_SCHEMA
        or document["absolute_path"] != absolute_path
        or any(type(value) is not int for value in identity_values)
        or identity_values != identity
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        return None
    return digest


def _write_digest_cache_receipt(path: Path, document: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(document)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _persistent_file_sha256(
    absolute_path: str,
    identity: tuple[int, int, int, int],
    root: Path,
) -> str:
    receipt = _digest_cache_receipt_path(root, absolute_path)
    with _digest_cache_lock(root):
        selected = Path(absolute_path)
        try:
            current = selected.lstat()
        except OSError as error:
            raise ValueError(
                f"N0-TWAM artifact changed while hashing: {absolute_path}"
            ) from error
        if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != identity:
            raise ValueError(f"N0-TWAM artifact changed while hashing: {absolute_path}")
        cached = _load_digest_cache_receipt(receipt, absolute_path, identity)
        if cached is not None:
            return cached
        digest = _cached_file_sha256(absolute_path, *identity)
        _write_digest_cache_receipt(
            receipt, _digest_cache_document(absolute_path, identity, digest)
        )
        return digest


@lru_cache(maxsize=None)
def _cached_file_sha256(
    absolute_path: str,
    st_dev: int,
    st_ino: int,
    st_size: int,
    st_mtime_ns: int,
) -> str:
    expected_identity = (st_dev, st_ino, st_size, st_mtime_ns)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute_path, flags)
    except OSError as error:
        raise ValueError(
            f"N0-TWAM artifact changed while hashing: {absolute_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _stat_identity(before) != expected_identity
        ):
            raise ValueError(f"N0-TWAM artifact changed while hashing: {absolute_path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(after) != expected_identity:
            raise ValueError(f"N0-TWAM artifact changed while hashing: {absolute_path}")
    finally:
        os.close(descriptor)
    try:
        current = Path(absolute_path).lstat()
    except OSError as error:
        raise ValueError(
            f"N0-TWAM artifact changed while hashing: {absolute_path}"
        ) from error
    if (
        not stat.S_ISREG(current.st_mode)
        or _stat_identity(current) != expected_identity
    ):
        raise ValueError(f"N0-TWAM artifact changed while hashing: {absolute_path}")
    return digest.hexdigest()


def _file_sha256_cache_clear() -> None:
    """Clear process-local digest state for isolated tests."""

    _cached_file_sha256.cache_clear()


def _file_sha256(path: Path, name: str) -> str:
    selected = Path(path).absolute()
    try:
        metadata = selected.lstat()
    except OSError as error:
        raise ValueError(
            f"N0-TWAM {name} must be a non-symlink regular file"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"N0-TWAM {name} must be a non-symlink regular file")
    identity = _stat_identity(metadata)
    cache_root = _digest_cache_root()
    if cache_root is None:
        digest = _cached_file_sha256(str(selected), *identity)
    else:
        digest = _persistent_file_sha256(str(selected), identity, cache_root)
    try:
        current = selected.lstat()
    except OSError as error:
        raise ValueError(
            f"N0-TWAM {name} changed while validating its cached digest"
        ) from error
    if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != identity:
        raise ValueError(f"N0-TWAM {name} changed while validating its cached digest")
    return digest


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _component_file_paths(name: str, source: Path) -> dict[str, Path]:
    if source.is_symlink():
        raise ValueError(f"N0 base component cannot be a symlink: {source}")
    if source.is_file():
        return {name: source}
    if not source.is_dir():
        raise ValueError(f"N0 base component is unavailable: {source}")
    files: dict[str, Path] = {}
    for current, directories, filenames in os.walk(source, followlinks=False):
        current_path = Path(current)
        if any((current_path / item).is_symlink() for item in directories):
            raise ValueError("N0 base component cannot contain directory symlinks")
        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("N0 base component must contain regular files")
            files[f"{name}/{path.relative_to(source).as_posix()}"] = path
    if not files:
        raise ValueError(f"N0 base component is empty: {source}")
    return files


def _validate_serve_bundle(manifest: "N0TWAMArtifactManifest") -> None:
    document = strict_json_bytes(
        manifest.serve_bundle_manifest_path.read_bytes(), "N0 serve bundle manifest"
    )
    required = {
        "action_mode",
        "base_files",
        "base_repository",
        "base_revision",
        "checkpoint_repository",
        "checkpoint_revision",
        "components",
        "normalizer_path",
        "prompt",
        "schema_version",
        "serve_pool_root",
        "serve_task_id",
        "task_id",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise ValueError("N0 serve bundle manifest fields mismatch")
    expected_scalars = {
        "action_mode": "delta",
        "base_repository": BASE_REPOSITORY,
        "base_revision": BASE_REVISION,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "normalizer_path": str(manifest.normalizer_path),
        "schema_version": "robotactile-n0-serve-bundle-v2",
        "serve_pool_root": str(manifest.serve_pool_root),
        "serve_task_id": manifest.serve_task_id,
        "task_id": manifest.task_id,
    }
    if any(document[name] != value for name, value in expected_scalars.items()):
        raise ValueError("N0 serve bundle identity mismatch")
    components = document["components"]
    if not isinstance(components, Mapping):
        raise ValueError("N0 serve components must be a mapping")
    expected_components = {
        "transformer": manifest.checkpoint_root / "transformer",
        "vae": manifest.base_root / "vae",
        "tokenizer": manifest.base_root / "tokenizer",
        "text_encoder": manifest.base_root / "text_encoder",
    }
    for optional in ("assets", "empty_emb.pt"):
        path = manifest.base_root / optional
        if path.exists():
            expected_components[optional] = path
    if set(components) != set(expected_components):
        raise ValueError("N0 serve component inventory mismatch")
    for name, source in expected_components.items():
        resolved = source.resolve(strict=True)
        if components[name] != str(resolved):
            raise ValueError(f"N0 serve component source mismatch: {name}")
        link = manifest.serve_bundle_root / name
        if not link.is_symlink() or link.resolve(strict=True) != resolved:
            raise ValueError(f"N0 serve component link mismatch: {name}")

    expected_files: dict[str, Path] = {}
    for name, source in expected_components.items():
        if name != "transformer":
            expected_files.update(_component_file_paths(name, source))
    declared_files = document["base_files"]
    if not isinstance(declared_files, Mapping) or set(declared_files) != set(
        expected_files
    ):
        raise ValueError("N0 base component file inventory mismatch")
    for relative, path in expected_files.items():
        entry = declared_files[relative]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "size"}
            or entry["size"] != path.stat().st_size
            or entry["sha256"] != _file_sha256(path, relative)
        ):
            raise ValueError(f"N0 base component hash mismatch: {relative}")


@dataclass(frozen=True)
class N0TWAMArtifactManifest:
    """Frozen source, Hub revisions, paths, and hashes for one served task."""

    bundle_root: Path
    base_root: Path
    checkpoint_root: Path
    serve_bundle_root: Path
    serve_pool_root: Path
    checkpoint_path: Path
    config_path: Path
    train_meta_path: Path
    normalizer_path: Path
    prompt_manifest_path: Path
    serve_bundle_manifest_path: Path
    serve_info_path: Path
    serve_tasks_path: Path
    checkpoint_sha256: str
    config_sha256: str
    train_meta_sha256: str
    normalizer_sha256: str
    prompt_manifest_sha256: str
    serve_bundle_sha256: str
    serve_info_sha256: str
    serve_tasks_sha256: str
    task_id: str
    serve_task_id: str
    external_commit: str
    base_repository: str = BASE_REPOSITORY
    base_revision: str = BASE_REVISION
    checkpoint_repository: str = CHECKPOINT_REPOSITORY
    checkpoint_revision: str = CHECKPOINT_REVISION
    action_mode: str = "delta"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        root = _absolute(self.bundle_root, "bundle_root")
        object.__setattr__(self, "bundle_root", root)
        for name in (
            "base_root",
            "checkpoint_root",
            "serve_bundle_root",
            "serve_pool_root",
            "checkpoint_path",
            "config_path",
            "train_meta_path",
            "normalizer_path",
            "prompt_manifest_path",
            "serve_bundle_manifest_path",
            "serve_info_path",
            "serve_tasks_path",
        ):
            selected = _absolute(getattr(self, name), name)
            _inside(root, selected, name)
            object.__setattr__(self, name, selected)
        if self.schema_version != SCHEMA_VERSION or self.action_mode != "delta":
            raise ValueError("N0-TWAM artifact version/action mode mismatch")
        if (
            self.base_repository != BASE_REPOSITORY
            or self.base_revision != BASE_REVISION
            or self.checkpoint_repository != CHECKPOINT_REPOSITORY
            or self.checkpoint_revision != CHECKPOINT_REVISION
        ):
            raise ValueError("official N0-TWAM Hub identity mismatch")
        if _COMMIT.fullmatch(self.external_commit) is None:
            raise ValueError("external_commit must be lowercase 40-character hex")
        expected_commit = load_integration_lock().by_id("n0_twam").commit_sha
        if self.external_commit != expected_commit:
            raise ValueError("N0-TWAM external commit mismatch")
        if self.task_id not in _TASK_IDS:
            raise ValueError("unsupported official N0-TWAM task")
        if self.serve_task_id != serve_task_id(self.task_id):
            raise ValueError("official N0-TWAM serve task ID mismatch")
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "train_meta_sha256",
            "normalizer_sha256",
            "prompt_manifest_sha256",
            "serve_bundle_sha256",
            "serve_info_sha256",
            "serve_tasks_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_mode": self.action_mode,
            "base_repository": self.base_repository,
            "base_revision": self.base_revision,
            "base_root": str(self.base_root),
            "bundle_root": str(self.bundle_root),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_repository": self.checkpoint_repository,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_root": str(self.checkpoint_root),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "external_commit": self.external_commit,
            "normalizer_path": str(self.normalizer_path),
            "normalizer_sha256": self.normalizer_sha256,
            "prompt_manifest_path": str(self.prompt_manifest_path),
            "prompt_manifest_sha256": self.prompt_manifest_sha256,
            "schema_version": self.schema_version,
            "serve_bundle_manifest_path": str(self.serve_bundle_manifest_path),
            "serve_bundle_sha256": self.serve_bundle_sha256,
            "serve_bundle_root": str(self.serve_bundle_root),
            "serve_info_path": str(self.serve_info_path),
            "serve_info_sha256": self.serve_info_sha256,
            "serve_pool_root": str(self.serve_pool_root),
            "serve_tasks_path": str(self.serve_tasks_path),
            "serve_tasks_sha256": self.serve_tasks_sha256,
            "serve_task_id": self.serve_task_id,
            "task_id": self.task_id,
            "train_meta_path": str(self.train_meta_path),
            "train_meta_sha256": self.train_meta_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "N0TWAMArtifactManifest":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("N0-TWAM artifact manifest fields mismatch")
        document: dict[str, Any] = dict(value)
        for name in (
            "bundle_root",
            "base_root",
            "checkpoint_root",
            "serve_bundle_root",
            "serve_pool_root",
            "checkpoint_path",
            "config_path",
            "train_meta_path",
            "normalizer_path",
            "prompt_manifest_path",
            "serve_bundle_manifest_path",
            "serve_info_path",
            "serve_tasks_path",
        ):
            if not isinstance(document[name], str):
                raise TypeError(f"{name} must be a string")
            document[name] = Path(document[name])
        return cls(**cast(dict[str, Any], document))


def build_n0_twam_artifact_manifest(
    *,
    bundle_root: Path,
    task_id: str,
    base_root: Path,
    checkpoint_root: Path,
    serve_bundle_root: Path,
    serve_pool_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    train_meta_path: Path,
    normalizer_path: Path,
    prompt_manifest_path: Path,
    serve_bundle_manifest_path: Path,
    serve_info_path: Path,
    serve_tasks_path: Path,
) -> N0TWAMArtifactManifest:
    """Hash official serving resources without importing its Torch runtime."""

    paths = {
        name: Path(value).absolute()
        for name, value in {
            "bundle_root": bundle_root,
            "base_root": base_root,
            "checkpoint_root": checkpoint_root,
            "serve_bundle_root": serve_bundle_root,
            "serve_pool_root": serve_pool_root,
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "train_meta_path": train_meta_path,
            "normalizer_path": normalizer_path,
            "prompt_manifest_path": prompt_manifest_path,
            "serve_bundle_manifest_path": serve_bundle_manifest_path,
            "serve_info_path": serve_info_path,
            "serve_tasks_path": serve_tasks_path,
        }.items()
    }
    return N0TWAMArtifactManifest(
        bundle_root=paths["bundle_root"],
        base_root=paths["base_root"],
        checkpoint_root=paths["checkpoint_root"],
        serve_bundle_root=paths["serve_bundle_root"],
        serve_pool_root=paths["serve_pool_root"],
        checkpoint_path=paths["checkpoint_path"],
        config_path=paths["config_path"],
        train_meta_path=paths["train_meta_path"],
        normalizer_path=paths["normalizer_path"],
        prompt_manifest_path=paths["prompt_manifest_path"],
        serve_bundle_manifest_path=paths["serve_bundle_manifest_path"],
        serve_info_path=paths["serve_info_path"],
        serve_tasks_path=paths["serve_tasks_path"],
        checkpoint_sha256=_file_sha256(paths["checkpoint_path"], "checkpoint"),
        config_sha256=_file_sha256(paths["config_path"], "config"),
        train_meta_sha256=_file_sha256(paths["train_meta_path"], "train metadata"),
        normalizer_sha256=_file_sha256(paths["normalizer_path"], "normalizer"),
        prompt_manifest_sha256=_file_sha256(
            paths["prompt_manifest_path"], "prompt manifest"
        ),
        serve_bundle_sha256=_file_sha256(
            paths["serve_bundle_manifest_path"], "serve bundle manifest"
        ),
        serve_info_sha256=_file_sha256(paths["serve_info_path"], "serve info"),
        serve_tasks_sha256=_file_sha256(paths["serve_tasks_path"], "serve tasks"),
        task_id=task_id,
        serve_task_id=serve_task_id(task_id),
        external_commit=load_integration_lock().by_id("n0_twam").commit_sha,
    )


def validate_n0_twam_artifact(
    manifest: N0TWAMArtifactManifest,
) -> Mapping[str, str]:
    if type(manifest) is not N0TWAMArtifactManifest:
        raise TypeError("manifest must be an exact N0TWAMArtifactManifest")
    directories = (
        manifest.base_root,
        manifest.checkpoint_root,
        manifest.serve_bundle_root,
        manifest.serve_pool_root,
    )
    if any(path.is_symlink() or not path.is_dir() for path in directories):
        raise ValueError("N0-TWAM artifact roots must be real directories")
    resources = (
        (manifest.checkpoint_path, manifest.checkpoint_sha256, "checkpoint"),
        (manifest.config_path, manifest.config_sha256, "config"),
        (manifest.train_meta_path, manifest.train_meta_sha256, "train metadata"),
        (manifest.normalizer_path, manifest.normalizer_sha256, "normalizer"),
        (
            manifest.prompt_manifest_path,
            manifest.prompt_manifest_sha256,
            "prompt manifest",
        ),
        (
            manifest.serve_bundle_manifest_path,
            manifest.serve_bundle_sha256,
            "serve bundle manifest",
        ),
        (manifest.serve_info_path, manifest.serve_info_sha256, "serve info"),
        (manifest.serve_tasks_path, manifest.serve_tasks_sha256, "serve tasks"),
    )
    for path, expected, name in resources:
        if _file_sha256(path, name) != expected:
            raise ValueError(f"N0-TWAM {name} SHA256 mismatch")
    _validate_serve_bundle(manifest)
    return {
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": manifest.config_sha256,
        "external_commit": manifest.external_commit,
        "normalizer_sha256": manifest.normalizer_sha256,
        "prompt_manifest_sha256": manifest.prompt_manifest_sha256,
        "serve_bundle_sha256": manifest.serve_bundle_sha256,
        "serve_info_sha256": manifest.serve_info_sha256,
        "serve_tasks_sha256": manifest.serve_tasks_sha256,
    }


def load_n0_twam_artifact_manifest(path: Path) -> N0TWAMArtifactManifest:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("N0-TWAM artifact manifest must be a regular file")
    raw = selected.read_bytes()
    value = strict_json_bytes(raw, "N0-TWAM artifact manifest")
    manifest = N0TWAMArtifactManifest.from_dict(value)
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise ValueError("N0-TWAM artifact manifest is not canonical")
    return manifest


__all__ = [
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_REVISION",
    "N0TWAMArtifactManifest",
    "SCHEMA_VERSION",
    "build_n0_twam_artifact_manifest",
    "load_n0_twam_artifact_manifest",
    "serve_task_id",
    "validate_n0_twam_artifact",
]
