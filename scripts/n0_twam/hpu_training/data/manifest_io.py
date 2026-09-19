"""No-clobber persistence for the train759 source and conversion receipts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import (
    SOURCE_MANIFEST_NAME,
    SOURCE_MANIFEST_PROTOCOL,
    SourceEpisode,
    build_source_manifest,
    canonical_json_sha256,
    discover_source_episodes,
)


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def atomic_json_no_clobber(path: Path, payload: object) -> None:
    """Create one JSON artifact atomically and refuse every overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_or_verify_json(path: Path, payload: object) -> None:
    """Make idempotent resume safe without silently replacing prior bytes."""

    if path.exists():
        if load_json_object(path) != payload:
            raise FileExistsError(f"existing artifact differs: {path}")
        return
    try:
        atomic_json_no_clobber(path, payload)
    except FileExistsError:
        if load_json_object(path) != payload:
            raise FileExistsError(f"existing artifact differs: {path}") from None


def _manifest_sha(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    actual = canonical_json_sha256(unsigned)
    if claimed != actual:
        raise ValueError("source split manifest SHA256 is invalid")
    return actual


def _saved_episode_map(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = payload.get("episodes")
    if not isinstance(raw, list):
        raise ValueError("source split manifest episodes must be a list")
    result: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise ValueError("source split manifest episode is invalid")
        relative_path = item["relative_path"]
        if relative_path in result:
            raise ValueError(f"duplicate source manifest path: {relative_path}")
        result[relative_path] = item
    return result


def _records_from_manifest(
    payload: Mapping[str, object], *, raw_root: Path
) -> tuple[SourceEpisode, ...]:
    if payload.get("protocol_id") != SOURCE_MANIFEST_PROTOCOL:
        raise ValueError("source split manifest protocol is unsupported")
    root = raw_root.resolve(strict=True)
    if payload.get("raw_root") != str(root):
        raise ValueError("source split manifest is bound to a different raw root")
    _manifest_sha(payload)
    saved = _saved_episode_map(payload)
    current = discover_source_episodes(root, hash_sources=False)
    if set(saved) != {record.relative_path for record in current}:
        raise ValueError("source HDF5 path set changed after split freeze")
    records: list[SourceEpisode] = []
    for observed in current:
        item = saved[observed.relative_path]
        expected = {
            "task": observed.task,
            "episode_id": observed.episode_id,
            "split": observed.split,
            "size_bytes": observed.size_bytes,
            "mtime_ns": observed.mtime_ns,
            "usable_source_range": (
                None
                if observed.usable_source_range is None
                else list(observed.usable_source_range)
            ),
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError(f"source identity changed: {observed.relative_path}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"source SHA256 is invalid: {observed.relative_path}")
        records.append(
            SourceEpisode(
                root=root,
                relative_path=observed.relative_path,
                task=observed.task,
                episode_id=observed.episode_id,
                split=observed.split,
                size_bytes=observed.size_bytes,
                mtime_ns=observed.mtime_ns,
                sha256=digest,
            )
        )
    rebuilt = build_source_manifest(records, raw_root=root)
    if rebuilt != payload:
        raise ValueError("source split manifest content does not match its records")
    return tuple(records)


def load_or_create_source_manifest(
    *, raw_root: Path, output_root: Path
) -> tuple[dict[str, object], tuple[SourceEpisode, ...]]:
    """Freeze all 800 identities once, then resume by cheap identity validation."""

    manifest_path = output_root / SOURCE_MANIFEST_NAME
    if manifest_path.exists():
        payload = load_json_object(manifest_path)
        return payload, _records_from_manifest(payload, raw_root=raw_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output root is non-empty but has no source split manifest: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    records = discover_source_episodes(raw_root, hash_sources=True)
    payload = build_source_manifest(records, raw_root=raw_root)
    atomic_json_no_clobber(manifest_path, payload)
    return payload, records


def receipt_sha256(payload: Mapping[str, object], *, field: str) -> str:
    unsigned = dict(payload)
    claimed = unsigned.pop(field, None)
    actual = canonical_json_sha256(unsigned)
    if claimed != actual:
        raise ValueError(f"receipt digest is invalid: {field}")
    return actual


def signed_payload(payload: Mapping[str, object], *, field: str) -> dict[str, object]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"digest field already exists: {field}")
    result[field] = canonical_json_sha256(result)
    return result


def select_train_records(
    records: Sequence[SourceEpisode], *, task: str
) -> tuple[SourceEpisode, ...]:
    selected = tuple(
        sorted(
            (
                record
                for record in records
                if record.task == task and record.split == "train"
            ),
            key=lambda item: item.episode_id,
        )
    )
    expected = 94 if task == "grasp_classify" else 95
    if len(selected) != expected:
        raise ValueError(f"{task} train count must be {expected}, got {len(selected)}")
    return selected


__all__ = [
    "atomic_json_no_clobber",
    "load_json_object",
    "load_or_create_source_manifest",
    "receipt_sha256",
    "select_train_records",
    "signed_payload",
    "write_or_verify_json",
]
