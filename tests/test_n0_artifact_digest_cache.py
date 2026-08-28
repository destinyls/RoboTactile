from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from robotactile_benchmark.integrations.n0_twam import artifacts


@pytest.fixture(autouse=True)
def _isolated_digest_cache() -> Iterator[None]:
    artifacts._file_sha256_cache_clear()
    yield
    artifacts._file_sha256_cache_clear()


def test_unchanged_file_reuses_digest_across_labels(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"stable model bytes")

    first = artifacts._file_sha256(artifact, "checkpoint")
    second = artifacts._file_sha256(artifact, "base component")

    assert first == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert second == first
    cache = artifacts._cached_file_sha256.cache_info()
    assert cache.misses == 1
    assert cache.hits == 1


def test_atomic_same_size_replacement_is_a_cache_miss(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    artifact.write_bytes(b"before")
    replacement.write_bytes(b"after!")

    before = artifacts._file_sha256(artifact, "checkpoint")
    os.replace(replacement, artifact)
    after = artifacts._file_sha256(artifact, "checkpoint")

    assert before != after
    cache = artifacts._cached_file_sha256.cache_info()
    assert cache.misses == 2
    assert cache.hits == 0


def test_cached_path_is_still_rejected_after_becoming_symlink(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.safetensors"
    target = tmp_path / "target.safetensors"
    artifact.write_bytes(b"original")
    target.write_bytes(b"target!!")
    artifacts._file_sha256(artifact, "checkpoint")
    artifact.unlink()
    artifact.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink regular file"):
        artifacts._file_sha256(artifact, "checkpoint")


def test_cache_hit_rechecks_path_identity_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    artifact.write_bytes(b"before")
    replacement.write_bytes(b"after!")
    artifacts._file_sha256(artifact, "checkpoint")
    cached_digest = artifacts._cached_file_sha256

    def replacing_cache_hit(*args: object) -> str:
        digest = cached_digest(*args)
        os.replace(replacement, artifact)
        return digest

    monkeypatch.setattr(artifacts, "_cached_file_sha256", replacing_cache_hit)

    with pytest.raises(ValueError, match="changed while validating"):
        artifacts._file_sha256(artifact, "checkpoint")


def test_atomic_replacement_during_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    artifact.write_bytes(b"a" * (2 * 1024 * 1024))
    replacement.write_bytes(b"b" * (2 * 1024 * 1024))
    original_read = artifacts.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            os.replace(replacement, artifact)
            replaced = True
        return chunk

    monkeypatch.setattr(artifacts.os, "read", replacing_read)

    with pytest.raises(ValueError, match="changed while hashing"):
        artifacts._file_sha256(artifact, "checkpoint")
