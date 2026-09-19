"""Checks for the hash-locked optional visualization environment."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _pillow_uv_block() -> str:
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    start = uv_lock.index('[[package]]\nname = "pillow"\n')
    end = uv_lock.index("\n[[package]]", start + 1)
    return uv_lock[start:end]


def test_visualization_lock_matches_pillow_uv_entry() -> None:
    lock_path = ROOT / "requirements/visualization.lock.txt"
    lock = lock_path.read_text(encoding="utf-8")
    requirement_lines = tuple(
        line.rstrip() for line in lock.splitlines() if line and not line.startswith("#")
    )

    assert requirement_lines[0] == "Pillow==11.3.0 \\"
    assert all(line.startswith("    --hash=sha256:") for line in requirement_lines[1:])
    assert all(line.endswith(" \\") for line in requirement_lines[1:-1])
    assert not requirement_lines[-1].endswith("\\")

    lock_hashes = SHA256_PATTERN.findall(lock)
    uv_hashes = SHA256_PATTERN.findall(_pillow_uv_block())
    assert lock_hashes
    assert len(lock_hashes) == len(set(lock_hashes))
    assert set(lock_hashes) == set(uv_hashes)


def test_visualization_docs_use_reproducible_noneditable_install() -> None:
    requirements_doc = (ROOT / "requirements/README.md").read_text(encoding="utf-8")
    isaac_doc = (ROOT / "docs/isaac_sim.md").read_text(encoding="utf-8")
    combined = requirements_doc + isaac_doc

    assert "requirements/visualization.lock.txt" in requirements_doc
    assert "requirements/visualization.lock.txt" in isaac_doc
    assert "--only-binary=:all: --require-hashes" in combined
    assert "--no-deps --no-build-isolation ." in combined
    assert "pip install -e" not in combined
    assert "robotactile_benchmark-*.whl" not in requirements_doc
    assert "robotactile_benchmark-0.6.0-py3-none-any.whl" in requirements_doc
