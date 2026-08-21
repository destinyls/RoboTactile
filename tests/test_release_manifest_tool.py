from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "release" / "update_source_manifest.py"


def _source_files() -> dict[str, str]:
    return {
        ".gitignore": "data/\noutputs/\n",
        "BENCHMARK_CARD.md": "# Benchmark card\n",
        "CITATION.cff": "cff-version: 1.2.0\n",
        "CONTRIBUTING.md": "# Contributing\n",
        "LICENSE": "license\n",
        "README.md": "readme\n",
        "SECURITY.md": "# Security\n",
        "THIRD_PARTY_NOTICES.md": "# Notices\n",
        "pyproject.toml": "[project]\nname = 'fixture'\n",
        "uv.lock": "version = 1\n",
        "configs/operators/core.json": "{}\n",
        "configs/source_manifest.sha256": "configuration payload\n",
        "requirements/README.md": "# pip locks\n",
        "requirements/core.lock.txt": "numpy==2.0.2 --hash=sha256:abc\n",
        "requirements/dev.lock.txt": "pytest==8.4.2 --hash=sha256:def\n",
        "schemas/result.schema.json": "{}\n",
        "src/example/__init__.py": '"""Example."""\n',
        "src/example/ignored.txt": "not shipped source\n",
        "tests/test_example.py": "def test_example():\n    assert True\n",
        "tests/helper.py": "HELPER = True\n",
        "docs/workflow.md": "# Workflow\n",
        "docs/diagram.svg": "<svg/>\n",
        "examples/act/request.json": "{}\n",
        "integrations/integrations.lock.json": "{}\n",
        "integrations/install.sh": "#!/bin/sh\nexit 0\n",
        "scripts/check.py": "raise SystemExit(0)\n",
        "scripts/deploy.sh": "#!/bin/sh\nexit 0\n",
        "scripts/README.md": "# Scripts\n",
        ".github/workflows/ci.yml": "name: ci\n",
    }


def _make_source_tree(root: Path) -> dict[str, str]:
    files = _source_files()
    for relative_path, payload in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    (root / "src" / "example" / "__pycache__").mkdir()
    (root / "src" / "example" / "__pycache__" / "cached.py").write_text(
        "ignored\n", encoding="utf-8"
    )
    (root / "scripts" / ".DS_Store").write_bytes(b"ignored")
    (root / "scripts" / "build" / "generated.py").parent.mkdir()
    (root / "scripts" / "build" / "generated.py").write_text(
        "ignored\n", encoding="utf-8"
    )
    return files


def _run(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(root), mode],
        check=False,
        capture_output=True,
        text=True,
    )


def _expected_manifest(files: dict[str, str]) -> bytes:
    included = {
        relative_path: payload
        for relative_path, payload in files.items()
        if relative_path
        in {
            ".gitignore",
            "BENCHMARK_CARD.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "README.md",
            "SECURITY.md",
            "THIRD_PARTY_NOTICES.md",
            "pyproject.toml",
            "uv.lock",
        }
        or relative_path.startswith(("configs/", "requirements/", "schemas/"))
        or (relative_path.startswith("src/") and relative_path.endswith(".py"))
        or (
            relative_path.startswith("tests/")
            and Path(relative_path).name.startswith("test_")
            and relative_path.endswith(".py")
        )
        or (relative_path.startswith("docs/") and relative_path.endswith(".md"))
        or (
            relative_path.startswith(("examples/", "integrations/", ".github/"))
            and Path(relative_path).suffix in {".json", ".md", ".sh", ".yaml", ".yml"}
        )
        or (
            relative_path.startswith("scripts/")
            and (
                relative_path.endswith((".py", ".sh"))
                or Path(relative_path).name in {"README", "README.md"}
            )
        )
    }
    lines = [
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {relative_path}\n"
        for relative_path, payload in sorted(included.items())
    ]
    return "".join(lines).encode("utf-8")


def test_write_emits_the_exact_sorted_inventory_and_is_byte_stable(
    tmp_path: Path,
) -> None:
    files = _make_source_tree(tmp_path)

    first = _run(tmp_path, "--write")

    assert first.returncode == 0, first.stderr
    manifest = tmp_path / "release" / "source_manifest.sha256"
    first_payload = manifest.read_bytes()
    assert first_payload == _expected_manifest(files)
    assert b"\r" not in first_payload

    second = _run(tmp_path, "--write")

    assert second.returncode == 0, second.stderr
    assert manifest.read_bytes() == first_payload
    checked = _run(tmp_path, "--check")
    assert checked.returncode == 0, checked.stderr


def test_check_detects_drift_without_changing_the_manifest(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    assert _run(tmp_path, "--write").returncode == 0
    manifest = tmp_path / "release" / "source_manifest.sha256"
    before = manifest.read_bytes()
    (tmp_path / "src" / "example" / "__init__.py").write_text(
        '"""Changed."""\n', encoding="utf-8"
    )

    checked = _run(tmp_path, "--check")

    assert checked.returncode == 1
    assert "out of date" in checked.stderr.lower()
    assert manifest.read_bytes() == before
    assert not tuple((tmp_path / "release").glob(".source_manifest.*"))


def test_check_missing_manifest_does_not_create_release_directory(
    tmp_path: Path,
) -> None:
    _make_source_tree(tmp_path)

    checked = _run(tmp_path, "--check")

    assert checked.returncode == 1
    assert not (tmp_path / "release").exists()


def test_inventory_rejects_a_symlink_that_can_escape_the_project(
    tmp_path: Path,
) -> None:
    _make_source_tree(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    link = tmp_path / "src" / "example" / "escaped.py"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    written = _run(tmp_path, "--write")

    assert written.returncode == 2
    assert "symlink" in written.stderr.lower()
    assert not (tmp_path / "release" / "source_manifest.sha256").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_inventory_rejects_non_regular_candidate(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    fifo = tmp_path / "scripts" / "broken.py"
    os.mkfifo(fifo)

    written = _run(tmp_path, "--write")

    assert written.returncode == 2
    assert "regular file" in written.stderr.lower()
    assert not (tmp_path / "release" / "source_manifest.sha256").exists()
