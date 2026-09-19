"""Source and package snapshot provenance checks."""

from pathlib import Path

import pytest

from robotactile_benchmark.n0_fault_campaign.stress_provenance import (
    code_sha256,
    verify_runtime_code,
)


def test_snapshot_hash_and_runtime_package_are_both_verified(tmp_path: Path):
    code, package = tmp_path / "code", tmp_path / "package"
    for name in (
        "src/robotactile_benchmark",
        "scripts",
        "configs",
        "schemas",
        "integrations",
    ):
        (code / name).mkdir(parents=True)
    (package / "robotactile_benchmark").mkdir(parents=True)
    src = code / "src/robotactile_benchmark/__init__.py"
    installed = package / "robotactile_benchmark/__init__.py"
    src.write_text("VALUE = 1\n")
    installed.write_text(src.read_text())
    digest = code_sha256(code)
    verify_runtime_code(code, package, digest)
    installed.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="package"):
        verify_runtime_code(code, package, digest)
    src.write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="code differs"):
        verify_runtime_code(code, package, digest)
