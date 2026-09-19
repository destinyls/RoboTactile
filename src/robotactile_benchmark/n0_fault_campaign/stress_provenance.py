"""Source snapshot and imported package binding for isolated stress campaigns."""

from __future__ import annotations

from pathlib import Path

from robotactile_benchmark.contracts import canonical_hash

from .io import file_sha256


def code_inventory(code_root: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    for directory, suffixes in (
        ("src", {".py"}),
        ("scripts", {".py"}),
        ("configs", {".json", ".yaml", ".yml"}),
        ("schemas", {".json"}),
        ("integrations", {".json"}),
    ):
        base = code_root / directory
        if not base.is_dir() or base.is_symlink():
            raise ValueError(f"missing regular snapshot directory: {directory}")
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                if path.is_symlink():
                    raise ValueError("source snapshot cannot contain linked members")
                members[path.relative_to(code_root).as_posix()] = file_sha256(path)
    if not members:
        raise ValueError("source snapshot is empty")
    return members


def code_sha256(code_root: Path) -> str:
    return canonical_hash(code_inventory(code_root))


def verify_runtime_code(
    code_root: Path, package_root: Path, expected_sha256: str
) -> None:
    inventory = code_inventory(code_root)
    if canonical_hash(inventory) != expected_sha256:
        raise ValueError("runtime code differs from frozen protocol")
    expected = {
        k.removeprefix("src/"): v
        for k, v in inventory.items()
        if k.startswith("src/robotactile_benchmark/")
    }
    package = package_root / "robotactile_benchmark"
    if not package.is_dir() or package.is_symlink():
        raise ValueError("runtime package directory is missing or linked")
    actual = {}
    for path in package.rglob("*.py"):
        if path.is_symlink():
            raise ValueError("runtime package member is linked")
        actual[path.relative_to(package_root).as_posix()] = file_sha256(path)
    if actual != expected:
        raise ValueError("imported package and frozen source snapshot differ")
