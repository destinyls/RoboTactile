#!/usr/bin/env python3
"""Generate or verify the deterministic RoboTactile source manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

ROOT_FILES: Tuple[str, ...] = (
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
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", "__pycache__", "build", "dist", "output", "outputs"}
)
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
MANIFEST_RELATIVE_PATH = Path("release") / "source_manifest.sha256"


class ManifestError(ValueError):
    """Raised when the source inventory cannot be constructed safely."""


def _all_files(_: Path) -> bool:
    return True


def _python_file(path: Path) -> bool:
    return path.suffix == ".py"


def _test_file(path: Path) -> bool:
    return path.name.startswith("test_") and path.suffix == ".py"


def _markdown_file(path: Path) -> bool:
    return path.suffix == ".md"


def _script_file(path: Path) -> bool:
    return path.suffix in {".py", ".sh"} or path.name in {"README", "README.md"}


def _release_file(path: Path) -> bool:
    return path.suffix in {".json", ".md", ".sh", ".yaml", ".yml"}


INVENTORY_RULES: Tuple[Tuple[str, Callable[[Path], bool]], ...] = (
    ("configs", _all_files),
    ("requirements", _all_files),
    ("schemas", _all_files),
    ("src", _python_file),
    ("tests", _test_file),
    ("docs", _markdown_file),
    ("examples", _release_file),
    ("integrations", _release_file),
    ("scripts", _script_file),
    (".github", _release_file),
)


def _project_root(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ManifestError(f"project root must not be a symlink: {lexical}")
    if not lexical.exists() or not lexical.is_dir():
        raise ManifestError(f"project root is not a directory: {lexical}")
    return lexical.resolve(strict=True)


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ManifestError(f"path escapes project root: {path}") from error
    if not relative or "\n" in relative or "\r" in relative:
        raise ManifestError(f"unsafe inventory path: {relative!r}")
    try:
        relative.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ManifestError(
            f"inventory path is not valid UTF-8: {relative!r}"
        ) from error
    return relative


def _validate_regular_file(root: Path, path: Path) -> str:
    if path.is_symlink():
        raise ManifestError(f"inventory candidate must not be a symlink: {path}")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise ManifestError(f"required inventory file is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ManifestError(f"inventory candidate is not a regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ManifestError(
            f"inventory candidate escapes project root: {path}"
        ) from error
    return _relative_path(root, path)


def _walk_inventory(
    root: Path,
    directory_name: str,
    accepts: Callable[[Path], bool],
) -> Iterable[Tuple[str, Path]]:
    directory = root / directory_name
    if directory.is_symlink():
        raise ManifestError(f"inventory directory must not be a symlink: {directory}")
    if not directory.exists() or not directory.is_dir():
        raise ManifestError(f"inventory directory is missing: {directory}")
    for current, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        retained_directories = []
        for name in sorted(directory_names):
            child = current_path / name
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            if child.is_symlink():
                raise ManifestError(
                    f"inventory directory must not be a symlink: {child}"
                )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            if name in EXCLUDED_FILE_NAMES:
                continue
            path = current_path / name
            relative_to_inventory = path.relative_to(directory)
            if accepts(relative_to_inventory):
                yield _validate_regular_file(root, path), path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(project_root: Path) -> bytes:
    """Return the canonical UTF-8/LF manifest for ``project_root``."""

    root = _project_root(project_root)
    entries = []
    for relative_path in ROOT_FILES:
        path = root / relative_path
        entries.append((_validate_regular_file(root, path), path))
    for directory_name, accepts in INVENTORY_RULES:
        entries.extend(_walk_inventory(root, directory_name, accepts))
    paths = [relative_path for relative_path, _ in entries]
    if len(paths) != len(set(paths)):
        raise ManifestError("source inventory contains duplicate paths")
    lines = [
        f"{_sha256(path)}  {relative_path}\n"
        for relative_path, path in sorted(entries, key=lambda item: item[0])
    ]
    return "".join(lines).encode("utf-8", errors="strict")


def _manifest_path(root: Path, *, create_parent: bool) -> Path:
    release_directory = root / MANIFEST_RELATIVE_PATH.parent
    if release_directory.exists():
        if release_directory.is_symlink():
            raise ManifestError(
                f"release directory must not be a symlink: {release_directory}"
            )
        if not release_directory.is_dir():
            raise ManifestError(f"release path is not a directory: {release_directory}")
    elif create_parent:
        release_directory.mkdir(mode=0o755)
    target = root / MANIFEST_RELATIVE_PATH
    if target.exists() or target.is_symlink():
        _validate_regular_file(root, target)
    _relative_path(root, target)
    return target


def check_manifest(project_root: Path) -> bool:
    """Return whether the checked-in manifest exactly matches the source tree."""

    root = _project_root(project_root)
    expected = build_manifest(root)
    target = _manifest_path(root, create_parent=False)
    if not target.exists():
        return False
    return target.read_bytes() == expected


def _atomic_replace(target: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".source_manifest.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest(project_root: Path) -> bool:
    """Atomically write the manifest, returning whether bytes changed."""

    root = _project_root(project_root)
    payload = build_manifest(root)
    target = _manifest_path(root, create_parent=True)
    if target.exists() and target.read_bytes() == payload:
        return False
    _atomic_replace(target, payload)
    return True


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="benchmark project root (defaults to the script's project)",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="verify without writing")
    modes.add_argument(
        "--write", action="store_true", help="atomically update manifest"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the manifest command-line interface."""

    arguments = _argument_parser().parse_args(argv)
    try:
        root = _project_root(arguments.project_root)
        target = root / MANIFEST_RELATIVE_PATH
        if arguments.check:
            if check_manifest(root):
                sys.stdout.write(f"source manifest is current: {target}\n")
                return 0
            sys.stderr.write(f"source manifest is missing or out of date: {target}\n")
            return 1
        changed = write_manifest(root)
    except (ManifestError, OSError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    state = "updated" if changed else "already current"
    sys.stdout.write(f"source manifest {state}: {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
