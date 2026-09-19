#!/usr/bin/env python3
"""Create an exact source sync list and a no-transfer asset inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXCLUDED_TOP_LEVEL = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "data",
    "deployment",
    "dist",
    "outputs",
    "plan",
    "temp",
}
EXCLUDED_DIRECTORY_NAMES = {"__pycache__"}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    ".env",
    "credentials.json",
    "settings.json",
}
SENSITIVE_SUFFIXES = {".key", ".pem"}
ASSET_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".mp4",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".zip",
}
HASH_LIMIT_BYTES = 64 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _is_secret(path: Path) -> bool:
    return (
        path.name in EXCLUDED_FILE_NAMES
        or path.name.startswith(".env.")
        or path.suffix.lower() in SENSITIVE_SUFFIXES
    )


def _source_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in EXCLUDED_DIRECTORY_NAMES
            and not (relative_current == Path(".") and directory in EXCLUDED_TOP_LEVEL)
        )
        for filename in sorted(filenames):
            path = current_path / filename
            relative = path.relative_to(root)
            if _is_secret(path) or path.suffix.lower() in ASSET_SUFFIXES:
                continue
            if path.is_symlink():
                target = os.readlink(path)
                entries.append(
                    {
                        "path": relative.as_posix(),
                        "type": "symlink",
                        "target": target,
                        "sha256": hashlib.sha256(target.encode()).hexdigest(),
                    }
                )
            elif path.is_file():
                entries.append(
                    {
                        "path": relative.as_posix(),
                        "type": "file",
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    return entries


def _asset_entries(root: Path) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    for top_level in ("data", "deployment", "outputs", "temp"):
        base = root / top_level
        if not base.exists():
            continue
        for current, directories, filenames in os.walk(base, followlinks=False):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in EXCLUDED_DIRECTORY_NAMES
                and directory not in {".git", ".venv", "runtime"}
            )
            for filename in filenames:
                path = Path(current) / filename
                if _is_secret(path):
                    continue
                if (
                    top_level == "data"
                    or path.suffix.lower() in ASSET_SUFFIXES
                    or path.stat().st_size >= HASH_LIMIT_BYTES
                ):
                    candidates.add(path)
    entries: list[dict[str, Any]] = []
    for path in sorted(candidates):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": _sha256(path) if size <= HASH_LIMIT_BYTES else None,
                "sha256_omitted_reason": None
                if size <= HASH_LIMIT_BYTES
                else "file_exceeds_64_mib_inventory_hash_limit",
                "transferred": False,
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_entries = _source_entries(root)
    assets = _asset_entries(root)
    git_head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_count = len(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    source_manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "repository_root": str(root),
        "git_head": git_head,
        "working_tree_dirty_entry_count": dirty_count,
        "file_count": len(source_entries),
        "total_bytes": sum(int(entry.get("size_bytes", 0)) for entry in source_entries),
        "selection": {
            "excluded_top_level": sorted(EXCLUDED_TOP_LEVEL),
            "excluded_asset_suffixes": sorted(ASSET_SUFFIXES),
            "sensitive_files_included": False,
        },
        "files": source_entries,
    }
    canonical_files = json.dumps(
        source_entries, sort_keys=True, separators=(",", ":")
    ).encode()
    source_manifest["source_tree_sha256"] = hashlib.sha256(canonical_files).hexdigest()
    asset_inventory = {
        "schema_version": 1,
        "generated_at": generated_at,
        "evidence_level": "INVENTORY",
        "transferred": False,
        "file_count": len(assets),
        "total_bytes": sum(entry["size_bytes"] for entry in assets),
        "files": assets,
    }
    _write_json(output_dir / "source_manifest.json", source_manifest)
    _write_json(output_dir / "excluded_assets_inventory.json", asset_inventory)
    (output_dir / "sync_files.txt").write_text(
        "".join(f"{entry['path']}\n" for entry in source_entries),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_files": len(source_entries),
                "source_bytes": source_manifest["total_bytes"],
                "asset_files": len(assets),
                "asset_bytes": asset_inventory["total_bytes"],
                "source_tree_sha256": source_manifest["source_tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
