"""Apply the bounded Dream-Tac HCU source overlay to an isolated checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .overlay_contract import (
    ATTENTION_RELATIVE_PATH,
    CLAIM_BOUNDARY,
    DISTRIBUTED_RELATIVE_PATH,
    EXPERIMENT_CONFIG_RELATIVE_PATH,
    IMAGINAIRE_CONFIG_RELATIVE_PATH,
    ITER_SPEED_RELATIVE_PATH,
    ORIGINAL_ATTENTION_SHA256,
    ORIGINAL_DISTRIBUTED_SHA256,
    ORIGINAL_EXPERIMENT_CONFIG_SHA256,
    ORIGINAL_FILE_SHA256,
    ORIGINAL_IMAGINAIRE_CONFIG_SHA256,
    ORIGINAL_ITER_SPEED_SHA256,
    OVERLAY_PROTOCOL,
    PATCHED_ATTENTION_SHA256,
    PATCHED_DISTRIBUTED_SHA256,
    PATCHED_EXPERIMENT_CONFIG_SHA256,
    PATCHED_FILE_SHA256,
    PATCHED_IMAGINAIRE_CONFIG_SHA256,
    PATCHED_ITER_SPEED_SHA256,
    PINNED_COMMIT,
    PURE_PYTHON_RUNTIME_PINS,
    SOURCE_PATCH_SPECS,
    TARGET_RELATIVE_PATH,
    RuntimeWheelPin,
    SourcePatchSpec,
)
from .overlay_render import render_patch
from .receipt import canonical_json_sha256, signed_receipt, write_or_verify_receipt


@dataclass(frozen=True)
class OverlayConfig:
    """Explicit checkouts and no-clobber receipt for one overlay application."""

    pinned_source_checkout: Path
    target_checkout: Path
    receipt: Path

    def __post_init__(self) -> None:
        for field_name, path in (
            ("pinned_source_checkout", self.pinned_source_checkout),
            ("target_checkout", self.target_checkout),
            ("receipt", self.receipt),
        ):
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be an absolute path")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def overlay_manifest() -> dict[str, object]:
    """Return the immutable source and pure-Python dependency contract."""

    return {
        "protocol_id": OVERLAY_PROTOCOL,
        "pinned_commit": PINNED_COMMIT,
        "source_patches": [spec.as_manifest_entry() for spec in SOURCE_PATCH_SPECS],
        "isolated_pure_python_runtime": {
            "installation": "external_wheels_not_vendored",
            "wheels": [pin.as_manifest_entry() for pin in PURE_PYTHON_RUNTIME_PINS],
        },
    }


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed in {checkout}: {detail}")
    return completed.stdout.strip()


def _validate_clean_checkout(path: Path, *, label: str) -> tuple[Path, str]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    top_level = Path(_git(resolved, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != resolved:
        raise ValueError(f"{label} must name the Git checkout root: {resolved}")
    commit = _git(resolved, "rev-parse", "HEAD")
    if commit != PINNED_COMMIT:
        raise ValueError(
            f"{label} commit mismatch: expected {PINNED_COMMIT}, found {commit}"
        )
    dirty = _git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if dirty:
        raise ValueError(f"{label} must be clean before applying the overlay")
    return resolved, commit


def _atomic_replace(path: Path, payload: bytes, *, expected_current: bytes) -> None:
    if path.read_bytes() != expected_current:
        raise RuntimeError(
            f"target changed after validation; refusing to write: {path}"
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".overlay.tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_overlay(config: OverlayConfig) -> dict[str, object]:
    """Validate identities, patch only the target, and emit a signed receipt."""

    if config.receipt.exists() or config.receipt.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {config.receipt}")
    pinned, _ = _validate_clean_checkout(
        config.pinned_source_checkout,
        label="pinned_source_checkout",
    )
    if config.receipt.resolve(strict=False).is_relative_to(pinned):
        raise ValueError("receipt must not be written inside pinned_source_checkout")
    target, commit = _validate_clean_checkout(
        config.target_checkout,
        label="target_checkout",
    )
    if pinned.samefile(target):
        raise ValueError("target_checkout must be distinct from pinned_source_checkout")
    patch_payloads: list[tuple[Path, bytes, bytes]] = []
    for spec in SOURCE_PATCH_SPECS:
        pinned_file = pinned / spec.relative_path
        target_file = target / spec.relative_path
        for label, path in (
            ("pinned source file", pinned_file),
            ("target file", target_file),
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{label} must be a regular non-symlink file: {path}")
        if pinned_file.samefile(target_file):
            raise ValueError(
                f"target file must not share an inode with pinned source: {spec.relative_path}"
            )
        pinned_bytes = pinned_file.read_bytes()
        target_bytes = target_file.read_bytes()
        if _sha256_bytes(pinned_bytes) != spec.preimage_sha256:
            raise ValueError(f"pinned source SHA256 mismatch: {spec.relative_path}")
        if target_bytes != pinned_bytes:
            raise ValueError(f"target differs from pinned source: {spec.relative_path}")
        patched_bytes = render_patch(spec, target_bytes)
        if _sha256_bytes(patched_bytes) != spec.patched_sha256:
            raise RuntimeError(f"rendered patch SHA256 mismatch: {spec.relative_path}")
        patch_payloads.append((target_file, target_bytes, patched_bytes))
    manifest = overlay_manifest()
    receipt = signed_receipt(
        {
            "schema_version": 1,
            "status": "applied",
            "protocol_id": OVERLAY_PROTOCOL,
            "claim_boundary": CLAIM_BOUNDARY,
            "training_launch_performed": False,
            "training_success_claimed": False,
            "pinned_source_checkout": str(pinned),
            "target_checkout": str(target),
            "pinned_commit": commit,
            "overlay_manifest_sha256": canonical_json_sha256(manifest),
            "overlay_manifest": manifest,
        }
    )
    applied: list[tuple[Path, bytes, bytes]] = []
    try:
        for target_file, target_bytes, patched_bytes in patch_payloads:
            _atomic_replace(target_file, patched_bytes, expected_current=target_bytes)
            applied.append((target_file, target_bytes, patched_bytes))
        if len(SOURCE_PATCH_SPECS) != len(applied):
            raise RuntimeError("not every HCU source patch was applied")
        for spec, (target_file, _, _) in zip(SOURCE_PATCH_SPECS, applied):
            if _sha256_bytes(target_file.read_bytes()) != spec.patched_sha256:
                raise RuntimeError(
                    f"target patch verification failed: {spec.relative_path}"
                )
        write_or_verify_receipt(config.receipt, receipt)
    except Exception:
        for target_file, target_bytes, patched_bytes in reversed(applied):
            if target_file.is_file() and target_file.read_bytes() == patched_bytes:
                _atomic_replace(
                    target_file, target_bytes, expected_current=patched_bytes
                )
        raise
    return receipt


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-source-checkout", required=True, type=Path)
    parser.add_argument("--target-checkout", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = apply_overlay(
        OverlayConfig(
            pinned_source_checkout=args.pinned_source_checkout,
            target_checkout=args.target_checkout,
            receipt=args.receipt,
        )
    )
    print(
        json.dumps(
            {
                "claim_boundary": receipt["claim_boundary"],
                "overlay_manifest_sha256": receipt["overlay_manifest_sha256"],
                "receipt": str(args.receipt),
                "receipt_sha256": receipt["receipt_sha256"],
                "status": receipt["status"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTENTION_RELATIVE_PATH",
    "CLAIM_BOUNDARY",
    "DISTRIBUTED_RELATIVE_PATH",
    "EXPERIMENT_CONFIG_RELATIVE_PATH",
    "IMAGINAIRE_CONFIG_RELATIVE_PATH",
    "ITER_SPEED_RELATIVE_PATH",
    "ORIGINAL_FILE_SHA256",
    "ORIGINAL_ATTENTION_SHA256",
    "ORIGINAL_EXPERIMENT_CONFIG_SHA256",
    "ORIGINAL_IMAGINAIRE_CONFIG_SHA256",
    "ORIGINAL_DISTRIBUTED_SHA256",
    "ORIGINAL_ITER_SPEED_SHA256",
    "OVERLAY_PROTOCOL",
    "OverlayConfig",
    "PATCHED_ATTENTION_SHA256",
    "PATCHED_FILE_SHA256",
    "PATCHED_EXPERIMENT_CONFIG_SHA256",
    "PATCHED_IMAGINAIRE_CONFIG_SHA256",
    "PATCHED_DISTRIBUTED_SHA256",
    "PATCHED_ITER_SPEED_SHA256",
    "PINNED_COMMIT",
    "PURE_PYTHON_RUNTIME_PINS",
    "RuntimeWheelPin",
    "SOURCE_PATCH_SPECS",
    "SourcePatchSpec",
    "TARGET_RELATIVE_PATH",
    "apply_overlay",
    "main",
    "overlay_manifest",
]
