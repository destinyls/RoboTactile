#!/usr/bin/env python3
"""Provision FTP-1's PaliGemma tokenizer into deployment-local OpenPI data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class TokenizerSpec:
    """Immutable identity of the upstream tokenizer object."""

    source_url: str
    sha256: str
    size_bytes: int
    relative_path: Path


PINNED_TOKENIZER = TokenizerSpec(
    source_url="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model",
    sha256="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
    size_bytes=4_264_023,
    relative_path=Path("big_vision/paligemma_tokenizer.model"),
)
OPENPI_DATA_HOME_RELATIVE = Path("artifacts/openpi-data/ftp1-policy")
RECEIPT_RELATIVE = Path(
    "artifacts/deployment/ftp1_policy_openpi_tokenizer_provision.json"
)
SCHEMA_VERSION = "robotactile.ftp1_policy.openpi_tokenizer.v1"


class ProvisioningError(RuntimeError):
    """Raised when a no-clobber or identity check fails."""


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_file(path: Path, spec: TokenizerSpec) -> None:
    if path.is_symlink() or not path.is_file():
        raise ProvisioningError(f"tokenizer target is not a regular file: {path}")
    observed_sha256, observed_size = _file_identity(path)
    if observed_sha256 != spec.sha256 or observed_size != spec.size_bytes:
        raise ProvisioningError(
            "existing tokenizer identity mismatch; refusing to overwrite "
            f"{path} (sha256={observed_sha256}, size={observed_size})"
        )


def _expected_receipt(
    *,
    deployment_root: Path,
    data_home: Path,
    target: Path,
    spec: TokenizerSpec,
) -> dict[str, str | int]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "provisioned",
        "source_url": spec.source_url,
        "file_sha256": spec.sha256,
        "file_size_bytes": spec.size_bytes,
        "openpi_data_home": str(data_home.relative_to(deployment_root)),
        "tokenizer_path": str(target.relative_to(deployment_root)),
    }


def _validate_receipt(path: Path, expected: dict[str, str | int]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ProvisioningError(f"tokenizer receipt is not a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisioningError(f"invalid tokenizer receipt: {path}") from error
    if not isinstance(document, dict):
        raise ProvisioningError(f"invalid tokenizer receipt: {path}")
    for key, expected_value in expected.items():
        if document.get(key) != expected_value:
            raise ProvisioningError(
                f"tokenizer receipt field {key!r} is incompatible; refusing to overwrite"
            )


def _download(source_url: str, output: BinaryIO) -> None:
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "RoboTactile-FTP1-tokenizer-provisioner/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _publish_receipt(path: Path, document: dict[str, str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".ftp1-tokenizer-receipt.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ProvisioningError(
                f"refusing to overwrite tokenizer receipt: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def provision_tokenizer(
    deployment_root: Path,
    *,
    spec: TokenizerSpec = PINNED_TOKENIZER,
) -> Path:
    """Provision and verify the tokenizer without touching a user/root cache."""

    root = deployment_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_home = root / OPENPI_DATA_HOME_RELATIVE
    target = data_home / spec.relative_path
    receipt = root / RECEIPT_RELATIVE
    expected_receipt = _expected_receipt(
        deployment_root=root,
        data_home=data_home,
        target=target,
        spec=spec,
    )

    if receipt.exists() or receipt.is_symlink():
        _validate_file(target, spec)
        _validate_receipt(receipt, expected_receipt)
        return data_home

    if target.exists() or target.is_symlink():
        _validate_file(target, spec)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.is_symlink():
            raise ProvisioningError(
                f"tokenizer directory must not be a symlink: {target.parent}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".paligemma_tokenizer.", suffix=".partial", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                _download(spec.source_url, stream)
            _validate_file(temporary, spec)
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ProvisioningError(
                    f"refusing to overwrite tokenizer target: {target}"
                ) from error
            target.chmod(0o644)
        finally:
            temporary.unlink(missing_ok=True)

    document = dict(expected_receipt)
    document["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    _publish_receipt(receipt, document)
    return data_home


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision the hash-bound FTP-1 OpenPI tokenizer."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="RoboTactile deployment root",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        data_home = provision_tokenizer(args.root)
    except (OSError, ProvisioningError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(data_home)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
