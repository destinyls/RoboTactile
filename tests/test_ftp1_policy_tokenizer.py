from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/ftp1_policy/provision_openpi_tokenizer.py"


def _load_provisioner() -> Any:
    name = "robotactile_ftp1_tokenizer_provisioner_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


provisioner = _load_provisioner()


def _spec(source: Path) -> Any:
    payload = source.read_bytes()
    return provisioner.TokenizerSpec(
        source_url=source.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        relative_path=Path("big_vision/paligemma_tokenizer.model"),
    )


def test_official_tokenizer_identity_and_deployment_paths_are_frozen() -> None:
    identity = provisioner.PINNED_TOKENIZER

    assert identity.source_url == (
        "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
    )
    assert identity.sha256 == (
        "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
    )
    assert identity.size_bytes == 4_264_023
    assert (
        Path("artifacts/openpi-data/ftp1-policy")
        == provisioner.OPENPI_DATA_HOME_RELATIVE
    )
    assert (
        Path("artifacts/deployment/ftp1_policy_openpi_tokenizer_provision.json")
        == provisioner.RECEIPT_RELATIVE
    )


def test_provisioner_downloads_hash_checks_and_writes_no_clobber_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.model"
    source.write_bytes(b"verified-tokenizer-payload")
    deployment = tmp_path / "deployment"
    spec = _spec(source)

    data_home = provisioner.provision_tokenizer(deployment, spec=spec)

    target = data_home / spec.relative_path
    receipt_path = deployment / provisioner.RECEIPT_RELATIVE
    assert data_home == (deployment / "artifacts/openpi-data/ftp1-policy").resolve()
    assert target.read_bytes() == source.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "provisioned"
    assert receipt["source_url"] == source.as_uri()
    assert receipt["file_sha256"] == spec.sha256
    assert receipt["file_size_bytes"] == spec.size_bytes
    assert receipt["openpi_data_home"] == "artifacts/openpi-data/ftp1-policy"
    assert receipt["tokenizer_path"] == (
        "artifacts/openpi-data/ftp1-policy/big_vision/paligemma_tokenizer.model"
    )


def test_provisioner_is_idempotent_without_rewriting_target_or_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.model"
    source.write_bytes(b"verified-tokenizer-payload")
    deployment = tmp_path / "deployment"
    spec = _spec(source)
    data_home = provisioner.provision_tokenizer(deployment, spec=spec)
    target = data_home / spec.relative_path
    receipt = deployment / provisioner.RECEIPT_RELATIVE
    target_stat = target.stat()
    receipt_bytes = receipt.read_bytes()

    returned = provisioner.provision_tokenizer(deployment, spec=spec)

    assert returned == data_home
    assert target.stat().st_ino == target_stat.st_ino
    assert target.stat().st_mtime_ns == target_stat.st_mtime_ns
    assert receipt.read_bytes() == receipt_bytes


def test_provisioner_refuses_to_overwrite_wrong_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.model"
    source.write_bytes(b"verified-tokenizer-payload")
    deployment = tmp_path / "deployment"
    spec = _spec(source)
    target = deployment / provisioner.OPENPI_DATA_HOME_RELATIVE / spec.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong-partial-payload")

    with pytest.raises(provisioner.ProvisioningError, match="refusing to overwrite"):
        provisioner.provision_tokenizer(deployment, spec=spec)

    assert target.read_bytes() == b"wrong-partial-payload"
    assert not (deployment / provisioner.RECEIPT_RELATIVE).exists()


def test_provisioner_refuses_incompatible_existing_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.model"
    source.write_bytes(b"verified-tokenizer-payload")
    deployment = tmp_path / "deployment"
    spec = _spec(source)
    provisioner.provision_tokenizer(deployment, spec=spec)
    receipt = deployment / provisioner.RECEIPT_RELATIVE
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["file_sha256"] = "0" * 64
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(provisioner.ProvisioningError, match="receipt field"):
        provisioner.provision_tokenizer(deployment, spec=spec)


def test_tokenizer_provisioner_help_requires_no_network() -> None:
    completed = subprocess.run(
        (sys.executable, str(SCRIPT), "--help"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--root" in completed.stdout
    assert "hash-bound FTP-1 OpenPI tokenizer" in completed.stdout
