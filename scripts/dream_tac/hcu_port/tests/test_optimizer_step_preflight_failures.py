"""Fail-closed cases for the HCU optimizer-step preflight."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from scripts.dream_tac.hcu_port import optimizer_step_preflight as preflight
from scripts.dream_tac.hcu_port.optimizer_step_preflight import (
    build_optimizer_step_preflight_receipt,
)
from scripts.dream_tac.hcu_port.optimizer_step_request import (
    HcuOptimizerStepRequest,
)
from scripts.dream_tac.hcu_port.overlay_contract import (
    PINNED_COMMIT,
    SOURCE_PATCH_SPECS,
)
from scripts.dream_tac.hcu_port.receipt import signed_receipt
from scripts.dream_tac.training.training_request import sha256_regular_file

from .test_optimizer_step_preflight import (
    _install_stubs,
    _make_fixture,
    _write_json,
)


def test_git_preserves_porcelain_leading_space(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.name", "RoboTactile Test"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "robotactile@example.invalid"],
        cwd=checkout,
        check=True,
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "fixture"], cwd=checkout, check=True
    )
    tracked.write_text("after\n", encoding="utf-8")

    status = preflight._git(
        checkout, "status", "--porcelain=v1", "--untracked-files=all"
    )

    assert status == " M tracked.py"


def test_overlay_source_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    monkeypatch.setattr(preflight, "sha256_regular_file", sha256_regular_file)

    with pytest.raises(ValueError, match="overlay source SHA256 mismatch"):
        build_optimizer_step_preflight_receipt(fixture.request)


def test_overlay_checkout_rejects_extra_dirty_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)

    def git(checkout: Path, *arguments: str) -> str:
        assert checkout.resolve() == fixture.source_root.resolve()
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(fixture.source_root)
        if arguments == ("rev-parse", "HEAD"):
            return cast(str, PINNED_COMMIT)
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            expected = [
                f" M {spec.relative_path.as_posix()}" for spec in SOURCE_PATCH_SPECS
            ]
            return "\n".join([*expected, "?? unexpected.py"])
        raise AssertionError(arguments)

    monkeypatch.setattr(preflight, "_git", git)

    with pytest.raises(ValueError, match="unexpected dirty entry"):
        build_optimizer_step_preflight_receipt(fixture.request)


def test_failed_casa_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    casa_path = fixture.request.casa_receipt.path
    payload = json.loads(casa_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload.pop("receipt_sha256")
    payload["overall_status"] = "failed"
    _write_json(casa_path, signed_receipt(payload))
    request_payload = dict(fixture.request_payload)
    request_payload["casa_receipt"] = {
        "path": str(casa_path),
        "sha256": sha256_regular_file(casa_path),
    }
    request = HcuOptimizerStepRequest.from_dict(request_payload)

    with pytest.raises(ValueError, match="CASA receipt contract mismatch"):
        build_optimizer_step_preflight_receipt(request)


def test_placeholder_artifact_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    payload = dict(fixture.request_payload)
    payload["tokenizer_checkpoint"] = {
        "path": "/path/to/tokenizer.pth",
        "sha256": "7" * 64,
    }
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match="placeholder path"):
        build_optimizer_step_preflight_receipt(request)


def test_bound_receipt_bytes_cannot_change_after_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    fixture.request.overlay_receipt.path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="overlay_receipt SHA256 mismatch"):
        build_optimizer_step_preflight_receipt(fixture.request)


def test_hcu_environment_script_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    fixture.request.hcu_environment_script.path.write_text(
        "export TAMPERED=1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hcu_environment_script SHA256 mismatch"):
        build_optimizer_step_preflight_receipt(fixture.request)


@pytest.mark.parametrize("kind", ("symlink", "placeholder"))
def test_hcu_environment_script_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    payload = dict(fixture.request_payload)
    if kind == "symlink":
        path = tmp_path / "runtime" / "linked-env.sh"
        path.symlink_to(fixture.request.hcu_environment_script.path)
        message = "must not be a symlink"
    else:
        path = Path("/path/to/hcu-env.sh")
        message = "placeholder path"
    payload["hcu_environment_script"] = {
        "path": str(path),
        "sha256": fixture.request.hcu_environment_script.sha256,
    }
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match=message):
        build_optimizer_step_preflight_receipt(request)


def test_overlay_receipt_must_name_runtime_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    receipt_path = fixture.request.overlay_receipt.path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    receipt.pop("receipt_sha256")
    receipt["target_checkout"] = str(tmp_path / "different-checkout")
    (tmp_path / "different-checkout").mkdir()
    _write_json(receipt_path, signed_receipt(receipt))
    payload = dict(fixture.request_payload)
    payload["overlay_receipt"] = {
        "path": str(receipt_path),
        "sha256": sha256_regular_file(receipt_path),
    }
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match="target does not match"):
        build_optimizer_step_preflight_receipt(request)


def test_flat_pt_cannot_be_used_as_dcp_load_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    payload = dict(fixture.request_payload)
    payload["base_dcp_root"] = str(tmp_path / "artifacts" / "wrong.pt")
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match="cannot be a flat .pt"):
        build_optimizer_step_preflight_receipt(request)


def test_dcp_payload_bytes_must_match_converter_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    dcp_payload = fixture.request.base_dcp_root / "model" / ".metadata"
    dcp_payload.write_bytes(b"tampered-dcp")

    with pytest.raises(ValueError, match="file size mismatch|file SHA256 mismatch"):
        build_optimizer_step_preflight_receipt(fixture.request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tokenizer_load_mean_std", True, "converter receipt contract mismatch"),
        (
            "tokenizer_checkpoint",
            {"path": "/wrong/tokenizer.pth", "sha256": "f" * 64},
            "converter tokenizer does not match request",
        ),
    ),
)
def test_dcp_converter_must_bind_exact_tokenizer_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    receipt_path = fixture.request.base_dcp_receipt.path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    receipt.pop("receipt_sha256")
    receipt[field] = value
    _write_json(receipt_path, signed_receipt(receipt))
    payload = dict(fixture.request_payload)
    payload["base_dcp_receipt"] = {
        "path": str(receipt_path),
        "sha256": sha256_regular_file(receipt_path),
    }
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match=message):
        build_optimizer_step_preflight_receipt(request)


def test_dcp_converter_must_bind_exact_flat_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    other = tmp_path / "artifacts" / "relocated-base.pt"
    other.write_bytes(fixture.request.base_checkpoint.path.read_bytes())
    receipt_path = fixture.request.base_dcp_receipt.path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    receipt.pop("receipt_sha256")
    input_checkpoint = receipt["input_checkpoint"]
    assert isinstance(input_checkpoint, dict)
    input_checkpoint["path"] = str(other)
    _write_json(receipt_path, signed_receipt(receipt))
    payload = dict(fixture.request_payload)
    payload["base_dcp_receipt"] = {
        "path": str(receipt_path),
        "sha256": sha256_regular_file(receipt_path),
    }
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match="input path does not match"):
        build_optimizer_step_preflight_receipt(request)


def test_runtime_pythonpath_root_must_be_real_directory_not_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    link = tmp_path / "runtime" / "linked-site"
    link.symlink_to(fixture.runtime_roots[0], target_is_directory=True)
    payload = dict(fixture.request_payload)
    payload["runtime_pythonpath_roots"] = [
        str(link),
        *(str(path) for path in fixture.runtime_roots[1:]),
    ]
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match="must not be a symlink"):
        build_optimizer_step_preflight_receipt(request)


@pytest.mark.parametrize("kind", ("missing", "regular_file"))
def test_runtime_pythonpath_root_must_exist_as_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    invalid = tmp_path / "runtime" / kind
    if kind == "regular_file":
        invalid.write_bytes(b"not-a-directory")
    payload = dict(fixture.request_payload)
    payload["runtime_pythonpath_roots"] = [
        str(invalid),
        *(str(path) for path in fixture.runtime_roots[1:]),
    ]
    request = HcuOptimizerStepRequest.from_dict(payload)

    with pytest.raises(ValueError, match="does not exist|must be a directory"):
        build_optimizer_step_preflight_receipt(request)
