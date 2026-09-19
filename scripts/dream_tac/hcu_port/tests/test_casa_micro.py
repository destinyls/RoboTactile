"""Contract tests for the source-bound Dream-Tac CASA BF16 micro."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.dream_tac.hcu_port import casa_micro
from scripts.dream_tac.hcu_port.casa_micro import CasaMicroConfig
from scripts.dream_tac.hcu_port.receipt import (
    signed_receipt,
    verify_receipt,
    write_or_verify_receipt,
)

_FAKE_CASA_SOURCE = b"def self_attention_with_tactile_outer_bias_chunked():\n    pass\n"


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _prepare_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, str]:
    checkout = tmp_path / "Dream-Tac"
    checkout.mkdir()
    source_file = checkout / casa_micro.CASA_SOURCE_RELATIVE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(_FAKE_CASA_SOURCE)
    _git(checkout, "init", "--quiet")
    _git(checkout, "config", "user.name", "RoboTactile Test")
    _git(checkout, "config", "user.email", "robotactile-test@example.invalid")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "--quiet", "-m", "fixture")
    commit = _git(checkout, "rev-parse", "HEAD")
    monkeypatch.setattr(casa_micro, "PINNED_COMMIT", commit)
    monkeypatch.setattr(
        casa_micro,
        "CASA_SOURCE_SHA256",
        hashlib.sha256(_FAKE_CASA_SOURCE).hexdigest(),
    )
    return checkout, source_file, commit


def _config(checkout: Path, output: Path) -> CasaMicroConfig:
    return CasaMicroConfig(
        dream_tac_checkout=checkout,
        output=output,
        device_index=0,
    )


def _passing_execution() -> dict[str, object]:
    required = (
        "q",
        "k",
        "v",
        "a",
        "b",
        "gamma",
        "projection_weight",
        "projection_bias",
    )
    return {
        "status": "passed",
        "seed": 0,
        "requested_backend": "flashbias_sdpa",
        "output": {"shape": [2, 64, 64], "finite": True, "nonzero": True},
        "loss": {
            "value": 1.0,
            "finite": True,
            "nonzero": True,
            "backward_completed": True,
        },
        "gradient_checks": {
            name: {"present": True, "finite": True, "nonzero": True}
            for name in required
        },
        "warnings": [
            {
                "category": "UserWarning",
                "message": "memory-efficient backend unavailable",
            }
        ],
        "warnings_are_nonfatal_for_correctness": True,
    }


def test_casa_micro_pass_receipt_is_bounded_and_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, source_file, commit = _prepare_source(tmp_path, monkeypatch)

    def executor(path: Path, device_index: int) -> tuple[dict[str, object], bool]:
        assert path == source_file
        assert device_index == 0
        return _passing_execution(), True

    receipt = casa_micro.build_casa_micro_receipt(
        _config(checkout, tmp_path / "receipt.json"),
        executor=executor,
    )

    verify_receipt(receipt)
    assert receipt["protocol_id"] == "dream_tac_hcu_casa_bf16_micro_v1"
    assert receipt["overall_status"] == "passed"
    assert receipt["claim_boundary"] == (
        "casa_random_tensor_forward_backward_only_not_training"
    )
    assert receipt["training_launch_performed"] is False
    assert receipt["training_success_claimed"] is False
    assert receipt["fused_kernel_or_performance_parity_claimed"] is False
    source_identity = receipt["source_identity"]
    assert isinstance(source_identity, dict)
    assert source_identity["pinned_commit"] == commit
    assert (
        source_identity["source_sha256"]
        == hashlib.sha256(_FAKE_CASA_SOURCE).hexdigest()
    )


def test_unrelated_dirty_overlay_is_recorded_but_does_not_block_casa_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _, _ = _prepare_source(tmp_path, monkeypatch)
    (checkout / "overlay-change.txt").write_text("dirty\n", encoding="utf-8")

    receipt = casa_micro.build_casa_micro_receipt(
        _config(checkout, tmp_path / "receipt.json"),
        executor=lambda *_: (_passing_execution(), True),
    )

    assert receipt["overall_status"] == "passed"
    source_identity = receipt["source_identity"]
    assert isinstance(source_identity, dict)
    assert source_identity["checkout_clean"] is False
    dirty = source_identity["checkout_dirty_entries"]
    assert isinstance(dirty, list)
    assert any("overlay-change.txt" in entry for entry in dirty)


def test_wrong_source_commit_produces_signed_failure_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _, _ = _prepare_source(tmp_path, monkeypatch)
    monkeypatch.setattr(casa_micro, "PINNED_COMMIT", "0" * 40)
    called = False

    def executor(*_args: object) -> tuple[dict[str, object], bool]:
        nonlocal called
        called = True
        return _passing_execution(), True

    receipt = casa_micro.build_casa_micro_receipt(
        _config(checkout, tmp_path / "receipt.json"),
        executor=executor,
    )

    verify_receipt(receipt)
    assert receipt["overall_status"] == "failed"
    assert called is False
    execution = receipt["execution"]
    assert isinstance(execution, dict)
    assert execution["status"] == "not_run"


def test_runtime_failure_is_evidence_not_a_training_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _, _ = _prepare_source(tmp_path, monkeypatch)

    def executor(*_args: object) -> tuple[dict[str, object], bool]:
        raise RuntimeError("synthetic HCU failure")

    receipt = casa_micro.build_casa_micro_receipt(
        _config(checkout, tmp_path / "receipt.json"),
        executor=executor,
    )

    assert receipt["overall_status"] == "failed"
    assert receipt["training_success_claimed"] is False
    execution = receipt["execution"]
    assert isinstance(execution, dict)
    error = execution["error"]
    assert isinstance(error, dict)
    assert error["message"] == "synthetic HCU failure"


def test_casa_receipt_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "casa.json"
    first = signed_receipt({"status": "first"})
    second = signed_receipt({"status": "second"})

    assert write_or_verify_receipt(output, first) == "created"
    assert write_or_verify_receipt(output, first) == "verified_existing"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_or_verify_receipt(output, second)
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_backend_environment_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    environ = {"COSMOS_TACTILE_SELF_ATTN_BACKEND": "eager"}
    key, previous = casa_micro._set_backend(environ)
    assert environ[key] == "flashbias_sdpa"
    casa_micro._restore_backend(environ, key, previous)
    assert environ[key] == "eager"

    empty: dict[str, str] = {}
    key, previous = casa_micro._set_backend(empty)
    casa_micro._restore_backend(empty, key, previous)
    assert key not in empty


def test_casa_config_rejects_relative_or_negative_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dream_tac_checkout must be an absolute"):
        CasaMicroConfig(
            dream_tac_checkout=Path("relative"),
            output=tmp_path / "receipt.json",
            device_index=0,
        )
    with pytest.raises(ValueError, match="device_index must be non-negative"):
        CasaMicroConfig(
            dream_tac_checkout=tmp_path,
            output=tmp_path / "receipt.json",
            device_index=-1,
        )
