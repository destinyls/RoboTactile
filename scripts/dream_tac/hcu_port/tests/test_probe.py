"""Contract tests for the bounded Dream-Tac HCU probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts.dream_tac.hcu_port import probe
from scripts.dream_tac.hcu_port.probe import ProbeConfig, build_probe_receipt
from scripts.dream_tac.hcu_port.receipt import (
    signed_receipt,
    verify_receipt,
    write_or_verify_receipt,
)


def _config(tmp_path: Path, *, casa_module: str | None = None) -> ProbeConfig:
    return ProbeConfig(
        output=tmp_path / "probe.json",
        device_index=0,
        matrix_size=16,
        required_modules=("required_backend",),
        casa_module=casa_module,
    )


def _fake_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__dict__["__version__"] = "1.0-test"
    module.__file__ = f"/fake/{name}.py"
    return module


def test_probe_passes_only_as_bounded_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = _fake_module("torch")
    runtime = {
        "status": "passed",
        "torch_version": "2.5.1+dtk",
        "hip_version": "6.2",
    }
    monkeypatch.setattr(
        probe,
        "_runtime_phase",
        lambda **_: (runtime, fake_torch, True),
    )
    monkeypatch.setattr(
        probe,
        "_bf16_phase",
        lambda *_args, **_kwargs: (
            {
                "status": "passed",
                "forward_completed": True,
                "backward_completed": True,
            },
            True,
        ),
    )

    receipt = build_probe_receipt(
        _config(tmp_path), importer=_fake_module, environ={"DTK_VERSION": "test"}
    )

    verify_receipt(receipt)
    assert receipt["overall_status"] == "core_probe_passed_casa_not_checked"
    assert receipt["claim_boundary"] == (
        "compatibility_probe_only_not_training_success"
    )
    assert receipt["training_launch_performed"] is False
    assert receipt["training_success_claimed"] is False
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    assert phases["optional_casa_import"] == {
        "status": "not_requested",
        "module": None,
    }


def test_runtime_failure_skips_bf16_and_fails_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probe,
        "_runtime_phase",
        lambda **_: (
            {"status": "failed", "error": {"message": "no HIP"}},
            None,
            False,
        ),
    )
    called = False

    def unexpected_bf16(
        *_args: object, **_kwargs: object
    ) -> tuple[dict[str, object], bool]:
        nonlocal called
        called = True
        return {"status": "passed"}, True

    monkeypatch.setattr(probe, "_bf16_phase", unexpected_bf16)

    receipt = build_probe_receipt(_config(tmp_path), importer=_fake_module)

    assert receipt["overall_status"] == "probe_failed"
    assert called is False
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    assert phases["bf16_forward_backward"] == {
        "status": "not_run",
        "reason": "runtime phase did not establish an HCU/HIP device",
    }


def test_requested_casa_import_is_a_real_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = _fake_module("torch")
    monkeypatch.setattr(
        probe,
        "_runtime_phase",
        lambda **_: ({"status": "passed"}, fake_torch, True),
    )
    monkeypatch.setattr(
        probe,
        "_bf16_phase",
        lambda *_args, **_kwargs: ({"status": "passed"}, True),
    )

    receipt = build_probe_receipt(
        _config(tmp_path, casa_module="casa_backend"), importer=_fake_module
    )

    assert receipt["overall_status"] == "probe_passed_with_casa_import"
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    casa = phases["optional_casa_import"]
    assert isinstance(casa, dict)
    assert casa["status"] == "passed"


def test_receipt_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "probe.json"
    first = signed_receipt({"status": "first"})
    second = signed_receipt({"status": "second"})

    assert write_or_verify_receipt(output, first) == "created"
    assert write_or_verify_receipt(output, first) == "verified_existing"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_or_verify_receipt(output, second)
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_probe_import_does_not_load_n0_or_act() -> None:
    code = """
import sys
import scripts.dream_tac.hcu_port.probe
forbidden = sorted(
    name for name in sys.modules
    if name.startswith('scripts.n0_twam') or name.startswith('scripts.act')
)
assert not forbidden, forbidden
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_probe_config_rejects_invalid_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="matrix_size"):
        ProbeConfig(
            output=tmp_path / "probe.json",
            device_index=0,
            matrix_size=1,
            required_modules=("backend",),
            casa_module=None,
        )
