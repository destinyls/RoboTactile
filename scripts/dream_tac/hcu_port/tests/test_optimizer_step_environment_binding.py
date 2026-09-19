"""Environment-script binding and receipt tests for HCU preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dream_tac.hcu_port import optimizer_step_preflight as preflight
from scripts.dream_tac.hcu_port.optimizer_step_preflight import (
    build_optimizer_step_preflight_receipt,
    write_optimizer_step_preflight_receipt,
)

from .test_optimizer_step_preflight import _install_stubs, _make_fixture


def test_preflight_receipt_binds_hcu_environment_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)

    receipt = build_optimizer_step_preflight_receipt(fixture.request)

    assert receipt["hcu_environment_script"] == {
        "path": str(fixture.request.hcu_environment_script.path),
        "sha256": fixture.request.hcu_environment_script.sha256,
    }


def test_preflight_receipt_is_no_clobber_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)

    path, first = write_optimizer_step_preflight_receipt(fixture.request)
    _, second = write_optimizer_step_preflight_receipt(fixture.request)

    assert path == fixture.request.receipt_root / preflight.PREFLIGHT_RECEIPT_NAME
    assert first == second

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA256 is invalid"):
        write_optimizer_step_preflight_receipt(fixture.request)
