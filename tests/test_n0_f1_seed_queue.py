"""Focused no-GPU checks for the official N0 F1 seed queue."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.n0_fault_campaign import N0FaultCellDisposition
from robotactile_benchmark.trials import Condition
from scripts.retrained_evaluation.n0_f1_seed_queue import (
    F1,
    _select_f1_request,
    _write_canonical_once,
)


def _cell(**overrides: object) -> SimpleNamespace:
    values = {
        "condition": Condition.FAULTED,
        "operator_id": F1,
        "severity_level": 5,
        "disposition": N0FaultCellDisposition.LIVE_REQUEST,
        "request_relpath": "requests/f1.json",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_select_f1_request_requires_one_live_level_five_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "requests/f1.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.retrained_evaluation.n0_f1_seed_queue.load_live_univtac_request",
        lambda _path: SimpleNamespace(condition=Condition.FAULTED),
    )
    bundle = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(cells=(_cell(),)),
    )
    assert _select_f1_request(bundle) == request


def test_select_f1_request_rejects_ambiguous_cells(tmp_path: Path) -> None:
    bundle = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(cells=(_cell(), _cell())),
    )
    with pytest.raises(ValueError, match="one live F1"):
        _select_f1_request(bundle)


def test_select_f1_request_rejects_clean_only(tmp_path: Path) -> None:
    bundle = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(
            cells=(_cell(condition=Condition.CLEAN, operator_id=None),)
        ),
    )
    with pytest.raises(ValueError, match="one live F1"):
        _select_f1_request(bundle)


def test_clean_source_is_written_as_strict_canonical_json(tmp_path: Path) -> None:
    path = tmp_path / "source" / "request.json"
    value = {"z": 1, "a": [True, None]}
    _write_canonical_once(path, value)
    assert path.read_bytes() == canonical_json_bytes(value)
