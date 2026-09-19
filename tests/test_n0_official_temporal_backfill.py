"""No-GPU safeguards for the official N0 temporal-only successor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.n0_twam import backfill_official_early_temporal as backfill
from scripts.n0_twam.run_official_early_fault_seed import write_json


def test_only_unscored_temporal_cells_are_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = tuple(
        SimpleNamespace(operator_id=operator, artifact_relpath=operator)
        for operator in backfill.TEMPORAL_OPERATORS
    )
    loaded = SimpleNamespace(root=tmp_path, manifest=SimpleNamespace(cells=cells))
    monkeypatch.setattr(backfill, "load_n0_fault_campaign_bundle", lambda _: loaded)
    write_json(
        tmp_path / "T1_fixed_source_delay/terminal_result.json",
        {
            "execution_status": "timeout",
            "validation_passed": True,
            "score_eligible": True,
            "score_success": False,
        },
    )
    write_json(
        tmp_path / "T2_held_last_freeze/terminal_result.json",
        {"execution_status": "crash", "failure_code": "delivery_failed"},
    )
    assert backfill._pending_operators(tmp_path) == (
        "T2_held_last_freeze",
        "T3_inter_sensor_skew",
    )


def test_unclassified_model_outcome_is_not_silently_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = tuple(
        SimpleNamespace(operator_id=operator, artifact_relpath=operator)
        for operator in backfill.TEMPORAL_OPERATORS
    )
    loaded = SimpleNamespace(root=tmp_path, manifest=SimpleNamespace(cells=cells))
    monkeypatch.setattr(backfill, "load_n0_fault_campaign_bundle", lambda _: loaded)
    write_json(
        tmp_path / "T1_fixed_source_delay/terminal_result.json",
        {"execution_status": "timeout", "failure_code": None},
    )
    with pytest.raises(ValueError, match="unclassified"):
        backfill._pending_operators(tmp_path)


def test_backfill_requires_prior_completion_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "successor"
    with pytest.raises(RuntimeError, match="must finish"):
        backfill.prepare(tmp_path / "prior", output)
    assert not output.exists()
