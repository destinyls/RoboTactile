from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.live_univtac.probe_univtac_stock_expert_isaac import (
    _emit_failure,
    _emit_stage,
    _expert_panel,
    _parser,
    _strict_predicate,
    _validate_args,
)


def test_stock_expert_probe_flushes_stage_and_failure_before_cleanup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _emit_stage("backend_reset")
    error = RuntimeError("reset failed")
    _emit_failure("backend_reset", error)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert '"event":"robotactile_stock_expert_probe_stage"' in captured.err
    assert '"stage":"backend_reset"' in captured.err
    assert '"event":"robotactile_stock_expert_probe_failure"' in captured.err
    assert '"error_type":"RuntimeError"' in captured.err
    assert "RuntimeError: reset failed" in captured.err


def test_stock_expert_probe_requires_explicit_task_and_seeds() -> None:
    args = _parser().parse_args(
        [
            "--upstream-root",
            "/upstream",
            "--runtime-dir",
            "/runtime",
            "--output-dir",
            "/output",
            "--task",
            "lift_bottle",
            "--initial-seed",
            "1000000",
            "--exogenous-seed",
            "1000000",
        ]
    )

    _validate_args(args)

    assert args.task == "lift_bottle"
    assert args.initial_seed == 1_000_000
    assert args.exogenous_seed == 1_000_000
    assert args.antialiasing_mode == "TAA"


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_stock_expert_probe_accepts_boolean_predicates(value: object) -> None:
    task = SimpleNamespace(check_success=lambda: value)

    assert _strict_predicate(task, "check_success") is True


def test_stock_expert_probe_rejects_non_boolean_predicates() -> None:
    task = SimpleNamespace(check_success=lambda: 1)

    with pytest.raises(RuntimeError, match="non-boolean"):
        _strict_predicate(task, "check_success")


@pytest.mark.parametrize("task_id", ["insert_hole", "insert_tube"])
def test_stock_expert_probe_normalizes_upstream_none_early_stop(
    task_id: str,
) -> None:
    task = SimpleNamespace(check_early_stop=lambda: None)

    assert _strict_predicate(task, "check_early_stop", task_id=task_id) is False


def test_stock_expert_probe_rejects_none_for_other_early_stop_tasks() -> None:
    task = SimpleNamespace(check_early_stop=lambda: None)

    with pytest.raises(RuntimeError, match="non-boolean"):
        _strict_predicate(task, "check_early_stop", task_id="pull_out_key")


def test_stock_expert_panel_contains_only_supplied_pixels() -> None:
    before = {
        name: np.full((12, 16, 3), 32, dtype=np.uint8)
        for name in ("top", "wrist_l", "tactile_a", "tactile_b")
    }
    after = {
        name: np.full((12, 16, 3), 192, dtype=np.uint8)
        for name in ("top", "wrist_l", "tactile_a", "tactile_b")
    }

    panel = Image.open(io.BytesIO(_expert_panel(before, after)))

    assert panel.format == "PNG"
    assert panel.size == (1280, 528)
