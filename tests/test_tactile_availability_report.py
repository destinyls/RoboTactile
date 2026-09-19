"""Fixed-denominator and protocol-integrity tests for availability reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.retrained_evaluation.availability_report import (
    main,
    summarize_availability,
)

A1 = "A1_stream_absence"
A2 = "A2_frame_erasure"


def put(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def group(campaign: Path, seed: int = 0, **updates: Any) -> Path:
    path = campaign / "seeds" / f"seed-{seed:03d}" / "groups/n0_vtla/lift_can"
    put(
        path / "group.json",
        {
            "model": "n0_vtla",
            "task": "lift_can",
            "seed": seed,
            "tactile_availability_mode": "native_missing_v1",
            "ordered_requests": [
                f"requests/{condition}.json" for condition in ("clean", A1, A2)
            ],
            "excluded": [],
            **updates,
        },
    )
    return path


def result(directory: Path, index: int, **updates: Any) -> None:
    put(
        directory / "results" / f"{index:02d}.json",
        {
            "score_eligible": True,
            "validation_passed": True,
            "failure_stage": None,
            "score_success": True,
            "terminal_status": "success",
            "control_cycle_count": 4,
            "observation_count": 156,
            **updates,
        },
    )


def summarize(path: Path, seeds: tuple[int, ...] = (0,)) -> dict[str, Any]:
    return summarize_availability(
        path, model="n0_vtla", task="lift_can", seeds=seeds, mode="native_missing_v1"
    )


def test_valid_failure_invalid_and_partial_results_keep_denominator(
    tmp_path: Path,
) -> None:
    directory = group(tmp_path)
    result(directory, 0)
    result(directory, 1, score_success=False, terminal_status="timeout")
    result(
        directory,
        2,
        score_eligible=False,
        validation_passed=False,
        terminal_status="validator_rejected",
    )
    report = summarize(tmp_path, (0, 1))
    assert report["planned_episode_count"] == 6
    assert report["completed_execution_count"] == 3
    assert report["eligible_episode_count"] == 2
    assert report["coverage"] == report["support_coverage"] == 0.5
    assert report["support_unknown_episode_count"] == 3
    clean, a1, a2 = report["cells"]
    assert clean["success_rate"] == 1.0 and clean["missing"] == 1
    assert a1["success_rate"] == 0.0 and a1["eligible"] == 1
    assert a2["success_rate"] is None and a2["invalid"] == 1
    assert report["episodes"][0]["executed_action_steps"] == 155
    assert report["episodes"][0]["control_cycle_count"] == 4


def test_absent_campaign_is_all_missing_not_failed(tmp_path: Path) -> None:
    report = summarize(tmp_path / "not_created", (0, 1, 2))
    assert report["completed_execution_count"] == report["eligible_episode_count"] == 0
    assert (
        report["planned_episode_count"] == report["support_unknown_episode_count"] == 9
    )
    assert all(
        cell["missing"] == 3 and cell["invalid"] == 0 for cell in report["cells"]
    )


def test_unsupported_is_explicit_and_not_a_failure(tmp_path: Path) -> None:
    directory = group(
        tmp_path,
        ordered_requests=["requests/clean.json"],
        excluded=[
            {"operator": operator, "status": "unsupported_contract"}
            for operator in (A1, A2)
        ],
    )
    result(directory, 0)
    report = summarize(tmp_path)
    assert report["support_coverage"] == pytest.approx(1 / 3)
    for cell in report["cells"][1:]:
        assert cell["unsupported"] == 1 and cell["missing"] == cell["invalid"] == 0
        assert cell["eligible"] == 0 and cell["success_rate"] is None


@pytest.mark.parametrize(
    "update",
    [
        {"model": "dream_tac"},
        {"task": "other"},
        {"seed": 1},
        {"tactile_availability_mode": "zero_fill_v1"},
    ],
)
def test_mixed_group_identity_or_protocol_rejected(
    tmp_path: Path, update: dict[str, Any]
) -> None:
    group(tmp_path, **update)
    # A mismatched seed must be at the requested seed directory, not a new group.
    if update.get("seed") == 1:
        directory = group(tmp_path)
        plan = json.loads((directory / "group.json").read_text())
        plan["seed"] = 1
        put(directory / "group.json", plan)
    with pytest.raises(ValueError, match="identity/protocol"):
        summarize(tmp_path)


@pytest.mark.parametrize(
    "ordered",
    [
        ["requests/clean.json", "requests/clean.json"],
        ["requests/clean.json", f"requests/v1/{A1}.json", f"requests/v2/{A1}.json"],
    ],
)
def test_duplicate_requests_or_condition_rejected(
    tmp_path: Path, ordered: list[str]
) -> None:
    group(tmp_path, ordered_requests=ordered)
    with pytest.raises(ValueError, match="duplicate"):
        summarize(tmp_path)


def test_unknown_conditions_not_counted_and_result_indices_preserved(
    tmp_path: Path,
) -> None:
    directory = group(
        tmp_path,
        ordered_requests=[
            "requests/clean.json",
            "requests/F1_global_response_drift.json",
            f"requests/{A1}.json",
        ],
    )
    result(directory, 0)
    result(directory, 1)
    result(directory, 2, score_success=False, terminal_status="early_stop")
    report = summarize(tmp_path)
    assert report["completed_execution_count"] == 2
    assert report["cells"][1]["success"] == 0
    assert report["cells"][2]["missing"] == 1
    assert report["ignored_conditions"][0]["condition"] == "F1_global_response_drift"


def test_scheduled_missing_result_has_known_support(tmp_path: Path) -> None:
    group(tmp_path)
    report = summarize(tmp_path)
    assert report["support_coverage"] == 1.0 and report["coverage"] == 0.0
    assert report["support_unknown_episode_count"] == 0


def test_cli_writes_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "availability_report",
            "--campaign",
            str(tmp_path),
            "--model",
            "n0_vtla",
            "--task",
            "lift_can",
            "--seeds",
            "0",
            "1",
            "2",
            "--mode",
            "native_missing_v1",
            "--output",
            str(output),
        ],
    )
    main()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == original
