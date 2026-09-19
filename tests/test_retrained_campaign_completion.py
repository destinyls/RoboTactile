"""A zero exit status cannot stand in for missing result publication."""

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.retrained_evaluation import campaign
from scripts.retrained_evaluation.group import write_json


@pytest.mark.parametrize(
    "returncode,count,receipt,complete",
    [
        (0, 2, False, False),
        (0, 3, False, False),
        (1, 3, True, False),
        (0, 3, True, True),
    ],
)
def test_group_completion_requires_all_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    count: int,
    receipt: bool,
    complete: bool,
) -> None:
    class Process:
        pid = 4242

        def wait(self) -> int:
            return returncode

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(campaign, "stop_owned", lambda _: None)
    write_json(tmp_path / "group.json", {"ordered_requests": ["clean", "A1", "A2"]})
    for index in range(count):
        write_json(tmp_path / "results" / f"{index:02d}.json", {})
    if receipt:
        write_json(tmp_path / "recovery_receipt.json", {})
    arguments: tuple[Any, ...] = (
        tmp_path / "group.json",
        tmp_path,
        tmp_path / "python.sh",
        tmp_path,
        {},
        0.0,
    )
    if complete:
        campaign._run_group(*arguments)
    else:
        with pytest.raises(RuntimeError, match="incomplete Isaac group"):
            campaign._run_group(*arguments)
    result = json.loads((tmp_path / "process_exit.json").read_text())
    assert result["complete"] is complete
    assert result["expected_result_count"] == 3
