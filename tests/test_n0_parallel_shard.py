"""Disjoint shard and process-handoff contracts without simulator processes."""

import json
import signal
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.n0_twam import parallel_shard as shard
from scripts.retrained_evaluation.group import write_json


@pytest.mark.parametrize(
    "assignments",
    [
        {},
        {"other": [0]},
        {"../lift_can": [0]},
        {"lift_can": []},
        {"lift_can": [0, 0]},
        {"lift_can": [True]},
        {"lift_can": [-1]},
        {"lift_can": [1.0]},
    ],
)
def test_assignment_rejections(assignments):
    with pytest.raises(ValueError):
        shard.validate_assignments(assignments, ["lift_can"])


@pytest.mark.parametrize("seeds", [list(range(10)), [3, 9, 1000]])
def test_assignments_accept_nonnegative_seeds_beyond_two(seeds):
    assert shard.validate_assignments({"lift_can": seeds}, ["lift_can"]) == {
        "lift_can": seeds
    }


def test_duplicate_json_tasks_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        json.loads(
            '{"lift_can":[0],"lift_can":[1]}', object_pairs_hook=shard.unique_object
        )


def test_snapshot_only_counts_assigned_seeds(tmp_path):
    result = shard.shard_snapshot(tmp_path, {"lift_can": [2]})
    data = json.loads(result.read_text())
    assert {item["seed"] for item in data["episodes"]} == {2}
    assert all(cell["seeds"] == [2] for cell in data["cells"])


def handoff_spec(tmp_path):
    group = tmp_path / "old-group"
    write_json(group / "launch.json", {"pid": 12})
    return {
        "hostname": shard.socket.gethostname(),
        "current_group": str(group),
        "supervisor": {"pid": 11, "start_ticks": 100, "cmd_marker": "supervisor"},
        "isaac": {"pid": 12, "start_ticks": 101, "cmd_marker": "isaac"},
        "worker": {"pid": 13, "start_ticks": 102, "cmd_marker": "torchrun", "pgid": 13},
    }


@pytest.mark.parametrize(
    "states", [{11: "S", 12: "S", 13: "S"}, {11: "T", 12: "T", 13: "S"}]
)
def test_handoff_refuses_running_parent_or_stopped_child(tmp_path, monkeypatch, states):
    spec = handoff_spec(tmp_path)
    monkeypatch.setattr(
        shard, "process_state", lambda identity: states[identity["pid"]]
    )
    kill = Mock()
    monkeypatch.setattr(shard.os, "kill", kill)
    monkeypatch.setattr(shard.os, "killpg", kill)
    with pytest.raises(RuntimeError):
        shard.drain_handoff(spec, tmp_path / "new", 10)
    kill.assert_not_called()


def test_handoff_does_not_signal_until_all_results_exist(tmp_path, monkeypatch):
    spec = handoff_spec(tmp_path)
    write_json(
        Path(spec["current_group"]) / "group.json",
        {"ordered_requests": list(range(13))},
    )
    states = {11: "T", 12: "Z", 13: "S"}
    monkeypatch.setattr(
        shard, "process_state", lambda identity: states[identity["pid"]]
    )
    kill = Mock()
    monkeypatch.setattr(shard, "guarded_signal", kill)
    with pytest.raises(FileNotFoundError):
        shard.drain_handoff(spec, tmp_path / "new", 10)
    kill.assert_not_called()


def test_completed_group_handoff_never_signals_isaac(tmp_path, monkeypatch):
    spec = handoff_spec(tmp_path)
    group = Path(spec["current_group"])
    write_json(group / "group.json", {"ordered_requests": list(range(13))})
    write_json(group / "paired_receipt.json", {"completed": True})
    for index in range(13):
        write_json(group / "results" / f"{index:02d}.json", {"valid_episode_count": 0})
    states = {11: "T", 12: "Z", 13: "S"}
    monkeypatch.setattr(
        shard, "process_state", lambda identity: states[identity["pid"]]
    )
    calls = []

    def send(identity, sig, *, group=False):
        calls.append((identity["pid"], sig, group))
        if sig == signal.SIGCONT or group:
            states[identity["pid"]] = "Z"

    monkeypatch.setattr(shard, "guarded_signal", send)
    shard.drain_handoff(spec, tmp_path / "new", 10)
    assert calls == [
        (11, signal.SIGTERM, False),
        (11, signal.SIGCONT, False),
        (13, signal.SIGTERM, True),
    ]
    assert (tmp_path / "new/adoption_receipt.json").is_file()
    assert not (group / "process_exit.json").exists()


def test_identity_failure_prevents_signal(monkeypatch):
    monkeypatch.setattr(
        shard, "process_state", Mock(side_effect=RuntimeError("changed"))
    )
    kill = Mock()
    monkeypatch.setattr(shard.os, "kill", kill)
    with pytest.raises(RuntimeError, match="changed"):
        shard.guarded_signal({"pid": 123}, signal.SIGTERM)
    kill.assert_not_called()


def test_existing_campaign_is_never_overwritten(tmp_path, monkeypatch):
    binding = tmp_path / "binding.json"
    write_json(binding, {"model": "n0_twam", "tasks": ["lift_can"]})
    campaign = tmp_path / "existing"
    campaign.mkdir()
    run = Mock()
    monkeypatch.setattr(shard, "run_task_seeds", run)
    with pytest.raises(FileExistsError):
        shard.run_shard(binding, campaign, tmp_path, tmp_path, {"lift_can": [0]})
    run.assert_not_called()


def test_signal_group_requires_leader_identity(monkeypatch):
    monkeypatch.setattr(shard, "process_state", lambda identity: "S")
    kill = Mock()
    monkeypatch.setattr(shard.os, "killpg", kill)
    with pytest.raises(RuntimeError, match="unverified"):
        shard.guarded_signal({"pid": 123, "pgid": 456}, signal.SIGTERM, group=True)
    kill.assert_not_called()


@pytest.mark.parametrize(
    "ticks, marker, pgid",
    [(101, "torchrun", 12), (100, "other", 12), (100, "torchrun", 99)],
)
def test_linux_identity_rejects_reused_pid_wrong_command_or_group(
    monkeypatch, ticks, marker, pgid
):
    fields = ["S", "1", "12"] + ["0"] * 16 + ["100"]
    monkeypatch.setattr(
        Path, "read_text", lambda self: "12 (name with spaces) " + " ".join(fields)
    )
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"python\x00torchrun\x00")
    with pytest.raises(RuntimeError):
        shard.process_state(
            {"pid": 12, "start_ticks": ticks, "cmd_marker": marker, "pgid": pgid}
        )


def test_handoff_current_seed_cannot_be_assigned_again(tmp_path):
    binding = tmp_path / "binding.json"
    write_json(binding, {"model": "n0_twam", "tasks": ["lift_can"]})
    group = tmp_path / "old-group"
    write_json(group / "group.json", {"task": "lift_can", "seed": 1})
    with pytest.raises(ValueError, match="overlap"):
        shard.run_shard(
            binding,
            tmp_path / "new",
            tmp_path,
            tmp_path,
            {"lift_can": [1]},
            {"current_group": str(group)},
        )
    assert not (tmp_path / "new").exists()


def test_drain_only_dispatches_no_new_episodes(tmp_path, monkeypatch):
    campaign = tmp_path / "drain"
    drain = Mock()
    run = Mock()
    monkeypatch.setattr(shard, "drain_handoff", drain)
    monkeypatch.setattr(shard, "run_task_seeds", run)
    shard.run_drain_only(campaign, {"test": "handoff"})
    drain.assert_called_once_with({"test": "handoff"}, campaign, 21600)
    run.assert_not_called()
    receipt = json.loads((campaign / "supervisor_exit.json").read_text())
    assert receipt["mode"] == "drain_only"
    assert receipt["new_episodes_launched"] == 0
    assert receipt["status"] == "finished"
    assert json.loads((campaign / "assignment.json").read_text())["assignments"] == {}


def test_drain_only_no_clobber(tmp_path, monkeypatch):
    drain = Mock()
    monkeypatch.setattr(shard, "drain_handoff", drain)
    with pytest.raises(FileExistsError):
        shard.run_drain_only(tmp_path, {})
    drain.assert_not_called()


def test_drain_only_requires_handoff(tmp_path):
    with pytest.raises(ValueError, match="handoff"):
        shard.run_drain_only(tmp_path / "drain", None)
    assert not (tmp_path / "drain").exists()


def test_drain_only_failure_receipt_and_no_dispatch(tmp_path, monkeypatch):
    campaign = tmp_path / "drain"
    monkeypatch.setattr(
        shard, "drain_handoff", Mock(side_effect=RuntimeError("identity"))
    )
    run = Mock()
    monkeypatch.setattr(shard, "run_task_seeds", run)
    with pytest.raises(RuntimeError, match="identity"):
        shard.run_drain_only(campaign, {})
    run.assert_not_called()
    assert (
        json.loads((campaign / "supervisor_exit.json").read_text())["status"]
        == "failed"
    )
    assert (campaign / "shard_error.json").is_file()


def test_drain_only_cli_missing_handoff_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["parallel_shard", "--drain-only", "--campaign", str(tmp_path / "drain")],
    )
    with pytest.raises(SystemExit) as exc:
        shard.main()
    assert exc.value.code == 2
    assert not (tmp_path / "drain").exists()
