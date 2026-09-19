"""Local supervisor contracts, using no model or simulator processes."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.n0_twam import multi_task_robustness as multi
from scripts.retrained_evaluation import campaign as runner
from scripts.retrained_evaluation.group import write_json


def binding_file(tmp_path: Path, *, rest: bool = True) -> Path:
    root = tmp_path / "repo/deployment"
    runtime = root / "runtime/python"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    isaac = root / "runtime/isaac-sim-4.5.0/python.sh"
    isaac.parent.mkdir()
    isaac.touch()
    path = tmp_path / "binding.json"
    write_json(
        path,
        {
            "deployment_root": str(root),
            "runtime_python": str(runtime),
            "model": "n0_twam",
            "tasks": ["lift_can"],
            "port": 0,
            "artifact": str(tmp_path / "weights"),
            "rest_references": {"lift_can": "/certified/task-rest"} if rest else {},
            "evaluation": {"measure_n0_rest": True},
        },
    )
    return path


@pytest.mark.parametrize("rest", [True, False])
def test_one_server_three_fresh_groups_and_at_most_one_calibration(
    tmp_path, monkeypatch, rest
):
    binding = binding_file(tmp_path, rest=rest)
    launches = []
    plans = []

    def popen(command, **kwargs):
        launches.append(command)
        if "scripts.retrained_evaluation.group" in command:
            plan = Path(command[command.index("--group") + 1])
            write_json(plan.parent / "results/00.json", {})
            write_json(plan.parent / "paired_receipt.json", {})
        return Mock(pid=123, poll=Mock(return_value=0), wait=Mock(return_value=0))

    def prepare(data, task, group, seed):
        plans.append((group, seed, data["rest_references"][task]))
        write_json(
            group / "group.json",
            {"seed": seed, "ordered_requests": ["requests/clean.json"]},
        )
        return group / "group.json"

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.socket, "socket", MagicMock())
    monkeypatch.setattr(runner, "wait_ready", Mock())
    monkeypatch.setattr(runner, "prepare_group", prepare)
    completed = []
    destination = tmp_path / "campaign/tasks/lift_can"
    runner.run_task_seeds(
        binding,
        "lift_can",
        destination,
        tmp_path,
        tmp_path,
        [0, 1, 2],
        lambda seed, group: completed.append((seed, group)),
    )
    assert sum("torch.distributed.run" in command for command in launches) == 1
    assert (
        sum("scripts.retrained_evaluation.group" in command for command in launches)
        == 3
    )
    assert sum("calibrate" in command for command in launches) == int(not rest)
    assert [item[1] for item in plans] == [0, 1, 2]
    assert len({str(item[0]) for item in plans}) == 3
    assert len({item[2] for item in plans}) == 1
    assert [item[0] for item in completed] == [0, 1, 2]
    for seed, group in completed:
        assert (
            group
            == destination / "seeds" / f"seed-{seed:03d}" / "groups/n0_twam/lift_can"
        )
        assert (group / "process_exit.json").exists()


def test_legacy_run_one_preserves_layout(tmp_path, monkeypatch):
    binding = binding_file(tmp_path)
    called = Mock()
    monkeypatch.setattr(runner, "_run_task_seeds", called)
    runner.run_one(binding, "lift_can", tmp_path, tmp_path, tmp_path, 4)
    assert called.call_args.args[-1] == [4]
    assert called.call_args.kwargs == {"recovery_group": None}


def test_report_failure_does_not_repeat_seed_or_abort_following_seed(
    tmp_path, monkeypatch
):
    binding = binding_file(tmp_path)
    seeds_seen = []

    def run(binding_path, task, destination, code, package, seeds, callback):
        config = read_object(binding_path)
        assert config["evaluation"]["fault_window_mode"] == "early_random_onset_v1"
        assert config["evaluation"]["fault_onset_max_index"] == 8
        assert "fault_start_index" not in config["evaluation"]
        assert config["evaluation"]["capture_profile"] == "paper_full_v1"
        for seed in seeds:
            seeds_seen.append(seed)
            callback(
                seed,
                destination / "seeds" / f"seed-{seed:03d}" / "groups/n0_twam" / task,
            )

    monkeypatch.setattr(multi, "run_task_seeds", run)
    monkeypatch.setattr(
        multi, "report", Mock(side_effect=OSError("export unavailable"))
    )
    target = tmp_path / "campaign"
    multi.run_campaign(binding, target, tmp_path, tmp_path, ["lift_can"], [0, 1, 2])
    assert seeds_seen == [0, 1, 2]
    events = [read_object(path) for path in (target / "events").glob("*.json")]
    assert sum(item["stage"] == "report_error" for item in events) == 3
    assert read_object(target / "supervisor_exit.json")["status"] == "finished"
    with pytest.raises(FileExistsError):
        multi.run_campaign(binding, target, tmp_path, tmp_path, ["lift_can"], [0, 1, 2])
    assert seeds_seen == [0, 1, 2]


def test_summary_separates_success_invalid_missing_and_unsupported(tmp_path):
    for seed in [0, 1]:
        group = (
            tmp_path
            / "tasks/lift_can/seeds"
            / f"seed-{seed:03d}"
            / "groups/n0_twam/lift_can"
        )
        write_json(
            group / "group.json",
            {
                "ordered_requests": [
                    "requests/clean.json",
                    "requests/optical_marker_extreme_v1/F6_history_residual_imprint.json",
                ]
            },
        )
        write_json(
            group / "results/00.json",
            {
                "score_eligible": True,
                "score_success": seed == 0,
                "failure_stage": None,
                "terminal_status": "success" if seed == 0 else "timeout",
                "validation_passed": True,
                "artifact": "/source/artifact",
            },
        )
        write_json(
            group / "results/01.json",
            {
                "score_eligible": True,
                "score_success": True,
                "failure_stage": None,
                "terminal_status": "success",
                "validation_passed": False,
                "artifact": "/source/invalid",
            },
        )
    summary = multi.summarize(tmp_path, ["lift_can"], [0, 1, 2])
    cells = {row["condition"]: row for row in summary["cells"]}
    assert cells["clean"]["eligible"] == 2
    assert cells["clean"]["success_rate"] == 0.5
    assert cells["clean"]["missing"] == 1
    assert cells["F6_history_residual_imprint"]["invalid"] == 2
    assert cells["F6_history_residual_imprint"]["success_rate"] is None
    assert cells["F6_history_residual_imprint"]["attempted"] == 2
    assert cells["A1_stream_absence"]["unsupported"] == 3
    assert cells["A1_stream_absence"]["attempted"] == 0


def test_infrastructure_failure_finishes_without_inventing_episodes(
    tmp_path, monkeypatch
):
    binding = binding_file(tmp_path)
    monkeypatch.setattr(
        multi, "run_task_seeds", Mock(side_effect=RuntimeError("server failed"))
    )
    target = tmp_path / "campaign"
    multi.run_campaign(binding, target, tmp_path, tmp_path, ["lift_can"], [0, 1, 2])
    result = read_object(Path(read_object(target / "supervisor_exit.json")["summary"]))
    clean = next(row for row in result["cells"] if row["condition"] == "clean")
    assert clean["attempted"] == clean["invalid"] == 0
    assert clean["missing"] == 3
