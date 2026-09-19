"""Supplement preparation never starts runtimes or overwrites previous results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_retrained_group_options import binding

from scripts.retrained_evaluation import availability_campaign
from scripts.retrained_evaluation.availability_campaign import (
    prepare_supplement,
    run_supplement,
)
from scripts.retrained_evaluation.group import prepare_group, write_json


def test_prepare_native_and_zero_are_separate_and_preserve_binding(
    tmp_path: Path,
) -> None:
    source = binding(tmp_path)
    original = json.dumps(source, sort_keys=True)
    prepared = []
    for mode in ("native_missing_v1", "zero_fill_v1"):
        path = prepare_supplement(
            source,
            tmp_path / mode,
            task="lift_can",
            seeds=(0, 1, 2),
            mode=mode,
            zero_shape=(240, 320, 3) if mode == "zero_fill_v1" else None,
        )
        raw = json.loads(path.read_text())
        prepared.append(raw)
        plan = json.loads((path.parent / "supplement.json").read_text())
        assert plan["status"] == "prepared_not_executed"
        assert plan["planned_episode_count"] == 9
        assert not (path.parent / "campaign").exists()
        assert len(list((path.parent / "prepared").glob("*/group.json"))) == 3
        with pytest.raises(FileExistsError):
            prepare_supplement(
                source,
                path.parent,
                task="lift_can",
                seeds=(0,),
                mode="native_missing_v1",
            )
    assert prepared[0]["binding_sha256"] != prepared[1]["binding_sha256"]
    assert json.dumps(source, sort_keys=True) == original


@pytest.mark.parametrize("seeds", [(), (0, 0), (-1,), (True,)])
def test_bad_seeds_create_nothing(tmp_path: Path, seeds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        prepare_supplement(
            binding(tmp_path),
            tmp_path / "bad",
            task="lift_can",
            seeds=seeds,
            mode="native_missing_v1",
        )
    assert not (tmp_path / "bad").exists()


def test_run_supplement_serial_end_to_end_and_duplicate_start_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "supplement"
    code, package = tmp_path / "code", tmp_path / "package"
    binding_path = prepare_supplement(
        binding(tmp_path),
        output,
        task="lift_can",
        seeds=(0, 1, 2),
        mode="native_missing_v1",
    )
    events: list[tuple[str, int, str]] = []
    result_paths: list[Path] = []
    video_paths: list[Path] = []

    def fake_run_one(
        supplied_binding: Path,
        task: str,
        seed_root: Path,
        supplied_code: Path,
        supplied_package: Path,
        seed: int,
    ) -> None:
        assert supplied_binding == binding_path
        assert supplied_code == code and supplied_package == package
        assert task == "lift_can"
        assert seed_root == output / "campaign/seeds" / f"seed-{seed:03d}"
        events.append(("run", seed, task))
        value = json.loads(supplied_binding.read_text())
        group_path = prepare_group(
            value, task, seed_root / "groups/n0_vtla" / task, seed
        )
        plan = json.loads(group_path.read_text())
        assert len(plan["ordered_requests"]) == 3
        for index, name in enumerate(plan["ordered_requests"]):
            request = json.loads((group_path.parent / name).read_text())
            artifact = Path(request["output_dir"])
            artifact.mkdir(parents=True)
            path = group_path.parent / "results" / f"{index:02d}.json"
            write_json(
                path,
                {
                    "artifact": str(artifact),
                    "score_eligible": True,
                    "validation_passed": True,
                    "failure_stage": None,
                    "terminal_status": "success",
                    "score_success": True,
                    "control_cycle_count": 4,
                    "observation_count": 156,
                },
            )
            result_paths.append(path)

    class FakeExport:
        def __init__(self, destination: Path) -> None:
            self.destination = destination

        def to_cli_dict(self) -> dict[str, Any]:
            return {"mock_export": True, "output": str(self.destination)}

    def fake_export(
        artifact: Path,
        destination: Path,
        *,
        fps: int,
        stride: int,
        video: bool,
    ) -> FakeExport:
        assert artifact.is_dir()
        assert artifact.is_relative_to(output / "campaign")
        assert fps == 10 and stride == 1 and video is True
        seed = int(destination.parent.name.removeprefix("seed-"))
        assert destination == output / "videos" / f"seed-{seed:03d}" / destination.name
        events.append(("export", seed, destination.name))
        destination.mkdir(parents=True)
        video_paths.append(destination)
        return FakeExport(destination)

    monkeypatch.setattr(availability_campaign, "run_one", fake_run_one)
    monkeypatch.setattr(
        "robotactile_benchmark.visualization.live_artifact.export_live_artifact_visualization",
        fake_export,
    )
    run_supplement(output, code=code, package=package)
    conditions = ("clean", "A1_stream_absence", "A2_frame_erasure")
    assert events == [
        event
        for seed in (0, 1, 2)
        for event in [
            ("run", seed, "lift_can"),
            *[("export", seed, condition) for condition in conditions],
        ]
    ]
    assert len(result_paths) == len(set(result_paths)) == 9
    assert len(video_paths) == len(set(video_paths)) == 9
    assert all(path.is_file() for path in result_paths)
    for seed in (0, 1, 2):
        report_path = output / "reports" / f"after-seed-{seed:03d}.json"
        report = json.loads(report_path.read_text())
        assert report["planned_episode_count"] == 9
        assert report["completed_execution_count"] == 3 * (seed + 1)
        for condition in conditions:
            receipt = output / "videos" / f"seed-{seed:03d}" / f"{condition}.json"
            assert json.loads(receipt.read_text())["mock_export"] is True
    completed_path = output / "completed.json"
    completed_bytes = completed_path.read_bytes()
    completed = json.loads(completed_bytes)
    assert completed["status"] == "execution_finished"
    assert completed["summary"]["completed_execution_count"] == 9
    assert completed["summary"]["eligible_episode_count"] == 9
    assert (output / "started.json").is_file()
    with pytest.raises(FileExistsError):
        run_supplement(output, code=code, package=package)
    assert len(events) == 12
    assert completed_path.read_bytes() == completed_bytes
