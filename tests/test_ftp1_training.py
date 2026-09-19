from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.ftp1_policy.training.commands import (
    dataset_config,
    norm_command,
    parser_command,
    task_zarr_path,
    training_command,
)
from scripts.ftp1_policy.training.contracts import (
    TASK_IDS,
    FTP1TrainingRequest,
    canonical_json_sha256,
    validate_source_manifest,
)
from scripts.ftp1_policy.training.orchestrate import run
from scripts.ftp1_policy.training.runtime import explicit_environment


def _request() -> FTP1TrainingRequest:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "experiments"
        / "ftp1_policy"
        / "train759_all8.example.json"
    )
    return FTP1TrainingRequest.load(path)


def _source_manifest() -> dict[str, object]:
    episodes: list[dict[str, object]] = []
    for task in TASK_IDS:
        for episode_id in range(100):
            if episode_id in {0, 1, 2, 3, 5}:
                split = "frozen"
            elif task == "grasp_classify" and episode_id == 90:
                split = "quarantine"
            else:
                split = "train"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "mtime_ns": 1,
                    "relative_path": f"{task}/clean/{episode_id}.hdf5",
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                    "split": split,
                    "task": task,
                    "usable_source_range": None,
                }
            )
    payload: dict[str, object] = {
        "action_per_frame": 4,
        "action_schema": "ee20_absolute_next_step",
        "episodes": episodes,
        "gripper_source": "embodiment/joint[:,7]",
        "materialized_splits": ["train"],
        "protocol_id": "univtac_train759_absee20_v1",
        "quaternion_order": "wxyz",
        "raw_root": "/source/UniVTAC",
        "schema_version": 1,
        "source_fps": 10,
        "split_counts": {"frozen": 40, "quarantine": 1, "train": 759},
        "used_action_channel_ids": list(range(10)),
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def test_training_request_round_trip_and_all_tasks() -> None:
    request = _request()

    assert request.tasks == TASK_IDS
    assert request.num_train_steps == 160_000
    assert request.batch_size == 16
    assert request.to_dict()["vision_tactile_policy"] == (
        "vision_plus_two_tactile_always_on_v1"
    )
    assert request.to_dict()["training_scope"] == "single_joint_checkpoint_all8_v1"
    assert FTP1TrainingRequest.from_dict(request.to_dict()) == request


def test_training_command_forces_vision_and_two_tactile_streams() -> None:
    request = _request()
    config_path = request.output_root / "configs" / "joint_all8.json"

    command = training_command(request, config_path, resume=False)

    assert "--model.use_tactile_input" in command
    assert "--non_tactile_dropout_ratio=0.0" in command
    assert (
        "--model.tactile_tokenizer_config.no_load_t3_pretrained_checkpoint" in command
    )
    assert "--val_ratio=0.0" in command
    assert "--no-wandb_enabled" in command
    assert "--pytorch_training_precision=bfloat16" in command
    assert f"--pytorch_weight_path={request.base_checkpoint_dir}" in command
    assert "--resume" not in command


def test_runtime_prefers_pinned_ftp1_source_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request()
    request = FTP1TrainingRequest.from_dict(
        {**request.to_dict(), "output_root": str(tmp_path / "output")}
    )
    monkeypatch.setenv("PYTHONPATH", "/inherited/pythonpath")

    environment = explicit_environment(request)
    pythonpath = environment["PYTHONPATH"].split(os.pathsep)

    assert pythonpath[:2] == [
        str(request.ftp1_root / "src"),
        str(request.ftp1_root),
    ]
    assert pythonpath[2] == "/inherited/pythonpath"
    assert environment["TMPDIR"].startswith("/tmp/rtf1-")
    assert len(environment["TMPDIR"]) < 32


def test_norm_command_uses_finalize_compatibility_entrypoint() -> None:
    request = _request()
    config_path = request.output_root / "configs" / "joint_all8.json"

    command = norm_command(request, config_path)

    assert Path(command[1]).name == "norm_entrypoint.py"
    assert f"--dataset_config_path={config_path}" in command


def test_resume_does_not_reload_pretrained_weights() -> None:
    request = _request()
    config_path = request.output_root / "configs" / "joint_all8.json"

    command = training_command(request, config_path, resume=True)

    assert "--resume" in command
    assert not any(item.startswith("--pytorch_weight_path=") for item in command)


def test_camera_routing_matches_upstream_univtac_tasks() -> None:
    request = _request()

    assert task_zarr_path(request, "insert_tube").name == "insert_tube_all.zarr"
    assert task_zarr_path(request, "lift_can").name == "lift_can_all.zarr"
    assert task_zarr_path(request, "insert_hole").name == "insert_hole_head.zarr"
    datasets = dataset_config(request)["datasets"]
    assert isinstance(datasets, list)
    assert len(datasets) == 8
    assert [entry["name"] for entry in datasets] == [
        f"UniVTAC_{task}" for task in TASK_IDS
    ]
    assert datasets[2] == {
        "name": "UniVTAC_insert_hole",
        "path": str((request.zarr_root / "insert_hole").resolve()),
        "use_trajectory_ratio": 1.0,
        "enabled": True,
    }


def test_parser_uses_robotactile_rgb_boundary() -> None:
    request = _request()

    command = parser_command(request, "insert_tube")

    assert Path(command[1]).name == "rgb_parser_entrypoint.py"
    assert f"--ftp1-root={request.ftp1_root}" in command
    assert "--task-list=insert_tube" in command
    assert "parse_data_module.parse_data_univtac" not in command


def test_dataset_config_contains_all_tasks_in_one_model() -> None:
    request = _request()

    assert dataset_config(request)["datasets"] == [
        {
            "name": f"UniVTAC_{task}",
            "path": str((request.zarr_root / task).resolve()),
            "use_trajectory_ratio": 1.0,
            "enabled": True,
        }
        for task in TASK_IDS
    ]


def test_source_manifest_selects_exact_train759() -> None:
    records = validate_source_manifest(_source_manifest())

    assert len(records) == 759
    assert sum(item["task"] == "grasp_classify" for item in records) == 94
    assert all(item["split"] == "train" for item in records)


def test_source_manifest_rejects_digest_or_split_drift() -> None:
    digest_drift = _source_manifest()
    digest_drift["raw_root"] = "/changed"
    with pytest.raises(ValueError, match="digest"):
        validate_source_manifest(digest_drift)

    split_drift = _source_manifest()
    episodes = split_drift["episodes"]
    assert isinstance(episodes, list)
    first = episodes[0]
    assert isinstance(first, dict)
    first["split"] = "train"
    split_drift["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in split_drift.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="per-task"):
        validate_source_manifest(split_drift)


def test_all_phase_parses_all_tasks_before_one_joint_norm_and_train(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request()
    request = FTP1TrainingRequest.from_dict(
        {**request.to_dict(), "output_root": str(tmp_path / "output")}
    )
    events: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.write_or_verify",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.write_status",
        lambda _request, *, phase, task, completed_tasks, prepared_tasks, error=None: (
            events.append((phase, task))
        ),
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.preflight",
        lambda _request: events.append(("preflight-call", None)),
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.stage_sources",
        lambda _request, *, verify_hashes: events.append(("stage-call", None)),
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.parse_task",
        lambda _request, task: events.append(("parse-call", task)),
    )

    def normalize_joint(_request: FTP1TrainingRequest) -> Path:
        events.append(("norm-call", None))
        return Path(str(request.output_root)) / "configs" / "joint_all8.json"

    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.normalize_joint",
        normalize_joint,
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.train_joint",
        lambda _request, _config: events.append(("train-call", None)),
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.completed_tasks",
        lambda _request: [],
    )
    monkeypatch.setattr(
        "scripts.ftp1_policy.training.orchestrate.prepared_tasks",
        lambda _request: [],
    )

    run(request, phase="all", task=None, verify_source_hashes=False)

    parse_events = [("parse-call", task) for task in TASK_IDS]
    assert all(event in events for event in parse_events)
    assert events.count(("norm-call", None)) == 1
    assert events.count(("train-call", None)) == 1
    assert max(events.index(event) for event in parse_events) < events.index(
        ("norm-call", None)
    )
    assert events.index(("norm-call", None)) < events.index(("train-call", None))


def test_task_selection_cannot_enter_joint_training(tmp_path: Path) -> None:
    request = _request()
    request = FTP1TrainingRequest.from_dict(
        {**request.to_dict(), "output_root": str(tmp_path / "output")}
    )

    with pytest.raises(ValueError, match="only valid"):
        run(
            request,
            phase="all",
            task="insert_hole",
            verify_source_hashes=False,
        )
