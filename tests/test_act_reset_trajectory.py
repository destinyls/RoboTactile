"""Strict ACT reset-trajectory artifact persistence tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import robotactile_benchmark.act_fault_campaign.reset_trajectory as trajectory_io
from robotactile_benchmark.act_fault_campaign.reset_trajectory import (
    ACTResetTrajectoryArtifactError,
    act_reset_trajectory_relpath,
    build_act_trajectory_replay_reference,
    load_act_reset_trajectory,
    write_act_reset_trajectory,
)
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_pairing import (
    _validate_trajectory_reset_reference,
)
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveSegment,
    UniVTACPreMoveTrajectory,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes


def _trajectory() -> UniVTACPreMoveTrajectory:
    gripper = UniVTACPreMoveSegment(
        kind="gripper",
        request_fingerprint="a" * 64,
        start_joint9=(0.0,) * 7 + (0.04, 0.04),
        position=((0.04,), (0.035,), (0.03,)),
        velocity=((-0.01,), (-0.01,), (-0.01,)),
    )
    arm = UniVTACPreMoveSegment(
        kind="arm",
        request_fingerprint="b" * 64,
        start_joint9=(0.0,) * 7 + (0.03, 0.03),
        position=((0.0,) * 7, (0.01,) * 7),
        velocity=((0.1,) * 7, (0.1,) * 7),
    )
    close_gripper = replace(
        gripper,
        request_fingerprint="5" * 64,
        start_joint9=(0.01,) * 7 + (0.03, 0.03),
    )
    lift_arm = replace(
        arm,
        request_fingerprint="6" * 64,
        start_joint9=(0.01,) * 7 + (0.01, 0.01),
    )
    return UniVTACPreMoveTrajectory(
        task_id="grasp_classify",
        action_spec="qpos8_next_step",
        initial_seed=938,
        exogenous_seed=1_000_000,
        pair_key="c" * 64,
        dataset_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
        config_sha256="f" * 64,
        source_run_content_sha256="1" * 64,
        reset_reference_sha256="2" * 64,
        capture_reset_receipt_sha256="7" * 64,
        capture_simulator_state_sha256="8" * 64,
        capture_native_step=240,
        capture_qpos8=(0.125,) * 8,
        source_qpos8_max_abs_error=0.001,
        upstream_commit="3" * 40,
        task_source_sha256="4" * 64,
        post_reset_hold_steps=0,
        segments=(gripper, arm, close_gripper, lift_arm),
    )


def _source_reference() -> UniVTACResetReference:
    return UniVTACResetReference(
        task_id="grasp_classify",
        initial_seed=938,
        exogenous_seed=1_000_000,
        pair_key="c" * 64,
        dataset_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
        config_sha256="f" * 64,
        source_artifact_root_sha256="9" * 64,
        source_result_sha256="a" * 64,
        source_run_content_sha256="1" * 64,
        expected_simulator_state_sha256="b" * 64,
        expected_native_step=239,
        expected_qpos8=(0.25,) * 8,
        qpos_atol=1e-3,
    )


def test_build_replay_reference_uses_exact_captured_endpoint() -> None:
    source = _source_reference()
    trajectory = replace(
        _trajectory(),
        reset_reference_sha256=source.sha256,
    )

    replay = build_act_trajectory_replay_reference(source, trajectory)

    assert replay is not source
    assert replay.expected_simulator_state_sha256 == "8" * 64
    assert replay.expected_native_step == 240
    assert replay.expected_qpos8 == (0.125,) * 8
    assert replay.qpos_atol == 1e-5
    assert replay.task_id == source.task_id
    assert replay.source_artifact_root_sha256 == source.source_artifact_root_sha256
    assert replay.source_result_sha256 == source.source_result_sha256
    assert replay.source_run_content_sha256 == source.source_run_content_sha256


def test_pairing_accepts_capture_bound_source_and_derived_replay_reference() -> None:
    config = build_univtac_backend_config("grasp_classify")
    source = _source_reference()
    trajectory = replace(
        _trajectory(),
        reset_reference_sha256=source.sha256,
        upstream_commit=config.upstream_commit,
        task_source_sha256=config.task.task_source_sha256,
    )
    replay = build_act_trajectory_replay_reference(source, trajectory)

    assert trajectory.reset_reference_sha256 == source.sha256
    assert trajectory.reset_reference_sha256 != replay.sha256
    _validate_trajectory_reset_reference(config, replay, trajectory)


@pytest.mark.parametrize(
    "mutation",
    [
        {"reset_reference_sha256": "0" * 64},
        {"pair_key": "0" * 64},
        {"source_run_content_sha256": "0" * 64},
    ],
)
def test_build_replay_reference_rejects_source_identity_mismatch(
    mutation: dict[str, object],
) -> None:
    source = _source_reference()
    trajectory = replace(
        _trajectory(),
        **({"reset_reference_sha256": source.sha256} | mutation),
    )

    with pytest.raises(
        ACTResetTrajectoryArtifactError,
        match="identity differs",
    ):
        build_act_trajectory_replay_reference(source, trajectory)


def test_relpath_is_canonical_and_rejects_unsafe_identity() -> None:
    pair_key = "a" * 64
    assert act_reset_trajectory_relpath("grasp_classify", pair_key) == (
        f"reset_trajectories/grasp_classify/{pair_key}.json"
    )

    for task in ("", "../task", "/task", "task/name", " task"):
        with pytest.raises(ACTResetTrajectoryArtifactError, match="safe identifier"):
            act_reset_trajectory_relpath(task, pair_key)
    for invalid_pair in ("A" * 64, "a" * 63, "../pair"):
        with pytest.raises(ACTResetTrajectoryArtifactError, match="lowercase SHA256"):
            act_reset_trajectory_relpath("grasp_classify", invalid_pair)


def test_canonical_roundtrip_and_no_clobber(tmp_path: Path) -> None:
    trajectory = _trajectory()
    target = tmp_path / "nested" / "trajectory.json"

    assert write_act_reset_trajectory(target, trajectory) is True
    assert target.read_bytes() == canonical_json_bytes(trajectory.to_dict())
    assert trajectory.to_dict()["post_reset_hold_steps"] == 0
    assert write_act_reset_trajectory(target, trajectory) is False
    loaded = load_act_reset_trajectory(target)
    assert loaded == trajectory
    assert isinstance(loaded.segments, tuple)
    assert isinstance(loaded.segments[0].position, tuple)
    assert isinstance(loaded.segments[0].position[0], tuple)

    different = replace(trajectory, reset_reference_sha256="5" * 64)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_act_reset_trajectory(target, different)
    assert load_act_reset_trajectory(target) == trajectory


def test_write_and_load_reject_symlink_paths(tmp_path: Path) -> None:
    trajectory = _trajectory()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ACTResetTrajectoryArtifactError, match="symlink"):
        write_act_reset_trajectory(linked_parent / "trajectory.json", trajectory)

    target = real_parent / "trajectory.json"
    write_act_reset_trajectory(target, trajectory)
    linked_file = tmp_path / "trajectory-link.json"
    linked_file.symlink_to(target)
    with pytest.raises(ACTResetTrajectoryArtifactError, match="symlink"):
        load_act_reset_trajectory(linked_file)


@pytest.mark.parametrize("mutation", ["extra", "shape", "nan"])
def test_loader_rejects_extra_fields_bad_shapes_and_nan(
    tmp_path: Path,
    mutation: str,
) -> None:
    document = _trajectory().to_dict()
    target = tmp_path / f"{mutation}.json"

    if mutation == "extra":
        document["unexpected"] = True
        target.write_bytes(canonical_json_bytes(document))
    elif mutation == "shape":
        segments = document["segments"]
        assert isinstance(segments, list)
        arm = segments[1]
        assert isinstance(arm, dict)
        arm["position"] = [[0.0] * 6]
        arm["velocity"] = [[0.0] * 6]
        target.write_bytes(canonical_json_bytes(document))
    else:
        segments = document["segments"]
        assert isinstance(segments, list)
        first = segments[0]
        assert isinstance(first, dict)
        first["position"] = [[float("nan")]]
        raw = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=True,
        )
        target.write_text(raw + "\n", encoding="utf-8")

    with pytest.raises(ACTResetTrajectoryArtifactError, match="strict canonical"):
        load_act_reset_trajectory(target)


def test_loader_rejects_noncanonical_json_and_non_files(tmp_path: Path) -> None:
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(_trajectory().to_dict(), indent=2) + "\n")
    with pytest.raises(ACTResetTrajectoryArtifactError, match="strict canonical"):
        load_act_reset_trajectory(pretty)

    with pytest.raises(ACTResetTrajectoryArtifactError, match="regular"):
        load_act_reset_trajectory(tmp_path)


def test_read_and_write_enforce_the_size_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trajectory_io, "MAX_ACT_RESET_TRAJECTORY_BYTES", 16)
    target = tmp_path / "trajectory.json"
    target.write_bytes(b"{" + b" " * 15 + b"}")
    with pytest.raises(ACTResetTrajectoryArtifactError, match="size"):
        load_act_reset_trajectory(target)

    with pytest.raises(ACTResetTrajectoryArtifactError, match="size"):
        write_act_reset_trajectory(tmp_path / "output.json", _trajectory())
