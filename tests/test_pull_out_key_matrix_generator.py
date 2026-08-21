"""Behavior tests for the first reproducible pull-out-key live matrix."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from robotactile_benchmark.execution import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import Condition, system_manifest_hash

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/live_univtac/generate_pull_out_key_matrix.py"
TACTILE_CHECKPOINT = "1" * 64
VISION_CHECKPOINT = "2" * 64
STATS = "3" * 64
ENCODER = "4" * 64
TACTILE_CONFIG = "acdab30e50fa7280918804c4a196f75a6db6e854a3c7a86a0ddd84791c533397"
VISION_CONFIG = "427c54337b56b0958606de4a235b6e13527877f75c21c2939045b5a6b1375cc2"


def _command(root: Path, output: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--deployment-root",
        str(root / "deployment"),
        "--univtac-root",
        str(root / "deployment/src/UniVTAC"),
        "--checkpoint-root",
        str(root / "checkpoints"),
        "--tactile-checkpoint-sha256",
        TACTILE_CHECKPOINT,
        "--vision-checkpoint-sha256",
        VISION_CHECKPOINT,
        "--stats-sha256",
        STATS,
        "--encoder-sha256",
        ENCODER,
        "--initial-seed",
        "17",
        "--exogenous-seed",
        "29",
        "--output",
        str(output),
        *extra,
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else f"{source}{os.pathsep}{environment['PYTHONPATH']}"
    )
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_emits_loadable_matched_four_condition_matrix() -> None:
    """Catches a control checkpoint leaking into the shared pair identity."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "matrix"
        completed = _run(_command(root, output))

        assert completed.returncode == 0, completed.stderr
        requests = {
            condition: load_live_univtac_request(
                output / "requests" / f"{condition.value}.json"
            )
            for condition in Condition
        }
        loaded = {
            condition: load_live_univtac_run(request)
            for condition, request in requests.items()
        }
        base_hash = system_manifest_hash(
            "official-univtac-act.pull_out_key.univtac.policy_last.v1",
            TACTILE_CHECKPOINT,
            TACTILE_CONFIG,
            "qpos8_next_step",
        )

        assert tuple(requests) == tuple(Condition)
        assert len({item.trial.pair_key for item in loaded.values()}) == 1
        assert {
            request.base_system_manifest_sha256 for request in requests.values()
        } == {base_hash}
        for condition in (Condition.CLEAN, Condition.FAULTED, Condition.RESTORED):
            assert requests[condition].checkpoint_sha256 == TACTILE_CHECKPOINT
            assert requests[condition].config_sha256 == TACTILE_CONFIG
        no_touch = requests[Condition.NO_TOUCH]
        assert no_touch.checkpoint_sha256 == VISION_CHECKPOINT
        assert no_touch.config_sha256 == VISION_CONFIG
        assert no_touch.matched_no_touch_system_id == (
            "official-univtac-act.pull_out_key.vision_only.policy_last.v1"
        )
        assert (
            no_touch.matched_no_touch_artifact_path
            == (
                root / "checkpoints/pull_out_key/vision_only/policy_last.ckpt"
            ).absolute()
        )


def test_generator_freezes_trial_identity_fault_windows_and_live_devices() -> None:
    """Catches data-file claims, truncated horizons, or physical GPU indices."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "matrix"
        completed = _run(_command(root, output))
        assert completed.returncode == 0, completed.stderr

        trial_set = output / "trial_set_manifest.json"
        receipt = _json(output / "matrix_receipt.json")
        dataset_identity = receipt["dataset_identity"]
        assert isinstance(dataset_identity, dict)
        assert dataset_identity["identity_kind"] == (
            "frozen_trial_set_manifest_content_sha256"
        )
        assert dataset_identity["sha256"] == _file_sha256(trial_set)
        assert dataset_identity["is_raw_dataset_file_hash"] is False
        policy_artifacts = receipt["policy_artifacts"]
        assert isinstance(policy_artifacts, dict)
        tactile_artifact = policy_artifacts["univtac"]
        vision_artifact = policy_artifacts["vision_only"]
        assert isinstance(tactile_artifact, dict)
        assert isinstance(vision_artifact, dict)
        assert tactile_artifact["checkpoint_sha256"] == TACTILE_CHECKPOINT
        assert vision_artifact["checkpoint_sha256"] == VISION_CHECKPOINT
        assert {tactile_artifact["stats_sha256"], vision_artifact["stats_sha256"]} == {
            STATS
        }
        assert {
            tactile_artifact["encoder_sha256"],
            vision_artifact["encoder_sha256"],
        } == {ENCODER}

        persistent = FaultManifest.from_dict(
            _json(output / "fault_manifests/persistent.json")
        )
        restored = FaultManifest.from_dict(
            _json(output / "fault_manifests/restored.json")
        )
        assert persistent.operator_id == "T1_fixed_source_delay"
        assert persistent.severity_level == 3
        assert persistent.start_index == 16
        assert persistent.stop_index == 301
        assert restored.stop_index == 180
        assert persistent.parameters["lag_frames"] == 4
        assert restored.parameters["lag_frames"] == 4

        requests = [
            load_live_univtac_request(output / "requests" / f"{value.value}.json")
            for value in Condition
        ]
        assert {request.dataset_sha256 for request in requests} == {
            _file_sha256(trial_set)
        }
        assert {request.max_control_cycles for request in requests} == {300}
        assert {request.max_observation_steps for request in requests} == {301}
        assert {request.act_device_name for request in requests} == {"cuda:0"}
        assert {request.simulator_device for request in requests} == {"cuda:0"}
        assert all(
            dict(request.launcher_args) == {"headless": True, "enable_cameras": True}
            for request in requests
        )
        assert len({request.output_dir for request in requests}) == 4
        assert len({request.runtime_dir for request in requests}) == 4


def test_generator_is_idempotent_and_refuses_different_content() -> None:
    """Catches partial rewrites or silent replacement of a frozen matrix."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "matrix"
        command = _command(root, output)
        first = _run(command)
        before = {
            path.relative_to(output): _file_sha256(path)
            for path in output.rglob("*")
            if path.is_file()
        }
        second = _run(command)
        conflict = _run(_command(root, output, "--exogenous-seed", "30"))
        after = {
            path.relative_to(output): _file_sha256(path)
            for path in output.rglob("*")
            if path.is_file()
        }

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert json.loads(second.stdout)["status"] == "already_present"
        assert conflict.returncode != 0
        assert "refusing to overwrite" in conflict.stderr
        assert after == before


def test_generator_rejects_unbound_rest_references_and_invalid_delay_warmup() -> None:
    """Catches manifests that could not be executed with their declared inputs."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rest_required = _run(
            _command(
                root,
                root / "rest-required",
                "--operator",
                "F2_spatial_sensitivity_loss",
            )
        )
        invalid_warmup = _run(
            _command(
                root,
                root / "invalid-warmup",
                "--fault-start",
                "3",
            )
        )

        assert rest_required.returncode != 0
        assert "requires --rest-reference-artifact" in rest_required.stderr
        assert not (root / "rest-required").exists()
        assert invalid_warmup.returncode != 0
        assert "delay warm-up" in invalid_warmup.stderr
        assert not (root / "invalid-warmup").exists()
