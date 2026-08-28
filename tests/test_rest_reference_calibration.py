"""Rest-reference calibration selection, persistence, CLI, and matrix tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration import (
    build_rest_reference_from_live_artifact,
    load_rest_reference_artifact,
    write_rest_reference_artifact,
)
from robotactile_benchmark.calibration.contracts import (
    RestReferenceArtifactError,
)
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.contracts import (
    array_sha256,
    build_evaluation_record,
    canonical_hash,
)
from robotactile_benchmark.execution import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    load_live_univtac_artifact,
    load_live_univtac_request,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.rest_references import ReferenceSplit
from robotactile_benchmark.trials import Condition

ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "scripts/live_univtac/generate_pull_out_key_matrix.py"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _calibrated_episode(task: str, length: int = 12):
    config = build_univtac_backend_config(task)
    calibration = config.aliases.calibration_config_sha256
    records = []
    for record in make_synthetic_episode(length):
        sensors = tuple(
            replace(
                sensor,
                payload=np.resize(sensor.payload, config.tactile_rgb_shape).astype(
                    np.uint8
                ),
            )
            for sensor in record.observation.tactile
        )
        observation = replace(record.observation, tactile=sensors)
        provenance = tuple(
            replace(
                item,
                calibration_sha256=calibration,
                payload_sha256=array_sha256(observation.sensor(item.slot_id).payload),
            )
            for item in record.provenance
        )
        records.append(build_evaluation_record(observation, provenance))
    return tuple(records)


def _request(root: Path, task: str = "pull_out_key") -> LiveUniVTACRunRequest:
    return LiveUniVTACRunRequest(
        task_id=task,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.ACT,
        base_system_id="rest-calibration-policy-v1",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        base_system_manifest_sha256=None,
        initial_seed=31,
        exogenous_seed=47,
        max_control_cycles=11,
        max_observation_steps=12,
        execute_action_steps=1,
        wall_timeout_s=10.0,
        upstream_root=root / "upstream",
        runtime_dir=root / "runtime",
        output_dir=None,
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu",
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
    )


def _source_artifact(root: Path, task: str = "pull_out_key") -> Path:
    loaded = load_live_univtac_run(_request(root, task))
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        DeterministicFakeBackend(
            _calibrated_episode(task),
            success_predicate_id=loaded.run_spec.success_predicate_id,
        ),
        DeterministicFakePolicy.for_trial(loaded.trial),
    )
    output = root / "source-live"
    write_live_univtac_artifact(output, loaded, evidence)
    return output


def _calibration_artifact(root: Path, task: str = "pull_out_key") -> Path:
    source = load_live_univtac_artifact(_source_artifact(root, task))
    references, validation = build_rest_reference_from_live_artifact(
        source, ReferenceSplit.CALIBRATION, 3
    )
    output = root / "rest-calibration"
    write_rest_reference_artifact(output, references, validation)
    return output


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_selects_earliest_longest_free_run_and_center_frame() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = load_live_univtac_artifact(_source_artifact(root))
        references, validation = build_rest_reference_from_live_artifact(
            source, ReferenceSplit.CALIBRATION, 3
        )

        assert (validation.qualified_start_index, validation.qualified_stop_index) == (
            0,
            3,
        )
        assert validation.selected_step_index == 1
        assert references.source_artifact_sha256 == source.root_receipt_sha256
        assert references.split_manifest_sha256 == source.trial.dataset_sha256
        selected = source.evidence.finalization.clean_records[1]  # type: ignore[union-attr]
        for slot in ("left", "right"):
            assert (
                references.payload_for(slot).tobytes()
                == selected.observation.sensor(slot).payload.tobytes()  # type: ignore[union-attr]
            )


def test_artifact_round_trip_is_strict_idempotent_and_byte_identical() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = load_live_univtac_artifact(_source_artifact(root))
        references, validation = build_rest_reference_from_live_artifact(
            source, ReferenceSplit.VALIDATION, 3
        )
        first = root / "first"
        second = root / "second"

        receipt = write_rest_reference_artifact(first, references, validation)
        write_rest_reference_artifact(second, references, validation)
        repeated = write_rest_reference_artifact(first, references, validation)
        loaded = load_rest_reference_artifact(first)

        assert receipt == repeated
        assert _inventory(first) == _inventory(second)
        assert loaded.references.sha256 == references.sha256
        assert loaded.validation.sha256 == validation.sha256
        assert loaded.root_receipt_sha256 == receipt.artifact_root_sha256
        assert loaded.root_receipt.simulator_qualification_claimed is False


def test_rejects_synthetic_split_short_run_and_semantic_tamper() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = load_live_univtac_artifact(_source_artifact(root))
        with pytest.raises(RestReferenceArtifactError):
            build_rest_reference_from_live_artifact(
                source, ReferenceSplit.SYNTHETIC_DEVELOPMENT, 3
            )
        with pytest.raises(RestReferenceArtifactError):
            build_rest_reference_from_live_artifact(
                source, ReferenceSplit.CALIBRATION, 4
            )

        output = _calibration_artifact(root / "tamper")
        validation_path = output / "no_contact_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["selected_payload_sha256"]["left"] = "f" * 64
        validation_raw = _canonical(validation)
        validation_path.write_bytes(validation_raw)
        root_path = output / "root_receipt.json"
        receipt = json.loads(root_path.read_text(encoding="utf-8"))
        receipt["no_contact_validation_sha256"] = canonical_hash(validation)
        for member in receipt["members"]:
            if member["path"] == "no_contact_validation.json":
                member["sha256"] = hashlib.sha256(validation_raw).hexdigest()
                member["size_bytes"] = len(validation_raw)
        root_path.write_bytes(_canonical(receipt))

        with pytest.raises(RestReferenceArtifactError):
            load_rest_reference_artifact(output)


def test_cli_builds_same_strict_artifact_twice() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source_artifact(root)
        output = root / "cli-output"
        command = [
            sys.executable,
            "-m",
            "robotactile_benchmark.cli",
            "build-rest-references",
            "--source-live-artifact",
            str(source),
            "--dataset-split",
            "calibration",
            "--minimum-consecutive-free-records",
            "3",
            "--output",
            str(output),
        ]
        first = subprocess.run(command, check=False, capture_output=True, text=True)
        second = subprocess.run(command, check=False, capture_output=True, text=True)

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert json.loads(first.stdout) == json.loads(second.stdout)
        assert json.loads(first.stdout)["artifact_root_sha256"] == (
            load_rest_reference_artifact(output).root_receipt_sha256
        )


def test_pull_out_key_generator_accepts_bound_rest_reference_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        calibration = _calibration_artifact(root / "calibration-source")
        output = root / "matrix"
        command = [
            sys.executable,
            str(GENERATOR),
            "--deployment-root",
            str(root / "deployment"),
            "--univtac-root",
            str(root / "deployment/src/UniVTAC"),
            "--checkpoint-root",
            str(root / "checkpoints"),
            "--tactile-checkpoint-sha256",
            "1" * 64,
            "--vision-checkpoint-sha256",
            "2" * 64,
            "--stats-sha256",
            "3" * 64,
            "--encoder-sha256",
            "4" * 64,
            "--initial-seed",
            "17",
            "--exogenous-seed",
            "29",
            "--operator",
            "F2_spatial_sensitivity_loss",
            "--rest-reference-artifact",
            str(calibration),
            "--output",
            str(output),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        assert completed.returncode == 0, completed.stderr
        calibration_loaded = load_rest_reference_artifact(output / "rest_references")
        assert calibration_loaded.references.sha256 == (
            load_rest_reference_artifact(calibration).references.sha256
        )
        for condition in (Condition.FAULTED, Condition.RESTORED):
            loaded = load_live_univtac_run(
                load_live_univtac_request(
                    output / "requests" / f"{condition.value}.json"
                )
            )
            assert loaded.rest_references is not None
            assert loaded.fault_manifest is not None
            assert loaded.fault_manifest.parameters["rest_reference_sha256"] == (
                loaded.rest_references.sha256
            )
