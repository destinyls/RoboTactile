"""Primary 14-by-5 request generation, strict reload, and CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration import write_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    NO_CONTACT_PREDICATE_ID,
    NoContactValidationReceipt,
)
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    REST_REFERENCE_OPERATOR_IDS,
    SENSOR_SLOTS,
)
from robotactile_benchmark.contracts import array_sha256, canonical_hash
from robotactile_benchmark.matrix import (
    PrimaryMatrixGenerationError,
    PrimaryMatrixGenerationSpec,
    generate_primary_matrix_bundle,
    load_primary_matrix_generation,
)
from robotactile_benchmark.matrix.primary_generation_io import scan_files
from robotactile_benchmark.rest_references import (
    ReferenceSplit,
    RestReferenceBundle,
)
from robotactile_benchmark.trials import Condition


def _rest_reference(root: Path, task_id: str = "pull_out_key") -> Path:
    config = build_univtac_backend_config(task_id)
    payloads = {
        slot: np.full(config.tactile_rgb_shape, 11 + index, dtype=np.uint8)
        for index, slot in enumerate(SENSOR_SLOTS)
    }
    payload_hashes = {slot: array_sha256(payloads[slot]) for slot in SENSOR_SLOTS}
    source_root = "a" * 64
    clean_hashes = tuple(canonical_hash({"clean_record": index}) for index in range(3))
    selected_step = 1
    record_ids = {
        slot: canonical_hash(
            {
                "namespace": "robotactile.rest-reference-record.v1",
                "source_live_artifact_root_sha256": source_root,
                "clean_record_sha256": clean_hashes[selected_step],
                "slot_id": slot,
                "source_index": selected_step,
                "payload_sha256": payload_hashes[slot],
            }
        )
        for slot in SENSOR_SLOTS
    }
    calibration = {
        slot: config.aliases.calibration_config_sha256 for slot in SENSOR_SLOTS
    }
    validation = NoContactValidationReceipt(
        source_live_artifact_root_sha256=source_root,
        source_trial_manifest_sha256="b" * 64,
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="c" * 64,
        task=task_id,
        episode_id=f"calibration-{task_id}",
        initial_seed=31,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        phase_tracker_sha256=canonical_hash(config.phase_tracker),
        minimum_consecutive_free_records=3,
        qualified_start_index=0,
        qualified_stop_index=3,
        selected_step_index=selected_step,
        qualified_clean_record_sha256s=clean_hashes,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        selected_payload_sha256=payload_hashes,
    )
    references = RestReferenceBundle(
        reference_id=f"rest-{task_id}-test",
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="c" * 64,
        source_artifact_sha256=source_root,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        no_contact_validation_sha256=validation.sha256,
        no_contact_verified=True,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        payloads=payloads,
    )
    output = root / f"rest-{task_id}"
    write_rest_reference_artifact(output, references, validation)
    return output


def _spec(root: Path, rest_reference: Path) -> PrimaryMatrixGenerationSpec:
    return PrimaryMatrixGenerationSpec(
        task_id="pull_out_key",
        dataset_sha256="d" * 64,
        base_system_id="official-act-pull-out-key-touch",
        tactile_checkpoint_sha256="e" * 64,
        no_touch_system_id="official-act-pull-out-key-vision",
        no_touch_checkpoint_sha256="f" * 64,
        stats_sha256="1" * 64,
        encoder_sha256="2" * 64,
        initial_seed=17,
        exogenous_seed=29,
        operator_seed_base=20260821,
        fault_start_index=16,
        fault_stop_index=301,
        max_control_cycles=300,
        max_observation_steps=301,
        wall_timeout_s=1800.0,
        upstream_root=root / "UniVTAC",
        runtime_root=root / "runtime",
        official_act_artifact_root=root / "official-act",
        rest_reference_artifact=rest_reference,
    )


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generates_complete_portable_primary_matrix_deterministically(
    tmp_path: Path,
) -> None:
    rest_reference = _rest_reference(tmp_path)
    spec = _spec(tmp_path, rest_reference)
    first = tmp_path / "primary-a"
    second = tmp_path / "primary-b"

    status, loaded = generate_primary_matrix_bundle(first, spec)
    repeated, loaded_repeated = generate_primary_matrix_bundle(first, spec)
    _, loaded_second = generate_primary_matrix_bundle(second, spec)

    assert status == "created"
    assert repeated == "already_present"
    assert loaded.receipt_file_sha256 == loaded_repeated.receipt_file_sha256
    assert loaded.receipt_file_sha256 == loaded_second.receipt_file_sha256
    assert _inventory(first) == _inventory(second)
    assert len(loaded.manifest.comparisons) == 70
    assert len(loaded.manifest.cells) == 72
    assert loaded.receipt.fault_manifest_count == 70
    assert loaded.receipt.simulator_execution_claimed is False
    assert loaded.receipt.task_success_claimed is False
    assert {
        (row.operator_id, row.severity_level) for row in loaded.manifest.comparisons
    } == {
        (operator_id, severity)
        for operator_id in CORE_OPERATOR_IDS
        for severity in range(1, 6)
    }
    assert (
        sum(cell.trial.condition is Condition.CLEAN for cell in loaded.manifest.cells)
        == 1
    )
    assert (
        sum(
            cell.trial.condition is Condition.NO_TOUCH for cell in loaded.manifest.cells
        )
        == 1
    )
    for cell in loaded.manifest.cells:
        resource = loaded.run_config.resources[cell.sha256]
        if cell.fault_manifest is not None:
            assert cell.fault_manifest.sensor_slots == SENSOR_SLOTS
            assert resource.fault_manifest_path is not None
        if cell.operator_id in REST_REFERENCE_OPERATOR_IDS:
            assert resource.rest_references_path is not None


def test_cli_generates_then_materializes_all_cells_without_execution(
    tmp_path: Path,
) -> None:
    rest_reference = _rest_reference(tmp_path)
    output = tmp_path / "cli-primary"
    command = [
        sys.executable,
        "-m",
        "robotactile_benchmark.cli",
        "generate-primary-matrix",
        "--task",
        "pull_out_key",
        "--dataset-sha256",
        "d" * 64,
        "--tactile-checkpoint-sha256",
        "e" * 64,
        "--no-touch-checkpoint-sha256",
        "f" * 64,
        "--stats-sha256",
        "1" * 64,
        "--encoder-sha256",
        "2" * 64,
        "--initial-seed",
        "17",
        "--exogenous-seed",
        "29",
        "--upstream-root",
        str(tmp_path / "UniVTAC"),
        "--runtime-root",
        str(tmp_path / "runtime"),
        "--official-act-artifact-root",
        str(tmp_path / "official-act"),
        "--rest-reference-artifact",
        str(rest_reference),
        "--output",
        str(output),
    ]
    generated = subprocess.run(command, capture_output=True, text=True, check=False)

    assert generated.returncode == 0, generated.stderr
    generated_payload = json.loads(generated.stdout)
    assert generated_payload["comparison_count"] == 70
    assert generated_payload["cell_count"] == 72
    assert generated_payload["simulator_execution_claimed"] is False
    materialize = subprocess.run(
        [
            sys.executable,
            "-m",
            "robotactile_benchmark.cli",
            "run-live-matrix",
            "--matrix-manifest",
            str(output / "matrix_manifest.json"),
            "--run-config",
            str(output / "live_matrix_run_config.json"),
            "--matrix-output",
            str(tmp_path / "matrix-output"),
            "--max-new-cells",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert materialize.returncode == 0, materialize.stderr
    matrix_payload = json.loads(materialize.stdout)
    assert matrix_payload["pending_cell_count"] == 72
    assert matrix_payload["executed_cell_count"] == 0
    assert matrix_payload["simulator_qualification_claimed"] is False


def test_rejects_warmup_task_and_content_tampering(tmp_path: Path) -> None:
    rest_reference = _rest_reference(tmp_path)
    spec = _spec(tmp_path, rest_reference)
    with pytest.raises(ValueError, match="warm-up"):
        generate_primary_matrix_bundle(
            tmp_path / "bad-warmup",
            replace(spec, fault_start_index=15),
        )

    output = tmp_path / "tamper"
    generate_primary_matrix_bundle(output, spec)
    fault_path = next((output / "fault_manifests").glob("*.json"))
    fault_path.write_bytes(fault_path.read_bytes() + b" ")
    with pytest.raises(PrimaryMatrixGenerationError, match="inventory"):
        load_primary_matrix_generation(output)

    wrong_rest = _rest_reference(tmp_path / "wrong", "insert_HDMI")
    with pytest.raises(PrimaryMatrixGenerationError, match="task differs"):
        generate_primary_matrix_bundle(
            tmp_path / "wrong-task", replace(spec, rest_reference_artifact=wrong_rest)
        )


def test_receipt_member_inventory_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "primary"
    generate_primary_matrix_bundle(output, _spec(tmp_path, _rest_reference(tmp_path)))
    loaded = load_primary_matrix_generation(output)

    assert scan_files(output, exclude="primary_matrix_receipt.json") == dict(
        loaded.receipt.members
    )
    assert set(loaded.receipt.members) == {
        "matrix_manifest.json",
        "live_matrix_run_config.json",
        "rest_references/rest_reference.json",
        "rest_references/no_contact_validation.json",
        "rest_references/root_receipt.json",
        *{
            f"fault_manifests/{cell.sha256}.json"
            for cell in loaded.manifest.cells
            if cell.fault_manifest is not None
        },
    }


def test_legacy_v1_primary_generation_contracts_fail_closed(tmp_path: Path) -> None:
    rest_reference = _rest_reference(tmp_path)

    with pytest.raises(PrimaryMatrixGenerationError, match="version mismatch"):
        replace(_spec(tmp_path, rest_reference), semantic_version="1.0")

    output = tmp_path / "primary"
    generate_primary_matrix_bundle(output, _spec(tmp_path, rest_reference))
    loaded = load_primary_matrix_generation(output)
    with pytest.raises(PrimaryMatrixGenerationError, match="version mismatch"):
        replace(loaded.receipt, semantic_version="1.0")
