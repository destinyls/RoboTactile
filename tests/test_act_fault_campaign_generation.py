"""Generation tests for official ACT Clean/Faulted campaign bundles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import robotactile_benchmark.integrations.act.requests as act_requests
from robotactile_benchmark.act_fault_campaign.contracts import (
    ACT_UNSUPPORTED_OPERATOR_IDS,
    ACTFaultCampaignError,
    ACTFaultCellDisposition,
)
from robotactile_benchmark.act_fault_campaign.generation import (
    ACTFaultCampaignGenerationSpec,
    derive_operator_template_seed,
    generate_act_fault_campaign_bundle,
)
from robotactile_benchmark.act_fault_campaign.io import (
    load_act_fault_campaign_bundle,
)
from robotactile_benchmark.act_fault_campaign.reset_reference import (
    act_reset_reference_relpath,
    load_act_reset_reference,
    write_act_reset_reference,
)
from robotactile_benchmark.act_fault_campaign.reset_trajectory import (
    act_reset_trajectory_relpath,
    load_act_reset_trajectory,
    write_act_reset_trajectory,
)
from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveSegment,
    UniVTACPreMoveTrajectory,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
)
from robotactile_benchmark.calibration import write_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    NO_CONTACT_PREDICATE_ID,
    NoContactValidationReceipt,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import CORE_OPERATOR_IDS, SENSOR_SLOTS
from robotactile_benchmark.contracts import array_sha256, canonical_hash
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.integrations.act.artifacts import (
    ACTArtifactManifest,
    act_artifact_manifest_to_dict,
)
from robotactile_benchmark.integrations.act.requests import (
    build_official_act_request,
    write_official_act_request,
)
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.rest_references import ReferenceSplit, RestReferenceBundle
from robotactile_benchmark.trials import Condition

TASK = "grasp_classify"


@pytest.fixture(autouse=True)
def _accept_test_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        act_requests,
        "validate_official_univtac_act_artifact",
        lambda manifest: {"profile": manifest.profile.value},
    )


def _manifest(root: Path) -> tuple[ACTArtifactManifest, Path]:
    manifest = ACTArtifactManifest.for_shared_root(
        task_id=TASK,
        profile=OfficialACTProfile.UNIVTAC,
        artifact_root=root / "artifacts/models/act",
        upstream_root=root / "sources/UniVTAC",
        checkpoint_sha256="1" * 64,
        stats_sha256="2" * 64,
        encoder_sha256="3" * 64,
    )
    path = root / "act_manifest.json"
    path.write_bytes(canonical_json_bytes(act_artifact_manifest_to_dict(manifest)))
    return manifest, path


def _clean_request(root: Path, manifest: ACTArtifactManifest) -> Path:
    request = build_official_act_request(
        base_manifest=manifest,
        layout=DeploymentLayout((root / "deployment").absolute()),
        condition=Condition.CLEAN,
        dataset_sha256="a" * 64,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=40,
        max_observation_steps=41,
        wall_timeout_s=1800.0,
        act_device_name="cuda:0",
        simulator_device="cuda:0",
        live_output_dir=root / "old-artifact",
    )
    path = root / "base-clean/request.json"
    write_official_act_request(path, request)
    return path


def _rest_reference(root: Path) -> Path:
    config = build_univtac_backend_config(TASK, action_spec=QPOS8_ACTION_SPEC)
    payloads = {
        slot: np.full(config.tactile_rgb_shape, 13 + index, dtype=np.uint8)
        for index, slot in enumerate(SENSOR_SLOTS)
    }
    payload_hashes = {slot: array_sha256(payloads[slot]) for slot in SENSOR_SLOTS}
    source_root = "4" * 64
    clean_hashes = tuple(canonical_hash({"clean": index}) for index in range(3))
    selected = 1
    record_ids = {
        slot: canonical_hash(
            {
                "namespace": "robotactile.rest-reference-record.v1",
                "source_live_artifact_root_sha256": source_root,
                "clean_record_sha256": clean_hashes[selected],
                "slot_id": slot,
                "source_index": selected,
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
        source_trial_manifest_sha256="5" * 64,
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="6" * 64,
        task=TASK,
        episode_id="calibration-pull-out-key",
        initial_seed=31,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        phase_tracker_sha256=canonical_hash(config.phase_tracker),
        minimum_consecutive_free_records=3,
        qualified_start_index=0,
        qualified_stop_index=3,
        selected_step_index=selected,
        qualified_clean_record_sha256s=clean_hashes,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        selected_payload_sha256=payload_hashes,
    )
    references = RestReferenceBundle(
        reference_id="rest-pull-out-key-test",
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="6" * 64,
        source_artifact_sha256=source_root,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        no_contact_validation_sha256=validation.sha256,
        no_contact_verified=True,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        payloads=payloads,
    )
    output = root / "rest-reference"
    write_rest_reference_artifact(output, references, validation)
    return output


def _reset_reference(root: Path, request_path: Path) -> Path:
    loaded = load_live_univtac_run(load_live_univtac_request(request_path))
    trial = loaded.trial
    reference = UniVTACResetReference(
        task_id=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        dataset_sha256=trial.dataset_sha256,
        checkpoint_sha256=trial.checkpoint_sha256,
        config_sha256=trial.config_sha256,
        source_artifact_root_sha256="7" * 64,
        source_result_sha256="8" * 64,
        source_run_content_sha256=loaded.content_sha256,
        expected_simulator_state_sha256="9" * 64,
        expected_native_step=240,
        expected_qpos8=(0.0,) * 8,
        qpos_atol=1e-5,
    )
    output = root / "reset-reference.json"
    write_act_reset_reference(output, reference)
    return output


def _reset_trajectory(
    root: Path,
    request_path: Path,
    reset_reference_path: Path,
) -> Path:
    loaded = load_live_univtac_run(load_live_univtac_request(request_path))
    trial = loaded.trial
    reference = load_act_reset_reference(reset_reference_path)

    def segment(kind: str, fingerprint: str) -> UniVTACPreMoveSegment:
        width = 1 if kind == "gripper" else 7
        return UniVTACPreMoveSegment(
            kind=kind,
            request_fingerprint=fingerprint,
            start_joint9=(0.0,) * 7 + (0.04, 0.04),
            position=((0.0,) * width,),
            velocity=((0.0,) * width,),
        )

    trajectory = UniVTACPreMoveTrajectory(
        task_id=trial.task,
        action_spec=trial.action_spec,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        dataset_sha256=trial.dataset_sha256,
        checkpoint_sha256=trial.checkpoint_sha256,
        config_sha256=trial.config_sha256,
        source_run_content_sha256=loaded.content_sha256,
        reset_reference_sha256=reference.sha256,
        capture_reset_receipt_sha256="b" * 64,
        capture_simulator_state_sha256="7" * 64,
        capture_native_step=reference.expected_native_step,
        capture_qpos8=reference.expected_qpos8,
        source_qpos8_max_abs_error=0.0,
        upstream_commit=loaded.backend_config.upstream_commit,
        task_source_sha256=loaded.backend_config.task.task_source_sha256,
        post_reset_hold_steps=0,
        segments=(
            segment("gripper", "c" * 64),
            segment("arm", "d" * 64),
            segment("gripper", "e" * 64),
            segment("arm", "f" * 64),
        ),
    )
    output = root / "reset-trajectory.json"
    write_act_reset_trajectory(output, trajectory)
    return output


def _spec(root: Path) -> ACTFaultCampaignGenerationSpec:
    manifest, manifest_path = _manifest(root)
    request_path = _clean_request(root, manifest)
    reset_reference_path = _reset_reference(root, request_path)
    return ACTFaultCampaignGenerationSpec(
        campaign_id="act-fault-pilot",
        base_clean_request_paths=(request_path,),
        artifact_manifest_paths={TASK: manifest_path},
        deployment_layout=DeploymentLayout((root / "deployment").absolute()),
        operator_ids=tuple(sorted(CORE_OPERATOR_IDS)),
        severity_levels=(3,),
        operator_seed_master=20260830,
        fault_start_index=20,
        fault_stop_index=40,
        rest_reference_artifacts={TASK: _rest_reference(root)},
        reset_reference_artifact_paths=(reset_reference_path,),
        reset_trajectory_artifact_paths=(
            _reset_trajectory(root, request_path, reset_reference_path),
        ),
    )


def test_full_single_severity_grid_is_15_cells_13_live_2_unsupported(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    spec = _spec(tmp_path)
    status, loaded = generate_act_fault_campaign_bundle(output, spec)
    repeated, reloaded = generate_act_fault_campaign_bundle(output, _spec(tmp_path))

    assert status == "created"
    assert repeated == "already_present"
    assert reloaded.receipt_file_sha256 == loaded.receipt_file_sha256
    assert loaded.manifest.cell_count == 15
    assert loaded.manifest.live_request_count == 13
    assert loaded.manifest.unsupported_contract_count == 2
    assert loaded.receipt.fault_manifest_count == 14
    unsupported = [
        cell
        for cell in loaded.manifest.cells
        if cell.disposition is ACTFaultCellDisposition.UNSUPPORTED_CONTRACT
    ]
    assert {cell.operator_id for cell in unsupported} == ACT_UNSUPPORTED_OPERATOR_IDS
    assert all(cell.request_relpath is None for cell in unsupported)
    assert (
        len(
            {
                path
                for cell in loaded.manifest.cells
                for path in (
                    cell.request_relpath,
                    cell.artifact_relpath,
                    cell.fault_manifest_relpath,
                )
                if path is not None
            }
        )
        == 40
    )
    live_faults = [
        cell
        for cell in loaded.manifest.cells
        if cell.condition is Condition.FAULTED
        and cell.disposition is ACTFaultCellDisposition.LIVE_REQUEST
    ]
    assert len(live_faults) == 12
    live_runtime_dirs: set[Path] = set()
    for cell in (
        item
        for item in loaded.manifest.cells
        if item.disposition is ACTFaultCellDisposition.LIVE_REQUEST
    ):
        request = load_live_univtac_request(output / str(cell.request_relpath))
        run = load_live_univtac_run(request)
        assert request.policy_kind is LivePolicyKind.ACT
        assert request.condition is cell.condition
        assert request.execute_action_steps == 1
        assert run.trial.pair_key == cell.pair_key
        assert (
            request.output_dir.resolve()
            == (output / str(cell.artifact_relpath)).resolve()
        )
        live_runtime_dirs.add(request.runtime_dir.resolve(strict=False))
    base = load_live_univtac_request(spec.base_clean_request_paths[0])
    assert live_runtime_dirs == {base.runtime_dir.resolve(strict=False)}
    reset_path = output / act_reset_reference_relpath(
        TASK,
        loaded.manifest.cells[0].pair_key,
    )
    assert reset_path.is_file()
    assert reset_path.as_posix() in {
        (output / path).as_posix() for path in loaded.receipt.members
    }
    reset_trajectory_path = output / act_reset_trajectory_relpath(
        TASK,
        loaded.manifest.cells[0].pair_key,
    )
    copied_trajectory = load_act_reset_trajectory(reset_trajectory_path)
    source_trajectory = load_act_reset_trajectory(
        spec.reset_trajectory_artifact_paths[0]
    )
    copied_reference = load_act_reset_reference(reset_path)
    assert copied_trajectory == source_trajectory
    assert copied_trajectory.reset_reference_sha256 == (copied_reference.sha256)
    assert copied_trajectory.capture_reset_receipt_sha256 == "b" * 64
    assert copied_trajectory.capture_simulator_state_sha256 == "7" * 64
    assert (
        copied_trajectory.capture_native_step == copied_reference.expected_native_step
    )
    assert copied_trajectory.capture_qpos8 == copied_reference.expected_qpos8
    assert copied_trajectory.source_qpos8_max_abs_error == 0.0
    assert copied_trajectory.post_reset_hold_steps == 0
    assert reset_trajectory_path.as_posix() in {
        (output / path).as_posix() for path in loaded.receipt.members
    }
    assert not any("no_touch" in path.as_posix() for path in output.rglob("*"))


def test_generation_is_deterministic_and_strict_loader_rejects_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    _, loaded = generate_act_fault_campaign_bundle(output, _spec(tmp_path))
    seed = derive_operator_template_seed(
        master_seed=20260830,
        pair_key=loaded.manifest.cells[0].pair_key,
        operator_id="F2_spatial_sensitivity_loss",
    )
    matching = [
        cell
        for cell in loaded.manifest.cells
        if cell.operator_id == "F2_spatial_sensitivity_loss"
    ]
    assert [cell.operator_template_seed for cell in matching] == [seed]
    clean = loaded.manifest.cells[0]
    trajectory_path = output / act_reset_trajectory_relpath(
        clean.task,
        clean.pair_key,
    )
    original_binding = b'"capture_reset_receipt_sha256":"' + b"b" * 64 + b'"'
    tampered_binding = b'"capture_reset_receipt_sha256":"' + b"0" * 64 + b'"'
    trajectory_path.write_bytes(
        trajectory_path.read_bytes().replace(original_binding, tampered_binding)
    )
    with pytest.raises(ACTFaultCampaignError, match="inventory mismatch"):
        load_act_fault_campaign_bundle(output)


def test_generation_rejects_trajectory_bound_to_another_reset_reference(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    trajectory = load_act_reset_trajectory(spec.reset_trajectory_artifact_paths[0])
    mismatched = replace(trajectory, reset_reference_sha256="0" * 64)
    mismatch_path = tmp_path / "mismatched-reset-trajectory.json"
    write_act_reset_trajectory(mismatch_path, mismatched)
    mismatched_spec = replace(
        spec,
        reset_trajectory_artifact_paths=(mismatch_path,),
    )

    with pytest.raises(ACTFaultCampaignError, match="trajectory identity"):
        generate_act_fault_campaign_bundle(tmp_path / "bad-bundle", mismatched_spec)
