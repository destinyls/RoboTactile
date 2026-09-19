"""Deterministic generation tests for N0 Clean/Faulted campaign bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration import write_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    NO_CONTACT_PREDICATE_ID,
    NoContactValidationReceipt,
)
from robotactile_benchmark.closed_loop.contracts import WallTimeoutRole
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    SENSOR_SLOTS,
)
from robotactile_benchmark.contracts import array_sha256, canonical_hash
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    N0ObservedTactileMode,
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.integrations.n0_twam.requests import (
    write_official_n0_clean_request,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.n0_fault_campaign import (
    N0FaultCampaignError,
    N0FaultCampaignGenerationSpec,
    N0FaultCellDisposition,
    derive_operator_template_seed,
    generate_n0_fault_campaign_bundle,
    load_n0_fault_campaign_bundle,
)
from robotactile_benchmark.n0_fault_campaign.generation_cells import (
    _optical_sample_period_s,
)
from robotactile_benchmark.rest_references import (
    ReferenceSplit,
    RestReferenceBundle,
)
from robotactile_benchmark.trials import Condition, system_manifest_hash


def _clean_request(
    root: Path, task: str = "pull_out_key", seed: int = 1_000_002
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_system_id = f"official-n0-twam.{task}.test"
    checkpoint = "a" * 64
    config = "b" * 64
    request = LiveUniVTACRunRequest(
        task_id=task,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.N0,
        base_system_id=base_system_id,
        dataset_sha256="c" * 64,
        checkpoint_sha256=checkpoint,
        config_sha256=config,
        base_system_manifest_sha256=system_manifest_hash(
            base_system_id, checkpoint, config, EE8_ACTION_SPEC
        ),
        initial_seed=seed,
        exogenous_seed=seed,
        max_control_cycles=40,
        max_observation_steps=41,
        execute_action_steps=24,
        wall_timeout_s=1800.0,
        upstream_root=root / "UniVTAC",
        runtime_dir=root / "runtime/n0",
        output_dir=root / "old-artifact",
        fault_manifest_path=None,
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=None,
        simulator_device="cuda:0",
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit="d" * 40,
        n0_normalizer_sha256="e" * 64,
        n0_serve_bundle_sha256="f" * 64,
        n0_prompt_manifest_sha256="1" * 64,
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
    )
    path = root / "base-clean/request.json"
    write_official_n0_clean_request(path, request)
    return path


def _rest_reference(root: Path, task: str = "pull_out_key") -> Path:
    config = build_univtac_backend_config(task, action_spec=EE8_ACTION_SPEC)
    payloads = {
        slot: np.full(config.tactile_rgb_shape, 13 + index, dtype=np.uint8)
        for index, slot in enumerate(SENSOR_SLOTS)
    }
    payload_hashes = {slot: array_sha256(payloads[slot]) for slot in SENSOR_SLOTS}
    source_root = "2" * 64
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
        source_trial_manifest_sha256="3" * 64,
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="4" * 64,
        task=task,
        episode_id=f"calibration-{task}",
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
        reference_id=f"rest-{task}-test",
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="4" * 64,
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


def _spec(root: Path) -> N0FaultCampaignGenerationSpec:
    return N0FaultCampaignGenerationSpec(
        campaign_id="n0-fault-pilot",
        base_clean_request_paths=(_clean_request(root),),
        operator_ids=(
            "T1_fixed_source_delay",
            "A2_frame_erasure",
            "F2_spatial_sensitivity_loss",
            "A1_stream_absence",
        ),
        severity_levels=(5, 1),
        operator_seed_master=20260828,
        fault_start_index=20,
        fault_stop_index=40,
        rest_reference_artifacts={"pull_out_key": _rest_reference(root)},
    )


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generates_clean_supported_and_receipt_only_unsupported_cells(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    first = tmp_path / "bundle-a"
    second = tmp_path / "bundle-b"

    status, loaded = generate_n0_fault_campaign_bundle(first, spec)
    repeated, loaded_repeated = generate_n0_fault_campaign_bundle(first, spec)
    _, loaded_second = generate_n0_fault_campaign_bundle(second, spec)

    assert status == "created"
    assert repeated == "already_present"
    assert loaded.receipt_file_sha256 == loaded_repeated.receipt_file_sha256
    assert loaded.receipt_file_sha256 == loaded_second.receipt_file_sha256
    assert _inventory(first) == _inventory(second)
    assert loaded.manifest.pair_count == 1
    assert loaded.manifest.cell_count == 9
    assert loaded.manifest.live_request_count == 5
    assert loaded.manifest.unsupported_contract_count == 4
    assert loaded.manifest.operator_ids == tuple(sorted(spec.operator_ids))
    assert loaded.manifest.severity_levels == (1, 5)
    assert sum(cell.condition is Condition.CLEAN for cell in loaded.manifest.cells) == 1

    unsupported = [
        cell
        for cell in loaded.manifest.cells
        if cell.disposition is N0FaultCellDisposition.UNSUPPORTED_CONTRACT
    ]
    assert {cell.operator_id for cell in unsupported} == {
        "A1_stream_absence",
        "A2_frame_erasure",
    }
    assert all(cell.request_relpath is None for cell in unsupported)
    live_faults = [
        cell
        for cell in loaded.manifest.cells
        if cell.condition is Condition.FAULTED
        and cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
    ]
    for cell in live_faults:
        request = load_live_univtac_request(first / str(cell.request_relpath))
        assert request.policy_kind is LivePolicyKind.N0
        assert request.condition is Condition.FAULTED
        assert request.output_dir is not None
        assert (
            request.output_dir.resolve()
            == (first / str(cell.artifact_relpath)).resolve()
        )


def test_early_random_onset_is_seed_bound_and_shared_by_operators(
    tmp_path: Path,
) -> None:
    sources = tuple(
        _clean_request(tmp_path / f"source-{seed}", seed=seed) for seed in range(6)
    )
    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-early-onset",
        base_clean_request_paths=sources,
        operator_ids=(
            "F2_spatial_sensitivity_loss",
            "T1_fixed_source_delay",
            "T2_held_last_freeze",
            "T3_inter_sensor_skew",
        ),
        severity_levels=(5,),
        operator_seed_master=17,
        fault_start_index=0,
        fault_stop_index=41,
        rest_reference_artifacts={"pull_out_key": _rest_reference(tmp_path)},
        fault_onset_mode="early_random_onset_v1",
        fault_onset_max_index=8,
    )
    _, loaded = generate_n0_fault_campaign_bundle(tmp_path / "early-bundle", spec)
    starts: set[int] = set()
    for seed in range(6):
        cells = [
            cell
            for cell in loaded.manifest.cells
            if cell.exogenous_seed == seed and cell.condition is Condition.FAULTED
        ]
        manifests = [
            load_live_univtac_run(
                load_live_univtac_request(loaded.root / str(cell.request_relpath))
            ).fault_manifest
            for cell in cells
        ]
        assert all(manifest is not None for manifest in manifests)
        expected = derive_early_random_onset(
            task="pull_out_key", seed=seed, stop=41, cap=8
        )
        assert {manifest.start_index for manifest in manifests} == {expected}
        assert {manifest.stop_index for manifest in manifests} == {41}
        assert 1 <= expected <= 8
        starts.add(expected)
        for manifest in manifests:
            if manifest.operator_id.startswith("T"):
                assert manifest.parameters["temporal_schedule"] == "window_to_end_v1"
    assert len(starts) >= 2


def test_early_random_onset_covers_all_fourteen_operator_manifests(
    tmp_path: Path,
) -> None:
    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-early-all14",
        base_clean_request_paths=(_clean_request(tmp_path, seed=7),),
        operator_ids=tuple(sorted(CORE_OPERATOR_IDS)),
        severity_levels=(5,),
        operator_seed_master=17,
        fault_start_index=0,
        fault_stop_index=41,
        rest_reference_artifacts={"pull_out_key": _rest_reference(tmp_path)},
        fault_onset_mode="early_random_onset_v1",
    )
    _, loaded = generate_n0_fault_campaign_bundle(tmp_path / "early-all14", spec)
    expected = derive_early_random_onset(task="pull_out_key", seed=7, stop=41)
    faults = [
        cell for cell in loaded.manifest.cells if cell.condition is Condition.FAULTED
    ]
    assert len(faults) == 14
    assert {cell.operator_id for cell in faults} == CORE_OPERATOR_IDS
    for cell in faults:
        assert cell.fault_manifest_relpath is not None
        manifest = FaultManifest.from_dict(
            json.loads((loaded.root / cell.fault_manifest_relpath).read_text())
        )
        assert manifest.start_index == expected
        assert manifest.stop_index == 41


def test_fixed_zero_onset_marks_temporal_faults_as_full_episode(
    tmp_path: Path,
) -> None:
    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-first-inference-all14",
        base_clean_request_paths=(_clean_request(tmp_path, seed=7),),
        operator_ids=(
            "T1_fixed_source_delay",
            "T2_held_last_freeze",
            "T3_inter_sensor_skew",
        ),
        severity_levels=(5,),
        operator_seed_master=17,
        fault_start_index=0,
        fault_stop_index=41,
        rest_reference_artifacts={},
        severity_registry=OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    )
    _, loaded = generate_n0_fault_campaign_bundle(
        tmp_path / "first-inference-all14", spec
    )
    faults = [
        cell for cell in loaded.manifest.cells if cell.condition is Condition.FAULTED
    ]
    assert len(faults) == 3
    for cell in faults:
        assert cell.fault_manifest_relpath is not None
        manifest = FaultManifest.from_dict(
            json.loads((loaded.root / cell.fault_manifest_relpath).read_text())
        )
        assert manifest.start_index == 0
        assert manifest.stop_index == 41
        assert manifest.parameters["temporal_schedule"] == "full_episode_v1"

    t2_cell = next(cell for cell in faults if cell.operator_id == "T2_held_last_freeze")
    assert t2_cell.fault_manifest_relpath is not None
    t2_manifest = FaultManifest.from_dict(
        json.loads((loaded.root / t2_cell.fault_manifest_relpath).read_text())
    )
    assert t2_manifest.parameters["hold_duration_frames"] == 41
    assert t2_manifest.parameters["hold_policy"] == "until_episode_end"


def test_early_optical_campaign_binds_native_sensor_period(tmp_path: Path) -> None:
    source = _clean_request(tmp_path / "source", seed=5)
    loaded_source = load_live_univtac_run(load_live_univtac_request(source))
    expected_period = 1 / loaded_source.backend_config.sim_hz
    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-early-optical-seed5",
        base_clean_request_paths=(source,),
        operator_ids=tuple(sorted(CORE_OPERATOR_IDS)),
        severity_levels=(5,),
        operator_seed_master=5,
        fault_start_index=0,
        fault_stop_index=41,
        rest_reference_artifacts={"pull_out_key": _rest_reference(tmp_path)},
        severity_registry=OPTICAL_MARKER_EXTREME_REGISTRY_ID,
        fault_onset_mode="early_random_onset_v1",
    )
    _, campaign = generate_n0_fault_campaign_bundle(tmp_path / "optical", spec)
    faults = [
        cell for cell in campaign.manifest.cells if cell.condition is Condition.FAULTED
    ]
    assert len(faults) == 14
    for cell in faults:
        assert cell.fault_manifest_relpath is not None
        manifest = FaultManifest.from_dict(
            json.loads((campaign.root / cell.fault_manifest_relpath).read_text())
        )
        assert manifest.parameters["sample_period_s"] == expected_period
        assert manifest.start_index == derive_early_random_onset(
            task="pull_out_key", seed=5, stop=41
        )


def test_optical_period_follows_retrained_observation_cadence() -> None:
    official = build_univtac_backend_config("pull_out_key", action_spec=EE8_ACTION_SPEC)
    retrained = build_univtac_backend_config(
        "pull_out_key",
        action_spec=EE8_ACTION_SPEC,
        n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    )
    assert _optical_sample_period_s(official) == 1 / official.sim_hz
    assert _optical_sample_period_s(retrained) == (
        retrained.physics_steps_per_action / retrained.sim_hz
    )


def test_template_seed_is_shared_across_severity_but_not_operator() -> None:
    pair = "9" * 64
    f2_l1 = derive_operator_template_seed(
        master_seed=17, pair_key=pair, operator_id="F2_spatial_sensitivity_loss"
    )
    f2_l5 = derive_operator_template_seed(
        master_seed=17, pair_key=pair, operator_id="F2_spatial_sensitivity_loss"
    )
    t1 = derive_operator_template_seed(
        master_seed=17, pair_key=pair, operator_id="T1_fixed_source_delay"
    )

    assert f2_l1 == f2_l5
    assert f2_l1 != t1
    assert 0 <= f2_l1 <= 0x7FFFFFFF


def test_strict_reload_rejects_tampering_and_generation_refuses_clobber(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    output = tmp_path / "bundle"
    _, loaded = generate_n0_fault_campaign_bundle(output, spec)
    request = next(
        cell for cell in loaded.manifest.cells if cell.condition is Condition.CLEAN
    )
    request_path = output / str(request.request_relpath)
    request_path.write_bytes(
        request_path.read_bytes().replace(b'"clean"', b'"faulted"')
    )

    with pytest.raises(N0FaultCampaignError, match="inventory mismatch"):
        load_n0_fault_campaign_bundle(output)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        generate_n0_fault_campaign_bundle(output, spec)


def test_execution_artifacts_do_not_invalidate_frozen_generation_inventory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    _, loaded = generate_n0_fault_campaign_bundle(output, _spec(tmp_path))
    clean = next(
        cell for cell in loaded.manifest.cells if cell.condition is Condition.CLEAN
    )
    artifact = output / str(clean.artifact_relpath) / "root_receipt.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"runtime":"result"}\n', encoding="utf-8")

    reloaded = load_n0_fault_campaign_bundle(output)

    assert reloaded.manifest.sha256 == loaded.manifest.sha256
    bad_link = output / "artifacts/bad-link"
    bad_link.symlink_to(output / "campaign_manifest.json")
    with pytest.raises(N0FaultCampaignError, match="symlinks"):
        load_n0_fault_campaign_bundle(output)


def test_tactile_null_generation_is_one_clean_plus_one_full_horizon_ablation(
    tmp_path: Path,
) -> None:
    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-tactile-null",
        base_clean_request_paths=(_clean_request(tmp_path),),
        operator_ids=("F1_global_response_drift",),
        severity_levels=(5,),
        operator_seed_master=20260829,
        fault_start_index=0,
        fault_stop_index=41,
        rest_reference_artifacts={},
        severity_registry=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    )
    _, loaded = generate_n0_fault_campaign_bundle(tmp_path / "null-bundle", spec)

    assert loaded.manifest.cell_count == 2
    assert loaded.manifest.live_request_count == 2
    fault_cell = next(
        cell for cell in loaded.manifest.cells if cell.condition is Condition.FAULTED
    )
    assert fault_cell.artifact_relpath is not None
    assert "tactile_null_black_frame_v1" in fault_cell.artifact_relpath
    request = load_live_univtac_request(loaded.root / str(fault_cell.request_relpath))
    run = load_live_univtac_run(request)
    assert run.fault_manifest is not None
    assert run.fault_manifest.start_index == 0
    assert run.fault_manifest.stop_index == request.max_observation_steps
    assert run.fault_manifest.sensor_slots == SENSOR_SLOTS
    assert run.fault_manifest.parameters["ablation_id"] == "tactile_null_black_frame_v1"
    assert run.rest_references is None
    assert loaded.manifest.rest_reference_bindings == {}
    assert loaded.manifest.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID


def test_observed_tactile_absence_generates_live_full_horizon_a1(
    tmp_path: Path,
) -> None:
    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-observed-tactile-absence",
        base_clean_request_paths=(_clean_request(tmp_path),),
        operator_ids=("A1_stream_absence",),
        severity_levels=(5,),
        operator_seed_master=20260829,
        fault_start_index=0,
        fault_stop_index=41,
        rest_reference_artifacts={},
        severity_registry=DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    )
    _, loaded = generate_n0_fault_campaign_bundle(tmp_path / "absence-bundle", spec)

    assert loaded.manifest.cell_count == 2
    assert loaded.manifest.live_request_count == 2
    assert loaded.manifest.unsupported_contract_count == 0
    fault_cell = next(
        cell for cell in loaded.manifest.cells if cell.condition is Condition.FAULTED
    )
    assert "observed_tactile_absent_v1" in str(fault_cell.artifact_relpath)
    request = load_live_univtac_request(loaded.root / str(fault_cell.request_relpath))
    run = load_live_univtac_run(request)
    assert request.n0_observed_tactile_mode is N0ObservedTactileMode.ABSENT
    assert run.policy_identity.consumes_tactile is True
    assert run.policy_identity.supports_structural_absence is True
    assert run.fault_manifest is not None
    assert run.fault_manifest.operator_id == "A1_stream_absence"
    assert tuple(run.fault_manifest.parameters["affected_offsets"]) == tuple(
        range(request.max_observation_steps)
    )
