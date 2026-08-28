"""Generate and strictly reload a complete primary live-matrix request bundle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    REST_REFERENCE_PATH,
    ROOT_RECEIPT_PATH,
    VALIDATION_PATH,
)
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    REST_REFERENCE_OPERATOR_IDS,
    SENSOR_SLOTS,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    production_univtac_launcher_args,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix.builders import build_primary_matrix_manifest
from robotactile_benchmark.matrix.io import (
    MATRIX_MANIFEST_PATH,
    load_matrix_manifest_file,
)
from robotactile_benchmark.matrix.live_run_config import (
    LiveMatrixResourceEntry,
    LiveMatrixRunConfig,
    load_live_matrix_run_config,
)
from robotactile_benchmark.matrix.primary_generation_contracts import (
    PRIMARY_RECEIPT_PATH,
    LoadedPrimaryMatrixGeneration,
    PrimaryMatrixGenerationError,
    PrimaryMatrixGenerationReceipt,
    PrimaryMatrixGenerationSpec,
)
from robotactile_benchmark.matrix.primary_generation_io import (
    copy_regular_file,
    discard_staging,
    file_sha256,
    publish_staging,
    read_json,
    relative_run_config_document,
    scan_files,
    staging_directory,
    write_json,
)
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
)
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TrialManifest,
    system_manifest_hash,
)

LIVE_MATRIX_RUN_CONFIG_PATH = "live_matrix_run_config.json"
FAULT_MANIFEST_DIRECTORY = "fault_manifests"
REST_REFERENCE_DIRECTORY = "rest_references"


def _official_manifest(
    spec: PrimaryMatrixGenerationSpec, profile: OfficialACTProfile, checkpoint: str
) -> OfficialUniVTACACTArtifactManifest:
    return OfficialUniVTACACTArtifactManifest.for_shared_root(
        task_id=spec.task_id,
        profile=profile,
        artifact_root=spec.official_act_artifact_root,
        upstream_root=spec.upstream_root,
        checkpoint_sha256=checkpoint,
        stats_sha256=spec.stats_sha256,
        encoder_sha256=spec.encoder_sha256,
    )


def _fault_manifests(
    spec: PrimaryMatrixGenerationSpec, rest_reference_sha256: str
) -> tuple[FaultManifest, ...]:
    faults: list[FaultManifest] = []
    for operator_offset, operator_id in enumerate(sorted(CORE_OPERATOR_IDS)):
        for severity_level in range(1, 6):
            parameters: dict[str, object] = {}
            if operator_id in REST_REFERENCE_OPERATOR_IDS:
                parameters["rest_reference_sha256"] = rest_reference_sha256
            if operator_id == "C2_frame_misregistration":
                parameters["realization"] = "registered_pixels"
            faults.append(
                FaultManifest(
                    operator_id=operator_id,
                    severity_level=severity_level,
                    operator_seed=(
                        spec.operator_seed_base
                        + operator_offset * 5
                        + severity_level
                        - 1
                    ),
                    start_index=spec.fault_start_index,
                    stop_index=spec.fault_stop_index,
                    sensor_slots=SENSOR_SLOTS,
                    observability=Observability.BLIND,
                    parameters=parameters,
                )
            )
    return tuple(faults)


def _clean_trial(
    spec: PrimaryMatrixGenerationSpec,
    tactile: OfficialUniVTACACTArtifactManifest,
) -> TrialManifest:
    base_hash = system_manifest_hash(
        spec.base_system_id,
        tactile.checkpoint_sha256,
        tactile.config_sha256,
        ACTION_SPEC,
    )
    return TrialManifest(
        task=spec.task_id,
        initial_seed=spec.initial_seed,
        exogenous_seed=spec.exogenous_seed,
        condition=Condition.CLEAN,
        base_system_id=spec.base_system_id,
        executed_system_id=spec.base_system_id,
        dataset_sha256=spec.dataset_sha256,
        base_system_manifest_sha256=base_hash,
        checkpoint_sha256=tactile.checkpoint_sha256,
        config_sha256=tactile.config_sha256,
        action_spec=ACTION_SPEC,
        fault_manifest_sha256=None,
        matched_no_touch_system_id=None,
        restoration_index=None,
        restoration_mode=None,
    )


def _matrix_id(
    spec: PrimaryMatrixGenerationSpec,
    rest_root_sha256: str,
    tactile_config_sha256: str,
    no_touch_config_sha256: str,
) -> str:
    if spec.matrix_id is not None:
        return spec.matrix_id
    identity = canonical_hash(
        {
            "generation_contract_sha256": spec.contract_sha256,
            "rest_reference_artifact_root_sha256": rest_root_sha256,
            "tactile_config_sha256": tactile_config_sha256,
            "no_touch_config_sha256": no_touch_config_sha256,
        }
    )
    return f"primary-{spec.task_id}-{identity[:16]}"


def _resource_paths(
    root: Path,
    manifest: Any,
    no_touch_checkpoint: Path,
) -> tuple[dict[str, LiveMatrixResourceEntry], dict[str, dict[str, object]]]:
    typed: dict[str, LiveMatrixResourceEntry] = {}
    portable: dict[str, dict[str, object]] = {}
    for cell in manifest.cells:
        fault_relative = (
            None
            if cell.fault_manifest is None
            else f"{FAULT_MANIFEST_DIRECTORY}/{cell.sha256}.json"
        )
        rest_relative = (
            f"{REST_REFERENCE_DIRECTORY}/{REST_REFERENCE_PATH}"
            if cell.operator_id in REST_REFERENCE_OPERATOR_IDS
            else None
        )
        no_touch = (
            no_touch_checkpoint if cell.trial.condition is Condition.NO_TOUCH else None
        )
        typed[cell.sha256] = LiveMatrixResourceEntry(
            fault_manifest_path=(
                None if fault_relative is None else root / fault_relative
            ),
            rest_references_path=(
                None if rest_relative is None else root / rest_relative
            ),
            matched_no_touch_artifact_path=no_touch,
        )
        portable[cell.sha256] = {
            "fault_manifest_path": fault_relative,
            "rest_references_path": rest_relative,
            "matched_no_touch_artifact_path": (
                None if no_touch is None else str(no_touch)
            ),
        }
    return typed, portable


def _write_bundle(staging: Path, spec: PrimaryMatrixGenerationSpec) -> None:
    rest = load_rest_reference_artifact(spec.rest_reference_artifact)
    if rest.validation.task != spec.task_id:
        raise PrimaryMatrixGenerationError(
            "rest-reference artifact task differs from the primary matrix task"
        )
    task_config = build_univtac_backend_config(spec.task_id)
    if spec.max_control_cycles != task_config.task.action_horizon:
        raise PrimaryMatrixGenerationError(
            "primary matrix control budget must equal the frozen task horizon"
        )
    tactile = _official_manifest(
        spec, OfficialACTProfile.UNIVTAC, spec.tactile_checkpoint_sha256
    )
    no_touch = _official_manifest(
        spec, OfficialACTProfile.VISION_ONLY, spec.no_touch_checkpoint_sha256
    )
    matrix = build_primary_matrix_manifest(
        matrix_id=_matrix_id(
            spec,
            rest.root_receipt_sha256,
            tactile.config_sha256,
            no_touch.config_sha256,
        ),
        clean=_clean_trial(spec, tactile),
        fault_manifests=_fault_manifests(spec, rest.references.sha256),
        restoration_index=spec.restoration_index,
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id=spec.no_touch_system_id,
        no_touch_checkpoint_sha256=no_touch.checkpoint_sha256,
        no_touch_config_sha256=no_touch.config_sha256,
    )
    write_json(staging / MATRIX_MANIFEST_PATH, matrix.to_dict())
    for cell in matrix.cells:
        if cell.fault_manifest is not None:
            write_json(
                staging / FAULT_MANIFEST_DIRECTORY / f"{cell.sha256}.json",
                cell.fault_manifest.to_dict(),
            )
    for filename in (REST_REFERENCE_PATH, VALIDATION_PATH, ROOT_RECEIPT_PATH):
        copy_regular_file(
            spec.rest_reference_artifact / filename,
            staging / REST_REFERENCE_DIRECTORY / filename,
        )
    typed_resources, portable_resources = _resource_paths(
        staging, matrix, no_touch.checkpoint_path
    )
    run_config = LiveMatrixRunConfig(
        matrix_manifest_sha256=matrix.sha256,
        policy_kind=LivePolicyKind.ACT,
        max_control_cycles=spec.max_control_cycles,
        max_observation_steps=spec.max_observation_steps,
        execute_action_steps=1,
        wall_timeout_s=spec.wall_timeout_s,
        upstream_root=spec.upstream_root,
        runtime_root=spec.runtime_root,
        act_device_name=spec.act_device_name,
        simulator_device=spec.simulator_device,
        launcher_args=production_univtac_launcher_args(),
        official_act_artifact_root=spec.official_act_artifact_root,
        stats_sha256=spec.stats_sha256,
        encoder_sha256=spec.encoder_sha256,
        resources=typed_resources,
    )
    write_json(
        staging / LIVE_MATRIX_RUN_CONFIG_PATH,
        relative_run_config_document(run_config.to_dict(), portable_resources),
    )
    load_live_matrix_run_config(staging / LIVE_MATRIX_RUN_CONFIG_PATH, matrix)
    members = scan_files(staging)
    receipt = PrimaryMatrixGenerationReceipt(
        matrix_id=matrix.matrix_id,
        matrix_manifest_sha256=matrix.sha256,
        generation_contract_sha256=spec.contract_sha256,
        pair_key=matrix.pair_key,
        task_id=spec.task_id,
        cell_count=len(matrix.cells),
        comparison_count=len(matrix.comparisons),
        fault_manifest_count=sum(
            cell.fault_manifest is not None for cell in matrix.cells
        ),
        rest_reference_artifact_root_sha256=rest.root_receipt_sha256,
        rest_reference_sha256=rest.references.sha256,
        tactile_config_sha256=tactile.config_sha256,
        no_touch_config_sha256=no_touch.config_sha256,
        members=members,
    )
    write_json(staging / PRIMARY_RECEIPT_PATH, receipt.to_dict())


def generate_primary_matrix_bundle(
    output: Path, spec: PrimaryMatrixGenerationSpec
) -> tuple[str, LoadedPrimaryMatrixGeneration]:
    """Generate atomically or reuse an exact existing primary request bundle."""

    if type(spec) is not PrimaryMatrixGenerationSpec:
        raise TypeError("spec must be an exact PrimaryMatrixGenerationSpec")
    target = Path(output).expanduser().absolute()
    staging = staging_directory(target)
    try:
        _write_bundle(staging, spec)
        load_primary_matrix_generation(staging)
        status = publish_staging(staging, target)
        return status, load_primary_matrix_generation(target)
    finally:
        discard_staging(staging)


def load_primary_matrix_generation(root: Path) -> LoadedPrimaryMatrixGeneration:
    """Fail closed on inventory, hashes, typed links, or resource drift."""

    directory = Path(root).expanduser().absolute()
    receipt = PrimaryMatrixGenerationReceipt.from_dict(
        read_json(directory / PRIMARY_RECEIPT_PATH, "primary matrix receipt")
    )
    actual = scan_files(directory, exclude=PRIMARY_RECEIPT_PATH)
    if actual != dict(receipt.members):
        raise PrimaryMatrixGenerationError("primary matrix member inventory mismatch")
    manifest = load_matrix_manifest_file(directory / MATRIX_MANIFEST_PATH)
    run_config = load_live_matrix_run_config(
        directory / LIVE_MATRIX_RUN_CONFIG_PATH, manifest
    )
    rest = load_rest_reference_artifact(directory / REST_REFERENCE_DIRECTORY)
    if (
        manifest.matrix_id != receipt.matrix_id
        or manifest.sha256 != receipt.matrix_manifest_sha256
        or manifest.pair_key != receipt.pair_key
        or len(manifest.cells) != receipt.cell_count
        or len(manifest.comparisons) != receipt.comparison_count
        or rest.root_receipt_sha256 != receipt.rest_reference_artifact_root_sha256
        or rest.references.sha256 != receipt.rest_reference_sha256
    ):
        raise PrimaryMatrixGenerationError("primary matrix receipt links disagree")
    _validate_generated_resources(directory, manifest, run_config, receipt)
    return LoadedPrimaryMatrixGeneration(
        root=directory,
        manifest=manifest,
        run_config=run_config,
        receipt=receipt,
        receipt_file_sha256=file_sha256(directory / PRIMARY_RECEIPT_PATH),
    )


def _validate_generated_resources(
    root: Path,
    manifest: Any,
    run_config: LiveMatrixRunConfig,
    receipt: PrimaryMatrixGenerationReceipt,
) -> None:
    no_touch_path = (
        run_config.official_act_artifact_root
        / receipt.task_id
        / OfficialACTProfile.VISION_ONLY.value
        / "policy_last.ckpt"
    )
    fault_count = 0
    for cell in manifest.cells:
        resource = run_config.resources[cell.sha256]
        if cell.trial.task != receipt.task_id:
            raise PrimaryMatrixGenerationError("matrix task identity drift")
        expected_config = (
            receipt.no_touch_config_sha256
            if cell.trial.condition is Condition.NO_TOUCH
            else receipt.tactile_config_sha256
        )
        if cell.trial.config_sha256 != expected_config:
            raise PrimaryMatrixGenerationError("policy config identity drift")
        if cell.fault_manifest is None:
            if resource.fault_manifest_path is not None:
                raise PrimaryMatrixGenerationError("baseline cell references a fault")
        else:
            fault_count += 1
            expected = root / FAULT_MANIFEST_DIRECTORY / f"{cell.sha256}.json"
            if resource.fault_manifest_path != expected:
                raise PrimaryMatrixGenerationError("fault resource path drift")
            loaded_fault = FaultManifest.from_dict(
                cast(
                    Mapping[str, Any],
                    read_json(expected, "generated fault manifest"),
                )
            )
            if loaded_fault != cell.fault_manifest:
                raise PrimaryMatrixGenerationError("fault resource content drift")
        expected_rest = (
            root / REST_REFERENCE_DIRECTORY / REST_REFERENCE_PATH
            if cell.operator_id in REST_REFERENCE_OPERATOR_IDS
            else None
        )
        if resource.rest_references_path != expected_rest:
            raise PrimaryMatrixGenerationError("rest-reference resource path drift")
        expected_no_touch = (
            no_touch_path if cell.trial.condition is Condition.NO_TOUCH else None
        )
        if resource.matched_no_touch_artifact_path != expected_no_touch:
            raise PrimaryMatrixGenerationError("no-touch resource path drift")
    if fault_count != receipt.fault_manifest_count:
        raise PrimaryMatrixGenerationError("fault resource count drift")
