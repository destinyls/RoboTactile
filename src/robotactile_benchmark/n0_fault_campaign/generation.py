"""Generate and strictly reload N0-TWAM Clean/Faulted campaign bundles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    REST_REFERENCE_PATH,
    ROOT_RECEIPT_PATH,
    VALIDATION_PATH,
)
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import canonical_hash, thaw_value
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.fault_timing import EARLY_RANDOM_ONSET_MODE
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0_FAULT_TEMPLATE_SEED_DERIVATION,
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCampaignManifest,
    N0FaultCellDisposition,
    require_integer,
)
from robotactile_benchmark.n0_fault_campaign.generation_cells import (
    write_clean_cell,
    write_fault_cell,
)
from robotactile_benchmark.n0_fault_campaign.generation_contracts import (
    N0_FAULT_GENERATION_SPEC_VERSION,
    N0FaultCampaignGenerationSpec,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    CAMPAIGN_MANIFEST_PATH,
    GENERATION_RECEIPT_PATH,
    LoadedN0FaultCampaign,
    N0FaultCampaignGenerationReceipt,
    copy_regular_file,
    discard_staging,
    file_sha256,
    load_n0_fault_campaign_bundle,
    publish_staging,
    read_json,
    scan_files,
    staging_directory,
    write_json,
)
from robotactile_benchmark.trials import Condition

__all__ = [
    "N0_FAULT_GENERATION_SPEC_VERSION",
    "N0FaultCampaignGenerationSpec",
    "derive_operator_template_seed",
    "generate_n0_fault_campaign_bundle",
]


def derive_operator_template_seed(
    *, master_seed: int, pair_key: str, operator_id: str
) -> int:
    require_integer(master_seed, "master_seed")
    if operator_id not in CORE_OPERATOR_IDS:
        raise N0FaultCampaignError("unknown operator for template seed")
    digest = canonical_hash(
        {
            "namespace": N0_FAULT_TEMPLATE_SEED_DERIVATION,
            "master_seed": master_seed,
            "pair_key": pair_key,
            "operator_id": operator_id,
        }
    )
    return int(digest[:8], 16) & 0x7FFFFFFF


def generate_n0_fault_campaign_bundle(
    output: Path, spec: N0FaultCampaignGenerationSpec
) -> tuple[str, LoadedN0FaultCampaign]:
    """Generate atomically or accept an exact existing campaign bundle."""

    if type(spec) is not N0FaultCampaignGenerationSpec:
        raise TypeError("spec must be an exact N0FaultCampaignGenerationSpec")
    target = Path(output).expanduser().absolute()
    staging = staging_directory(target)
    try:
        _write_bundle(staging, spec)
        load_n0_fault_campaign_bundle(staging)
        status = publish_staging(staging, target)
        return status, load_n0_fault_campaign_bundle(target)
    finally:
        discard_staging(staging)


def _write_bundle(root: Path, spec: N0FaultCampaignGenerationSpec) -> None:
    sources = _load_clean_sources(spec.base_clean_request_paths)
    if spec.fault_onset_mode == EARLY_RANDOM_ONSET_MODE and any(
        request.max_observation_steps != spec.fault_stop_index
        for _, request, _ in sources
    ):
        raise N0FaultCampaignError(
            "early random onset must continue to each request observation horizon"
        )
    if spec.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID and any(
        request.max_observation_steps != spec.fault_stop_index
        for _, request, _ in sources
    ):
        raise N0FaultCampaignError(
            "tactile-null stop index must equal every request observation horizon"
        )
    if (
        spec.severity_registry == DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID
        and any(
            request.max_observation_steps != spec.fault_stop_index
            for _, request, _ in sources
        )
    ):
        raise N0FaultCampaignError(
            "observed-tactile absence must cover every request observation horizon"
        )
    task_ids = {loaded.trial.task for _, _, loaded in sources}
    rest = _copy_rest_artifacts(root, task_ids, spec)
    cells: list[N0FaultCampaignCellSpec] = []
    source_hashes: list[str] = []
    for source_path, base_request, loaded_clean in sources:
        source_hashes.append(file_sha256(source_path))
        cells.append(
            write_clean_cell(root, spec, base_request, loaded_clean, len(cells))
        )
        for operator_id in spec.operator_ids:
            template_seed = derive_operator_template_seed(
                master_seed=spec.operator_seed_master,
                pair_key=loaded_clean.trial.pair_key,
                operator_id=operator_id,
            )
            for severity_level in spec.severity_levels:
                cells.append(
                    write_fault_cell(
                        root=root,
                        spec=spec,
                        base_request=base_request,
                        loaded_clean=loaded_clean,
                        rest=rest,
                        operator_id=operator_id,
                        severity_level=severity_level,
                        template_seed=template_seed,
                        ordinal=len(cells),
                    )
                )
    manifest = N0FaultCampaignManifest(
        campaign_id=spec.campaign_id,
        policy_kind=LivePolicyKind.N0,
        operator_ids=spec.operator_ids,
        severity_levels=spec.severity_levels,
        operator_template_seed_derivation=N0_FAULT_TEMPLATE_SEED_DERIVATION,
        pair_count=len(sources),
        cell_count=len(cells),
        live_request_count=sum(
            cell.disposition is N0FaultCellDisposition.LIVE_REQUEST for cell in cells
        ),
        unsupported_contract_count=sum(
            cell.disposition is N0FaultCellDisposition.UNSUPPORTED_CONTRACT
            for cell in cells
        ),
        rest_reference_bindings={task: binding for task, (_, binding) in rest.items()},
        cells=tuple(cells),
        severity_registry=spec.severity_registry,
    )
    write_json(root / CAMPAIGN_MANIFEST_PATH, manifest.to_dict())
    members = scan_files(root)
    generation_contract: dict[str, object] = {
        "campaign_id": spec.campaign_id,
        "source_request_sha256s": sorted(source_hashes),
        "operator_ids": list(spec.operator_ids),
        "severity_levels": list(spec.severity_levels),
        "operator_seed_master": spec.operator_seed_master,
        "fault_window": [spec.fault_start_index, spec.fault_stop_index],
        "sensor_slots": list(spec.sensor_slots),
        "observability": spec.observability.value,
        "severity_registry": spec.severity_registry,
        "rest_reference_bindings": manifest.to_dict()["rest_reference_bindings"],
        "semantic_version": spec.semantic_version,
    }
    if spec.fault_onset_mode == EARLY_RANDOM_ONSET_MODE:
        generation_contract["fault_onset_mode"] = spec.fault_onset_mode
        generation_contract["fault_onset_max_index"] = spec.fault_onset_max_index
    if spec.spatial_calibration:
        generation_contract["spatial_calibration"] = thaw_value(
            spec.spatial_calibration
        )
    generation_hash = canonical_hash(generation_contract)
    receipt = N0FaultCampaignGenerationReceipt(
        campaign_id=spec.campaign_id,
        campaign_manifest_sha256=manifest.sha256,
        generation_contract_sha256=generation_hash,
        pair_count=manifest.pair_count,
        cell_count=manifest.cell_count,
        live_request_count=manifest.live_request_count,
        unsupported_contract_count=manifest.unsupported_contract_count,
        fault_manifest_count=sum(
            cell.fault_manifest_sha256 is not None for cell in cells
        ),
        members=members,
    )
    write_json(root / GENERATION_RECEIPT_PATH, receipt.to_dict())


def _load_clean_sources(
    paths: Sequence[Path],
) -> list[tuple[Path, LiveUniVTACRunRequest, Any]]:
    sources: list[tuple[Path, LiveUniVTACRunRequest, Any]] = []
    identities: dict[tuple[str, int, int], str] = {}
    for path in paths:
        read_json(path, "base N0 Clean request")
        request = load_live_univtac_request(path)
        loaded = load_live_univtac_run(request)
        if (
            request.policy_kind is not LivePolicyKind.N0
            or request.condition is not Condition.CLEAN
        ):
            raise N0FaultCampaignError("base request must be N0 Clean")
        identity = (
            loaded.trial.task,
            loaded.trial.initial_seed,
            loaded.trial.exogenous_seed,
        )
        if identity in identities or loaded.trial.pair_key in identities.values():
            raise N0FaultCampaignError("base requests duplicate a task/seed or pair")
        identities[identity] = loaded.trial.pair_key
        sources.append((path, request, loaded))
    return sorted(
        sources,
        key=lambda item: (
            item[2].trial.task,
            item[2].trial.initial_seed,
            item[2].trial.exogenous_seed,
        ),
    )


def _copy_rest_artifacts(
    root: Path,
    tasks: set[str],
    spec: N0FaultCampaignGenerationSpec,
) -> dict[str, tuple[Path, dict[str, str]]]:
    required = (
        tasks
        if any(
            operator_requires_rest_reference(
                operator_id,
                severity_registry=spec.severity_registry,
            )
            for operator_id in spec.operator_ids
        )
        else set()
    )
    if set(spec.rest_reference_artifacts) != required:
        raise N0FaultCampaignError(
            "rest-reference artifacts must exactly cover required tasks"
        )
    result: dict[str, tuple[Path, dict[str, str]]] = {}
    for task in sorted(required):
        source = spec.rest_reference_artifacts[task]
        loaded = load_rest_reference_artifact(source)
        if loaded.validation.task != task:
            raise N0FaultCampaignError("rest-reference task binding mismatch")
        relative_root = f"rest_references/{task}"
        for filename in (REST_REFERENCE_PATH, VALIDATION_PATH, ROOT_RECEIPT_PATH):
            copy_regular_file(source / filename, root / relative_root / filename)
        copied_root = root / relative_root
        copied = load_rest_reference_artifact(copied_root)
        result[task] = (
            copied_root / REST_REFERENCE_PATH,
            {
                "artifact_relpath": relative_root,
                "artifact_root_sha256": copied.root_receipt_sha256,
                "rest_reference_sha256": copied.references.sha256,
            },
        )
    return result
