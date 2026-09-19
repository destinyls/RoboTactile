"""Deterministically generate official ACT Clean/Faulted campaign bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Tuple

from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    REST_REFERENCE_PATH,
    ROOT_RECEIPT_PATH,
    VALIDATION_PATH,
)
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    SENSOR_SLOTS,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.integrations.act.artifacts import (
    load_act_artifact_manifest,
)
from robotactile_benchmark.manifests import Observability
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition

from .contracts import (
    ACT_FAULT_TEMPLATE_SEED_DERIVATION,
    ACTFaultCampaignCellSpec,
    ACTFaultCampaignError,
    ACTFaultCampaignManifest,
    ACTFaultCellDisposition,
    _identifier,
    _integer,
)
from .generation_cells import write_clean_cell, write_fault_cell
from .io import (
    CAMPAIGN_MANIFEST_PATH,
    GENERATION_RECEIPT_PATH,
    ACTFaultCampaignGenerationReceipt,
    LoadedACTFaultCampaign,
    copy_regular_file,
    discard_staging,
    file_sha256,
    load_act_fault_campaign_bundle,
    publish_staging,
    scan_files,
    staging_directory,
    write_json,
)
from .reset_reference import (
    MAX_ACT_RESET_QPOS_ATOL,
    act_reset_reference_relpath,
    load_act_reset_reference,
)
from .reset_trajectory import (
    act_reset_trajectory_relpath,
    load_act_reset_trajectory,
)

ACT_FAULT_GENERATION_SPEC_VERSION = "1.0"


@dataclass(frozen=True)
class ACTFaultCampaignGenerationSpec:
    """Frozen sources and axes for one full official ACT campaign."""

    campaign_id: str
    base_clean_request_paths: Tuple[Path, ...]
    artifact_manifest_paths: Mapping[str, Path]
    deployment_layout: DeploymentLayout
    operator_ids: Tuple[str, ...]
    severity_levels: Tuple[int, ...]
    operator_seed_master: int
    fault_start_index: int
    fault_stop_index: int
    rest_reference_artifacts: Mapping[str, Path]
    reset_reference_artifact_paths: Tuple[Path, ...]
    reset_trajectory_artifact_paths: Tuple[Path, ...]
    sensor_slots: Tuple[str, ...] = SENSOR_SLOTS
    observability: Observability = Observability.BLIND
    profile: OfficialACTProfile = OfficialACTProfile.UNIVTAC
    severity_registry: str = SEVERITY_REGISTRY_ID
    semantic_version: str = ACT_FAULT_GENERATION_SPEC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        paths = tuple(
            Path(path).expanduser().absolute() for path in self.base_clean_request_paths
        )
        if not paths or len(paths) != len(set(paths)):
            raise ACTFaultCampaignError(
                "base Clean request paths must be non-empty and unique"
            )
        operators = tuple(sorted(self.operator_ids))
        levels = tuple(sorted(self.severity_levels))
        if (
            operators != tuple(sorted(CORE_OPERATOR_IDS))
            or len(self.operator_ids) != 14
        ):
            raise ACTFaultCampaignError("ACT fault campaign requires all 14 operators")
        if len(levels) != 1 or levels[0] not in range(1, 6):
            raise ACTFaultCampaignError(
                "ACT fault campaign requires exactly one severity"
            )
        for name in ("operator_seed_master", "fault_start_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(
            self,
            "fault_stop_index",
            _integer(self.fault_stop_index, "fault_stop_index", 1),
        )
        if self.fault_start_index >= self.fault_stop_index:
            raise ACTFaultCampaignError("fault window must satisfy start < stop")
        if tuple(self.sensor_slots) != SENSOR_SLOTS:
            raise ACTFaultCampaignError("ACT campaign requires both tactile slots")
        observability = Observability(self.observability)
        profile = OfficialACTProfile(self.profile)
        if observability is not Observability.BLIND:
            raise ACTFaultCampaignError("ACT primary robustness faults must be blind")
        if profile is not OfficialACTProfile.UNIVTAC:
            raise ACTFaultCampaignError("ACT fault campaign requires profile=univtac")
        if self.severity_registry != SEVERITY_REGISTRY_ID:
            raise ACTFaultCampaignError(
                "ACT campaign requires the primary severity registry"
            )
        manifests = _path_mapping(self.artifact_manifest_paths, "artifact manifest")
        references = _path_mapping(self.rest_reference_artifacts, "rest reference")
        reset_references = tuple(
            Path(path).expanduser().absolute()
            for path in self.reset_reference_artifact_paths
        )
        if not reset_references or len(reset_references) != len(set(reset_references)):
            raise ACTFaultCampaignError(
                "reset reference paths must be non-empty and unique"
            )
        reset_trajectories = tuple(
            Path(path).expanduser().absolute()
            for path in self.reset_trajectory_artifact_paths
        )
        if not reset_trajectories or len(reset_trajectories) != len(
            set(reset_trajectories)
        ):
            raise ACTFaultCampaignError(
                "reset trajectory paths must be non-empty and unique"
            )
        if self.semantic_version != ACT_FAULT_GENERATION_SPEC_VERSION:
            raise ACTFaultCampaignError("generation spec version mismatch")
        object.__setattr__(self, "base_clean_request_paths", paths)
        object.__setattr__(self, "artifact_manifest_paths", manifests)
        object.__setattr__(self, "rest_reference_artifacts", references)
        object.__setattr__(
            self,
            "reset_reference_artifact_paths",
            reset_references,
        )
        object.__setattr__(
            self,
            "reset_trajectory_artifact_paths",
            reset_trajectories,
        )
        object.__setattr__(self, "operator_ids", operators)
        object.__setattr__(self, "severity_levels", levels)
        object.__setattr__(self, "sensor_slots", SENSOR_SLOTS)
        object.__setattr__(self, "observability", observability)
        object.__setattr__(self, "profile", profile)


def _path_mapping(value: Mapping[str, Path], label: str) -> Mapping[str, Path]:
    result = {
        _identifier(task, f"{label} task"): Path(path).expanduser().absolute()
        for task, path in value.items()
    }
    if len(result) != len(value):
        raise ACTFaultCampaignError(f"duplicate {label} task")
    return MappingProxyType(result)


def derive_operator_template_seed(
    *, master_seed: int, pair_key: str, operator_id: str
) -> int:
    _integer(master_seed, "master_seed")
    if operator_id not in CORE_OPERATOR_IDS:
        raise ACTFaultCampaignError("unknown operator for template seed")
    digest = canonical_hash(
        {
            "namespace": ACT_FAULT_TEMPLATE_SEED_DERIVATION,
            "master_seed": master_seed,
            "pair_key": pair_key,
            "operator_id": operator_id,
        }
    )
    return int(digest[:8], 16) & 0x7FFFFFFF


def generate_act_fault_campaign_bundle(
    output: Path, spec: ACTFaultCampaignGenerationSpec
) -> tuple[str, LoadedACTFaultCampaign]:
    """Generate atomically, or accept an exact existing bundle."""

    if type(spec) is not ACTFaultCampaignGenerationSpec:
        raise TypeError("spec must be an exact ACTFaultCampaignGenerationSpec")
    target = Path(output).expanduser().absolute()
    staging = staging_directory(target)
    try:
        _write_bundle(staging, spec)
        load_act_fault_campaign_bundle(staging)
        status = publish_staging(staging, target)
        return status, load_act_fault_campaign_bundle(target)
    finally:
        discard_staging(staging)


def _write_bundle(root: Path, spec: ACTFaultCampaignGenerationSpec) -> None:
    sources = _load_clean_sources(spec.base_clean_request_paths)
    tasks = {loaded.trial.task for _, _, loaded in sources}
    if set(spec.artifact_manifest_paths) != tasks:
        raise ACTFaultCampaignError(
            "artifact manifests must exactly cover source tasks"
        )
    manifests = {
        task: load_act_artifact_manifest(spec.artifact_manifest_paths[task])
        for task in sorted(tasks)
    }
    for task, manifest in manifests.items():
        if (
            manifest.task_id != task
            or manifest.profile is not OfficialACTProfile.UNIVTAC
        ):
            raise ACTFaultCampaignError("ACT artifact manifest task/profile mismatch")
    reset_references = _copy_reset_references(root, sources, spec)
    reset_trajectories = _copy_reset_trajectories(
        root,
        sources,
        spec,
        reset_references,
    )
    rest = _copy_rest_artifacts(root, tasks, spec)
    cells: list[ACTFaultCampaignCellSpec] = []
    request_hashes: list[str] = []
    for source_path, base, loaded in sources:
        request_hashes.append(file_sha256(source_path))
        manifest = manifests[loaded.trial.task]
        cells.append(write_clean_cell(root, spec, manifest, base, loaded, len(cells)))
        for operator_id in spec.operator_ids:
            template_seed = derive_operator_template_seed(
                master_seed=spec.operator_seed_master,
                pair_key=loaded.trial.pair_key,
                operator_id=operator_id,
            )
            cells.append(
                write_fault_cell(
                    root,
                    spec,
                    manifest,
                    base,
                    loaded,
                    rest,
                    operator_id,
                    spec.severity_levels[0],
                    template_seed,
                    len(cells),
                )
            )
    campaign = ACTFaultCampaignManifest(
        campaign_id=spec.campaign_id,
        policy_kind=LivePolicyKind.ACT,
        profile=OfficialACTProfile.UNIVTAC,
        operator_ids=spec.operator_ids,
        severity_levels=spec.severity_levels,
        operator_template_seed_derivation=ACT_FAULT_TEMPLATE_SEED_DERIVATION,
        pair_count=len(sources),
        cell_count=len(cells),
        live_request_count=sum(
            cell.disposition is ACTFaultCellDisposition.LIVE_REQUEST for cell in cells
        ),
        unsupported_contract_count=sum(
            cell.disposition is ACTFaultCellDisposition.UNSUPPORTED_CONTRACT
            for cell in cells
        ),
        rest_reference_bindings={task: binding for task, (_, binding) in rest.items()},
        cells=tuple(cells),
    )
    write_json(root / CAMPAIGN_MANIFEST_PATH, campaign.to_dict())
    generation_hash = canonical_hash(
        {
            "campaign_id": spec.campaign_id,
            "source_request_sha256s": sorted(request_hashes),
            "artifact_manifest_sha256s": {
                task: file_sha256(spec.artifact_manifest_paths[task])
                for task in sorted(tasks)
            },
            "deployment_layout": spec.deployment_layout.to_dict(),
            "operator_ids": list(spec.operator_ids),
            "severity_levels": list(spec.severity_levels),
            "operator_seed_master": spec.operator_seed_master,
            "fault_window": [spec.fault_start_index, spec.fault_stop_index],
            "sensor_slots": list(spec.sensor_slots),
            "observability": spec.observability.value,
            "profile": spec.profile.value,
            "severity_registry": spec.severity_registry,
            "rest_reference_bindings": campaign.to_dict()["rest_reference_bindings"],
            "reset_reference_bindings": reset_references,
            "reset_trajectory_bindings": reset_trajectories,
            "semantic_version": spec.semantic_version,
        }
    )
    receipt = ACTFaultCampaignGenerationReceipt(
        campaign_id=spec.campaign_id,
        campaign_manifest_sha256=campaign.sha256,
        generation_contract_sha256=generation_hash,
        pair_count=campaign.pair_count,
        cell_count=campaign.cell_count,
        live_request_count=campaign.live_request_count,
        unsupported_contract_count=campaign.unsupported_contract_count,
        fault_manifest_count=campaign.cell_count - campaign.pair_count,
        members=scan_files(root),
    )
    write_json(root / GENERATION_RECEIPT_PATH, receipt.to_dict())


def _load_clean_sources(
    paths: Sequence[Path],
) -> list[tuple[Path, LiveUniVTACRunRequest, Any]]:
    result: list[tuple[Path, LiveUniVTACRunRequest, Any]] = []
    identities: dict[tuple[str, int, int], str] = {}
    for path in paths:
        request = load_live_univtac_request(path)
        loaded = load_live_univtac_run(request)
        if (
            request.policy_kind is not LivePolicyKind.ACT
            or request.condition is not Condition.CLEAN
        ):
            raise ACTFaultCampaignError("base request must be official ACT Clean")
        if (
            request.execute_action_steps != 1
            or request.act_device_name is None
            or request.simulator_device is None
        ):
            raise ACTFaultCampaignError(
                "base ACT request execution contract is invalid"
            )
        identity = (
            loaded.trial.task,
            loaded.trial.initial_seed,
            loaded.trial.exogenous_seed,
        )
        if identity in identities or loaded.trial.pair_key in identities.values():
            raise ACTFaultCampaignError("base requests duplicate a task/seed or pair")
        identities[identity] = loaded.trial.pair_key
        result.append((path, request, loaded))
    return sorted(
        result,
        key=lambda item: (
            item[2].trial.task,
            item[2].trial.initial_seed,
            item[2].trial.exogenous_seed,
        ),
    )


def _copy_reset_references(
    root: Path,
    sources: Sequence[tuple[Path, LiveUniVTACRunRequest, Any]],
    spec: ACTFaultCampaignGenerationSpec,
) -> dict[str, dict[str, str]]:
    expected = {loaded.trial.pair_key: loaded for _, _, loaded in sources}
    loaded_references = {}
    source_paths = {}
    for path in spec.reset_reference_artifact_paths:
        reference = load_act_reset_reference(path)
        if reference.pair_key in loaded_references:
            raise ACTFaultCampaignError("duplicate reset reference pair_key")
        loaded_references[reference.pair_key] = reference
        source_paths[reference.pair_key] = path
    if set(loaded_references) != set(expected):
        raise ACTFaultCampaignError(
            "reset references must exactly cover base Clean pairs"
        )
    bindings: dict[str, dict[str, str]] = {}
    for pair_key, loaded in sorted(expected.items()):
        reference = loaded_references[pair_key]
        trial = loaded.trial
        if (
            reference.task_id != trial.task
            or reference.initial_seed != trial.initial_seed
            or reference.exogenous_seed != trial.exogenous_seed
            or reference.dataset_sha256 != trial.dataset_sha256
            or reference.checkpoint_sha256 != trial.checkpoint_sha256
            or reference.config_sha256 != trial.config_sha256
            or reference.source_run_content_sha256 != loaded.content_sha256
            or reference.qpos_atol > MAX_ACT_RESET_QPOS_ATOL
        ):
            raise ACTFaultCampaignError(
                "reset reference identity differs from the base Clean trial"
            )
        relative = act_reset_reference_relpath(trial.task, pair_key)
        destination = root / relative
        copy_regular_file(source_paths[pair_key], destination)
        copied = load_act_reset_reference(destination)
        if copied != reference:
            raise ACTFaultCampaignError("copied reset reference changed")
        bindings[pair_key] = {
            "artifact_relpath": relative,
            "file_sha256": file_sha256(destination),
            "reference_sha256": reference.sha256,
        }
    return bindings


def _copy_reset_trajectories(
    root: Path,
    sources: Sequence[tuple[Path, LiveUniVTACRunRequest, Any]],
    spec: ACTFaultCampaignGenerationSpec,
    reset_reference_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Copy one source-bound dense pre-move trajectory per Clean pair."""

    expected = {loaded.trial.pair_key: loaded for _, _, loaded in sources}
    trajectories = {}
    source_paths = {}
    for path in spec.reset_trajectory_artifact_paths:
        trajectory = load_act_reset_trajectory(path)
        if trajectory.pair_key in trajectories:
            raise ACTFaultCampaignError("duplicate reset trajectory pair_key")
        trajectories[trajectory.pair_key] = trajectory
        source_paths[trajectory.pair_key] = path
    if set(trajectories) != set(expected):
        raise ACTFaultCampaignError(
            "reset trajectories must exactly cover base Clean pairs"
        )
    bindings: dict[str, dict[str, str]] = {}
    for pair_key, loaded in sorted(expected.items()):
        trajectory = trajectories[pair_key]
        trial = loaded.trial
        reference_binding = reset_reference_bindings[pair_key]
        reference_path = root / reference_binding["artifact_relpath"]
        reference = load_act_reset_reference(reference_path)
        if (
            trajectory.task_id != trial.task
            or trajectory.initial_seed != trial.initial_seed
            or trajectory.exogenous_seed != trial.exogenous_seed
            or trajectory.pair_key != trial.pair_key
            or trajectory.dataset_sha256 != trial.dataset_sha256
            or trajectory.checkpoint_sha256 != trial.checkpoint_sha256
            or trajectory.config_sha256 != trial.config_sha256
            or trajectory.source_run_content_sha256 != loaded.content_sha256
            or trajectory.reset_reference_sha256 != reference.sha256
            or trajectory.upstream_commit != loaded.backend_config.upstream_commit
            or trajectory.task_source_sha256
            != loaded.backend_config.task.task_source_sha256
        ):
            raise ACTFaultCampaignError(
                "reset trajectory identity differs from the base Clean trial"
            )
        relative = act_reset_trajectory_relpath(trial.task, pair_key)
        destination = root / relative
        copy_regular_file(source_paths[pair_key], destination)
        copied = load_act_reset_trajectory(destination)
        if copied != trajectory:
            raise ACTFaultCampaignError("copied reset trajectory changed")
        bindings[pair_key] = {
            "artifact_relpath": relative,
            "file_sha256": file_sha256(destination),
            "trajectory_sha256": trajectory.sha256,
        }
    return bindings


def _copy_rest_artifacts(
    root: Path, tasks: set[str], spec: ACTFaultCampaignGenerationSpec
) -> dict[str, tuple[Path, dict[str, str]]]:
    if set(spec.rest_reference_artifacts) != tasks:
        raise ACTFaultCampaignError(
            "rest-reference artifacts must exactly cover source tasks"
        )
    result: dict[str, tuple[Path, dict[str, str]]] = {}
    for task in sorted(tasks):
        source = spec.rest_reference_artifacts[task]
        loaded = load_rest_reference_artifact(source)
        if loaded.validation.task != task:
            raise ACTFaultCampaignError("rest-reference task binding mismatch")
        relative_root = f"rest_references/{task}"
        for filename in (REST_REFERENCE_PATH, VALIDATION_PATH, ROOT_RECEIPT_PATH):
            copy_regular_file(source / filename, root / relative_root / filename)
        copied = load_rest_reference_artifact(root / relative_root)
        result[task] = (
            root / relative_root / REST_REFERENCE_PATH,
            {
                "artifact_relpath": relative_root,
                "artifact_root_sha256": copied.root_receipt_sha256,
                "rest_reference_sha256": copied.references.sha256,
            },
        )
    return result
