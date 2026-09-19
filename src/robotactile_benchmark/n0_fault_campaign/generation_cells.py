"""Write immutable Clean/Faulted cells for an N0 fault campaign bundle."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACBackendConfig,
)
from robotactile_benchmark.constants import (
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPTICAL_CONTACT_STRESS_REGISTRY_ID,
    OPTICAL_MARKER_REGISTRY_IDS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import thaw_value
from robotactile_benchmark.execution.contracts import (
    LiveUniVTACRunRequest,
    N0ObservedTactileMode,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.fault_timing import (
    EARLY_RANDOM_ONSET_MODE,
    FIXED_FAULT_ONSET_MODE,
    derive_early_random_onset,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import Condition

from .contracts import (
    N0_SUPPORTED_OPERATOR_IDS,
    N0_UNSUPPORTED_OPERATOR_IDS,
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCellDisposition,
    N0UnsupportedContractSpec,
)
from .generation_contracts import N0FaultCampaignGenerationSpec
from .io import file_sha256, write_json


def write_clean_cell(
    root: Path,
    spec: N0FaultCampaignGenerationSpec,
    base: LiveUniVTACRunRequest,
    loaded: Any,
    ordinal: int,
) -> N0FaultCampaignCellSpec:
    """Write the single shared Clean request for one task/seed pair."""

    pair = loaded.trial.pair_key
    request_rel = f"requests/{loaded.trial.task}/{pair}/clean.json"
    artifact_rel = f"artifacts/{loaded.trial.task}/{pair}/clean"
    request = replace(
        base,
        output_dir=root / artifact_rel,
        runtime_dir=_runtime_dir(base, spec.campaign_id, pair, "clean"),
    )
    request_hash, reloaded = _write_portable_request(
        root, request_rel, request, artifact_rel
    )
    return N0FaultCampaignCellSpec(
        ordinal=ordinal,
        task=loaded.trial.task,
        initial_seed=loaded.trial.initial_seed,
        exogenous_seed=loaded.trial.exogenous_seed,
        pair_key=pair,
        condition=Condition.CLEAN,
        disposition=N0FaultCellDisposition.LIVE_REQUEST,
        operator_id=None,
        severity_level=None,
        operator_template_seed=None,
        request_relpath=request_rel,
        artifact_relpath=artifact_rel,
        request_file_sha256=request_hash,
        trial_manifest_sha256=reloaded.trial.sha256,
        fault_manifest_sha256=None,
        fault_manifest_relpath=None,
        unsupported_receipt_relpath=None,
        unsupported_receipt_sha256=None,
    )


def write_fault_cell(
    *,
    root: Path,
    spec: N0FaultCampaignGenerationSpec,
    base_request: LiveUniVTACRunRequest,
    loaded_clean: Any,
    rest: Mapping[str, tuple[Path, dict[str, str]]],
    operator_id: str,
    severity_level: int,
    template_seed: int,
    ordinal: int,
) -> N0FaultCampaignCellSpec:
    """Write one supported live request or A1/A2 unsupported receipt."""

    task, pair = loaded_clean.trial.task, loaded_clean.trial.pair_key
    rest_entry = rest.get(task)
    rest_sha = None if rest_entry is None else rest_entry[1]["rest_reference_sha256"]
    start_index = (
        derive_early_random_onset(
            task=task,
            seed=loaded_clean.trial.exogenous_seed,
            stop=spec.fault_stop_index,
            cap=spec.fault_onset_max_index,
        )
        if spec.fault_onset_mode == EARLY_RANDOM_ONSET_MODE
        else spec.fault_start_index
    )
    fault = _fault_manifest(
        spec,
        rest_sha,
        operator_id,
        severity_level,
        template_seed,
        start_index,
        sample_period_s=_optical_sample_period_s(loaded_clean.backend_config),
    )
    fault_rel = f"fault_manifests/{pair}/{fault.sha256}.json"
    write_json(root / fault_rel, fault.to_dict())
    fault_trial = replace(
        loaded_clean.trial,
        condition=Condition.FAULTED,
        fault_manifest_sha256=fault.sha256,
    )
    observed_tactile_absence = (
        spec.severity_registry == DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID
        and operator_id == "A1_stream_absence"
    )
    if operator_id in N0_UNSUPPORTED_OPERATOR_IDS and not observed_tactile_absence:
        return _write_unsupported_cell(
            root=root,
            spec=spec,
            trial=fault_trial,
            fault=fault,
            fault_rel=fault_rel,
            operator_id=operator_id,
            severity_level=severity_level,
            template_seed=template_seed,
            ordinal=ordinal,
        )
    if operator_id not in N0_SUPPORTED_OPERATOR_IDS and not observed_tactile_absence:
        raise N0FaultCampaignError("operator applicability is not registered")
    if observed_tactile_absence:
        label = "observed_tactile_absent_v1"
    elif spec.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID:
        label = "tactile_null_black_frame_v1"
    else:
        label = f"{operator_id}/s{severity_level}"
    request_rel = f"requests/{task}/{pair}/{label}.json"
    artifact_rel = f"artifacts/{task}/{pair}/{label}"
    request = replace(
        base_request,
        condition=Condition.FAULTED,
        output_dir=root / artifact_rel,
        fault_manifest_path=root / fault_rel,
        rest_references_path=(
            rest_entry[0]
            if operator_requires_rest_reference(
                operator_id,
                severity_registry=spec.severity_registry,
            )
            and rest_entry is not None
            else None
        ),
        runtime_dir=_runtime_dir(
            base_request, spec.campaign_id, pair, f"{operator_id}-s{severity_level}"
        ),
        n0_observed_tactile_mode=(
            N0ObservedTactileMode.ABSENT
            if observed_tactile_absence
            else N0ObservedTactileMode.REQUIRED
        ),
    )
    request_hash, reloaded = _write_portable_request(
        root, request_rel, request, artifact_rel
    )
    if reloaded.trial != fault_trial:
        raise N0FaultCampaignError("generated fault request changed the paired trial")
    return N0FaultCampaignCellSpec(
        ordinal=ordinal,
        task=task,
        initial_seed=fault_trial.initial_seed,
        exogenous_seed=fault_trial.exogenous_seed,
        pair_key=pair,
        condition=Condition.FAULTED,
        disposition=N0FaultCellDisposition.LIVE_REQUEST,
        operator_id=operator_id,
        severity_level=severity_level,
        operator_template_seed=template_seed,
        request_relpath=request_rel,
        artifact_relpath=artifact_rel,
        request_file_sha256=request_hash,
        trial_manifest_sha256=fault_trial.sha256,
        fault_manifest_sha256=fault.sha256,
        fault_manifest_relpath=fault_rel,
        unsupported_receipt_relpath=None,
        unsupported_receipt_sha256=None,
    )


def _write_unsupported_cell(
    *,
    root: Path,
    spec: N0FaultCampaignGenerationSpec,
    trial: Any,
    fault: FaultManifest,
    fault_rel: str,
    operator_id: str,
    severity_level: int,
    template_seed: int,
    ordinal: int,
) -> N0FaultCampaignCellSpec:
    unsupported = N0UnsupportedContractSpec(
        campaign_id=spec.campaign_id,
        task=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        operator_id=operator_id,
        severity_level=severity_level,
        operator_template_seed=template_seed,
        fault_manifest_sha256=fault.sha256,
        trial_manifest_sha256=trial.sha256,
    )
    receipt_rel = (
        f"unsupported_contracts/{trial.task}/{trial.pair_key}/"
        f"{operator_id}/s{severity_level}.json"
    )
    write_json(root / receipt_rel, unsupported.to_dict())
    return N0FaultCampaignCellSpec(
        ordinal=ordinal,
        task=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        condition=Condition.FAULTED,
        disposition=N0FaultCellDisposition.UNSUPPORTED_CONTRACT,
        operator_id=operator_id,
        severity_level=severity_level,
        operator_template_seed=template_seed,
        request_relpath=None,
        artifact_relpath=None,
        request_file_sha256=None,
        trial_manifest_sha256=trial.sha256,
        fault_manifest_sha256=fault.sha256,
        fault_manifest_relpath=fault_rel,
        unsupported_receipt_relpath=receipt_rel,
        unsupported_receipt_sha256=file_sha256(root / receipt_rel),
    )


def _fault_manifest(
    spec: N0FaultCampaignGenerationSpec,
    rest_sha256: str | None,
    operator_id: str,
    severity_level: int,
    template_seed: int,
    start_index: int,
    *,
    sample_period_s: float,
) -> FaultManifest:
    parameters: dict[str, object] = {}
    if operator_id == "F3_persistent_surface_artifact" and spec.spatial_calibration:
        parameters["spatial_calibration"] = thaw_value(spec.spatial_calibration)
    if spec.severity_registry in OPTICAL_MARKER_REGISTRY_IDS:
        parameters["sample_period_s"] = sample_period_s
    if operator_requires_rest_reference(
        operator_id,
        severity_registry=spec.severity_registry,
    ):
        if rest_sha256 is None:
            raise N0FaultCampaignError("rest-reference SHA is required")
        parameters["rest_reference_sha256"] = rest_sha256
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    if operator_id.startswith("T"):
        if (
            spec.fault_onset_mode == EARLY_RANDOM_ONSET_MODE
            or spec.severity_registry == OPTICAL_CONTACT_STRESS_REGISTRY_ID
        ):
            parameters["temporal_schedule"] = "window_to_end_v1"
        elif spec.fault_onset_mode == FIXED_FAULT_ONSET_MODE and start_index == 0:
            parameters["temporal_schedule"] = "full_episode_v1"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity_level,
        operator_seed=template_seed,
        start_index=start_index,
        stop_index=spec.fault_stop_index,
        sensor_slots=spec.sensor_slots,
        observability=spec.observability,
        parameters=parameters,
        severity_registry=spec.severity_registry,
    )


def _optical_sample_period_s(config: UniVTACBackendConfig) -> float:
    """Match the source timestamps emitted by the UniVTAC record builder."""

    retrained = (
        config.action_execution_contract == N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT
        or config.action_execution_contract.startswith("robotactile_retrained_")
    )
    physics_ticks = config.physics_steps_per_action if retrained else 1
    return physics_ticks / config.sim_hz


def _runtime_dir(
    base: LiveUniVTACRunRequest, campaign_id: str, pair_key: str, label: str
) -> Path:
    return base.runtime_dir / "fault-campaigns" / campaign_id / pair_key / label


def _relative(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start=start)).as_posix()


def _write_portable_request(
    root: Path,
    request_rel: str,
    request: LiveUniVTACRunRequest,
    artifact_rel: str,
) -> tuple[str, Any]:
    target = root / request_rel
    document = live_univtac_request_to_dict(request)
    document["output_dir"] = _relative(root / artifact_rel, target.parent)
    for field in ("fault_manifest_path", "rest_references_path"):
        value = getattr(request, field)
        document[field] = None if value is None else _relative(value, target.parent)
    write_json(target, document)
    loaded_request = load_live_univtac_request(target)
    loaded = load_live_univtac_run(loaded_request)
    return file_sha256(target), loaded
