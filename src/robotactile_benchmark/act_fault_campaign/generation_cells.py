"""Cell writers for official ACT fault campaign generation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.act.artifacts import ACTArtifactManifest
from robotactile_benchmark.integrations.act.requests import build_official_act_request
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import Condition

from .contracts import (
    ACT_SUPPORTED_OPERATOR_IDS,
    ACT_UNSUPPORTED_OPERATOR_IDS,
    ACTFaultCampaignCellSpec,
    ACTFaultCampaignError,
    ACTFaultCellDisposition,
    ACTUnsupportedContractSpec,
)
from .io import file_sha256, write_json

if TYPE_CHECKING:
    from .generation import ACTFaultCampaignGenerationSpec


def write_clean_cell(
    root: Path,
    spec: ACTFaultCampaignGenerationSpec,
    manifest: ACTArtifactManifest,
    base: LiveUniVTACRunRequest,
    loaded: Any,
    ordinal: int,
) -> ACTFaultCampaignCellSpec:
    task, pair = loaded.trial.task, loaded.trial.pair_key
    request_rel = f"requests/{task}/{pair}/clean.json"
    artifact_rel = f"artifacts/{task}/{pair}/clean"
    request = _build_request(
        root=root,
        spec=spec,
        manifest=manifest,
        base=base,
        condition=Condition.CLEAN,
        artifact_rel=artifact_rel,
    )
    request_hash, reloaded = _write_portable_request(
        root, request_rel, request, artifact_rel
    )
    if reloaded.trial != loaded.trial:
        raise ACTFaultCampaignError("ACT artifact manifest differs from base request")
    return _cell(
        ordinal,
        reloaded.trial,
        ACTFaultCellDisposition.LIVE_REQUEST,
        None,
        None,
        None,
        request_rel,
        artifact_rel,
        request_hash,
        None,
        None,
        None,
        None,
    )


def write_fault_cell(
    root: Path,
    spec: ACTFaultCampaignGenerationSpec,
    manifest: ACTArtifactManifest,
    base: LiveUniVTACRunRequest,
    clean: Any,
    rest: Mapping[str, tuple[Path, dict[str, str]]],
    operator_id: str,
    severity: int,
    template_seed: int,
    ordinal: int,
) -> ACTFaultCampaignCellSpec:
    task, pair = clean.trial.task, clean.trial.pair_key
    rest_entry = rest[task]
    parameters: dict[str, object] = {}
    if operator_requires_rest_reference(operator_id):
        parameters["rest_reference_sha256"] = rest_entry[1]["rest_reference_sha256"]
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    fault = FaultManifest(
        operator_id=operator_id,
        severity_level=severity,
        operator_seed=template_seed,
        start_index=spec.fault_start_index,
        stop_index=spec.fault_stop_index,
        sensor_slots=spec.sensor_slots,
        observability=spec.observability,
        parameters=parameters,
        severity_registry=spec.severity_registry,
    )
    label = f"{operator_id}/s{severity}"
    fault_rel = f"fault_manifests/{task}/{pair}/{label}.json"
    write_json(root / fault_rel, fault.to_dict())
    fault_trial = replace(
        clean.trial,
        condition=Condition.FAULTED,
        fault_manifest_sha256=fault.sha256,
    )
    if operator_id in ACT_UNSUPPORTED_OPERATOR_IDS:
        return _write_unsupported(
            root,
            spec,
            fault_trial,
            fault,
            fault_rel,
            operator_id,
            severity,
            template_seed,
            ordinal,
        )
    if operator_id not in ACT_SUPPORTED_OPERATOR_IDS:
        raise ACTFaultCampaignError("operator applicability is not registered")
    request_rel = f"requests/{task}/{pair}/{label}.json"
    artifact_rel = f"artifacts/{task}/{pair}/{label}"
    request = _build_request(
        root=root,
        spec=spec,
        manifest=manifest,
        base=base,
        condition=Condition.FAULTED,
        artifact_rel=artifact_rel,
        fault_path=root / fault_rel,
        rest_path=(
            rest_entry[0] if operator_requires_rest_reference(operator_id) else None
        ),
    )
    request_hash, reloaded = _write_portable_request(
        root, request_rel, request, artifact_rel
    )
    if reloaded.trial != fault_trial:
        raise ACTFaultCampaignError("generated fault request changed the paired trial")
    return _cell(
        ordinal,
        fault_trial,
        ACTFaultCellDisposition.LIVE_REQUEST,
        operator_id,
        severity,
        template_seed,
        request_rel,
        artifact_rel,
        request_hash,
        fault.sha256,
        fault_rel,
        None,
        None,
    )


def _write_unsupported(
    root: Path,
    spec: ACTFaultCampaignGenerationSpec,
    trial: Any,
    fault: FaultManifest,
    fault_rel: str,
    operator_id: str,
    severity: int,
    template_seed: int,
    ordinal: int,
) -> ACTFaultCampaignCellSpec:
    receipt = ACTUnsupportedContractSpec(
        campaign_id=spec.campaign_id,
        task=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        operator_id=operator_id,
        severity_level=severity,
        operator_template_seed=template_seed,
        fault_manifest_sha256=fault.sha256,
        trial_manifest_sha256=trial.sha256,
    )
    relative = (
        f"unsupported_contracts/{trial.task}/{trial.pair_key}/"
        f"{operator_id}/s{severity}.json"
    )
    write_json(root / relative, receipt.to_dict())
    return _cell(
        ordinal,
        trial,
        ACTFaultCellDisposition.UNSUPPORTED_CONTRACT,
        operator_id,
        severity,
        template_seed,
        None,
        None,
        None,
        fault.sha256,
        fault_rel,
        relative,
        file_sha256(root / relative),
    )


def _build_request(
    *,
    root: Path,
    spec: ACTFaultCampaignGenerationSpec,
    manifest: ACTArtifactManifest,
    base: LiveUniVTACRunRequest,
    condition: Condition,
    artifact_rel: str,
    fault_path: Path | None = None,
    rest_path: Path | None = None,
) -> LiveUniVTACRunRequest:
    request = build_official_act_request(
        base_manifest=manifest,
        layout=spec.deployment_layout,
        condition=condition,
        dataset_sha256=base.dataset_sha256,
        initial_seed=base.initial_seed,
        exogenous_seed=base.exogenous_seed,
        max_control_cycles=base.max_control_cycles,
        max_observation_steps=base.max_observation_steps,
        wall_timeout_s=base.wall_timeout_s,
        act_device_name=str(base.act_device_name),
        simulator_device=str(base.simulator_device),
        live_output_dir=root / artifact_rel,
        fault_manifest_path=fault_path,
        rest_references_path=rest_path,
        success_profile_id=base.success_profile_id,
    )
    return replace(
        request,
        # ``runtime_dir`` is part of the simulator physics contract: UniVTAC
        # forwards it to the UIPC workspace.  Every condition in one paired
        # session must therefore retain the exact base Clean workspace.
        runtime_dir=base.runtime_dir,
        initial_state_policy=base.initial_state_policy,
        wall_timeout_role=base.wall_timeout_role,
    )


def _cell(
    ordinal: int,
    trial: Any,
    disposition: ACTFaultCellDisposition,
    operator_id: str | None,
    severity: int | None,
    template_seed: int | None,
    request_rel: str | None,
    artifact_rel: str | None,
    request_sha: str | None,
    fault_sha: str | None,
    fault_rel: str | None,
    unsupported_rel: str | None,
    unsupported_sha: str | None,
) -> ACTFaultCampaignCellSpec:
    return ACTFaultCampaignCellSpec(
        ordinal=ordinal,
        task=trial.task,
        initial_seed=trial.initial_seed,
        exogenous_seed=trial.exogenous_seed,
        pair_key=trial.pair_key,
        condition=trial.condition,
        disposition=disposition,
        operator_id=operator_id,
        severity_level=severity,
        operator_template_seed=template_seed,
        request_relpath=request_rel,
        artifact_relpath=artifact_rel,
        request_file_sha256=request_sha,
        trial_manifest_sha256=trial.sha256,
        fault_manifest_sha256=fault_sha,
        fault_manifest_relpath=fault_rel,
        unsupported_receipt_relpath=unsupported_rel,
        unsupported_receipt_sha256=unsupported_sha,
    )


def _write_portable_request(
    root: Path,
    request_rel: str,
    request: LiveUniVTACRunRequest,
    artifact_rel: str,
) -> tuple[str, Any]:
    target = root / request_rel
    document = live_univtac_request_to_dict(request)
    document["output_dir"] = Path(
        os.path.relpath(root / artifact_rel, target.parent)
    ).as_posix()
    for field in ("fault_manifest_path", "rest_references_path"):
        value = getattr(request, field)
        document[field] = (
            None
            if value is None
            else Path(os.path.relpath(value, target.parent)).as_posix()
        )
    write_json(target, document)
    loaded_request = load_live_univtac_request(target)
    return file_sha256(target), load_live_univtac_run(loaded_request)


__all__ = ["write_clean_cell", "write_fault_cell"]
