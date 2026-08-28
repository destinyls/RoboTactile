"""Upgrade a legacy qualification into task-source-bound v3 evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignManifest,
)
from robotactile_benchmark.clean_baseline.io import (
    read_canonical_json_file,
    write_canonical_no_clobber,
)
from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_V1_SEMANTIC_VERSION,
    QUALIFICATION_V3_SEMANTIC_VERSION,
    TASK_SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL,
    VerifiedAllTaskQualification,
    verify_all_task_qualification,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationEvidenceBinding,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_io import (
    load_observation_parity_artifact,
    resolve_evidence_member,
    sha256_file,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

_INSTALLATION_KINDS = (
    "isaac_install_receipt",
    "n0_client_install_receipt",
    "n0_runtime_receipt",
    "tacex_install_receipt",
)
_SHARED_RUNTIME_SOURCE_FIELDS = (
    "robotactile_source_manifest_sha256",
    "robotactile_wheel_sha256",
    "integrations_lock_sha256",
    "univtac_source_commit",
    "n0_source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "prompt_manifest_sha256",
    "input_profile_sha256",
    "action_execution_contract",
    "native_step_contract",
)


def build_source_bound_qualification(
    *,
    deployment_root: Path,
    legacy_qualification_relpath: str,
    campaign_manifest_relpath: str,
    installation_receipt_relpaths: Mapping[str, str],
    observation_parity_relpaths: Mapping[str, str],
    output_path: Path,
) -> VerifiedAllTaskQualification:
    """Publish v3 with shared runtime identity and task-local model bundles."""

    root = Path(deployment_root).resolve(strict=True)
    target = Path(output_path).absolute()
    try:
        output_relative = target.relative_to(root)
    except ValueError as error:
        raise CleanCampaignError(
            "qualification output must remain below deployment root"
        ) from error
    if output_relative.parts[:2] != ("artifacts", "deployment"):
        raise CleanCampaignError("qualification output must be in artifacts/deployment")
    legacy_path = resolve_evidence_member(
        root, legacy_qualification_relpath, "legacy qualification path"
    )
    legacy = verify_all_task_qualification(root, legacy_path)
    if legacy.semantic_version != QUALIFICATION_V1_SEMANTIC_VERSION:
        raise CleanCampaignError("qualification builder requires legacy v1 input")
    if legacy.action_spec != "ee8_absolute":
        raise CleanCampaignError("source-bound N0 qualification requires EE8")
    document, _ = read_canonical_json_file(legacy_path, "legacy qualification")
    tasks = document.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise CleanCampaignError("legacy qualification tasks must be a sequence")
    if set(observation_parity_relpaths) != set(legacy.tasks):
        raise CleanCampaignError("observation parity task inventory mismatch")
    if set(installation_receipt_relpaths) != set(_INSTALLATION_KINDS):
        raise CleanCampaignError("installation receipt inventory mismatch")

    parity_by_task: dict[str, tuple[str, str, RuntimeSourceBinding]] = {}
    shared_source_identity: tuple[str, ...] | None = None
    for task_id in legacy.tasks:
        relative = observation_parity_relpaths[task_id]
        path = resolve_evidence_member(root, relative, "observation parity path")
        artifact = load_observation_parity_artifact(root, path)
        if not artifact.passed or artifact.task_id != task_id:
            raise CleanCampaignError(f"observation parity failed for {task_id}")
        artifact.source_binding.verify_against_current_runtime()
        task_shared_identity = tuple(
            getattr(artifact.source_binding, field)
            for field in _SHARED_RUNTIME_SOURCE_FIELDS
        )
        if shared_source_identity is None:
            shared_source_identity = task_shared_identity
        elif task_shared_identity != shared_source_identity:
            raise CleanCampaignError(
                "observation parity source bindings disagree on shared runtime"
            )
        parity_by_task[task_id] = (
            relative,
            sha256_file(path),
            artifact.source_binding,
        )
    if shared_source_identity is None:
        raise CleanCampaignError("observation parity inventory is empty")

    campaign_path = resolve_evidence_member(
        root, campaign_manifest_relpath, "campaign manifest path"
    )
    campaign_document, _ = read_canonical_json_file(campaign_path, "campaign manifest")
    campaign_manifest = CleanCampaignManifest.from_dict(campaign_document)
    installation_receipts = []
    for kind in _INSTALLATION_KINDS:
        relative = installation_receipt_relpaths[kind]
        receipt_path = resolve_evidence_member(root, relative, f"{kind} path")
        installation_receipts.append(
            ObservationEvidenceBinding(
                kind=kind,
                relpath=relative,
                sha256=sha256_file(receipt_path),
                content_sha256=None,
            ).to_dict()
        )

    upgraded_tasks = []
    for raw_task in tasks:
        if not isinstance(raw_task, Mapping):
            raise CleanCampaignError("legacy qualification task must be an object")
        task = dict(cast(Mapping[str, object], raw_task))
        raw_task_id = task.get("task_id")
        if not isinstance(raw_task_id, str):
            raise CleanCampaignError("legacy qualification task ID is invalid")
        parity_relative, parity_sha256, source_binding = parity_by_task[raw_task_id]
        task.update(
            {
                "observation_parity_relpath": parity_relative,
                "observation_parity_sha256": parity_sha256,
                "source_binding": source_binding.to_dict(),
            }
        )
        upgraded_tasks.append(task)
    upgraded = dict(document)
    upgraded.update(
        {
            "campaign_manifest_relpath": campaign_manifest_relpath,
            "campaign_manifest_sha256": campaign_manifest.sha256,
            "evidence_level": TASK_SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL,
            "installation_receipts": installation_receipts,
            "semantic_version": QUALIFICATION_V3_SEMANTIC_VERSION,
            "tasks": upgraded_tasks,
        }
    )
    write_canonical_no_clobber(target, upgraded)
    return verify_all_task_qualification(root, target)


__all__ = ["build_source_bound_qualification"]
