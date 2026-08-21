"""Deterministic CPU policy protocol qualification orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.qualification.act_protocol import run_act_protocol
from robotactile_benchmark.qualification.n0_protocol import run_n0_protocol
from robotactile_benchmark.qualification.receipt import (
    ACT_CHECKS,
    ACT_EVIDENCE_TYPE,
    N0_CHECKS,
    N0_EVIDENCE_TYPE,
    PolicyQualificationError,
    PolicyQualificationReceipt,
)
from robotactile_benchmark.qualification.sources import (
    TASK6A_SOURCE_HASHES,
    TASK6B_SOURCE_HASHES,
    SourceIdentityError,
    collect_act_runtime_source,
    collect_reviewed_sources,
    source_manifest_sha256,
)


def _identity_document(identity: PolicyIdentity) -> Mapping[str, object]:
    return {
        "system_id": identity.system_id,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "config_sha256": identity.config_sha256,
        "action_spec": identity.action_spec,
        "consumes_tactile": identity.consumes_tactile,
        "supports_structural_absence": identity.supports_structural_absence,
    }


def run_cpu_policy_qualification(
    policy_kind: str,
    *,
    identity: PolicyIdentity,
    workspace_root: Path,
    task_id: str,
) -> PolicyQualificationReceipt:
    """Run one model-free ACT or N0 protocol and bind reviewed sources."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if policy_kind not in {"act", "n0"}:
        raise PolicyQualificationError("policy_kind must be act or n0")
    checks: tuple[str, ...]
    task6b: Mapping[str, str]
    runtime_path: Optional[str]
    runtime_sha: Optional[str]
    absence: Mapping[str, Any]
    try:
        task6a = collect_reviewed_sources(workspace_root, TASK6A_SOURCE_HASHES)
        if policy_kind == "act":
            task6b = {}
            runtime_path, runtime_sha = collect_act_runtime_source(workspace_root)
            result, absence, no_touch = run_act_protocol(identity, task_id)
            evidence_type = ACT_EVIDENCE_TYPE
            checks = ACT_CHECKS
        else:
            if TASK6B_SOURCE_HASHES is None:
                raise PolicyQualificationError(
                    "Task 6B source hashes are not review-locked"
                )
            task6b = collect_reviewed_sources(workspace_root, TASK6B_SOURCE_HASHES)
            runtime_path = None
            runtime_sha = None
            result, absence, no_touch = run_n0_protocol(
                identity, workspace_root, task_id
            )
            evidence_type = N0_EVIDENCE_TYPE
            checks = N0_CHECKS
    except SourceIdentityError as error:
        raise PolicyQualificationError(str(error)) from error
    identity_document = _identity_document(identity)
    return PolicyQualificationReceipt(
        evidence_type=evidence_type,
        policy_kind=policy_kind,
        task_id=task_id,
        policy_identity=identity_document,
        policy_identity_sha256=canonical_hash(identity_document),
        task6a_source_hashes=task6a,
        task6a_source_manifest_sha256=source_manifest_sha256(task6a),
        task6b_source_hashes=task6b,
        task6b_source_manifest_sha256=(
            source_manifest_sha256(task6b) if task6b else None
        ),
        act_runtime_source_path=runtime_path,
        act_runtime_source_sha256=runtime_sha,
        artifact_status="artifact_not_loaded",
        no_touch_status=no_touch,
        structural_absence_results=absence,
        structural_absence_sha256=canonical_hash(absence),
        protocol_result=result,
        protocol_result_sha256=canonical_hash(result),
        checks=checks,
        passed=True,
        failure_codes=(),
        live_model_executed=False,
        twam_server_executed=False,
        isaac_sim_executed=False,
        task_success_measured=False,
        tls_authenticated=False,
    )
