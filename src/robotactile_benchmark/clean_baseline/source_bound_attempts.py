"""Source-bound context and receipt fields for Clean attempt v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_V3_SEMANTIC_VERSION,
    verify_all_task_qualification,
)
from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacRuntimeAttestation,
    isaac_attestation_sha256,
    load_isaac_runtime_attestation,
)
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleIdentity
from robotactile_benchmark.runtime_attestation import (
    N0ServerRuntimeAttestation,
    load_n0_server_runtime_attestation,
    verify_live_n0_server_attestation,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding


def _relative(root: Path, path: Path, name: str) -> str:
    target = Path(path).absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain below deployment root") from error
    value = relative.as_posix()
    member = PurePosixPath(value)
    if not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"{name} path is unsafe")
    current = root
    for part in member.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    return value


@dataclass(frozen=True)
class SourceBoundCampaignContext:
    """Verified qualification and live N0 server for one task shard."""

    deployment_root: Path
    campaign_id: str
    campaign_manifest_sha256: str
    task_id: str
    qualification_path: Path
    qualification_relpath: str
    qualification_sha256: str
    runtime_source_binding: RuntimeSourceBinding
    n0_server_attestation_path: Path
    n0_server_attestation_relpath: str
    n0_server_attestation_sha256: str
    n0_server_attestation: N0ServerRuntimeAttestation


def source_bound_arguments_requested(
    qualification_path: Path | None,
    n0_server_attestation_path: Path | None,
    n0_server_attestation_sha256: str | None,
) -> bool:
    """Return all-or-none selection and reject partial source evidence flags."""

    values = (
        qualification_path,
        n0_server_attestation_path,
        n0_server_attestation_sha256,
    )
    if any(value is not None for value in values) and any(
        value is None for value in values
    ):
        raise ValueError("source-bound campaign arguments must be complete")
    return all(value is not None for value in values)


def load_source_bound_campaign_context(
    *,
    deployment_root: Path,
    campaign_id: str,
    campaign_manifest_sha256: str,
    task_id: str,
    qualification_path: Path,
    n0_server_attestation_path: Path,
    n0_server_attestation_sha256: str,
    require_live_server: bool = True,
) -> SourceBoundCampaignContext:
    """Fail closed unless qualification, task source, and server agree exactly."""

    root = Path(deployment_root).resolve(strict=True)
    qualification = verify_all_task_qualification(root, qualification_path)
    if qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION:
        raise ValueError("source-bound Clean campaign requires qualification v3")
    if qualification.campaign_manifest_sha256 != campaign_manifest_sha256:
        raise ValueError("qualification campaign manifest SHA256 mismatch")
    try:
        task_index = qualification.tasks.index(task_id)
    except ValueError as error:
        raise ValueError("qualification does not cover selected task") from error
    source = qualification.task_source_bindings[task_index]
    source.verify_against_current_runtime()
    server_path = Path(n0_server_attestation_path).absolute()
    server = load_n0_server_runtime_attestation(
        root,
        server_path,
        expected_sha256=n0_server_attestation_sha256,
        verify_members=True,
    )
    if (
        server.task_id != task_id
        or server.qualification_sha256 != qualification.sha256
        or server.task_source_binding != source
    ):
        raise ValueError("N0 server attestation source binding mismatch")
    if require_live_server:
        verify_live_n0_server_attestation(
            server,
            expected_task_id=task_id,
            expected_session_id=server.session_id,
            expected_process_group_id=server.server_process_group_id,
        )
    qualification_absolute = qualification.path.absolute()
    return SourceBoundCampaignContext(
        deployment_root=root,
        campaign_id=campaign_id,
        campaign_manifest_sha256=campaign_manifest_sha256,
        task_id=task_id,
        qualification_path=qualification_absolute,
        qualification_relpath=_relative(root, qualification_absolute, "qualification"),
        qualification_sha256=qualification.sha256,
        runtime_source_binding=source,
        n0_server_attestation_path=server_path,
        n0_server_attestation_relpath=_relative(
            root, server_path, "N0 server attestation"
        ),
        n0_server_attestation_sha256=n0_server_attestation_sha256,
        n0_server_attestation=server,
    )


def isaac_attestation_output_path(
    context: SourceBoundCampaignContext,
    *,
    ordinal: int,
    attempt_id: str,
) -> Path:
    """Derive the immutable child-owned attestation path for one attempt."""

    return (
        context.deployment_root
        / "outputs"
        / "clean-campaigns"
        / context.campaign_id
        / "lifecycle"
        / context.task_id
        / f"{ordinal:04d}-{attempt_id}.isaac-attestation.json"
    )


def source_bound_attempt_fields(
    context: SourceBoundCampaignContext,
    *,
    identity: LifecycleIdentity,
    isaac_attestation_path: Path,
) -> tuple[dict[str, object], IsaacRuntimeAttestation]:
    """Load child evidence and return exact additional attempt-v3 fields."""

    attestation_path = Path(isaac_attestation_path).absolute()
    digest = isaac_attestation_sha256(attestation_path)
    attestation = load_isaac_runtime_attestation(
        context.deployment_root,
        attestation_path,
        expected_sha256=digest,
        verify_members=True,
    )
    if (
        attestation.campaign_id != context.campaign_id
        or attestation.lifecycle_identity != identity
        or attestation.qualification_sha256 != context.qualification_sha256
        or attestation.runtime_source_binding != context.runtime_source_binding
        or attestation.n0_server_attestation_sha256
        != context.n0_server_attestation_sha256
        or attestation.n0_server_content_sha256
        != context.n0_server_attestation.content_sha256
    ):
        raise ValueError("Isaac child attestation identity mismatch")
    return (
        {
            "campaign_semantic_version": "2.0",
            "isaac_attestation_relpath": _relative(
                context.deployment_root,
                attestation_path,
                "Isaac attestation",
            ),
            "isaac_attestation_sha256": digest,
            "n0_server_attestation_relpath": (context.n0_server_attestation_relpath),
            "n0_server_attestation_sha256": (context.n0_server_attestation_sha256),
            "qualification_relpath": context.qualification_relpath,
            "qualification_sha256": context.qualification_sha256,
            "runtime_source_binding": context.runtime_source_binding.to_dict(),
        },
        attestation,
    )


__all__ = [
    "SourceBoundCampaignContext",
    "isaac_attestation_output_path",
    "load_source_bound_campaign_context",
    "source_bound_arguments_requested",
    "source_bound_attempt_fields",
]
