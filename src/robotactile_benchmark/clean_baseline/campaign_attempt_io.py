"""Strict raw attempt-receipt loading for resumable campaign execution."""

from __future__ import annotations

from collections.abc import Mapping

from robotactile_benchmark.clean_baseline.contracts import CleanCampaignTrialSpec
from robotactile_benchmark.closed_loop.artifact_io import strict_json_bytes
from robotactile_benchmark.deployment.layout import DeploymentLayout

ATTEMPT_FIELDS_V1 = frozenset(
    {
        "artifact",
        "artifact_validation_error_type",
        "campaign_id",
        "campaign_manifest_sha256",
        "command_kind",
        "duration_s",
        "finished_at_utc",
        "log_relpath",
        "log_sha256",
        "ordinal",
        "request_file_sha256",
        "return_code",
        "semantic_version",
        "started_at_utc",
        "task_id",
        "trial_manifest_sha256",
    }
)
ATTEMPT_FIELDS_V2 = ATTEMPT_FIELDS_V1 | frozenset(
    {"candidate_disposition", "exception_code"}
)
ATTEMPT_FIELDS_V3 = ATTEMPT_FIELDS_V2 | frozenset(
    {
        "campaign_semantic_version",
        "isaac_attestation_relpath",
        "isaac_attestation_sha256",
        "n0_server_attestation_relpath",
        "n0_server_attestation_sha256",
        "qualification_relpath",
        "qualification_sha256",
        "runtime_source_binding",
    }
)
ATTEMPT_FIELDS_BY_VERSION = {
    "1.0": ATTEMPT_FIELDS_V1,
    "2.0": ATTEMPT_FIELDS_V2,
    "3.0": ATTEMPT_FIELDS_V3,
}


def load_prior_attempts(
    *,
    layout: DeploymentLayout,
    campaign_id: str,
    campaign_manifest_sha256: str,
    entry: CleanCampaignTrialSpec,
    semantic_version: str = "1.0",
    required_semantic_version: str | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Load at most one exact-version receipt for campaign resume preflight."""

    selected_version = required_semantic_version or semantic_version
    try:
        expected_fields = ATTEMPT_FIELDS_BY_VERSION[selected_version]
    except KeyError as error:
        raise ValueError("unsupported campaign attempt semantic version") from error
    directory = (
        layout.outputs / "clean-campaigns" / campaign_id / "attempts" / entry.task
    )
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("campaign attempt directory must be a real directory")
    prefix = f"{entry.ordinal:04d}-"
    paths = tuple(sorted(directory.glob(f"{prefix}*.json")))
    attempts: list[Mapping[str, object]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("campaign attempt receipt must be a regular file")
        document = strict_json_bytes(path.read_bytes(), "campaign attempt receipt")
        if not isinstance(document, Mapping) or set(document) != expected_fields:
            raise ValueError("campaign attempt receipt fields mismatch")
        if (
            document["campaign_id"] != campaign_id
            or document["campaign_manifest_sha256"] != campaign_manifest_sha256
            or document["task_id"] != entry.task
            or document["ordinal"] != entry.ordinal
            or document["request_file_sha256"] != entry.request_file_sha256
            or document["trial_manifest_sha256"] != entry.trial_manifest_sha256
            or document["semantic_version"] != selected_version
        ):
            raise ValueError("campaign attempt receipt identity mismatch")
        if selected_version == "3.0" and document["campaign_semantic_version"] != "2.0":
            raise ValueError("source-bound attempt campaign version mismatch")
        attempts.append(document)
    return tuple(attempts)


__all__ = [
    "ATTEMPT_FIELDS_BY_VERSION",
    "ATTEMPT_FIELDS_V1",
    "ATTEMPT_FIELDS_V2",
    "ATTEMPT_FIELDS_V3",
    "load_prior_attempts",
]
