from __future__ import annotations

from pathlib import Path

import pytest
from test_n0_observation_parity_artifact import source_binding

from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacAttestationRequest,
    IsaacRuntimeAttestation,
)
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleIdentity


def _attestation() -> IsaacRuntimeAttestation:
    source = source_binding()
    return IsaacRuntimeAttestation.build(
        campaign_id="campaign-v3",
        lifecycle_identity=LifecycleIdentity(
            campaign_manifest_sha256="1" * 64,
            request_file_sha256="2" * 64,
            trial_manifest_sha256="3" * 64,
            task_id="lift_bottle",
            ordinal=0,
            attempt_id="attempt-v3",
        ),
        qualification_relpath="artifacts/deployment/qualification-v3.json",
        qualification_sha256="4" * 64,
        runtime_source_binding=source,
        n0_server_attestation_relpath="outputs/n0/server-v3.json",
        n0_server_attestation_sha256="5" * 64,
        n0_server_content_sha256="6" * 64,
        univtac_source_relpath="sources/UniVTAC",
        univtac_source_commit=source.univtac_source_commit,
        n0_source_relpath="sources/N0-TWAM",
        n0_source_commit=source.n0_source_commit,
        process_id=101,
        process_group_id=102,
        posix_session_id=102,
        recorded_at_utc="2026-08-25T00:00:00+00:00",
        evidence_level="isaac_child_runtime_attestation_v1",
        semantic_version="1.0",
    )


def test_isaac_attestation_roundtrip_and_content_tamper() -> None:
    attestation = _attestation()
    assert IsaacRuntimeAttestation.from_dict(attestation.to_dict()) == attestation

    changed = attestation.to_dict()
    changed["campaign_id"] = "another-campaign"
    with pytest.raises(ValueError, match="content hash mismatch"):
        IsaacRuntimeAttestation.from_dict(changed)


def test_isaac_attestation_request_carries_explicit_deployment_root(
    tmp_path: Path,
) -> None:
    request = IsaacAttestationRequest(
        campaign_id="campaign-v3",
        deployment_root=tmp_path / "deployment",
        output_path=tmp_path / "deployment/outputs/isaac.json",
        qualification_path=tmp_path / "deployment/artifacts/qualification.json",
        n0_server_attestation_path=tmp_path / "deployment/outputs/server.json",
        n0_server_attestation_sha256="a" * 64,
    )

    assert request.deployment_root == (tmp_path / "deployment").absolute()
