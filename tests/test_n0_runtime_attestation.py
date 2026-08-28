from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import robotactile_benchmark.runtime_attestation as runtime_attestation
from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.clean_baseline.io import write_canonical_no_clobber
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationEvidenceBinding,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock
from robotactile_benchmark.runtime_attestation import (
    N0ServerRuntimeAttestation,
    load_n0_server_runtime_attestation,
    n0_server_attestation_sha256,
    verify_live_n0_server_attestation,
)
from robotactile_benchmark.runtime_source import (
    RuntimeSourceBinding,
    current_integrations_lock_sha256,
    current_n0_input_profile_sha256,
    current_source_manifest_sha256,
)


@pytest.mark.parametrize(
    "statement",
    (
        "from robotactile_benchmark.runtime_attestation import "
        "load_n0_server_runtime_attestation",
        "from robotactile_benchmark.execution.isaac_runtime_attestation "
        "import IsaacRuntimeAttestation",
        "from robotactile_benchmark.clean_baseline import "
        "verify_all_task_qualification",
    ),
)
def test_runtime_attestation_imports_in_a_fresh_server_process(
    statement: str,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            statement,
        ),
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _source() -> RuntimeSourceBinding:
    lock = load_integration_lock()
    return RuntimeSourceBinding(
        robotactile_source_manifest_sha256=current_source_manifest_sha256(),
        robotactile_wheel_sha256="1" * 64,
        integrations_lock_sha256=current_integrations_lock_sha256(),
        univtac_source_commit=lock.by_id("univtac").commit_sha,
        n0_source_commit=lock.by_id("n0_twam").commit_sha,
        checkpoint_sha256="2" * 64,
        config_sha256="3" * 64,
        normalizer_sha256="4" * 64,
        serve_bundle_sha256="5" * 64,
        prompt_manifest_sha256="6" * 64,
        input_profile_sha256=current_n0_input_profile_sha256(),
        action_execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        native_step_contract=FIXED_NATIVE_STEP_CONTRACT,
    )


def _attestation() -> N0ServerRuntimeAttestation:
    source = _source()
    receipts = tuple(
        ObservationEvidenceBinding(
            kind=kind,
            relpath=f"artifacts/deployment/{kind}.json",
            sha256=f"{index + 7:x}" * 64,
            content_sha256=None,
        )
        for index, kind in enumerate(
            (
                "isaac_install_receipt",
                "n0_client_install_receipt",
                "n0_runtime_receipt",
                "tacex_install_receipt",
            )
        )
    )
    return N0ServerRuntimeAttestation.build(
        task_id="lift_bottle",
        session_id="session-v1",
        server_pid=1234,
        server_process_group_id=1200,
        server_posix_session_id=1200,
        rank=0,
        world_size=2,
        debug_offload=False,
        paper_gate_passed=True,
        n0_source_relpath="sources/N0-TWAM",
        n0_source_commit=source.n0_source_commit,
        n0_source_clean=True,
        robotactile_source_manifest_sha256=(source.robotactile_source_manifest_sha256),
        integrations_lock_sha256=source.integrations_lock_sha256,
        qualification_relpath="artifacts/deployment/qualification-v3.json",
        qualification_sha256="b" * 64,
        qualification_semantic_version="3.0",
        task_source_binding=source,
        integration_config_relpath=(
            "artifacts/models/n0_twam/configs/lift_bottle/integration_config.json"
        ),
        integration_config_sha256="c" * 64,
        artifact_manifest_relpath=(
            "artifacts/models/n0_twam/configs/lift_bottle/artifact_manifest.json"
        ),
        artifact_manifest_sha256="d" * 64,
        n0_artifact_hashes={
            "checkpoint_sha256": source.checkpoint_sha256,
            "config_sha256": source.config_sha256,
            "normalizer_sha256": source.normalizer_sha256,
            "prompt_manifest_sha256": source.prompt_manifest_sha256,
            "serve_bundle_sha256": source.serve_bundle_sha256,
        },
        installation_receipts=receipts,
        recorded_at_utc="2026-08-25T00:00:00+00:00",
        evidence_level="n0_rank0_runtime_attestation_v1",
        semantic_version="1.0",
    )


def test_attestation_is_canonical_no_clobber_and_hash_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    path = root / "outputs/n0-twam/lift_bottle/attestation.json"
    attestation = _attestation()
    write_canonical_no_clobber(path, attestation.to_dict())

    digest = n0_server_attestation_sha256(path)
    assert (
        load_n0_server_runtime_attestation(
            root,
            path,
            expected_sha256=digest,
            verify_members=False,
        )
        == attestation
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_n0_server_runtime_attestation(
            root,
            path,
            expected_sha256="0" * 64,
            verify_members=False,
        )


def test_attestation_rejects_debug_offload_and_content_tampering() -> None:
    document = _attestation().to_dict()
    document["debug_offload"] = True
    with pytest.raises(ValueError, match="paper gate"):
        N0ServerRuntimeAttestation.from_dict(document)

    document = _attestation().to_dict()
    document["task_id"] = "lift_can"
    with pytest.raises(ValueError, match="content hash"):
        N0ServerRuntimeAttestation.from_dict(document)


def test_historical_stock_binding_parses_but_cannot_attest_current_runtime() -> None:
    document = _source().to_dict()
    document["action_execution_contract"] = N0_STOCK_EE_ACTION_EXECUTION_CONTRACT
    document["native_step_contract"] = N0_STOCK_EE_NATIVE_STEP_CONTRACT

    historical = RuntimeSourceBinding.from_dict(document)

    assert historical.action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT
    with pytest.raises(ValueError, match="execution contract mismatch"):
        historical.verify_against_current_runtime()


def test_live_attestation_requires_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = _attestation()
    monkeypatch.setattr("os.getpgid", lambda pid: 1200)
    monkeypatch.setattr("os.getsid", lambda pid: 1200)
    verify_live_n0_server_attestation(
        attestation,
        expected_task_id="lift_bottle",
        expected_session_id="session-v1",
        expected_process_group_id=1200,
    )
    with pytest.raises(ValueError, match="process/session identity"):
        verify_live_n0_server_attestation(
            attestation,
            expected_task_id="lift_can",
            expected_session_id="session-v1",
            expected_process_group_id=1200,
        )


def test_git_identity_includes_untracked_source_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    commit = _source().n0_source_commit

    def run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        del kwargs
        calls.append(command)
        output = commit if command[-2:] == ("rev-parse", "HEAD") else ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(runtime_attestation.subprocess, "run", run)
    assert runtime_attestation._git_identity(tmp_path) == (commit, True)
    assert any(
        command[-3:]
        == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        for command in calls
    )


def test_member_verification_rehashes_n0_artifact_contents() -> None:
    source = Path(runtime_attestation.__file__).read_text(encoding="utf-8")
    assert "validate_n0_twam_artifact(manifest)" in source
