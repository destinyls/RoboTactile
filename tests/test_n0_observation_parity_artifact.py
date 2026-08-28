from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    REQUIRED_EVIDENCE_KINDS,
    REQUIRED_GATE_IDS,
    ObservationEvidenceBinding,
    ObservationParityArtifact,
    ObservationParityGate,
    ObservationParityStatus,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_io import (
    load_observation_parity_artifact,
    sha256_file,
    write_observation_parity_artifact,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock
from robotactile_benchmark.runtime_source import (
    RuntimeSourceBinding,
    current_integrations_lock_sha256,
    current_n0_input_profile_sha256,
    current_source_manifest_sha256,
)


def source_binding() -> RuntimeSourceBinding:
    lock = load_integration_lock()
    return RuntimeSourceBinding(
        robotactile_source_manifest_sha256="1" * 64,
        robotactile_wheel_sha256="2" * 64,
        integrations_lock_sha256=current_integrations_lock_sha256(),
        univtac_source_commit=lock.by_id("univtac").commit_sha,
        n0_source_commit=lock.by_id("n0_twam").commit_sha,
        checkpoint_sha256="3" * 64,
        config_sha256="4" * 64,
        normalizer_sha256="5" * 64,
        serve_bundle_sha256="6" * 64,
        prompt_manifest_sha256="7" * 64,
        input_profile_sha256=current_n0_input_profile_sha256(),
        action_execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        native_step_contract=FIXED_NATIVE_STEP_CONTRACT,
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def make_parity_artifact(
    root: Path, task_id: str = "grasp_classify"
) -> tuple[ObservationParityArtifact, Path]:
    evidence = []
    for index, kind in enumerate(REQUIRED_EVIDENCE_KINDS):
        path = root / f"artifacts/deployment/{kind}-{index}.json"
        document: dict[str, object] = {"kind": kind}
        content_sha256 = None
        if kind in {"experiment_lock", "teacher_forced_probe"}:
            content_sha256 = canonical_hash(document)
            document["content_sha256"] = content_sha256
        _write(path, document)
        evidence.append(
            ObservationEvidenceBinding(
                kind=kind,
                relpath=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                content_sha256=content_sha256,
            )
        )
    gates = tuple(
        ObservationParityGate(
            gate_id=gate_id,
            status=ObservationParityStatus.PASS,
            code="passed",
            detail=f"{gate_id} verified",
        )
        for gate_id in REQUIRED_GATE_IDS
    )
    artifact = ObservationParityArtifact.build(
        task_id=task_id,
        source_binding=source_binding(),
        evidence=evidence,
        gates=gates,
        limitations=("not pixel-registered ground truth",),
    )
    output = root / f"artifacts/deployment/parity-{task_id}.json"
    write_observation_parity_artifact(output, artifact)
    return artifact, output


def test_source_bound_parity_roundtrip_and_member_verification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    artifact, output = make_parity_artifact(root)
    loaded = load_observation_parity_artifact(root, output)
    assert loaded == artifact
    assert loaded.passed is True
    assert loaded.failure_codes == ()

    bound = root / artifact.evidence[-1].relpath
    bound.write_bytes(canonical_json_bytes({"kind": "changed"}))
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_observation_parity_artifact(root, output)


def test_parity_content_hash_and_gate_outcome_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    artifact, output = make_parity_artifact(root)
    document = artifact.to_dict()
    document["passed"] = False
    output.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError, match="pass flag mismatch"):
        load_observation_parity_artifact(root, output, verify_members=False)


def test_runtime_binding_parses_history_then_explicitly_checks_runtime() -> None:
    document = source_binding().to_dict()
    document["robotactile_source_manifest_sha256"] = current_source_manifest_sha256()
    current = RuntimeSourceBinding.from_dict(document)
    current.verify_against_current_runtime()

    document["n0_source_commit"] = "f" * 40
    historical = RuntimeSourceBinding.from_dict(document)
    assert historical.n0_source_commit == "f" * 40
    with pytest.raises(ValueError, match="N0-TWAM commit mismatch"):
        historical.verify_against_current_runtime()


def test_file_hash_is_sha256(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    assert sha256_file(path) == hashlib.sha256(b"value").hexdigest()
