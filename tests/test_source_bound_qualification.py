from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_n0_observation_parity_artifact import (
    make_parity_artifact,
    source_binding,
)

from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_POLICY_SEED_ROLE,
    CLEAN_SEED_DERIVATION,
    CLEAN_SIMULATOR_SEED_ROLE,
    CleanCampaignManifest,
    CleanCampaignProtocol,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.io import write_canonical_no_clobber
from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_EVIDENCE_LEVEL,
    SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL,
    verify_all_task_qualification,
)
from robotactile_benchmark.clean_baseline.qualification_builder import (
    build_source_bound_qualification,
)
from robotactile_benchmark.clean_baseline.seeds import derive_clean_campaign_seed
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationEvidenceBinding,
    ObservationParityArtifact,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_io import (
    load_observation_parity_artifact,
    sha256_file,
    write_observation_parity_artifact,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding


def _write(path: Path, value: object) -> None:
    write_canonical_no_clobber(path, value)


def _campaign_manifest() -> CleanCampaignManifest:
    task = load_registry().tasks[0]
    master_seed = 20260825
    initial_seed = derive_clean_campaign_seed(
        protocol_id="diagnostic_v1",
        master_seed=master_seed,
        task_id=task.task_id,
        task_ordinal=0,
        role="initial",
    )
    exogenous_seed = derive_clean_campaign_seed(
        protocol_id="diagnostic_v1",
        master_seed=master_seed,
        task_id=task.task_id,
        task_ordinal=0,
        role="exogenous",
    )
    trial = CleanCampaignTrialSpec(
        ordinal=0,
        task=task.task_id,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        base_system_id="n0-test",
        dataset_sha256="1" * 64,
        base_system_manifest_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        config_sha256="4" * 64,
        trial_manifest_sha256="5" * 64,
        pair_key="6" * 64,
        run_spec_sha256="7" * 64,
        run_content_sha256="8" * 64,
        request_file_sha256="9" * 64,
        request_relpath=f"requests/clean/{task.task_id}/request.json",
        artifact_relpath=f"artifacts/clean/{task.task_id}",
        max_control_cycles=task.action_horizon,
        max_observation_steps=task.action_horizon + 1,
        execute_action_steps=1,
    )
    return CleanCampaignManifest(
        campaign_id="campaign-v2",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        policy_kind=LivePolicyKind.N0,
        task_registry_sha256=load_registry().resource_sha256,
        master_seed=master_seed,
        seed_derivation=CLEAN_SEED_DERIVATION,
        simulator_seed_role=CLEAN_SIMULATOR_SEED_ROLE,
        policy_seed_role=CLEAN_POLICY_SEED_ROLE,
        planned_trial_count=1,
        confidence_level=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=29,
        trials=(trial,),
    )


def _qualification(root: Path) -> tuple[Path, dict[str, Path]]:
    source = source_binding()
    campaign_path = root / "requests/clean-campaigns/campaign-v2/manifest.json"
    _write(campaign_path, _campaign_manifest().to_dict())

    runtime_path = root / "artifacts/deployment/n0-runtime.json"
    _write(
        runtime_path,
        {
            "component": "n0_twam_runtime",
            "source_commit": source.n0_source_commit,
            "status": "installed",
        },
    )
    receipt_documents = {
        "isaac_install_receipt": {
            "component": "robotactile_isaac",
            "source_manifest_sha256": source.robotactile_source_manifest_sha256,
            "status": "installed",
            "wheel_sha256": source.robotactile_wheel_sha256,
        },
        "n0_client_install_receipt": {
            "component": "n0_twam_robotactile_client",
            "runtime_receipt_sha256": sha256_file(runtime_path),
            "source_manifest_sha256": source.robotactile_source_manifest_sha256,
            "status": "installed",
            "wheel_sha256": source.robotactile_wheel_sha256,
        },
        "tacex_install_receipt": {
            "component": "tacex",
            "status": "installed",
            "univtac_source_commit": source.univtac_source_commit,
        },
    }
    receipt_paths = {"n0_runtime_receipt": runtime_path}
    for kind, document in receipt_documents.items():
        path = root / f"artifacts/deployment/{kind}.json"
        _write(path, document)
        receipt_paths[kind] = path
    installation_receipts = [
        ObservationEvidenceBinding(
            kind=kind,
            relpath=receipt_paths[kind].relative_to(root).as_posix(),
            sha256=sha256_file(receipt_paths[kind]),
            content_sha256=None,
        ).to_dict()
        for kind in (
            "isaac_install_receipt",
            "n0_client_install_receipt",
            "n0_runtime_receipt",
            "tacex_install_receipt",
        )
    ]

    tasks = []
    for index, task in enumerate(load_registry().tasks):
        initial_seed = 100 + index
        exogenous_seed = 200 + index
        import_path = root / f"artifacts/deployment/import-{task.task_id}.json"
        _write(
            import_path,
            {
                "component": "univtac_task_import",
                "status": "qualified",
                "task_id": task.task_id,
                "task_source_sha256": task.task_source_sha256,
                "univtac_source_commit": source.univtac_source_commit,
            },
        )
        common = {
            "action_spec": "ee8_absolute",
            "exogenous_seed": str(exogenous_seed),
            "initial_seed": str(initial_seed),
            "runtime_source_manifest_sha256": (
                source.robotactile_source_manifest_sha256
            ),
            "status": "qualified",
            "task_id": task.task_id,
            "task_source_sha256": task.task_source_sha256,
            "univtac_source_commit": source.univtac_source_commit,
        }
        reset_path = root / f"artifacts/deployment/reset-{task.task_id}.json"
        _write(reset_path, {**common, "component": "univtac_task_reset"})
        pairing_path = root / f"artifacts/deployment/pairing-{task.task_id}.json"
        _write(
            pairing_path,
            {
                **common,
                "all_exact": "true",
                "canonical_state_sha256": "8" * 64,
                "component": "univtac_task_pairing",
                # The pairing smoke owns an independent in-process reset. Its
                # receipt hash is not the deployment reset qualification file.
                "paired_reset_receipt_sha256": "7" * 64,
                "replay_state_sha256": "8" * 64,
            },
        )
        _, parity_path = make_parity_artifact(root, task.task_id)
        tasks.append(
            {
                "exogenous_seed": exogenous_seed,
                "import_receipt_relpath": import_path.relative_to(root).as_posix(),
                "import_receipt_sha256": sha256_file(import_path),
                "initial_seed": initial_seed,
                "observation_parity_relpath": parity_path.relative_to(root).as_posix(),
                "observation_parity_sha256": sha256_file(parity_path),
                "pairing_receipt_relpaths": [pairing_path.relative_to(root).as_posix()],
                "pairing_receipt_sha256s": [sha256_file(pairing_path)],
                "reset_receipt_relpaths": [reset_path.relative_to(root).as_posix()],
                "reset_receipt_sha256s": [sha256_file(reset_path)],
                "task_id": task.task_id,
            }
        )
    qualification = root / "artifacts/deployment/qualification-v2.json"
    _write(
        qualification,
        {
            "action_spec": "ee8_absolute",
            "campaign_id": "qualification-v2",
            "campaign_manifest_relpath": campaign_path.relative_to(root).as_posix(),
            "campaign_manifest_sha256": sha256_file(campaign_path),
            "closed_loop_episode_executed": False,
            "evidence_level": SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL,
            "installation_receipts": installation_receipts,
            "master_seed": 20260825,
            "policy_loaded": False,
            "recorded_at_utc": "2026-08-25T00:00:00+00:00",
            "repetitions": 1,
            "seed_derivation": "sha256_signed31_v1",
            "semantic_version": "2.0",
            "source_binding": source.to_dict(),
            "success_rate_claimed": False,
            "tasks": tasks,
        },
    )
    return qualification, receipt_paths


def test_source_bound_v2_verifies_runtime_and_all_task_parity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    path, _ = _qualification(root)
    verified = verify_all_task_qualification(root, path)
    assert verified.source_bound is True
    assert verified.semantic_version == "2.0"
    assert verified.source_binding == source_binding()
    assert verified.task_source_bindings == (source_binding(),) * 8
    assert len(verified.observation_parity_sha256s) == 8
    campaign_path = root / "requests/clean-campaigns/campaign-v2/manifest.json"
    assert verified.campaign_manifest_sha256 == _campaign_manifest().sha256
    assert verified.campaign_manifest_sha256 != sha256_file(campaign_path)


def test_source_bound_v2_rejects_changed_installation_receipt(tmp_path: Path) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    path, receipts = _qualification(root)
    receipts["isaac_install_receipt"].write_text(
        json.dumps({"status": "changed"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_all_task_qualification(root, path)


def test_source_bound_v2_rejects_unrelated_campaign_manifest_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    path, _ = _qualification(root)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["campaign_manifest_sha256"] = "0" * 64
    rejected = root / "artifacts/deployment/qualification-v2-bad-hash.json"
    _write(rejected, document)

    with pytest.raises(ValueError, match="campaign manifest binding mismatch"):
        verify_all_task_qualification(root, rejected)


def _legacy_v1_and_task_parities(
    root: Path,
) -> tuple[str, dict[str, str], dict[str, str]]:
    v2_path, receipt_paths = _qualification(root)
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    legacy_tasks = []
    parity_paths: dict[str, str] = {}
    for index, raw_task in enumerate(v2["tasks"]):
        task = dict(raw_task)
        task_id = task["task_id"]
        parity_path = root / task.pop("observation_parity_relpath")
        task.pop("observation_parity_sha256")
        legacy_tasks.append(task)

        original = load_observation_parity_artifact(root, parity_path)
        task_source = replace(
            original.source_binding,
            normalizer_sha256=f"{index + 8:x}" * 64,
            serve_bundle_sha256=f"{15 - index:x}" * 64,
        )
        task_artifact = ObservationParityArtifact.build(
            task_id=task_id,
            source_binding=task_source,
            evidence=original.evidence,
            gates=original.gates,
            limitations=original.limitations,
        )
        task_parity_path = root / f"artifacts/deployment/parity-v3-{task_id}.json"
        write_observation_parity_artifact(task_parity_path, task_artifact)
        parity_paths[task_id] = task_parity_path.relative_to(root).as_posix()

    legacy = {
        key: value
        for key, value in v2.items()
        if key
        not in {
            "campaign_manifest_relpath",
            "campaign_manifest_sha256",
            "installation_receipts",
            "source_binding",
        }
    }
    legacy.update(
        {
            "evidence_level": QUALIFICATION_EVIDENCE_LEVEL,
            "semantic_version": "1.0",
            "tasks": legacy_tasks,
        }
    )
    legacy_path = root / "artifacts/deployment/qualification-v1.json"
    _write(legacy_path, legacy)
    receipt_relpaths = {
        kind: path.relative_to(root).as_posix() for kind, path in receipt_paths.items()
    }
    return (
        legacy_path.relative_to(root).as_posix(),
        receipt_relpaths,
        parity_paths,
    )


def test_v3_builder_allows_task_local_normalizer_and_serve_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    legacy, receipts, parities = _legacy_v1_and_task_parities(root)
    verified_bindings: list[str] = []

    def verify_current(source: RuntimeSourceBinding) -> None:
        verified_bindings.append(source.normalizer_sha256)

    monkeypatch.setattr(
        RuntimeSourceBinding,
        "verify_against_current_runtime",
        verify_current,
    )
    output = root / "artifacts/deployment/qualification-v3.json"
    verified = build_source_bound_qualification(
        deployment_root=root,
        legacy_qualification_relpath=legacy,
        campaign_manifest_relpath=(
            "requests/clean-campaigns/campaign-v2/manifest.json"
        ),
        installation_receipt_relpaths=receipts,
        observation_parity_relpaths=parities,
        output_path=output,
    )

    assert verified.semantic_version == "3.0"
    assert verified.source_bound is True
    assert verified.source_binding is None
    assert len(verified.task_source_bindings) == 8
    assert len(set(verified_bindings)) == 8
    assert len({item.normalizer_sha256 for item in verified.task_source_bindings}) == 8
    assert (
        len({item.serve_bundle_sha256 for item in verified.task_source_bindings}) == 8
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert "source_binding" not in document
    assert all("source_binding" in task for task in document["tasks"])
    campaign_path = root / "requests/clean-campaigns/campaign-v2/manifest.json"
    assert document["campaign_manifest_sha256"] == _campaign_manifest().sha256
    assert document["campaign_manifest_sha256"] != sha256_file(campaign_path)
    assert verified.campaign_manifest_sha256 == _campaign_manifest().sha256


def test_v3_builder_rejects_shared_checkpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    legacy, receipts, parities = _legacy_v1_and_task_parities(root)
    monkeypatch.setattr(
        RuntimeSourceBinding,
        "verify_against_current_runtime",
        lambda self: None,
    )
    task_id = load_registry().tasks[-1].task_id
    path = root / parities[task_id]
    original = load_observation_parity_artifact(root, path)
    divergent = ObservationParityArtifact.build(
        task_id=task_id,
        source_binding=replace(
            original.source_binding,
            checkpoint_sha256="0" * 64,
        ),
        evidence=original.evidence,
        gates=original.gates,
        limitations=original.limitations,
    )
    divergent_path = root / "artifacts/deployment/parity-v3-divergent.json"
    write_observation_parity_artifact(divergent_path, divergent)
    parities[task_id] = divergent_path.relative_to(root).as_posix()

    output = root / "artifacts/deployment/rejected-v3.json"
    with pytest.raises(ValueError, match="disagree on shared runtime"):
        build_source_bound_qualification(
            deployment_root=root,
            legacy_qualification_relpath=legacy,
            campaign_manifest_relpath=(
                "requests/clean-campaigns/campaign-v2/manifest.json"
            ),
            installation_receipt_relpaths=receipts,
            observation_parity_relpaths=parities,
            output_path=output,
        )
    assert not output.exists()


def test_v3_verifier_accepts_legacy_file_hash_and_normalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    legacy, receipts, parities = _legacy_v1_and_task_parities(root)
    monkeypatch.setattr(
        RuntimeSourceBinding,
        "verify_against_current_runtime",
        lambda self: None,
    )
    output = root / "artifacts/deployment/qualification-v3-content-hash.json"
    build_source_bound_qualification(
        deployment_root=root,
        legacy_qualification_relpath=legacy,
        campaign_manifest_relpath=(
            "requests/clean-campaigns/campaign-v2/manifest.json"
        ),
        installation_receipt_relpaths=receipts,
        observation_parity_relpaths=parities,
        output_path=output,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    campaign_path = root / "requests/clean-campaigns/campaign-v2/manifest.json"
    document["campaign_manifest_sha256"] = sha256_file(campaign_path)
    legacy_raw = root / "artifacts/deployment/qualification-v3-legacy-raw.json"
    _write(legacy_raw, document)

    verified = verify_all_task_qualification(root, legacy_raw)
    assert verified.campaign_manifest_sha256 == _campaign_manifest().sha256
    assert verified.campaign_manifest_sha256 != sha256_file(campaign_path)


def test_v3_verifier_rejects_campaign_manifest_content_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    legacy, receipts, parities = _legacy_v1_and_task_parities(root)
    monkeypatch.setattr(
        RuntimeSourceBinding,
        "verify_against_current_runtime",
        lambda self: None,
    )
    output = root / "artifacts/deployment/qualification-v3-before-tamper.json"
    build_source_bound_qualification(
        deployment_root=root,
        legacy_qualification_relpath=legacy,
        campaign_manifest_relpath=(
            "requests/clean-campaigns/campaign-v2/manifest.json"
        ),
        installation_receipt_relpaths=receipts,
        observation_parity_relpaths=parities,
        output_path=output,
    )
    campaign_path = root / "requests/clean-campaigns/campaign-v2/manifest.json"
    tampered = replace(_campaign_manifest(), campaign_id="campaign-v2-tampered")
    campaign_path.write_bytes(canonical_json_bytes(tampered.to_dict()))

    with pytest.raises(ValueError, match="campaign manifest binding mismatch"):
        verify_all_task_qualification(root, output)
