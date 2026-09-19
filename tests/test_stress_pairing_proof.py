"""Persisted pairing proof must bind results and exact reset witnesses."""

import copy
import json
from types import SimpleNamespace

import pytest

from robotactile_benchmark.backends.univtac_pairing import (
    UniVTACPairedResetReceipt,
    UniVTACResetWitness,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.paired_live_univtac import (
    PAIRED_EXECUTION_EVIDENCE_LEVEL,
    PAIRED_EXECUTION_SEMANTIC_VERSION,
)
from robotactile_benchmark.n0_fault_campaign.stress_group import (
    _verify_stress_pairing_proof,
)
from robotactile_benchmark.trials import Condition


def proof_fixture():
    witness = UniVTACResetWitness(
        0, "canonical_reset", 1, "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, True
    )
    reset = UniVTACPairedResetReceipt(
        "session", "lift_bottle", 100, 100, "f" * 64, (witness,), True
    )
    request = SimpleNamespace(
        task_id="lift_bottle",
        initial_seed=100,
        exogenous_seed=100,
        condition=Condition.CLEAN,
    )
    result = SimpleNamespace(
        sha256="1" * 64,
        terminal_status=SimpleNamespace(value="success"),
        validation_passed=True,
        initial_state_sha256="a" * 64,
    )
    artifact = SimpleNamespace(
        evidence=SimpleNamespace(result=result), run_content_sha256="2" * 64
    )
    receipt = {
        "evidence_level": PAIRED_EXECUTION_EVIDENCE_LEVEL,
        "semantic_version": PAIRED_EXECUTION_SEMANTIC_VERSION,
        "simulator_qualification_claimed": False,
        "reset_receipt": reset.to_dict(),
        "reset_receipt_sha256": reset.sha256,
        "executions": [
            {
                "condition": "clean",
                "run_content_sha256": "2" * 64,
                "result_sha256": "1" * 64,
                "terminal_status": "success",
                "validation_passed": True,
                "witness_index": 0,
            }
        ],
        "group_content_sha256": canonical_hash(
            {
                "namespace": PAIRED_EXECUTION_EVIDENCE_LEVEL,
                "run_content_sha256": ("2" * 64,),
                "result_sha256": ("1" * 64,),
                "reset_receipt_sha256": reset.sha256,
                "witness_indices": (0,),
            }
        ),
    }
    return receipt, [request], [artifact]


def test_proof_reconstructs_actual_reset_and_execution():
    receipt, requests, artifacts = proof_fixture()
    assert _verify_stress_pairing_proof(receipt, requests, artifacts, "f" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("result_sha256", "3" * 64),
        ("run_content_sha256", "3" * 64),
        ("witness_index", True),
        ("witness_index", -1),
        ("terminal_status", "timeout"),
        ("condition", "faulted"),
    ],
)
def test_rejects_mismatched_execution_links(field, value):
    receipt, requests, artifacts = proof_fixture()
    receipt["executions"][0][field] = value
    with pytest.raises(ValueError):
        _verify_stress_pairing_proof(receipt, requests, artifacts, "f" * 64)


def test_hash_recomputation_does_not_hide_non_equivalent_witness():
    receipt, requests, artifacts = proof_fixture()
    other = copy.deepcopy(receipt["reset_receipt"]["witnesses"][0])
    other.update(
        ordinal=1, reset_mode="snapshot_restore", snapshot_state_sha256="9" * 64
    )
    receipt["reset_receipt"]["witnesses"].append(other)
    receipt["reset_receipt_sha256"] = canonical_hash(receipt["reset_receipt"])
    with pytest.raises(ValueError, match="not exactly"):
        _verify_stress_pairing_proof(receipt, requests, artifacts, "f" * 64)


def test_proof_rejects_seed_and_group_hash_drift():
    receipt, requests, artifacts = proof_fixture()
    requests[0].initial_seed += 1
    with pytest.raises(ValueError, match="requests"):
        _verify_stress_pairing_proof(receipt, requests, artifacts, "f" * 64)
    receipt, requests, artifacts = proof_fixture()
    receipt["group_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="group content"):
        _verify_stress_pairing_proof(receipt, requests, artifacts, "f" * 64)


def test_read_rows_reconstructs_flags_and_rejects_saved_boolean_tamper(
    tmp_path, monkeypatch
):
    from robotactile_benchmark.execution import live_artifacts_values, loading
    from robotactile_benchmark.n0_fault_campaign import stress_group as module

    receipt, requests, artifacts = proof_fixture()
    request, artifact = requests[0], artifacts[0]
    request.output_dir = tmp_path / "artifact"
    request.output_dir.mkdir()
    (request.output_dir / "root_receipt.json").write_text("{}")
    artifact.request_identity = {"task_id": "lift_bottle"}
    artifact.trial = "trial"
    artifact.external_root_sha256 = "3" * 64
    loaded = SimpleNamespace(
        content_sha256="2" * 64,
        trial="trial",
        backend_config=SimpleNamespace(sha256="f" * 64),
    )
    plan = {"protocol_sha256": "4" * 64}
    group = {
        "group_sha256": "5" * 64,
        "seed": 100,
        "selected": [{"condition": "clean"}],
    }
    monkeypatch.setattr(
        module, "load_stress_group", lambda root: (plan, group, requests)
    )
    monkeypatch.setattr(module, "load_live_univtac_artifact", lambda root: artifact)
    monkeypatch.setattr(loading, "load_live_univtac_run", lambda req: loaded)
    monkeypatch.setattr(
        live_artifacts_values,
        "live_request_identity",
        lambda obj: {"task_id": "lift_bottle"},
    )
    terminal = {
        "score_eligible": True,
        "validation_passed": True,
        "terminal_status": "success",
        "observation_count": 1,
        "score_success": True,
    }
    monkeypatch.setattr(module, "result_to_dict", lambda result: terminal)
    module.write_once(tmp_path / "paired_execution_receipt.json", receipt)
    row = {
        **terminal,
        "seed": 100,
        "condition": "clean",
        "protocol_sha256": "4" * 64,
        "attempt_id": "5" * 64,
        "artifact_root_sha256": "3" * 64,
        "group_accepted": True,
        "exact_snapshot_reset_verified": True,
    }
    result = {
        "group_sha256": "5" * 64,
        "protocol_sha256": "4" * 64,
        "group_accepted": True,
        "paired_receipt_file_sha256": module.file_sha256(
            tmp_path / "paired_execution_receipt.json"
        ),
        "rows": [row],
    }
    result_path = tmp_path / "group_result.json"
    result_path.write_text(json.dumps(result))
    assert module.read_stress_rows(tmp_path)[0] == {
        **row,
        "group_root": str(tmp_path.resolve()),
    }
    for field in ("group_accepted", "exact_snapshot_reset_verified"):
        changed = copy.deepcopy(result)
        changed["rows"][0][field] = False
        result_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="row flags"):
            module.read_stress_rows(tmp_path)
    result_path.write_text(json.dumps(result))
    artifact.request_identity = {"task_id": "wrong_task"}
    with pytest.raises(ValueError, match="request identity"):
        module.read_stress_rows(tmp_path)
