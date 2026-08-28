from __future__ import annotations

from pathlib import Path

import pytest

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.execution.persistent_worker_receipt import (
    EpisodeDispatch,
    PersistentWorkerSessionReceipt,
    load_persistent_worker_session_receipt,
    persistent_worker_receipt_sha256,
    write_persistent_worker_session_receipt,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock
from robotactile_benchmark.runtime_source import RuntimeSourceBinding


def _source() -> RuntimeSourceBinding:
    lock = load_integration_lock()
    return RuntimeSourceBinding(
        robotactile_source_manifest_sha256="1" * 64,
        robotactile_wheel_sha256="2" * 64,
        integrations_lock_sha256="3" * 64,
        univtac_source_commit=lock.by_id("univtac").commit_sha,
        n0_source_commit=lock.by_id("n0_twam").commit_sha,
        checkpoint_sha256="4" * 64,
        config_sha256="5" * 64,
        normalizer_sha256="6" * 64,
        serve_bundle_sha256="7" * 64,
        prompt_manifest_sha256="8" * 64,
        input_profile_sha256="9" * 64,
        action_execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        native_step_contract=FIXED_NATIVE_STEP_CONTRACT,
    )


def _dispatch(sequence: int) -> EpisodeDispatch:
    return EpisodeDispatch(
        task_id="lift_bottle",
        campaign_id="paper-v1",
        sequence=sequence,
        ordinal=sequence,
        attempt_id=f"attempt-{sequence}",
        initial_seed=1_000_000 + sequence,
        exogenous_seed=1_000_000 + sequence,
        request_file_sha256=f"{sequence + 10:x}" * 64,
        trial_manifest_sha256=f"{sequence + 12:x}" * 64,
        dispatched_at_utc=f"2026-08-27T00:00:0{sequence}+00:00",
        completed_at_utc=f"2026-08-27T00:00:1{sequence}+00:00",
    )


def _receipt() -> PersistentWorkerSessionReceipt:
    return PersistentWorkerSessionReceipt.build(
        task_id="lift_bottle",
        campaign_id="paper-v1",
        worker_session_id="worker-0",
        worker_pid=1234,
        worker_process_group_id=1200,
        worker_posix_session_id=1200,
        restart_generation=0,
        runtime_source_binding=_source(),
        qualification_relpath="artifacts/deployment/qualification-v3.json",
        qualification_sha256="a" * 64,
        n0_server_attestation_relpath="outputs/n0-twam/server.json",
        n0_server_attestation_sha256="b" * 64,
        reset_equivalence_receipt_relpath=(
            "artifacts/deployment/persistent-reset-equivalence.json"
        ),
        reset_equivalence_receipt_sha256="c" * 64,
        episode_dispatches=(_dispatch(0), _dispatch(1)),
        started_at_utc="2026-08-27T00:00:00+00:00",
        finished_at_utc="2026-08-27T00:01:00+00:00",
        shutdown_status="clean",
        evidence_level="persistent_worker_session_receipt_v1",
        semantic_version="1.0",
    )


def _values(
    receipt: PersistentWorkerSessionReceipt,
) -> dict[str, object]:
    return {
        name: getattr(receipt, name)
        for name in receipt.__dataclass_fields__
        if name != "content_sha256"
    }


def test_receipt_round_trip_is_hash_bound_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "outputs/persistent-workers/worker-0.json"
    receipt = _receipt()

    assert write_persistent_worker_session_receipt(path, receipt) is True
    assert write_persistent_worker_session_receipt(path, receipt) is False
    digest = persistent_worker_receipt_sha256(path)
    assert (
        load_persistent_worker_session_receipt(path, expected_sha256=digest) == receipt
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_persistent_worker_session_receipt(path, expected_sha256="0" * 64)

    different = _values(receipt)
    different["shutdown_status"] = "forced"
    different_receipt = PersistentWorkerSessionReceipt.build(**different)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_persistent_worker_session_receipt(path, different_receipt)


def test_receipt_rejects_cross_task_and_non_unique_sequences() -> None:
    values = _values(_receipt())
    cross_task = _dispatch(1).to_dict()
    cross_task["task_id"] = "lift_can"
    values["episode_dispatches"] = (
        _dispatch(0),
        EpisodeDispatch.from_dict(cross_task),
    )
    with pytest.raises(ValueError, match="share task and campaign"):
        PersistentWorkerSessionReceipt.build(**values)

    values["episode_dispatches"] = (_dispatch(0), _dispatch(0))
    with pytest.raises(ValueError, match="unique and contiguous"):
        PersistentWorkerSessionReceipt.build(**values)


def test_receipt_rejects_content_tampering_and_bad_lifetime() -> None:
    document = _receipt().to_dict()
    document["restart_generation"] = 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        PersistentWorkerSessionReceipt.from_dict(document)

    values = _values(_receipt())
    values["finished_at_utc"] = "2026-08-26T23:59:59+00:00"
    with pytest.raises(ValueError, match="finish precedes start"):
        PersistentWorkerSessionReceipt.build(**values)
