"""Atomic, content-addressed artifacts for recorded N0 experiments."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union, cast

from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_MAX_JSON_BYTES,
    LiveArtifactValidationError,
)
from robotactile_benchmark.execution.live_artifacts_fs import (
    hash_live_member,
    read_live_member,
    scan_live_bundle,
)
from robotactile_benchmark.execution.live_artifacts_io import (
    LiveArrayWriter,
    canonical_live_json_bytes,
    live_sha256_bytes,
    load_live_array,
    strict_live_json_bytes,
)
from robotactile_benchmark.recorded.evaluator import (
    RecordedConditionResult,
    RecordedExperimentResult,
)
from robotactile_benchmark.recorded.metrics import ActionErrorMetrics

SCHEMA_VERSION = "robotactile-recorded-n0-experiment-v1"
ROOT_RECEIPT = "root_receipt.json"
EXPERIMENT_DOCUMENT = "experiment.json"


def _identity(result: RecordedExperimentResult) -> dict[str, object]:
    identity = result.policy_identity
    return {
        "action_spec": identity.action_spec,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "config_sha256": identity.config_sha256,
        "consumes_tactile": identity.consumes_tactile,
        "supports_structural_absence": identity.supports_structural_absence,
        "system_id": identity.system_id,
        "sha256": identity.sha256,
    }


def _metric(
    value: Optional[ActionErrorMetrics],
) -> Optional[dict[str, Union[float, int]]]:
    if value is None:
        return None
    return value.to_dict()


def _condition(
    item: RecordedConditionResult, arrays: LiveArrayWriter
) -> dict[str, object]:
    return {
        "anchor_index": item.anchor_index,
        "clean_drift_full": _metric(item.clean_drift_full),
        "condition_id": item.condition_id,
        "delivery_failure_codes": list(item.delivery_failure_codes),
        "delivery_trace_sha256": item.delivery_trace_sha256,
        "delivery_validation_passed": item.delivery_validation_passed,
        "expert_anchor_h0": _metric(item.expert_anchor),
        "expert_full_h0_h11": _metric(item.expert_full),
        "expert_future_h1_h11": _metric(item.expert_future_only),
        "fault_manifest": None if item.manifest is None else item.manifest.to_dict(),
        "operator_id": item.operator_id,
        "native_anchor_step": item.native_anchor_step,
        "prediction": None if item.prediction is None else arrays.add(item.prediction),
        "reason_code": item.reason_code,
        "severity_identifiable_at_anchor": item.severity_identifiable_at_anchor,
        "severity_level": item.severity_level,
        "status": item.status,
    }


def _document(
    result: RecordedExperimentResult, arrays: LiveArrayWriter
) -> dict[str, object]:
    episode = result.episode
    release_episode = result.release_episode
    return {
        "action_chunk_contract": {
            "cold_frame_0": "conditioning_frame_not_emitted",
            "emitted_frame": 1,
            "h0_semantics": "anchor_aligned_target",
            "h1_h11_semantics": "future_targets",
            "horizon": 12,
        },
        "anchor": {
            "benchmark_index": episode.anchor_index,
            "native_step": episode.native_steps[episode.anchor_index],
        },
        "conditions": [_condition(item, arrays) for item in result.results],
        "evidence_level": result.evidence_level,
        "exogenous_seed": result.exogenous_seed,
        "expert_actions": arrays.add(episode.expert_actions),
        "fault_start_index": result.fault_start_index,
        "initial_seed": episode.initial_seed,
        "policy_identity": _identity(result),
        "rest_index": episode.rest_index,
        "rest_native_step": episode.native_steps[episode.rest_index],
        "rest_references_sha256": result.rest_references_sha256,
        "release_anchor": (
            None
            if release_episode is None
            else {
                "benchmark_index": release_episode.anchor_index,
                "native_step": release_episode.native_steps[
                    release_episode.anchor_index
                ],
            }
        ),
        "release_clean_prediction": (
            None
            if result.release_clean_prediction is None
            else arrays.add(result.release_clean_prediction)
        ),
        "release_expert_actions": (
            None
            if release_episode is None
            else arrays.add(release_episode.expert_actions)
        ),
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file_name": episode.source_path.name,
            "loaded_record_count": len(episode.records),
            "sha256": episode.source_sha256,
            "total_record_count": episode.total_record_count,
        },
        "task_id": episode.task_id,
        "episode_id": episode.episode_id,
    }


def _write_member(path: Path, value: object) -> None:
    raw = canonical_live_json_bytes(value)
    if len(raw) > LIVE_MAX_JSON_BYTES:
        raise LiveArtifactValidationError("recorded JSON member exceeds size cap")
    with path.open("xb") as stream:
        stream.write(raw)


def _write_staging(staging: Path, result: RecordedExperimentResult) -> None:
    arrays = LiveArrayWriter(staging)
    experiment = _document(result, arrays)
    _write_member(staging / EXPERIMENT_DOCUMENT, experiment)
    snapshot = scan_live_bundle(staging)
    members = [
        {
            "path": path,
            "sha256": hash_live_member(snapshot, path),
            "size_bytes": metadata.size_bytes,
        }
        for path, metadata in sorted(snapshot.files.items())
    ]
    experiment_member = next(
        item for item in members if item["path"] == EXPERIMENT_DOCUMENT
    )
    receipt = {
        "evidence_level": result.evidence_level,
        "experiment_sha256": experiment_member["sha256"],
        "members": members,
        "policy_identity_sha256": result.policy_identity.sha256,
        "schema_version": SCHEMA_VERSION,
        "source_sha256": result.episode.source_sha256,
    }
    _write_member(staging / ROOT_RECEIPT, receipt)


def write_recorded_experiment_artifact(
    output: Path, result: RecordedExperimentResult
) -> str:
    """Write, validate, and atomically publish one no-clobber artifact."""

    target = Path(output)
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise FileExistsError("recorded artifact target is not a real directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".robotactile-recorded-staging-", dir=target.parent)
    )
    try:
        _write_staging(staging, result)
        staged = load_recorded_experiment_artifact(staging)
        if target.exists() and any(target.iterdir()):
            existing = load_recorded_experiment_artifact(target)
            if existing["root_receipt_sha256"] != staged["root_receipt_sha256"]:
                raise FileExistsError("target contains another recorded experiment")
            return cast(str, existing["root_receipt_sha256"])
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
        published = load_recorded_experiment_artifact(target)
        if published["root_receipt_sha256"] != staged["root_receipt_sha256"]:
            raise LiveArtifactValidationError("published recorded artifact changed")
        return cast(str, published["root_receipt_sha256"])
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LiveArtifactValidationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def load_recorded_experiment_artifact(output: Path) -> dict[str, object]:
    """Verify the exact bundle inventory and all content-addressed arrays."""

    snapshot = scan_live_bundle(Path(output))
    root_raw = read_live_member(snapshot, ROOT_RECEIPT, LIVE_MAX_JSON_BYTES)
    receipt = _mapping(strict_live_json_bytes(root_raw, ROOT_RECEIPT), ROOT_RECEIPT)
    required = {
        "evidence_level",
        "experiment_sha256",
        "members",
        "policy_identity_sha256",
        "schema_version",
        "source_sha256",
    }
    if set(receipt) != required or receipt["schema_version"] != SCHEMA_VERSION:
        raise LiveArtifactValidationError("recorded root receipt fields mismatch")
    raw_members = receipt["members"]
    if not isinstance(raw_members, list):
        raise LiveArtifactValidationError("recorded root members must be a list")
    expected_paths = {ROOT_RECEIPT}
    for raw_member in raw_members:
        member = _mapping(raw_member, "recorded member")
        if set(member) != {"path", "sha256", "size_bytes"}:
            raise LiveArtifactValidationError("recorded member fields mismatch")
        path = member["path"]
        size = member["size_bytes"]
        digest = member["sha256"]
        if (
            not isinstance(path, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
            or path not in snapshot.files
            or snapshot.files[path].size_bytes != size
            or hash_live_member(snapshot, path) != digest
        ):
            raise LiveArtifactValidationError("recorded member does not match bytes")
        expected_paths.add(path)
    if set(snapshot.files) != expected_paths:
        raise LiveArtifactValidationError("recorded artifact inventory mismatch")
    expected_directories = (
        {"arrays"}
        if any(path.startswith("arrays/") for path in expected_paths)
        else set()
    )
    if set(snapshot.directories) != expected_directories:
        raise LiveArtifactValidationError("recorded artifact directory mismatch")
    experiment_raw = read_live_member(
        snapshot, EXPERIMENT_DOCUMENT, LIVE_MAX_JSON_BYTES
    )
    if live_sha256_bytes(experiment_raw) != receipt["experiment_sha256"]:
        raise LiveArtifactValidationError("recorded experiment hash mismatch")
    experiment = _mapping(
        strict_live_json_bytes(experiment_raw, EXPERIMENT_DOCUMENT),
        EXPERIMENT_DOCUMENT,
    )
    if (
        experiment.get("schema_version") != SCHEMA_VERSION
        or experiment.get("evidence_level") != "recorded_model_only_n0_v1"
    ):
        raise LiveArtifactValidationError("recorded experiment identity mismatch")
    referenced: set[str] = set()
    load_live_array(experiment["expert_actions"], snapshot, referenced)
    for name in ("release_clean_prediction", "release_expert_actions"):
        descriptor = experiment.get(name)
        if descriptor is not None:
            load_live_array(descriptor, snapshot, referenced)
    conditions = experiment.get("conditions")
    if not isinstance(conditions, list):
        raise LiveArtifactValidationError("recorded conditions must be a list")
    for raw_condition in conditions:
        condition = _mapping(raw_condition, "recorded condition")
        prediction = condition.get("prediction")
        if prediction is not None:
            load_live_array(prediction, snapshot, referenced)
    actual_arrays = {path for path in snapshot.files if path.startswith("arrays/")}
    if referenced != actual_arrays:
        raise LiveArtifactValidationError("recorded artifact has orphan arrays")
    return {
        "condition_count": len(conditions),
        "evidence_level": experiment["evidence_level"],
        "experiment": dict(experiment),
        "root_receipt_sha256": live_sha256_bytes(root_raw),
        "schema_version": SCHEMA_VERSION,
    }


__all__ = [
    "load_recorded_experiment_artifact",
    "write_recorded_experiment_artifact",
]
