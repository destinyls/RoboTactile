"""Atomic exporter and strict loader for unqualified live UniVTAC traces."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from robotactile_benchmark.closed_loop.artifact_values import (
    result_from_dict,
    result_to_dict,
    run_spec_from_dict,
    run_spec_to_dict,
)
from robotactile_benchmark.closed_loop.capture import (
    ClosedLoopExecutionEvidence,
)
from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_JSON_RESERVE_BYTES,
    LIVE_MAX_BUNDLE_BYTES,
    LIVE_MAX_JSON_BYTES,
    LIVE_ROOT_RECEIPT_PATH,
    LiveArtifactMember,
    LiveArtifactRootReceipt,
    LiveArtifactValidationError,
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_artifacts_fs import (
    hash_live_member,
    read_live_member,
    scan_live_bundle,
)
from robotactile_benchmark.execution.live_artifacts_io import (
    LiveArrayWriter,
    canonical_live_json_bytes,
    estimate_unique_live_array_bytes,
    live_sha256_bytes,
    strict_live_json_bytes,
)
from robotactile_benchmark.execution.live_artifacts_records import (
    action_trace_from_live_dict,
    action_trace_to_live_dict,
    delivery_trace_from_live_dict,
    delivery_trace_to_live_dict,
)
from robotactile_benchmark.execution.live_artifacts_semantics import (
    validate_live_write_inputs,
    validate_loaded_live_links,
)
from robotactile_benchmark.execution.live_artifacts_values import (
    live_request_identity,
    rest_references_from_live_dict,
    rest_references_to_live_dict,
    source_binding_sha256,
    validate_live_request_identity,
    validation_from_live_dict,
    validation_to_live_dict,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExportReceipt,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import TrialManifest


def write_live_univtac_artifact(
    output: Path,
    loaded: LoadedLiveUniVTACRun,
    evidence: ClosedLoopExecutionEvidence,
) -> LiveArtifactExportReceipt:
    """Stage, revalidate, and atomically publish one path-free live trace."""

    output = Path(output)
    validate_live_write_inputs(loaded, evidence)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FileExistsError("live artifact target is not a real directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    _, estimated_arrays = estimate_unique_live_array_bytes(
        _iter_live_arrays(loaded, evidence)
    )
    required_free = estimated_arrays + LIVE_JSON_RESERVE_BYTES
    if required_free > LIVE_MAX_BUNDLE_BYTES:
        raise LiveArtifactValidationError("live artifact preallocation exceeds cap")
    if shutil.disk_usage(output.parent).free < required_free:
        raise LiveArtifactValidationError("insufficient space for live preallocation")
    staging = Path(
        tempfile.mkdtemp(prefix=".robotactile-live-staging-", dir=output.parent)
    )
    try:
        _write_staged_bundle(staging, loaded, evidence)
        staged = load_live_univtac_artifact(staging)
        if output.exists() and any(output.iterdir()):
            existing = load_live_univtac_artifact(output)
            if existing.root_receipt_sha256 != staged.root_receipt_sha256:
                raise FileExistsError("non-empty target belongs to another live trace")
            return LiveArtifactExportReceipt.for_execution(loaded, evidence)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        published = load_live_univtac_artifact(output)
        if published.root_receipt_sha256 != staged.root_receipt_sha256:
            raise LiveArtifactValidationError("published live root hash changed")
        return LiveArtifactExportReceipt.for_execution(loaded, evidence)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_live_univtac_artifact(output: Path) -> LoadedLiveUniVTACArtifact:
    """Reconstruct typed records and reject every unpinned or inconsistent byte."""

    try:
        return _load_live_univtac_artifact(Path(output))
    except LiveArtifactValidationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        raise LiveArtifactValidationError(
            "live UniVTAC artifact failed typed validation"
        ) from error


def _write_staged_bundle(
    staging: Path,
    loaded: LoadedLiveUniVTACRun,
    evidence: ClosedLoopExecutionEvidence,
) -> None:
    identity = live_request_identity(loaded)
    arrays = LiveArrayWriter(staging)
    documents: dict[str, object] = {
        "request_identity.json": identity,
        "trial_manifest.json": loaded.trial.to_dict(),
        "run_spec.json": run_spec_to_dict(loaded.run_spec),
        "fault_manifest.json": (
            None if loaded.fault_manifest is None else loaded.fault_manifest.to_dict()
        ),
        "rest_references.json": rest_references_to_live_dict(
            loaded.rest_references, arrays
        ),
        "terminal_result.json": result_to_dict(evidence.result),
        "action_trace.json": action_trace_to_live_dict(evidence.action_entries, arrays),
        "delivery_trace.json": delivery_trace_to_live_dict(
            evidence.finalization, arrays
        ),
        "validation_report.json": validation_to_live_dict(evidence.finalization),
    }
    for name, value in documents.items():
        raw = canonical_live_json_bytes(value)
        if len(raw) > LIVE_MAX_JSON_BYTES:
            raise LiveArtifactValidationError(f"live JSON member too large: {name}")
        with (staging / name).open("xb") as stream:
            stream.write(raw)
    snapshot = scan_live_bundle(staging)
    members = tuple(
        LiveArtifactMember(path, hash_live_member(snapshot, path), metadata.size_bytes)
        for path, metadata in sorted(snapshot.files.items())
    )
    result = evidence.result
    receipt = LiveArtifactRootReceipt(
        request_sha256=canonical_hash(identity),
        run_content_sha256=loaded.content_sha256,
        source_binding_sha256=source_binding_sha256(identity),
        trial_manifest_sha256=loaded.trial.sha256,
        pair_key=loaded.trial.pair_key,
        run_spec_sha256=loaded.run_spec.sha256,
        fault_manifest_sha256=(
            None if loaded.fault_manifest is None else loaded.fault_manifest.sha256
        ),
        rest_references_sha256=(
            None if loaded.rest_references is None else loaded.rest_references.sha256
        ),
        result_sha256=result.sha256,
        terminal_trace_sha256=result.terminal_trace_sha256,
        action_trace_sha256=result.action_trace_sha256,
        clean_trace_sha256=result.clean_trace_sha256,
        delivered_trace_sha256=result.delivered_trace_sha256,
        members=members,
    )
    root_raw = canonical_live_json_bytes(receipt.to_dict())
    if sum(item.size_bytes for item in members) + len(root_raw) > LIVE_MAX_BUNDLE_BYTES:
        raise LiveArtifactValidationError("staged live artifact exceeds total cap")
    with (staging / LIVE_ROOT_RECEIPT_PATH).open("xb") as stream:
        stream.write(root_raw)


def _load_live_univtac_artifact(output: Path) -> LoadedLiveUniVTACArtifact:
    snapshot = scan_live_bundle(output)
    root_raw = read_live_member(snapshot, LIVE_ROOT_RECEIPT_PATH, LIVE_MAX_JSON_BYTES)
    receipt = LiveArtifactRootReceipt.from_dict(
        strict_live_json_bytes(root_raw, LIVE_ROOT_RECEIPT_PATH)
    )
    expected_files = {item.path for item in receipt.members} | {LIVE_ROOT_RECEIPT_PATH}
    if set(snapshot.files) != expected_files:
        raise LiveArtifactValidationError("actual live files do not match receipt")
    expected_directories = {
        parent for path in expected_files for parent in _parent_directories(path)
    }
    if set(snapshot.directories) != expected_directories:
        raise LiveArtifactValidationError("live artifact has unknown directories")
    for member in receipt.members:
        metadata = snapshot.files[member.path]
        if (
            metadata.size_bytes != member.size_bytes
            or hash_live_member(snapshot, member.path) != member.sha256
        ):
            raise LiveArtifactValidationError(f"live member mismatch: {member.path}")
    documents = {
        member.path: strict_live_json_bytes(
            read_live_member(snapshot, member.path, LIVE_MAX_JSON_BYTES), member.path
        )
        for member in receipt.members
        if not member.path.startswith("arrays/")
    }
    trial_document = documents["trial_manifest.json"]
    if not isinstance(trial_document, Mapping):
        raise LiveArtifactValidationError("live trial manifest must be an object")
    trial = TrialManifest.from_dict(trial_document)
    run_spec = run_spec_from_dict(documents["run_spec.json"])
    fault_document = documents["fault_manifest.json"]
    if fault_document is not None and not isinstance(fault_document, Mapping):
        raise LiveArtifactValidationError("live fault manifest must be object or null")
    fault = None if fault_document is None else FaultManifest.from_dict(fault_document)
    referenced_arrays: set[str] = set()
    rest_references = rest_references_from_live_dict(
        documents["rest_references.json"], snapshot, referenced_arrays
    )
    report, validation_links = validation_from_live_dict(
        documents["validation_report.json"]
    )
    action_entries = action_trace_from_live_dict(
        documents["action_trace.json"], snapshot, referenced_arrays
    )
    finalization = delivery_trace_from_live_dict(
        documents["delivery_trace.json"], report, snapshot, referenced_arrays
    )
    result = result_from_dict(documents["terminal_result.json"])
    evidence = ClosedLoopExecutionEvidence(result, finalization, action_entries)
    identity = validate_live_request_identity(
        documents["request_identity.json"],
        trial,
        run_spec,
        None if fault is None else fault.sha256,
        None if rest_references is None else rest_references.sha256,
    )
    validate_loaded_live_links(
        identity,
        trial,
        run_spec,
        fault,
        rest_references,
        evidence,
        validation_links,
        receipt,
    )
    array_paths = {
        item.path for item in receipt.members if item.path.startswith("arrays/")
    }
    if referenced_arrays != array_paths:
        raise LiveArtifactValidationError("live array descriptors are not bijective")
    if scan_live_bundle(output) != snapshot:
        raise LiveArtifactValidationError("live artifact changed during load")
    return LoadedLiveUniVTACArtifact(
        request_identity=identity,
        run_content_sha256=receipt.run_content_sha256,
        trial=trial,
        run_spec=run_spec,
        fault_manifest=fault,
        rest_references=rest_references,
        evidence=evidence,
        root_receipt=receipt,
        root_receipt_sha256=live_sha256_bytes(root_raw),
    )


def _iter_live_arrays(
    loaded: LoadedLiveUniVTACRun, evidence: ClosedLoopExecutionEvidence
) -> Iterable[Array]:
    if loaded.rest_references is not None:
        for slot in sorted(loaded.rest_references.payloads):
            yield loaded.rest_references.payload_for(slot)
    for entry in evidence.action_entries:
        yield entry.executed_actions
    if evidence.finalization is None:
        return
    for records in (
        evidence.finalization.clean_records,
        evidence.finalization.delivered_records,
    ):
        for record in records:
            for sensor in record.observation.tactile:
                if sensor.payload is not None:
                    yield sensor.payload
            yield from record.observation.vision.values()
            yield record.observation.proprio


def _parent_directories(path: str) -> tuple[str, ...]:
    parts = path.split("/")[:-1]
    return tuple("/".join(parts[:index]) for index in range(1, len(parts) + 1))
