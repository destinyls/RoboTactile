"""Write and strictly reload one bounded closed-loop evidence bundle."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_arrays import ArrayFileStore
from robotactile_benchmark.closed_loop.artifact_contracts import (
    EVIDENCE_LEVEL,
    ROOT_RECEIPT_PATH,
    ArtifactMember,
    ArtifactValidationError,
    LoadedClosedLoopBundle,
    RootReceipt,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    read_bundle_inventory,
    sha256_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.closed_loop.artifact_records import (
    action_trace_from_dict,
    action_trace_to_dict,
    delivery_trace_from_dict,
    delivery_trace_to_dict,
)
from robotactile_benchmark.closed_loop.artifact_semantics import (
    validate_trial_execution_semantics,
)
from robotactile_benchmark.closed_loop.artifact_validation import (
    validation_report_from_dict,
    validation_report_to_dict,
)
from robotactile_benchmark.closed_loop.artifact_values import (
    result_from_dict,
    result_to_dict,
    run_spec_from_dict,
    run_spec_to_dict,
)
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.result_hashes import action_trace_sha256
from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash, thaw_value
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import Condition, TrialManifest
from robotactile_benchmark.validators import ValidationReport, validate_delivery


def write_closed_loop_bundle(
    output: Path,
    *,
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault_manifest: FaultManifest,
    evidence: ClosedLoopExecutionEvidence,
) -> LoadedClosedLoopBundle:
    """Stage, verify, and atomically publish one single-trial CPU-sized bundle."""

    files = _build_bundle_files(trial, run_spec, fault_manifest, evidence)
    output = Path(output)
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise FileExistsError("closed-loop output exists and is not a real directory")
    if output.exists() and any(output.iterdir()):
        try:
            loaded = load_closed_loop_bundle(output)
            _, _, current = read_bundle_inventory(output)
        except (ArtifactValidationError, OSError) as error:
            raise FileExistsError(
                "non-empty output is not an exact verified closed-loop bundle"
            ) from error
        if current != files:
            raise FileExistsError("non-empty output belongs to a different bundle")
        return loaded
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".robotactile-staging-", dir=output.parent))
    try:
        _write_files(staging, files)
        staged = load_closed_loop_bundle(staging)
        if staged.root_receipt_sha256 != sha256_bytes(files[ROOT_RECEIPT_PATH]):
            raise ArtifactValidationError("staged root receipt hash changed")
        if output.exists():
            if any(output.iterdir()):
                raise FileExistsError(
                    "closed-loop output became non-empty during staging"
                )
            output.rmdir()
        os.replace(staging, output)
        return load_closed_loop_bundle(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_closed_loop_bundle(output: Path) -> LoadedClosedLoopBundle:
    """Fail closed unless every byte, contract, hash, and validator link agrees."""

    try:
        return _load_closed_loop_bundle(Path(output))
    except ArtifactValidationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        raise ArtifactValidationError(
            "closed-loop bundle failed typed validation"
        ) from error


def _build_bundle_files(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault_manifest: FaultManifest,
    evidence: ClosedLoopExecutionEvidence,
) -> dict[str, bytes]:
    _validate_write_inputs(trial, run_spec, fault_manifest, evidence)
    finalization = evidence.finalization
    if finalization is None or finalization.validation is None:
        raise ArtifactValidationError(
            "fault bundle requires delivery validation evidence"
        )
    arrays = ArrayFileStore()
    documents: dict[str, object] = {
        "trial_manifest.json": trial.to_dict(),
        "run_spec.json": run_spec_to_dict(run_spec),
        "fault_manifest.json": fault_manifest.to_dict(),
        "terminal_result.json": result_to_dict(evidence.result),
        "action_trace.json": action_trace_to_dict(evidence.action_entries, arrays),
        "delivery_trace.json": delivery_trace_to_dict(finalization, arrays),
        "validation_report.json": validation_report_to_dict(
            finalization.validation, finalization
        ),
    }
    files = {name: canonical_json_bytes(value) for name, value in documents.items()}
    files.update(arrays.files)
    members = tuple(
        ArtifactMember(path=path, sha256=sha256_bytes(raw), size_bytes=len(raw))
        for path, raw in sorted(files.items())
    )
    result = evidence.result
    receipt = RootReceipt(
        evidence_level=EVIDENCE_LEVEL,
        trial_manifest_sha256=trial.sha256,
        pair_key=trial.pair_key,
        run_spec_sha256=run_spec.sha256,
        fault_manifest_sha256=fault_manifest.sha256,
        result_sha256=result.sha256,
        terminal_trace_sha256=result.terminal_trace_sha256,
        action_trace_sha256=result.action_trace_sha256,
        clean_trace_sha256=result.clean_trace_sha256,
        delivered_trace_sha256=result.delivered_trace_sha256,
        members=members,
    )
    files[ROOT_RECEIPT_PATH] = canonical_json_bytes(receipt.to_dict())
    return files


def _load_closed_loop_bundle(output: Path) -> LoadedClosedLoopBundle:
    receipt, root_hash, files = read_bundle_inventory(output)
    documents = {
        name: strict_json_bytes(files[name], name)
        for name in (
            "trial_manifest.json",
            "run_spec.json",
            "fault_manifest.json",
            "terminal_result.json",
            "action_trace.json",
            "delivery_trace.json",
            "validation_report.json",
        )
    }
    trial_document = documents["trial_manifest.json"]
    fault_document = documents["fault_manifest.json"]
    if not isinstance(trial_document, dict) or not isinstance(fault_document, dict):
        raise ArtifactValidationError("manifest members must contain JSON objects")
    trial = TrialManifest.from_dict(trial_document)
    run_spec = run_spec_from_dict(documents["run_spec.json"])
    fault_manifest = FaultManifest.from_dict(fault_document)
    result = result_from_dict(documents["terminal_result.json"])
    validation, validation_links = validation_report_from_dict(
        documents["validation_report.json"]
    )
    referenced_arrays: set[str] = set()
    action_entries = action_trace_from_dict(
        documents["action_trace.json"], files, referenced_arrays
    )
    finalization = delivery_trace_from_dict(
        documents["delivery_trace.json"],
        validation,
        files,
        referenced_arrays,
    )
    array_members = {
        member.path for member in receipt.members if member.path.startswith("arrays/")
    }
    if referenced_arrays != array_members:
        raise ArtifactValidationError(
            "array descriptors and root inventory are not bijective"
        )
    _validate_loaded_cross_links(
        trial,
        run_spec,
        fault_manifest,
        result,
        finalization,
        action_entries,
        validation_links,
        receipt,
    )
    return LoadedClosedLoopBundle(
        trial=trial,
        run_spec=run_spec,
        fault_manifest=fault_manifest,
        result=result,
        finalization=finalization,
        action_entries=action_entries,
        root_receipt=receipt,
        root_receipt_sha256=root_hash,
    )


def _validate_write_inputs(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault_manifest: FaultManifest,
    evidence: ClosedLoopExecutionEvidence,
) -> None:
    if not isinstance(trial, TrialManifest) or not isinstance(
        run_spec, ClosedLoopRunSpec
    ):
        raise TypeError("writer requires typed trial and run spec contracts")
    if not isinstance(fault_manifest, FaultManifest) or not isinstance(
        evidence, ClosedLoopExecutionEvidence
    ):
        raise TypeError("writer requires typed fault and execution evidence")
    finalization = evidence.finalization
    if (
        trial.condition not in {Condition.FAULTED, Condition.RESTORED}
        or trial.fault_manifest_sha256 != fault_manifest.sha256
        or evidence.result.trial_manifest_sha256 != trial.sha256
        or evidence.result.run_spec_sha256 != run_spec.sha256
        or finalization is None
        or finalization.validation is None
        or not finalization.validation.passed
        or evidence.result.validation_passed is not True
    ):
        raise ArtifactValidationError(
            "writer inputs do not form a validated faulted trial"
        )
    if fault_manifest.operator_id in REST_REFERENCE_OPERATOR_IDS:
        raise ArtifactValidationError(
            "initial single-trial bundle does not support rest-reference operators"
        )
    validate_trial_execution_semantics(
        trial,
        run_spec,
        evidence.result,
        finalization,
        evidence.action_entries,
    )


def _validate_loaded_cross_links(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault: FaultManifest,
    result: object,
    finalization: object,
    action_entries: object,
    validation_links: dict[str, str],
    receipt: RootReceipt,
) -> None:
    from robotactile_benchmark.closed_loop.capture import ActionTraceEntry
    from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
    from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult

    if not isinstance(result, ClosedLoopTrialResult):
        raise ArtifactValidationError("terminal result was not reconstructed")
    if not isinstance(finalization, DeliveryFinalization):
        raise ArtifactValidationError("delivery finalization was not reconstructed")
    validation = finalization.validation
    if validation is None:
        raise ArtifactValidationError("fault delivery validation is missing")
    if not isinstance(action_entries, tuple) or not all(
        isinstance(entry, ActionTraceEntry) for entry in action_entries
    ):
        raise ArtifactValidationError("action entries were not reconstructed")
    validate_trial_execution_semantics(
        trial,
        run_spec,
        result,
        finalization,
        action_entries,
    )
    if trial.condition not in {Condition.FAULTED, Condition.RESTORED}:
        raise ArtifactValidationError("bounded bundle must contain a faulted trial")
    if fault.operator_id in REST_REFERENCE_OPERATOR_IDS:
        raise ArtifactValidationError("bundle requires unsupported rest-reference data")
    rerun = validate_delivery(
        finalization.clean_records,
        finalization.delivered_records,
        fault,
    )
    if _report_hash(rerun) != _report_hash(validation):
        raise ArtifactValidationError("independent delivery validation disagrees")
    action_hash = action_trace_sha256(
        tuple(entry.to_hash_entry() for entry in action_entries)
    )
    executed_rows = sum(entry.executed_actions.shape[0] for entry in action_entries)
    expected_links = {
        "manifest_sha256": fault.sha256,
        "clean_trace_sha256": finalization.clean_trace_sha256,
        "delivered_trace_sha256": finalization.delivered_trace_sha256,
    }
    cross_links = (
        trial.fault_manifest_sha256 == fault.sha256,
        result.trial_manifest_sha256 == trial.sha256,
        result.pair_key == trial.pair_key,
        result.run_spec_sha256 == run_spec.sha256,
        result.action_trace_sha256 == action_hash,
        result.clean_trace_sha256 == finalization.clean_trace_sha256,
        result.delivered_trace_sha256 == finalization.delivered_trace_sha256,
        result.observation_count == len(finalization.clean_records),
        result.observation_count == executed_rows + 1,
        result.control_cycle_count == len(action_entries),
        result.validation_passed is validation.passed,
        result.validation_failure_codes == validation.failure_codes,
        finalization.manifest_sha256 == fault.sha256,
        validation_links == expected_links,
        receipt.trial_manifest_sha256 == trial.sha256,
        receipt.pair_key == trial.pair_key,
        receipt.run_spec_sha256 == run_spec.sha256,
        receipt.fault_manifest_sha256 == fault.sha256,
        receipt.result_sha256 == result.sha256,
        receipt.terminal_trace_sha256 == result.terminal_trace_sha256,
        receipt.action_trace_sha256 == action_hash,
        receipt.clean_trace_sha256 == finalization.clean_trace_sha256,
        receipt.delivered_trace_sha256 == finalization.delivered_trace_sha256,
    )
    if not all(cross_links):
        raise ArtifactValidationError("closed-loop semantic cross-link mismatch")
    if not validation.passed or int(validation.metrics.get("affected_records", 0)) < 1:
        raise ArtifactValidationError("fault operator was not active and validated")


def _report_hash(report: Optional[ValidationReport]) -> str:
    if report is None:
        raise ArtifactValidationError("fault bundle validation report is missing")
    return canonical_hash(
        {
            "passed": report.passed,
            "failure_codes": report.failure_codes,
            "failures": report.failures,
            "metrics": thaw_value(report.metrics),
        }
    )


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for relative, raw in sorted(files.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
