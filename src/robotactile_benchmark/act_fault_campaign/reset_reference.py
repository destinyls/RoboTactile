"""Build and persist source-bound ACT reset references."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
    UniVTACResetReferenceError,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.contracts import EvaluationRecord
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.trials import Condition, TerminalStatus

DEFAULT_ACT_RESET_QPOS_ATOL = 1e-5
MAX_ACT_RESET_QPOS_ATOL = 1e-3
MAX_ACT_RESET_REFERENCE_BYTES = 64 * 1024

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ACTResetReferenceArtifactError(ValueError):
    """An ACT reset-reference source or JSON artifact failed validation."""


def act_reset_reference_relpath(task: str, pair_key: str) -> str:
    """Return the canonical campaign-relative path for one task/seed pair."""

    if not isinstance(task, str) or _SAFE_IDENTIFIER.fullmatch(task) is None:
        raise ACTResetReferenceArtifactError("task must be a safe identifier")
    if not isinstance(pair_key, str) or _SHA256.fullmatch(pair_key) is None:
        raise ACTResetReferenceArtifactError("pair_key must be a lowercase SHA256")
    return f"reset_references/{task}/{pair_key}.json"


def build_act_reset_reference_from_artifact(
    artifact: LoadedLiveUniVTACArtifact,
    *,
    qpos_atol: float = DEFAULT_ACT_RESET_QPOS_ATOL,
) -> UniVTACResetReference:
    """Derive one reset reference from a strict successful Clean artifact."""

    if type(artifact) is not LoadedLiveUniVTACArtifact:
        raise TypeError("artifact must be an exact LoadedLiveUniVTACArtifact")
    trial = artifact.trial
    result = artifact.evidence.result
    if trial.condition is not Condition.CLEAN or artifact.fault_manifest is not None:
        raise ACTResetReferenceArtifactError(
            "reset reference source must be a Clean artifact"
        )
    if (
        result.terminal_status is not TerminalStatus.SUCCESS
        or result.score_eligible is not True
        or result.score_success is not True
        or result.validation_passed is not True
    ):
        raise ACTResetReferenceArtifactError(
            "reset reference source must be a validated successful Clean result"
        )
    _validate_source_links(artifact)
    if (
        isinstance(qpos_atol, bool)
        or not isinstance(qpos_atol, (int, float))
        or not np.isfinite(float(qpos_atol))
        or not 0.0 < float(qpos_atol) <= MAX_ACT_RESET_QPOS_ATOL
    ):
        raise ACTResetReferenceArtifactError(
            "qpos_atol must be positive and no greater than the formal maximum"
        )
    simulator_state_sha256 = result.initial_state_sha256
    if (
        not isinstance(simulator_state_sha256, str)
        or _SHA256.fullmatch(simulator_state_sha256) is None
    ):
        raise ACTResetReferenceArtifactError(
            "successful Clean result must bind its initial simulator state"
        )
    native_step, qpos8 = _initial_reset_values(
        artifact,
        simulator_state_sha256=simulator_state_sha256,
    )
    try:
        return UniVTACResetReference(
            task_id=trial.task,
            initial_seed=trial.initial_seed,
            exogenous_seed=trial.exogenous_seed,
            pair_key=trial.pair_key,
            dataset_sha256=trial.dataset_sha256,
            checkpoint_sha256=trial.checkpoint_sha256,
            config_sha256=trial.config_sha256,
            source_artifact_root_sha256=artifact.external_root_sha256,
            source_result_sha256=artifact.root_receipt.result_sha256,
            source_run_content_sha256=artifact.run_content_sha256,
            expected_simulator_state_sha256=simulator_state_sha256,
            expected_native_step=native_step,
            expected_qpos8=qpos8,
            qpos_atol=qpos_atol,
        )
    except UniVTACResetReferenceError as error:
        raise ACTResetReferenceArtifactError(
            "derived ACT reset reference is invalid"
        ) from error


def write_act_reset_reference(
    path: Path,
    reference: UniVTACResetReference,
) -> bool:
    """Atomically publish canonical JSON; identical content is idempotent."""

    if type(reference) is not UniVTACResetReference:
        raise TypeError("reference must be an exact UniVTACResetReference")
    target = Path(path).expanduser().absolute()
    payload = canonical_json_bytes(reference.to_dict())
    _reject_symlink_components(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    if target.exists() or target.is_symlink():
        return _verify_existing(target, payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return _verify_existing(target, payload)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_act_reset_reference(path: Path) -> UniVTACResetReference:
    """Strictly load one stable, canonical, non-symlink reference JSON."""

    source = Path(path).expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ACTResetReferenceArtifactError(
            "ACT reset reference must be a regular non-symlink file"
        )
    try:
        before = source.stat()
        if not 1 <= before.st_size <= MAX_ACT_RESET_REFERENCE_BYTES:
            raise ACTResetReferenceArtifactError(
                "ACT reset reference size is outside bounds"
            )
        raw = source.read_bytes()
        after = source.stat()
    except OSError as error:
        raise ACTResetReferenceArtifactError(
            "ACT reset reference cannot be read"
        ) from error
    if (
        len(raw) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ACTResetReferenceArtifactError(
            "ACT reset reference changed while being read"
        )
    try:
        document = strict_json_bytes(raw, "ACT reset reference")
        if not isinstance(document, Mapping):
            raise ACTResetReferenceArtifactError(
                "ACT reset reference must contain a JSON object"
            )
        reference = UniVTACResetReference.from_dict(document)
    except (TypeError, ValueError) as error:
        if isinstance(error, ACTResetReferenceArtifactError):
            raise
        raise ACTResetReferenceArtifactError(
            "ACT reset reference is not strict canonical JSON"
        ) from error
    if canonical_json_bytes(reference.to_dict()) != raw:
        raise ACTResetReferenceArtifactError(
            "ACT reset reference does not round-trip canonically"
        )
    return reference


def _validate_source_links(artifact: LoadedLiveUniVTACArtifact) -> None:
    trial = artifact.trial
    result = artifact.evidence.result
    receipt = artifact.root_receipt
    if (
        result.trial_manifest_sha256 != trial.sha256
        or result.pair_key != trial.pair_key
        or receipt.trial_manifest_sha256 != trial.sha256
        or receipt.pair_key != trial.pair_key
        or receipt.result_sha256 != result.sha256
        or artifact.external_root_sha256 != artifact.root_receipt_sha256
    ):
        raise ACTResetReferenceArtifactError(
            "Clean source artifact provenance links are inconsistent"
        )


def _initial_record(artifact: LoadedLiveUniVTACArtifact) -> EvaluationRecord:
    if artifact.capture_profile is LiveCaptureProfile.PAPER_FULL:
        finalization = artifact.evidence.finalization
        if finalization is None or finalization.validation is not None:
            raise ACTResetReferenceArtifactError(
                "paper_full Clean artifact has no identity finalization"
            )
        records = finalization.delivered_records
    elif artifact.capture_profile is LiveCaptureProfile.PREVIEW:
        preview = artifact.preview_trace
        if (
            preview is None
            or not preview.selected_indices
            or preview.selected_indices[0] != 0
        ):
            raise ACTResetReferenceArtifactError(
                "preview artifact does not retain selected index 0"
            )
        records = preview.delivered_records
    else:
        raise ACTResetReferenceArtifactError("unsupported ACT reset capture profile")
    if not records or type(records[0]) is not EvaluationRecord:
        raise ACTResetReferenceArtifactError("initial Clean frame is missing")
    return records[0]


def _initial_reset_values(
    artifact: LoadedLiveUniVTACArtifact,
    *,
    simulator_state_sha256: str,
) -> tuple[int, tuple[float, ...]]:
    diagnostics = artifact.evidence.initial_diagnostics
    if isinstance(diagnostics, Mapping):
        task = diagnostics.get("task")
        if isinstance(task, Mapping) and "reset_witness" in task:
            return _reset_witness_values(
                task["reset_witness"],
                artifact=artifact,
                simulator_state_sha256=simulator_state_sha256,
            )
    if artifact.capture_profile is LiveCaptureProfile.METRICS_ONLY:
        raise ACTResetReferenceArtifactError(
            "metrics_only source requires a strict initial reset witness"
        )
    first = _initial_record(artifact)
    observation = first.observation
    trial = artifact.trial
    if (
        observation.task != trial.task
        or observation.seed != trial.initial_seed
        or observation.step_index != 0
        or first.clean_record_sha256 != first.delivered_record_sha256
    ):
        raise ACTResetReferenceArtifactError("initial Clean record identity mismatch")
    qpos8 = _qpos8(observation.proprio, "initial observation proprio8")
    native_step = diagnostics.get("native_step_id")
    return _native_step(native_step), qpos8


def _reset_witness_values(
    value: object,
    *,
    artifact: LoadedLiveUniVTACArtifact,
    simulator_state_sha256: str,
) -> tuple[int, tuple[float, ...]]:
    if not isinstance(value, Mapping):
        raise ACTResetReferenceArtifactError("initial reset witness must be a mapping")
    trial = artifact.trial
    if (
        value.get("schema") != "univtac-reset-witness-v1"
        or value.get("task_id") != trial.task
        or value.get("initial_seed") != trial.initial_seed
        or value.get("exogenous_seed") != trial.exogenous_seed
        or value.get("simulator_state_sha256") != simulator_state_sha256
    ):
        raise ACTResetReferenceArtifactError(
            "initial reset witness identity differs from successful Clean"
        )
    viable = value.get("reset_viable")
    if viable is not None and viable is not True:
        raise ACTResetReferenceArtifactError(
            "successful Clean reset witness is explicitly unqualified"
        )
    return _native_step(value.get("native_step")), _qpos8(
        value.get("qpos8"),
        "initial reset witness qpos8",
    )


def _native_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ACTResetReferenceArtifactError(
            "initial native step must be a non-negative integer"
        )
    return value


def _qpos8(value: object, name: str) -> tuple[float, ...]:
    array = np.asarray(value)
    if (
        array.shape != (8,)
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
    ):
        raise ACTResetReferenceArtifactError(
            f"{name} must contain eight finite floating values"
        )
    return tuple(float(item) for item in array)


def _verify_existing(path: Path, payload: bytes) -> bool:
    if path.is_symlink() or not path.is_file():
        raise ACTResetReferenceArtifactError(
            "ACT reset reference output is not a regular file"
        )
    try:
        current = path.read_bytes()
    except OSError as error:
        raise ACTResetReferenceArtifactError(
            "ACT reset reference output cannot be read"
        ) from error
    if current != payload:
        raise FileExistsError("refusing to replace a different ACT reset reference")
    return False


def _reject_symlink_components(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists() and candidate.is_symlink():
            raise ACTResetReferenceArtifactError(
                "ACT reset reference path cannot traverse a symlink"
            )


__all__ = [
    "ACTResetReferenceArtifactError",
    "DEFAULT_ACT_RESET_QPOS_ATOL",
    "MAX_ACT_RESET_QPOS_ATOL",
    "MAX_ACT_RESET_REFERENCE_BYTES",
    "act_reset_reference_relpath",
    "build_act_reset_reference_from_artifact",
    "load_act_reset_reference",
    "write_act_reset_reference",
]
