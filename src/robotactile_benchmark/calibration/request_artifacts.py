"""Generate, persist, and strictly reload clean calibration requests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from robotactile_benchmark.calibration.request_contracts import (
    CALIBRATION_REQUEST_PATH,
    CALIBRATION_REQUEST_RECEIPT_PATH,
    CalibrationRequestError,
    CalibrationRequestReceipt,
    CalibrationRequestSpec,
    LoadedCalibrationRequest,
)
from robotactile_benchmark.closed_loop.artifact_contracts import ArtifactValidationError
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.trials import Condition, system_manifest_hash

_EXPECTED_FILES = frozenset(
    {CALIBRATION_REQUEST_PATH, CALIBRATION_REQUEST_RECEIPT_PATH}
)
_MAX_FILE_BYTES = 256 * 1024


def build_clean_calibration_request(
    spec: CalibrationRequestSpec,
) -> LiveUniVTACRunRequest:
    """Materialize a clean ACT request without importing model or simulator code."""

    if type(spec) is not CalibrationRequestSpec:
        raise TypeError("spec must be an exact CalibrationRequestSpec")
    base_manifest = system_manifest_hash(
        spec.base_system_id,
        spec.checkpoint_sha256,
        spec.config_sha256,
        ACTION_SPEC,
    )
    request = LiveUniVTACRunRequest(
        task_id=spec.task_id,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.ACT,
        base_system_id=spec.base_system_id,
        dataset_sha256=spec.split_manifest_sha256,
        checkpoint_sha256=spec.checkpoint_sha256,
        config_sha256=spec.config_sha256,
        base_system_manifest_sha256=base_manifest,
        initial_seed=spec.initial_seed,
        exogenous_seed=spec.exogenous_seed,
        max_control_cycles=spec.max_control_cycles,
        max_observation_steps=spec.max_observation_steps,
        execute_action_steps=1,
        wall_timeout_s=spec.wall_timeout_s,
        upstream_root=spec.upstream_root,
        runtime_dir=spec.runtime_dir,
        output_dir=spec.live_artifact_output_dir,
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=spec.act_device_name,
        simulator_device=spec.simulator_device,
        launcher_args=spec.launcher_args,
    )
    load_live_univtac_run(request)
    return request


def write_calibration_request_bundle(
    output: Path, spec: CalibrationRequestSpec
) -> LoadedCalibrationRequest:
    """Write an atomic two-file request bundle, or reuse exact existing bytes."""

    output = Path(output).absolute()
    request = build_clean_calibration_request(spec)
    document = live_univtac_request_to_dict(request)
    request_raw = canonical_json_bytes(document)
    loaded_run = load_live_univtac_run(request)
    receipt = CalibrationRequestReceipt(
        request_sha256=canonical_hash(document),
        request_file_sha256=sha256_bytes(request_raw),
        trial_manifest_sha256=loaded_run.trial.sha256,
        task_id=request.task_id,
        dataset_split=spec.dataset_split,
        split_manifest_sha256=spec.split_manifest_sha256,
        base_system_manifest_sha256=loaded_run.trial.base_system_manifest_sha256,
    )
    receipt_raw = canonical_json_bytes(receipt.to_dict())
    if output.exists() or output.is_symlink():
        loaded = load_calibration_request_bundle(output)
        if loaded.request != request or loaded.receipt != receipt:
            raise CalibrationRequestError(
                "existing calibration request cannot be clobbered"
            )
        return loaded
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (staging / CALIBRATION_REQUEST_PATH).write_bytes(request_raw)
        (staging / CALIBRATION_REQUEST_RECEIPT_PATH).write_bytes(receipt_raw)
        os.rename(staging, output)
    except FileExistsError as error:
        loaded = load_calibration_request_bundle(output)
        if loaded.request != request or loaded.receipt != receipt:
            raise CalibrationRequestError(
                "calibration request publication race disagrees"
            ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_calibration_request_bundle(output)


def load_calibration_request_bundle(output: Path) -> LoadedCalibrationRequest:
    """Strictly load the exact bundle and recompute every content link."""

    root = Path(output).absolute()
    if root.is_symlink() or not root.is_dir():
        raise CalibrationRequestError(
            "calibration request root must be a real directory"
        )
    files: dict[str, bytes] = {}
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise CalibrationRequestError(
                "calibration request forbids links/directories"
            )
        stat = entry.stat(follow_symlinks=False)
        if not 1 <= stat.st_size <= _MAX_FILE_BYTES:
            raise CalibrationRequestError("calibration request member size is invalid")
        raw = Path(entry.path).read_bytes()
        if len(raw) != stat.st_size:
            raise CalibrationRequestError(
                "calibration request member changed during read"
            )
        files[entry.name] = raw
    if set(files) != _EXPECTED_FILES:
        raise CalibrationRequestError("calibration request inventory mismatch")
    try:
        request = load_live_univtac_request(root / CALIBRATION_REQUEST_PATH)
        receipt = CalibrationRequestReceipt.from_dict(
            strict_json_bytes(
                files[CALIBRATION_REQUEST_RECEIPT_PATH],
                CALIBRATION_REQUEST_RECEIPT_PATH,
            )
        )
    except (ArtifactValidationError, OSError, TypeError, ValueError) as error:
        raise CalibrationRequestError(
            "calibration request failed strict loading"
        ) from error
    document = live_univtac_request_to_dict(request)
    loaded_run = load_live_univtac_run(request)
    checks = (
        request.condition is Condition.CLEAN,
        receipt.request_sha256 == canonical_hash(document),
        receipt.request_file_sha256 == sha256_bytes(files[CALIBRATION_REQUEST_PATH]),
        receipt.trial_manifest_sha256 == loaded_run.trial.sha256,
        receipt.task_id == request.task_id,
        receipt.split_manifest_sha256 == request.dataset_sha256,
        receipt.base_system_manifest_sha256
        == loaded_run.trial.base_system_manifest_sha256,
    )
    if not all(checks):
        raise CalibrationRequestError("calibration request cross-link mismatch")
    return LoadedCalibrationRequest(
        request=request,
        receipt=receipt,
        receipt_file_sha256=sha256_bytes(files[CALIBRATION_REQUEST_RECEIPT_PATH]),
    )
