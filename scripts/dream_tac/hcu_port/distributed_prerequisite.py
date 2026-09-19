"""Bind distributed training to the completed Dream-Tac HCU v11 step."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from scripts.dream_tac.training.training_request import sha256_regular_file

from .distributed_request import DistributedTrainingRequest
from .optimizer_step_request import HcuOptimizerStepRequest
from .optimizer_step_validators import load_bound_receipt, verify_nonempty_bound_file

RESULT_PROTOCOL: Final[str] = "dream_tac_hcu_optimizer_step_result_v1"
OPTIMIZER_STEP_LAUNCH_RESULT_NAME: Final[str] = "optimizer_step_launch_result.json"


def _mapping(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"optimizer-step result {name} must be an object")
    return value


def verify_optimizer_step_prerequisite(
    request: DistributedTrainingRequest,
) -> tuple[HcuOptimizerStepRequest, dict[str, object]]:
    """Verify the exact successful single-step request/result pair.

    The single-card checkpoint is evidence that this source/data/runtime tuple can
    optimize. Distributed training still starts from the converted base DCP so a
    one-rank optimizer state is never treated as a 16-rank resume checkpoint.
    """

    request_path = verify_nonempty_bound_file(
        request.optimizer_step_request, "optimizer_step_request"
    )
    source_request = HcuOptimizerStepRequest.load(request_path)
    if sha256_regular_file(request_path) != request.optimizer_step_request.sha256:
        raise ValueError("optimizer-step request SHA256 changed during load")
    result_path, result = load_bound_receipt(
        request.optimizer_step_result, "optimizer_step_result"
    )
    expected_result_path = (
        source_request.receipt_root / OPTIMIZER_STEP_LAUNCH_RESULT_NAME
    )
    if result_path != expected_result_path.resolve(strict=True):
        raise ValueError("optimizer-step result path does not match its request")
    required: dict[str, object] = {
        "protocol_id": RESULT_PROTOCOL,
        "status": "completed",
        "request_sha256": source_request.request_sha256,
        "process_returncode": 0,
        "optimizer_step_completed": True,
        "exit_code": 0,
    }
    if any(result.get(name) != value for name, value in required.items()):
        raise ValueError("optimizer-step result is not a completed v11 prerequisite")
    evidence = _mapping(result, "evidence")
    if evidence.get("status") != "passed":
        raise ValueError("optimizer-step checkpoint evidence did not pass")
    markers = _mapping(evidence, "required_log_markers")
    if not markers or any(value is not True for value in markers.values()):
        raise ValueError("optimizer-step result is missing required log markers")
    if (
        evidence.get("trained_from_scratch") is not False
        or evidence.get("invalid_numeric_token") is not None
        or evidence.get("latest_marker_value") != "iter_000000001"
    ):
        raise ValueError("optimizer-step result has invalid numerical/resume evidence")
    expected_checkpoint = source_request.job_root / "checkpoints" / "iter_000000001"
    checkpoint_value = evidence.get("checkpoint_root")
    if (
        not isinstance(checkpoint_value, str)
        or Path(checkpoint_value) != expected_checkpoint
    ):
        raise ValueError("optimizer-step checkpoint path does not match its request")
    components = _mapping(evidence, "components")
    required_components = {"model", "optim", "scheduler", "trainer"}
    if set(components) != required_components or any(
        not isinstance(components[name], list) or not components[name]
        for name in required_components
    ):
        raise ValueError("optimizer-step result lacks complete DCP component evidence")
    return source_request, result


def prerequisite_receipt_fields(
    request: DistributedTrainingRequest,
    source_request: HcuOptimizerStepRequest,
    result: Mapping[str, object],
) -> dict[str, object]:
    """Return compact provenance fields for distributed launch receipts."""

    return {
        "optimizer_step_request": request.optimizer_step_request.to_dict(),
        "optimizer_step_request_sha256": source_request.request_sha256,
        "optimizer_step_result": request.optimizer_step_result.to_dict(),
        "optimizer_step_result_receipt_sha256": result["receipt_sha256"],
        "optimizer_step_completed": True,
        "distributed_resume_from_single_rank_state": False,
        "distributed_load_root": str(source_request.base_dcp_root),
    }


__all__ = [
    "RESULT_PROTOCOL",
    "prerequisite_receipt_fields",
    "verify_optimizer_step_prerequisite",
]
