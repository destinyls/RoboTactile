"""Validate immutable FTP-1 joint-training artifacts and progress."""

from __future__ import annotations

from pathlib import Path

from .contracts import FTP1TrainingRequest, canonical_json_sha256
from .runtime import load_object


def verify_signed(path: Path, signature_field: str) -> dict[str, object]:
    payload = load_object(path)
    unsigned = dict(payload)
    claimed = unsigned.pop(signature_field, None)
    if claimed != canonical_json_sha256(unsigned):
        raise ValueError(f"invalid signed artifact: {path}")
    return payload


def numeric_checkpoints(root: Path) -> list[tuple[int, Path]]:
    if not root.is_dir():
        return []
    return sorted(
        (int(path.name), path)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def completed_tasks(request: FTP1TrainingRequest) -> list[str]:
    path = request.output_root / "joint" / "training_receipt.json"
    if not path.is_file():
        return []
    receipt = verify_signed(path, "training_receipt_sha256")
    if (
        receipt.get("request_sha256") != request.request_sha256
        or receipt.get("joint_model") is not True
        or receipt.get("checkpoint_count") != 1
        or receipt.get("task_ids") != list(request.tasks)
    ):
        raise ValueError("joint training receipt does not satisfy the request")
    return list(request.tasks)


def prepared_tasks(request: FTP1TrainingRequest) -> list[str]:
    prepared: list[str] = []
    for task in request.tasks:
        conversion_path = (
            request.output_root / "dataset" / task / "conversion_receipt.json"
        )
        rgb_path = request.output_root / "dataset" / task / "rgb_contract_receipt.json"
        if not conversion_path.is_file() or not rgb_path.is_file():
            continue
        conversion = verify_signed(conversion_path, "conversion_receipt_sha256")
        rgb = verify_signed(rgb_path, "rgb_contract_receipt_sha256")
        if (
            conversion.get("request_sha256") != request.request_sha256
            or conversion.get("task_id") != task
            or conversion.get("color_contract") != "canonical_rgb_exact_v1"
            or rgb.get("request_sha256") != request.request_sha256
            or rgb.get("task_id") != task
            or rgb.get("status") != "passed"
        ):
            raise ValueError(f"{task} preparation receipts do not satisfy the request")
        prepared.append(task)
    return prepared


__all__ = [
    "completed_tasks",
    "numeric_checkpoints",
    "prepared_tasks",
    "verify_signed",
]
