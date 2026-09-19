"""Launch one source-bound Dream-Tac train759 job on 2 nodes x 8 HCUs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scripts.dream_tac.training.training_request import sha256_regular_file

from .distributed_prerequisite import (
    prerequisite_receipt_fields,
    verify_optimizer_step_prerequisite,
)
from .distributed_remote import (
    CONTAINER_RUNTIME,
    RANK_ENTRYPOINT,
    launch_all,
    preflight_all,
)
from .distributed_request import IMAGE_ID, IMAGE_REF, DistributedTrainingRequest
from .distributed_runtime import build_resolved_experiment
from .receipt import canonical_json_sha256, signed_receipt, write_or_verify_receipt

PLAN_NAME: Final[str] = "launch_plan.json"
RESOLVED_NAME: Final[str] = "resolved_experiment.json"
RESULT_NAME: Final[str] = "launch_result.json"


def _implementation_identity(request: DistributedTrainingRequest) -> dict[str, object]:
    actual_root = Path(__file__).resolve().parents[3]
    if request.robotactile_root.resolve(strict=True) != actual_root:
        raise ValueError("robotactile_root does not match the executing checkout")
    module_root = Path(__file__).resolve().parent
    files = (
        module_root / "distributed_request.py",
        module_root / "distributed_prerequisite.py",
        module_root / "distributed_runtime.py",
        module_root / "distributed_remote.py",
        module_root / "distributed_collective_probe.py",
        Path(__file__).resolve(),
        CONTAINER_RUNTIME.resolve(),
        RANK_ENTRYPOINT.resolve(),
    )
    entries = [
        {"path": str(path), "sha256": sha256_regular_file(path)} for path in files
    ]
    return {
        "robotactile_root": str(actual_root),
        "files": entries,
        "aggregate_source_sha256": canonical_json_sha256(entries),
    }


def build_launch_plan(
    request: DistributedTrainingRequest,
    prerequisite: Mapping[str, object],
    resolved: Mapping[str, object],
) -> dict[str, object]:
    return signed_receipt(
        {
            "schema_version": 1,
            "status": "planned",
            "protocol_id": "dream_tac_hcu_2node_launch_plan_v1",
            "claim_boundary": "planned_only_not_distributed_training",
            "request": request.to_dict(),
            "request_sha256": request.request_sha256,
            "prerequisite": dict(prerequisite),
            "resolved_experiment_receipt_sha256": resolved["receipt_sha256"],
            "implementation": _implementation_identity(request),
            "container": {
                "image": IMAGE_REF,
                "image_id": IMAGE_ID,
                "no_clobber_names": True,
                "only_own_transient_containers_cleaned_after_exit": True,
            },
            "logs": [
                str(request.run_root / "logs" / f"train-node-{rank:02d}-{node}.log")
                for rank, node in enumerate(request.nodes)
            ],
        }
    )


def _checkpoint_evidence(request: DistributedTrainingRequest) -> dict[str, object]:
    latest = request.job_root / "checkpoints" / "latest_checkpoint.txt"
    latest_value = (
        latest.read_text(encoding="utf-8").strip()
        if latest.is_file() and not latest.is_symlink()
        else None
    )
    checkpoint = (
        latest.parent / latest_value if latest_value else latest.parent / "missing"
    )
    components: dict[str, dict[str, int | bool]] = {}
    for name in ("model", "optim", "scheduler", "trainer"):
        root = checkpoint / name
        files = (
            [path for path in root.rglob("*") if path.is_file()]
            if root.is_dir()
            else []
        )
        components[name] = {
            "present": bool(files),
            "file_count": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
        }
    iteration: int | None = None
    if latest_value and latest_value.startswith("iter_"):
        try:
            iteration = int(latest_value.removeprefix("iter_"))
        except ValueError:
            iteration = None
    passed = (
        iteration is not None
        and iteration == request.max_iter
        and all(item["present"] for item in components.values())
    )
    return {
        "status": "passed" if passed else "failed",
        "latest_marker": str(latest),
        "latest_marker_value": latest_value,
        "iteration": iteration,
        "checkpoint_root": str(checkpoint),
        "components": components,
    }


def _write_status(
    request: DistributedTrainingRequest, name: str, **fields: object
) -> None:
    payload = signed_receipt(
        {
            "schema_version": 1,
            "protocol_id": "dream_tac_hcu_2node_status_v1",
            "request_sha256": request.request_sha256,
            "status": name,
            **fields,
        }
    )
    write_or_verify_receipt(request.run_root / "status" / f"{name}.json", payload)


def execute(request: DistributedTrainingRequest) -> int:
    """Create one run, preflight both nodes, and launch exactly once."""

    source, source_result = verify_optimizer_step_prerequisite(request)
    prerequisite = prerequisite_receipt_fields(request, source, source_result)
    resolved = build_resolved_experiment(request, source)
    plan = build_launch_plan(request, prerequisite, resolved)
    if not request.output_root.is_dir():
        raise FileNotFoundError(
            f"output_root must already exist: {request.output_root}"
        )
    if request.job_root.exists():
        raise FileExistsError(
            f"refusing to reuse existing job_root: {request.job_root}"
        )
    request.run_root.mkdir(parents=True, exist_ok=False)
    log_dir = request.run_root / "logs"
    log_dir.mkdir()
    write_or_verify_receipt(request.run_root / RESOLVED_NAME, resolved)
    write_or_verify_receipt(request.run_root / PLAN_NAME, plan)
    _write_status(request, "planned")
    try:
        preflight_codes = preflight_all(request, source, log_dir=log_dir)
        _write_status(request, "preflight_passed", node_returncodes=preflight_codes)
        _write_status(request, "running")
        train_codes = launch_all(request, source, log_dir=log_dir)
        evidence = _checkpoint_evidence(request)
        passed = (
            all(code == 0 for code in train_codes.values())
            and evidence["status"] == "passed"
        )
        result = signed_receipt(
            {
                "schema_version": 1,
                "status": "completed" if passed else "failed",
                "protocol_id": "dream_tac_hcu_2node_launch_result_v1",
                "claim_boundary": "distributed_training_only_not_model_quality",
                "request_sha256": request.request_sha256,
                "launch_plan_receipt_sha256": plan["receipt_sha256"],
                "node_returncodes": train_codes,
                "checkpoint_evidence": evidence,
                "exit_code": 0 if passed else 2,
            }
        )
        write_or_verify_receipt(request.run_root / RESULT_NAME, result)
        _write_status(
            request,
            "completed" if passed else "failed",
            result_receipt_sha256=result["receipt_sha256"],
        )
        return 0 if passed else 2
    except BaseException as error:
        _write_status(
            request,
            "failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    request = DistributedTrainingRequest.load(args.request)
    if args.print_plan:
        source, source_result = verify_optimizer_step_prerequisite(request)
        prerequisite = prerequisite_receipt_fields(request, source, source_result)
        resolved = build_resolved_experiment(request, source)
        print(
            json.dumps(
                build_launch_plan(request, prerequisite, resolved),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return execute(request)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_launch_plan", "execute", "main"]
