"""Public CLI for official ACT Clean/Faulted robustness campaigns."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact

from .generation import (
    ACTFaultCampaignGenerationSpec,
    generate_act_fault_campaign_bundle,
)
from .io import load_act_fault_campaign_bundle
from .reporting import build_act_fault_campaign_report
from .reset_reference import (
    DEFAULT_ACT_RESET_QPOS_ATOL,
    build_act_reset_reference_from_artifact,
    write_act_reset_reference,
)
from .reset_trajectory_calibration import capture_act_reset_trajectory
from .runner import run_act_fault_task

ACT_FAULT_REPORT_FILENAME = "act_fault_campaign_report.json"


def add_act_fault_campaign_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ACT campaign generation, execution, and reporting commands."""

    generate = subparsers.add_parser(
        "generate-act-fault-campaign",
        help="generate one-severity official ACT Clean/Faulted campaign",
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--campaign-id", required=True)
    generate.add_argument(
        "--base-clean-request", type=Path, action="append", required=True
    )
    generate.add_argument(
        "--artifact-manifest",
        action="append",
        required=True,
        metavar="TASK=PATH",
    )
    generate.add_argument("--deployment-root", type=Path, required=True)
    generate.add_argument("--severity", type=int, choices=range(1, 6), required=True)
    generate.add_argument("--operator-seed-master", type=int, required=True)
    generate.add_argument("--fault-start-index", type=int, required=True)
    generate.add_argument("--fault-stop-index", type=int, required=True)
    generate.add_argument(
        "--rest-reference",
        action="append",
        required=True,
        metavar="TASK=PATH",
    )
    generate.add_argument(
        "--reset-reference",
        type=Path,
        action="append",
        required=True,
        help="successful Clean-derived reset reference JSON; repeat per pair",
    )
    generate.add_argument(
        "--reset-trajectory",
        type=Path,
        action="append",
        required=True,
        help="qualified dense pre-move trajectory JSON; repeat per pair",
    )

    reset_reference = subparsers.add_parser(
        "build-act-reset-reference",
        help="derive a source-bound ACT reset reference from successful Clean",
    )
    reset_reference.add_argument("--source-artifact", type=Path, required=True)
    reset_reference.add_argument("--output", type=Path, required=True)
    reset_reference.add_argument(
        "--qpos-atol",
        type=float,
        default=DEFAULT_ACT_RESET_QPOS_ATOL,
    )

    reset_trajectory = subparsers.add_parser(
        "capture-act-reset-trajectory",
        help="capture one policy-free, reference-qualified dense reset trajectory",
    )
    reset_trajectory.add_argument("--base-clean-request", type=Path, required=True)
    reset_trajectory.add_argument("--reset-reference", type=Path, required=True)
    reset_trajectory.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser(
        "run-act-fault-campaign",
        help="run one ACT task/seed as Clean plus 12 executable faults",
    )
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--integration-config", type=Path, required=True)
    run.add_argument("--task")
    run.add_argument("--pair-key")
    run.add_argument("--receipt-dir", type=Path)
    run.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        default=LiveCaptureProfile.PAPER_FULL.value,
    )

    report = subparsers.add_parser(
        "report-act-fault-campaign",
        help="report ACT Clean/Faulted SR, degradation, and retention",
    )
    report.add_argument("--campaign-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)


def handle_act_fault_campaign_command(
    args: argparse.Namespace,
) -> Optional[dict[str, object]]:
    """Handle an ACT campaign command or return ``None`` when unrelated."""

    if args.command == "generate-act-fault-campaign":
        spec = ACTFaultCampaignGenerationSpec(
            campaign_id=args.campaign_id,
            base_clean_request_paths=tuple(args.base_clean_request),
            artifact_manifest_paths=_task_path_mapping(
                args.artifact_manifest, "artifact-manifest"
            ),
            deployment_layout=DeploymentLayout(
                args.deployment_root.expanduser().absolute()
            ),
            operator_ids=tuple(sorted(CORE_OPERATOR_IDS)),
            severity_levels=(args.severity,),
            operator_seed_master=args.operator_seed_master,
            fault_start_index=args.fault_start_index,
            fault_stop_index=args.fault_stop_index,
            rest_reference_artifacts=_task_path_mapping(
                args.rest_reference, "rest-reference"
            ),
            reset_reference_artifact_paths=tuple(args.reset_reference),
            reset_trajectory_artifact_paths=tuple(args.reset_trajectory),
        )
        status, loaded = generate_act_fault_campaign_bundle(args.output, spec)
        return {
            "campaign_id": loaded.manifest.campaign_id,
            "campaign_manifest_sha256": loaded.manifest.sha256,
            "campaign_root": str(loaded.root),
            "cell_count": loaded.manifest.cell_count,
            "generation_receipt_file_sha256": loaded.receipt_file_sha256,
            "live_request_count": loaded.manifest.live_request_count,
            "pair_count": loaded.manifest.pair_count,
            "status": status,
            "unsupported_contract_count": (loaded.manifest.unsupported_contract_count),
        }
    if args.command == "build-act-reset-reference":
        artifact = load_live_univtac_artifact(args.source_artifact)
        reference = build_act_reset_reference_from_artifact(
            artifact,
            qpos_atol=args.qpos_atol,
        )
        created = write_act_reset_reference(args.output, reference)
        return {
            "output": str(args.output.expanduser().absolute()),
            "reference_sha256": reference.sha256,
            "source_artifact_root_sha256": (reference.source_artifact_root_sha256),
            "source_result_sha256": reference.source_result_sha256,
            "status": "created" if created else "already_present",
            "task_id": reference.task_id,
        }
    if args.command == "capture-act-reset-trajectory":
        status, trajectory, reset_receipt_sha256 = capture_act_reset_trajectory(
            clean_request_path=args.base_clean_request,
            reset_reference_path=args.reset_reference,
            output_path=args.output,
        )
        return {
            "capture_mode": trajectory.capture_mode,
            "output": str(args.output.expanduser().absolute()),
            "policy_action_count": 0,
            "policy_loaded": False,
            "reset_receipt_sha256": reset_receipt_sha256,
            "status": status,
            "task_id": trajectory.task_id,
            "trajectory_sha256": trajectory.sha256,
        }
    if args.command == "run-act-fault-campaign":
        receipt = run_act_fault_task(
            args.campaign_root,
            integration_config=args.integration_config,
            task=args.task,
            pair_key=args.pair_key,
            capture_profile=LiveCaptureProfile(args.capture_profile),
            receipt_dir=args.receipt_dir,
        )
        return {
            **receipt.to_dict(),
            "task_run_receipt_sha256": receipt.sha256,
        }
    if args.command == "report-act-fault-campaign":
        loaded = load_act_fault_campaign_bundle(args.campaign_root)
        artifacts: dict[object, object] = {}
        for cell in loaded.manifest.cells:
            if cell.artifact_relpath is None:
                continue
            artifact_path = loaded.root / cell.artifact_relpath
            if artifact_path.is_dir() and any(artifact_path.iterdir()):
                artifacts[cell.artifact_relpath] = load_live_univtac_artifact(
                    artifact_path
                )
        report = build_act_fault_campaign_report(loaded.manifest, artifacts)
        output = args.output.expanduser().absolute()
        report_path = output / ACT_FAULT_REPORT_FILENAME
        payload = canonical_json_bytes(report.to_dict())
        status = _publish_bytes(report_path, payload)
        return {
            "campaign_id": report.campaign_id,
            "campaign_manifest_sha256": report.campaign_manifest_sha256,
            "clean_success_rate": report.clean_success_rate,
            "complete": report.complete,
            "completeness_blockers": list(report.completeness_blockers),
            "delta_success_rate": report.delta_success_rate,
            "faulted_success_rate": report.faulted_success_rate,
            "output": str(report_path),
            "report_file_sha256": hashlib.sha256(payload).hexdigest(),
            "report_sha256": report.sha256,
            "retention": report.retention,
            "status": status,
        }
    return None


def _task_path_mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError(f"{label} must use TASK=PATH")
        task, path = value.split("=", 1)
        if not task or task.strip() != task or not path:
            raise ValueError(f"{label} must use a non-empty TASK=PATH")
        if task in result:
            raise ValueError(f"duplicate {label} task: {task}")
        result[task] = Path(path).expanduser().absolute()
    return result


def _publish_bytes(target: Path, payload: bytes) -> str:
    if target.is_symlink():
        raise FileExistsError("ACT report path cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different ACT report")
        return "already_present"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError:
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(
                "concurrent ACT report publication disagrees"
            ) from None
        return "already_present"
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


__all__ = [
    "ACT_FAULT_REPORT_FILENAME",
    "add_act_fault_campaign_subcommands",
    "handle_act_fault_campaign_command",
]
