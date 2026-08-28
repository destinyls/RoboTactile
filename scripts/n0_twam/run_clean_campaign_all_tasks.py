#!/usr/bin/env python3
"""Run every task shard in one N0 Clean campaign end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_EE_ACTION_EXECUTION_CONTRACTS,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    validate_n0_ee_action_execution_contract,
)
from robotactile_benchmark.clean_baseline import (
    CleanCampaignManifest,
    CleanCampaignProtocol,
    VerifiedAllTaskQualification,
    load_clean_campaign_manifest,
    verify_all_task_qualification,
)
from robotactile_benchmark.clean_baseline.io import write_canonical_no_clobber
from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_V3_SEMANTIC_VERSION,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.same_task_worker_protocol import (
    SAME_TASK_WORKER_CONTRACT,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_FRESH_WORKER_CONTRACT = "fresh_process_v1"


class ExecutionProfile(str, Enum):
    """Execution strictness without changing the frozen campaign manifest."""

    QUICK = "quick"
    DIAGNOSTIC = "diagnostic"
    CLAIM = "claim"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--execution-profile",
        choices=tuple(item.value for item in ExecutionProfile),
        help=(
            "quick/diagnostic skip mandatory qualification; claim requires "
            "source-bound qualification v3. Default follows the manifest protocol."
        ),
    )
    parser.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        help=(
            "artifact capture surface; defaults to metrics_only_v1 for quick, "
            "preview_v1 for diagnostic, and paper_full_v1 for claim"
        ),
    )
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n0-port", type=int, default=29601)
    parser.add_argument("--master-port", type=int, default=29988)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-new-trials-per-task", type=int)
    parser.add_argument("--integration-config-label")
    parser.add_argument(
        "--worker-contract",
        choices=(_FRESH_WORKER_CONTRACT, SAME_TASK_WORKER_CONTRACT),
        default=_FRESH_WORKER_CONTRACT,
        help="fresh Isaac process per episode or one persistent worker per task",
    )
    parser.add_argument(
        "--reset-equivalence-dir",
        type=Path,
        help=(
            "directory containing one formal reset-equivalence receipt named "
            "<task>.json; required for persistent pilot/paper campaigns and "
            "optional for persistent diagnostic campaigns"
        ),
    )
    parser.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        default=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        help=(
            "N0 EE executor; formal campaigns require the training-aligned "
            "60 Hz contract"
        ),
    )
    parser.add_argument("--continue-on-infrastructure-failure", action="store_true")
    parser.add_argument("--allow-partial-report", action="store_true")
    parser.add_argument("--publish-paper", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labeled_integration_config(
    layout: DeploymentLayout, task_id: str, label: str | None
) -> Path | None:
    if label is None:
        return None
    if _IDENTIFIER.fullmatch(label) is None:
        raise ValueError("integration_config_label contains unsupported characters")
    selected = (
        layout.model_artifacts
        / "n0_twam/configs"
        / f"{task_id}-{label}"
        / "integration_config.json"
    )
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("labeled integration config must be a regular file")
    return selected.absolute()


def _native_step_contract(action_execution_contract: str) -> str:
    if action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT:
        return N0_STOCK_EE_NATIVE_STEP_CONTRACT
    return FIXED_NATIVE_STEP_CONTRACT


def _resolve_execution_profile(
    requested: str | None,
    manifest: CleanCampaignManifest,
) -> ExecutionProfile:
    """Resolve the backward-compatible default from the frozen protocol."""

    if requested is not None:
        return ExecutionProfile(requested)
    if manifest.protocol_id is CleanCampaignProtocol.DIAGNOSTIC:
        return ExecutionProfile.DIAGNOSTIC
    return ExecutionProfile.CLAIM


def _target_trials_per_task(manifest: CleanCampaignManifest) -> tuple[int, ...]:
    if manifest.sampling is not None:
        return (manifest.sampling.target_valid_trials_per_task,)
    return tuple(Counter(item.task for item in manifest.trials).values())


def _resolve_capture_profile(
    requested: str | None,
    execution_profile: ExecutionProfile,
) -> LiveCaptureProfile:
    """Resolve capture independently while preserving strict claim evidence."""

    if requested is not None:
        return LiveCaptureProfile(requested)
    defaults = {
        ExecutionProfile.QUICK: LiveCaptureProfile.METRICS_ONLY,
        ExecutionProfile.DIAGNOSTIC: LiveCaptureProfile.PREVIEW,
        ExecutionProfile.CLAIM: LiveCaptureProfile.PAPER_FULL,
    }
    return defaults[execution_profile]


def _validate_capture_profile(
    *,
    execution_profile: ExecutionProfile,
    capture_profile: LiveCaptureProfile,
    worker_contract: str,
    publish_paper: bool,
) -> None:
    """Reject insufficient evidence before any model or simulator process starts."""

    if (
        execution_profile is ExecutionProfile.CLAIM
        and capture_profile is not LiveCaptureProfile.PAPER_FULL
    ):
        raise ValueError("claim execution requires paper_full_v1 capture")
    if publish_paper and capture_profile is not LiveCaptureProfile.PAPER_FULL:
        raise ValueError("--publish-paper requires paper_full_v1 capture")
    if (
        worker_contract == SAME_TASK_WORKER_CONTRACT
        and capture_profile is not LiveCaptureProfile.PAPER_FULL
    ):
        raise ValueError("light capture profiles do not support same_task_worker_v1")


def _validate_execution_profile(
    *,
    profile: ExecutionProfile,
    manifest: CleanCampaignManifest,
    worker_contract: str,
    qualification_path: Path | None,
    max_new_trials_per_task: int | None,
    publish_paper: bool,
) -> int | None:
    """Fail before process launch and return the effective per-task budget."""

    is_diagnostic = manifest.protocol_id is CleanCampaignProtocol.DIAGNOSTIC
    if profile in {ExecutionProfile.QUICK, ExecutionProfile.DIAGNOSTIC}:
        if not is_diagnostic:
            raise ValueError(
                f"{profile.value} execution requires a diagnostic_v1 manifest"
            )
        if publish_paper:
            raise ValueError("quick/diagnostic execution cannot publish paper results")
    elif is_diagnostic:
        raise ValueError("claim execution requires a pilot_v1 or paper_v1 manifest")

    if profile is ExecutionProfile.QUICK:
        if any(target != 1 for target in _target_trials_per_task(manifest)):
            raise ValueError("quick execution requires one target trial per task")
        if worker_contract != _FRESH_WORKER_CONTRACT:
            raise ValueError("quick execution requires fresh_process_v1")
        if max_new_trials_per_task is not None and max_new_trials_per_task > 1:
            raise ValueError("quick execution permits at most one new trial per task")
        return 1 if max_new_trials_per_task is None else max_new_trials_per_task

    if profile is ExecutionProfile.CLAIM and qualification_path is None:
        raise ValueError("claim execution requires --qualification")
    if publish_paper and manifest.protocol_id is not CleanCampaignProtocol.PAPER:
        raise ValueError("--publish-paper requires claim execution over paper_v1")
    return max_new_trials_per_task


def _optional_qualification(
    *,
    layout: DeploymentLayout,
    requested: Path | None,
    profile: ExecutionProfile,
    tasks: Sequence[str],
    action_execution_contract: str,
) -> VerifiedAllTaskQualification | None:
    """Verify an explicitly supplied qualification or allow diagnostic execution."""

    if requested is None:
        return None
    qualification = verify_all_task_qualification(layout.root, requested.absolute())
    if qualification.action_spec != "ee8_absolute":
        raise ValueError("N0 campaign qualification must use EE8 actions")
    if not set(tasks) <= set(qualification.tasks):
        raise ValueError("campaign task scope is not covered by qualification")
    _require_qualification_execution_contract(
        qualification,
        tasks,
        action_execution_contract,
    )
    if (
        profile is ExecutionProfile.CLAIM
        and qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION
    ):
        raise ValueError("claim execution requires source-bound qualification v3")
    return qualification


def _reset_equivalence_receipts(
    *,
    deployment_root: Path,
    tasks: Sequence[str],
    protocol: CleanCampaignProtocol,
    worker_contract: str,
    directory: Path | None,
) -> tuple[Path | None, Mapping[str, tuple[Path, str]]]:
    """Resolve task-local formal reset proofs without accepting symlinks."""

    if worker_contract == _FRESH_WORKER_CONTRACT:
        if directory is not None:
            raise ValueError(
                "reset_equivalence_dir requires the persistent worker contract"
            )
        return None, {}
    if worker_contract != SAME_TASK_WORKER_CONTRACT:
        raise ValueError("worker_contract is unsupported")
    if directory is None:
        if protocol is CleanCampaignProtocol.DIAGNOSTIC:
            return None, {}
        raise ValueError("pilot/paper persistent worker requires reset_equivalence_dir")
    root = deployment_root.resolve(strict=True)
    selected = directory.absolute()
    try:
        selected.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "reset_equivalence_dir must remain below the deployment root"
        ) from error
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError("reset_equivalence_dir must be a regular directory")
    resolved: dict[str, tuple[Path, str]] = {}
    for task_id in tasks:
        receipt = selected / f"{task_id}.json"
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError(f"reset-equivalence receipt is unavailable for {task_id}")
        resolved[task_id] = (receipt, _sha256_file(receipt))
    return selected, resolved


def _require_qualification_execution_contract(
    qualification: VerifiedAllTaskQualification,
    tasks: Sequence[str],
    action_execution_contract: str,
) -> None:
    """Fail before launch when qualification and selected cadence disagree."""

    expected_native = _native_step_contract(action_execution_contract)
    for task_id in tasks:
        try:
            task_index = qualification.tasks.index(task_id)
            source = qualification.task_source_bindings[task_index]
        except (IndexError, ValueError):
            if qualification.source_bound:
                raise ValueError(
                    "qualification source binding does not cover campaign task"
                ) from None
            continue
        if (
            source.action_execution_contract != action_execution_contract
            or source.native_step_contract != expected_native
        ):
            raise ValueError("qualification action execution contract mismatch")


def _run_json(command: Sequence[str]) -> tuple[Mapping[str, object], int]:
    completed = subprocess.run(
        tuple(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.decode("utf-8", errors="replace").strip()
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"subcommand did not return JSON: {output[-2000:]}"
        ) from error
    if not isinstance(document, Mapping):
        raise RuntimeError("subcommand JSON must be an object")
    return document, completed.returncode


def run_all_tasks(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    if _IDENTIFIER.fullmatch(args.run_id) is None:
        raise ValueError("run_id contains unsupported characters")
    if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpus) is None:
        raise ValueError("gpus must be a comma-separated integer list")
    if args.max_new_trials_per_task is not None and args.max_new_trials_per_task < 0:
        raise ValueError("max_new_trials_per_task must be non-negative")
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    manifest_path = args.manifest.absolute()
    manifest = load_clean_campaign_manifest(manifest_path)
    if manifest.policy_kind is not LivePolicyKind.N0:
        raise ValueError("N0 all-task runner requires an N0 campaign")
    if manifest.semantic_version == "2.0" and args.continue_on_infrastructure_failure:
        raise ValueError(
            "official campaigns replace only classified exceptions with a fresh server"
        )
    action_execution_contract = validate_n0_ee_action_execution_contract(
        args.action_execution_contract
    )
    if (
        manifest.protocol_id
        in {CleanCampaignProtocol.PILOT, CleanCampaignProtocol.PAPER}
        and action_execution_contract != N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    ):
        raise ValueError(
            f"{manifest.protocol_id.value} requires "
            f"{N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT}"
        )
    profile = _resolve_execution_profile(args.execution_profile, manifest)
    capture_profile = _resolve_capture_profile(args.capture_profile, profile)
    effective_max_new_trials = _validate_execution_profile(
        profile=profile,
        manifest=manifest,
        worker_contract=args.worker_contract,
        qualification_path=args.qualification,
        max_new_trials_per_task=args.max_new_trials_per_task,
        publish_paper=args.publish_paper,
    )
    _validate_capture_profile(
        execution_profile=profile,
        capture_profile=capture_profile,
        worker_contract=args.worker_contract,
        publish_paper=args.publish_paper,
    )
    tasks = tuple(dict.fromkeys(item.task for item in manifest.trials))
    qualification = _optional_qualification(
        layout=layout,
        requested=args.qualification,
        profile=profile,
        tasks=tasks,
        action_execution_contract=action_execution_contract,
    )
    if args.worker_contract == SAME_TASK_WORKER_CONTRACT and (
        qualification is None
        or qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION
    ):
        raise ValueError("same_task_worker_v1 requires source-bound qualification v3")
    reset_equivalence_dir, reset_equivalence = _reset_equivalence_receipts(
        deployment_root=layout.root,
        tasks=tasks,
        protocol=manifest.protocol_id,
        worker_contract=args.worker_contract,
        directory=args.reset_equivalence_dir,
    )
    receipt_path = (
        layout.outputs
        / "clean-campaigns"
        / manifest.campaign_id
        / "runs"
        / f"{args.run_id}.json"
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("campaign run receipt exists; select a new run_id")
    repository_root = Path(__file__).resolve().parents[2]
    shard_script = repository_root / "scripts/n0_twam/run_clean_task_shard.py"
    started = datetime.now(timezone.utc)
    shard_results: list[dict[str, object]] = []
    completed_task_ids: set[str] = set()
    overall_return_code = 0
    failure_reason: dict[str, object] | None = None
    for task_id in tasks:
        integration_config = _labeled_integration_config(
            layout, task_id, args.integration_config_label
        )
        command = [
            sys.executable,
            str(shard_script),
            "--root",
            str(layout.root),
            "--manifest",
            str(manifest_path),
            "--task",
            task_id,
            "--gpus",
            args.gpus,
            "--n0-port",
            str(args.n0_port),
            "--master-port",
            str(args.master_port),
            "--server-ready-timeout-s",
            str(args.server_ready_timeout_s),
            "--action-execution-contract",
            action_execution_contract,
            "--worker-contract",
            args.worker_contract,
            "--capture-profile",
            capture_profile.value,
        ]
        reset_receipt = reset_equivalence.get(task_id)
        if reset_receipt is not None:
            command.extend(
                (
                    "--reset-equivalence-receipt",
                    str(reset_receipt[0]),
                    "--reset-equivalence-receipt-sha256",
                    reset_receipt[1],
                )
            )
        if (
            qualification is not None
            and qualification.semantic_version == QUALIFICATION_V3_SEMANTIC_VERSION
        ):
            command.extend(
                (
                    "--qualification",
                    str(qualification.path),
                    "--qualification-sha256",
                    qualification.sha256,
                )
            )
        if integration_config is not None:
            command.extend(("--integration-config", str(integration_config)))
        if effective_max_new_trials is not None:
            command.extend(("--max-new-trials", str(effective_max_new_trials)))
        if args.continue_on_infrastructure_failure:
            command.append("--continue-on-infrastructure-failure")
        previous_remaining: int | None = None
        while True:
            try:
                result, return_code = _run_json(command)
                if result.get("capture_profile") != capture_profile.value:
                    raise RuntimeError("task shard capture profile mismatch")
                receipt = Path(str(result.get("receipt", "")))
                if receipt.is_symlink() or not receipt.is_file():
                    raise RuntimeError("task shard did not publish a receipt")
                receipt_relpath = receipt.relative_to(layout.root).as_posix()
                receipt_sha256 = _sha256_file(receipt)
            except (OSError, RuntimeError, ValueError) as error:
                failure_reason = {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "stage": "task_shard",
                    "task_id": task_id,
                }
                shard_results.append(
                    {
                        "error": failure_reason,
                        "receipt_relpath": None,
                        "receipt_sha256": None,
                        "return_code": 2,
                        "task_id": task_id,
                    }
                )
                overall_return_code = 2
                break
            runner_summary = result.get("runner_summary")
            if not isinstance(runner_summary, Mapping):
                runner_summary = {}
            shard_results.append(
                {
                    "candidate_pool_exhausted": runner_summary.get(
                        "candidate_pool_exhausted"
                    ),
                    "completed_for_task": runner_summary.get("completed_for_task"),
                    "fatal_error": runner_summary.get("fatal_error"),
                    "receipt_relpath": receipt_relpath,
                    "receipt_sha256": receipt_sha256,
                    "remaining_unattempted_candidates": runner_summary.get(
                        "remaining_unattempted_candidates"
                    ),
                    "replacement_requires_fresh_server": result.get(
                        "replacement_requires_fresh_server", False
                    ),
                    "return_code": return_code,
                    "reset_equivalence_receipt_relpath": (
                        None
                        if reset_receipt is None
                        else reset_receipt[0].relative_to(layout.root).as_posix()
                    ),
                    "reset_equivalence_receipt_sha256": (
                        None if reset_receipt is None else reset_receipt[1]
                    ),
                    "task_id": task_id,
                    "target_complete": runner_summary.get("target_complete"),
                }
            )
            if return_code == 0:
                if runner_summary.get("target_complete") is not True:
                    failure_reason = {
                        "error_type": "RunnerContractError",
                        "message": "successful task shard did not complete its target",
                        "stage": "task_shard",
                        "task_id": task_id,
                    }
                    overall_return_code = 2
                    break
                completed_task_ids.add(task_id)
                break
            if return_code == 3 and manifest.semantic_version == "2.0":
                remaining = runner_summary.get("remaining_unattempted_candidates")
                if (
                    type(remaining) is not int
                    or remaining < 1
                    or result.get("replacement_requires_fresh_server") is not True
                    or (
                        previous_remaining is not None
                        and remaining >= previous_remaining
                    )
                ):
                    failure_reason = {
                        "error_type": "CandidatePoolExhausted",
                        "message": (
                            "replacement requested without a smaller non-empty "
                            "candidate suffix"
                        ),
                        "stage": "replacement_gate",
                        "task_id": task_id,
                    }
                    overall_return_code = 2
                    break
                previous_remaining = remaining
                continue
            if return_code == 4:
                failure_reason = {
                    "error_type": "TrialBudgetStopped",
                    "message": "task shard stopped at the requested trial budget",
                    "stage": "trial_budget",
                    "task_id": task_id,
                }
                overall_return_code = 4
                break
            failure_reason = {
                "error_type": "TaskShardFailed",
                "message": "task shard returned a non-recoverable status",
                "stage": "task_shard",
                "task_id": task_id,
            }
            overall_return_code = 2
            break
        if (
            task_id not in completed_task_ids
            and not args.continue_on_infrastructure_failure
        ):
            break
    summary_path = (
        layout.outputs
        / "clean-campaigns"
        / manifest.campaign_id
        / "clean_baseline_summary.json"
    )
    report_result: Mapping[str, object] | None = None
    report_return_code: int | None = None
    if completed_task_ids == set(tasks):
        report_command = [
            sys.executable,
            "-m",
            "robotactile_benchmark.cli",
            "clean-campaign-report",
            "--root",
            str(layout.root),
            "--manifest",
            str(manifest_path),
            "--output",
            str(summary_path),
        ]
        if profile is not ExecutionProfile.CLAIM:
            report_command.append("--allow-compact-capture")
        if args.allow_partial_report:
            report_command.append("--allow-partial")
        report_result, report_return_code = _run_json(report_command)
        if report_return_code != 0:
            overall_return_code = 2
    paper_result: Mapping[str, object] | None = None
    paper_return_code: int | None = None
    if args.publish_paper and report_return_code == 0:
        if qualification is None:
            raise RuntimeError("paper publication requires qualification")
        paper_result, paper_return_code = _run_json(
            (
                sys.executable,
                "-m",
                "robotactile_benchmark.cli",
                "clean-campaign-publish",
                "--root",
                str(layout.root),
                "--summary",
                str(summary_path),
                "--qualification",
                str(qualification.path),
            )
        )
        if paper_return_code != 0:
            overall_return_code = 2
    receipt_document = {
        "action_execution_contract": action_execution_contract,
        "campaign_id": manifest.campaign_id,
        "campaign_manifest_sha256": manifest.sha256,
        "capture_profile": capture_profile.value,
        "execution_profile": profile.value,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure_reason": failure_reason,
        "gpus": args.gpus,
        "paper_result": paper_result,
        "paper_return_code": paper_return_code,
        "qualification_relpath": (
            None
            if qualification is None
            else qualification.path.relative_to(layout.root).as_posix()
        ),
        "qualification_sha256": (
            None if qualification is None else qualification.sha256
        ),
        "reset_equivalence_dir_relpath": (
            None
            if reset_equivalence_dir is None
            else reset_equivalence_dir.relative_to(layout.root).as_posix()
        ),
        "reset_equivalence_receipts": [
            {
                "relpath": path.relative_to(layout.root).as_posix(),
                "sha256": digest,
                "task_id": task_id,
            }
            for task_id, (path, digest) in reset_equivalence.items()
        ],
        "report_result": report_result,
        "report_return_code": report_return_code,
        "run_id": args.run_id,
        "semantic_version": "2.0",
        "shards": shard_results,
        "started_at_utc": started.isoformat(),
        "source_bound": (
            qualification is not None
            and qualification.semantic_version == QUALIFICATION_V3_SEMANTIC_VERSION
        ),
        "task_count": len(tasks),
        "tasks_completed": len(completed_task_ids),
        "worker_contract": args.worker_contract,
    }
    write_canonical_no_clobber(receipt_path, receipt_document)
    return (
        {
            "action_execution_contract": action_execution_contract,
            "campaign_id": manifest.campaign_id,
            "capture_profile": capture_profile.value,
            "execution_profile": profile.value,
            "failure_reason": failure_reason,
            "paper_return_code": paper_return_code,
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256_file(receipt_path),
            "report_return_code": report_return_code,
            "shards_completed": len(shard_results),
            "source_bound": (
                qualification is not None
                and qualification.semantic_version == QUALIFICATION_V3_SEMANTIC_VERSION
            ),
            "task_count": len(tasks),
            "tasks_completed": len(completed_task_ids),
            "worker_contract": args.worker_contract,
        },
        overall_return_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    result, return_code = run_all_tasks(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
