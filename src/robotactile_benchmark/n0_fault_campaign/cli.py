"""Public CLI for N0-TWAM Clean/Faulted robustness campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import (
    N0_EE_ACTION_EXECUTION_CONTRACTS,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.fault_timing import (
    DEFAULT_EARLY_ONSET_MAX_INDEX,
    EARLY_RANDOM_ONSET_MODE,
    FIXED_FAULT_ONSET_MODE,
)
from robotactile_benchmark.n0_fault_campaign.aggregation import (
    aggregate_n0_fault_campaign,
)
from robotactile_benchmark.n0_fault_campaign.generation import (
    N0FaultCampaignGenerationSpec,
    generate_n0_fault_campaign_bundle,
)
from robotactile_benchmark.n0_fault_campaign.report_adapter import (
    load_n0_campaign_outcomes,
    reporting_spec_for_campaign,
)
from robotactile_benchmark.n0_fault_campaign.report_bundle import (
    write_n0_fault_report_bundle,
)
from robotactile_benchmark.n0_fault_campaign.runner import run_n0_fault_pair


def add_n0_fault_campaign_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register request generation and one-pair execution commands."""

    generate = subparsers.add_parser(
        "generate-n0-fault-campaign",
        help="expand canonical N0 Clean requests into a paired fault campaign",
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--campaign-id", required=True)
    generate.add_argument(
        "--base-clean-request", type=Path, action="append", required=True
    )
    generate.add_argument(
        "--operator",
        action="append",
        choices=sorted(CORE_OPERATOR_IDS),
        help="repeat to select operators; default is all 14 contract operators",
    )
    generate.add_argument(
        "--severity",
        action="append",
        type=int,
        choices=range(1, 6),
        help="repeat to select severities; default is S1-S5",
    )
    generate.add_argument("--operator-seed-master", type=int, required=True)
    generate.add_argument("--fault-start-index", type=int)
    generate.add_argument("--fault-stop-index", type=int, required=True)
    generate.add_argument(
        "--fault-onset-mode",
        choices=(FIXED_FAULT_ONSET_MODE, EARLY_RANDOM_ONSET_MODE),
        default=FIXED_FAULT_ONSET_MODE,
    )
    generate.add_argument(
        "--fault-onset-max-index",
        type=int,
        default=DEFAULT_EARLY_ONSET_MAX_INDEX,
    )
    diagnostic = generate.add_mutually_exclusive_group()
    diagnostic.add_argument(
        "--diagnostic-stress-max",
        action="store_true",
        help=(
            "use the non-paper diagnostic_stress_max_v1 profile; requires --severity 5"
        ),
    )
    diagnostic.add_argument(
        "--diagnostic-tactile-null",
        action="store_true",
        help=(
            "run the non-paper tactile_null_black_frame_v1 ablation; "
            "requires only F1, severity 5, start 0, and a full-horizon window"
        ),
    )
    diagnostic.add_argument(
        "--diagnostic-observed-tactile-absence",
        action="store_true",
        help=(
            "run non-paper A1 full-horizon structural absence using the "
            "training-consistent N0 tactile-drop path"
        ),
    )
    generate.add_argument(
        "--rest-reference",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="strict measured rest-reference artifact for one selected task",
    )

    run = subparsers.add_parser(
        "run-n0-fault-campaign",
        help="execute one pair shard from a generated N0 fault campaign",
    )
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--pair-key")
    run.add_argument("--integration-config", type=Path, required=True)
    run.add_argument("--n0-source-root", type=Path, required=True)
    run.add_argument("--n0-host", default="127.0.0.1")
    run.add_argument("--n0-port", type=int, default=29601)
    run.add_argument("--receipt-dir", type=Path)
    run.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        default=LiveCaptureProfile.PAPER_FULL.value,
    )
    run.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        default=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    )

    report = subparsers.add_parser(
        "report-n0-fault-campaign",
        help="strict-load one N0 campaign and publish deterministic JSON/CSV results",
    )
    report.add_argument("--campaign-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--bootstrap-seed", type=int, required=True)


def handle_n0_fault_campaign_command(
    args: argparse.Namespace,
) -> Optional[dict[str, object]]:
    """Handle an N0 campaign command or return ``None`` when unrelated."""

    if args.command == "generate-n0-fault-campaign":
        if args.fault_onset_mode == FIXED_FAULT_ONSET_MODE:
            if args.fault_start_index is None:
                raise ValueError("fixed_v1 requires --fault-start-index")
            start_index = args.fault_start_index
        else:
            if args.fault_start_index is not None:
                raise ValueError(
                    "early_random_onset_v1 does not accept --fault-start-index"
                )
            start_index = 0
        severity_levels = tuple(args.severity or range(1, 6))
        if args.diagnostic_observed_tactile_absence:
            if args.operator not in (None, ["A1_stream_absence"]):
                raise ValueError("observed-tactile absence accepts only A1 realization")
            severity_registry = DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID
            operator_ids = ("A1_stream_absence",)
            severity_levels = tuple(args.severity or (5,))
        elif args.diagnostic_tactile_null:
            if args.operator not in (None, ["F1_global_response_drift"]):
                raise ValueError("diagnostic tactile-null accepts only F1 realization")
            severity_registry = DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID
            operator_ids = ("F1_global_response_drift",)
            severity_levels = tuple(args.severity or (5,))
        else:
            severity_registry = (
                DIAGNOSTIC_STRESS_MAX_REGISTRY_ID
                if args.diagnostic_stress_max
                else SEVERITY_REGISTRY_ID
            )
            operator_ids = tuple(args.operator or sorted(CORE_OPERATOR_IDS))
        generation_spec = N0FaultCampaignGenerationSpec(
            campaign_id=args.campaign_id,
            base_clean_request_paths=tuple(args.base_clean_request),
            operator_ids=operator_ids,
            severity_levels=severity_levels,
            operator_seed_master=args.operator_seed_master,
            fault_start_index=start_index,
            fault_stop_index=args.fault_stop_index,
            rest_reference_artifacts=_rest_reference_mapping(args.rest_reference),
            severity_registry=severity_registry,
            fault_onset_mode=args.fault_onset_mode,
            fault_onset_max_index=args.fault_onset_max_index,
        )
        status, generation_loaded = generate_n0_fault_campaign_bundle(
            args.output, generation_spec
        )
        payload: dict[str, object] = {
            "status": status,
            "campaign_id": generation_loaded.manifest.campaign_id,
            "campaign_root": str(generation_loaded.root),
            "campaign_manifest_sha256": generation_loaded.manifest.sha256,
            "generation_receipt_file_sha256": generation_loaded.receipt_file_sha256,
            "pair_count": generation_loaded.manifest.pair_count,
            "cell_count": generation_loaded.manifest.cell_count,
            "live_request_count": generation_loaded.manifest.live_request_count,
            "unsupported_contract_count": (
                generation_loaded.manifest.unsupported_contract_count
            ),
        }
        if severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID:
            payload.update(
                diagnostic_ablation="tactile_null_black_frame_v1",
                paper_s1_s5_claim=False,
            )
        elif severity_registry == DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID:
            payload.update(
                diagnostic_ablation="observed_tactile_absent_v1",
                paper_s1_s5_claim=False,
            )
        return payload
    if args.command == "run-n0-fault-campaign":
        receipt = run_n0_fault_pair(
            args.campaign_root,
            integration_config=args.integration_config,
            n0_source_root=args.n0_source_root,
            host=args.n0_host,
            port=args.n0_port,
            pair_key=args.pair_key,
            capture_profile=LiveCaptureProfile(args.capture_profile),
            action_execution_contract=args.action_execution_contract,
            receipt_dir=args.receipt_dir,
        )
        payload = receipt.to_dict()
        payload["pair_run_receipt_sha256"] = receipt.sha256
        return payload
    if args.command == "report-n0-fault-campaign":
        report_loaded = load_n0_campaign_outcomes(args.campaign_root)
        report_spec = reporting_spec_for_campaign(
            report_loaded,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary = aggregate_n0_fault_campaign(report_loaded.outcomes, report_spec)
        result = write_n0_fault_report_bundle(args.output, summary)
        payload = {
            "status": result.publication_status,
            "campaign_id": report_loaded.campaign_id,
            "campaign_manifest_sha256": report_loaded.campaign_manifest_sha256,
            "campaign_complete": report_loaded.complete,
            "missing_live_cell_count": len(report_loaded.missing_live_cell_sha256s),
            "statistically_complete": summary.statistically_complete,
            "completeness_blockers": list(summary.completeness_blockers),
            "macro_clean_success_rate": summary.macro_clean_success_rate,
            "macro_fault_success_rate": summary.macro_fault_success_rate,
            "macro_degradation_clean_minus_fault": summary.macro_degradation,
            "report_receipt_sha256": result.receipt.sha256,
            "report_receipt_file_sha256": result.receipt_file_sha256,
            "output": str(args.output.absolute()),
        }
        if report_spec.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID:
            payload.update(
                diagnostic_ablation="tactile_null_black_frame_v1",
                tactile_null_success_rate=summary.macro_fault_success_rate,
                tactile_null_degradation_clean_minus_null=(summary.macro_degradation),
                paper_s1_s5_claim=False,
            )
        elif report_spec.severity_registry == (
            DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID
        ):
            payload.update(
                diagnostic_ablation="observed_tactile_absent_v1",
                observed_tactile_absence_success_rate=(
                    summary.macro_fault_success_rate
                ),
                observed_tactile_absence_degradation_clean_minus_absence=(
                    summary.macro_degradation
                ),
                paper_s1_s5_claim=False,
            )
        return payload
    return None


def _rest_reference_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError("rest-reference must use TASK=PATH")
        task, path = value.split("=", 1)
        if not task or task.strip() != task or not path:
            raise ValueError("rest-reference must use a non-empty TASK=PATH")
        if task in result:
            raise ValueError(f"duplicate rest-reference task: {task}")
        result[task] = Path(path).expanduser().absolute()
    return result


__all__ = [
    "add_n0_fault_campaign_subcommands",
    "handle_n0_fault_campaign_command",
]
