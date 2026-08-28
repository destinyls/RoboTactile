"""CLI registration for frozen clean campaign manifests and summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robotactile_benchmark.clean_baseline.aggregation import (
    build_clean_baseline_summary,
    load_clean_artifact_inventory,
)
from robotactile_benchmark.clean_baseline.contracts import CleanCampaignProtocol
from robotactile_benchmark.clean_baseline.generation import (
    build_clean_campaign_manifest,
    discover_clean_request_paths,
)
from robotactile_benchmark.clean_baseline.io import (
    load_clean_baseline_summary,
    load_clean_campaign_manifest,
    write_clean_baseline_summary,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.clean_baseline.paper_bundle import (
    build_paper_result_bundle,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)

CleanCliResult = tuple[dict[str, object], int]


def add_clean_baseline_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    build = subparsers.add_parser(
        "clean-campaign-build",
        help="freeze strict clean live requests into one campaign manifest",
    )
    build.add_argument("--root", type=Path)
    build.add_argument("--campaign-id", required=True)
    build.add_argument(
        "--protocol",
        choices=tuple(item.value for item in CleanCampaignProtocol),
        required=True,
    )
    build.add_argument("--master-seed", type=int, required=True)
    sources = build.add_mutually_exclusive_group(required=True)
    sources.add_argument("--request", type=Path, action="append")
    sources.add_argument("--request-root", type=Path)
    build.add_argument("--confidence-level", type=float, default=0.95)
    build.add_argument("--bootstrap-resamples", type=int, default=10_000)
    build.add_argument("--bootstrap-seed", type=int, default=20260823)
    build.add_argument("--output", type=Path)

    report = subparsers.add_parser(
        "clean-campaign-report",
        help="strictly aggregate clean artifacts and successful attempt receipts",
    )
    report.add_argument("--root", type=Path)
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--output", type=Path)
    report.add_argument(
        "--allow-partial",
        action="store_true",
        help="write an explicitly non-claimable summary with missing/invalid trials",
    )
    report.add_argument(
        "--allow-compact-capture",
        action="store_true",
        help=(
            "accept metrics-only or preview artifacts for diagnostic_v1 reporting; "
            "paper and pilot reports remain full-capture only"
        ),
    )
    publish = subparsers.add_parser(
        "clean-campaign-publish",
        help="bind a Clean summary to all-task qualification and paper exports",
    )
    publish.add_argument("--root", type=Path)
    publish.add_argument("--summary", type=Path, required=True)
    publish.add_argument("--qualification", type=Path, required=True)
    publish.add_argument("--output-dir", type=Path)


def handle_clean_baseline_command(args: argparse.Namespace) -> Optional[CleanCliResult]:
    if args.command == "clean-campaign-build":
        return _handle_build(args)
    if args.command == "clean-campaign-report":
        return _handle_report(args)
    if args.command == "clean-campaign-publish":
        return _handle_publish(args)
    return None


def _handle_build(args: argparse.Namespace) -> CleanCliResult:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    if args.request_root is not None:
        request_paths = discover_clean_request_paths(args.request_root)
    else:
        request_paths = tuple(args.request or ())
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=request_paths,
        campaign_id=args.campaign_id,
        protocol_id=CleanCampaignProtocol(args.protocol),
        master_seed=args.master_seed,
        confidence_level=args.confidence_level,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = args.output or (
        layout.requests
        / "clean-campaigns"
        / manifest.campaign_id
        / "campaign_manifest.json"
    )
    output = _contained_output(layout.root, output, "requests")
    created = write_clean_campaign_manifest(output, manifest)
    return (
        {
            "campaign_id": manifest.campaign_id,
            "campaign_manifest": str(output),
            "campaign_manifest_sha256": manifest.sha256,
            "created": created,
            "planned_trial_count": manifest.planned_trial_count,
            "protocol_id": manifest.protocol_id.value,
            "protocol_allows_paper_claim": manifest.protocol_allows_paper_claim,
        },
        0,
    )


def _handle_report(args: argparse.Namespace) -> CleanCliResult:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    manifest = load_clean_campaign_manifest(args.manifest)
    inventory = load_clean_artifact_inventory(
        layout.root,
        manifest,
        allow_compact=args.allow_compact_capture,
    )
    if (
        inventory.missing_trials or inventory.protocol_invalid_trials
    ) and not args.allow_partial:
        return (
            {
                "campaign_id": manifest.campaign_id,
                "campaign_manifest_sha256": manifest.sha256,
                "missing_artifact_count": len(inventory.missing_trials),
                "protocol_invalid_count": len(inventory.protocol_invalid_trials),
                "status": "incomplete_no_summary_written",
            },
            2,
        )
    summary = build_clean_baseline_summary(manifest, inventory)
    output = args.output or (
        layout.outputs
        / "clean-campaigns"
        / manifest.campaign_id
        / "clean_baseline_summary.json"
    )
    output = _contained_output(layout.root, output, "outputs")
    created = write_clean_baseline_summary(output, summary)
    return (
        {
            "campaign_id": summary.campaign_id,
            "created": created,
            "eligible_success_rate": summary.eligible_success_rate,
            "loaded_artifact_count": summary.loaded_artifact_count,
            "macro_task_success_rate": summary.macro_task_success_rate,
            "missing_artifact_count": summary.missing_artifact_count,
            "paper_claim_eligible": summary.paper_claim_eligible,
            "protocol_invalid_count": summary.protocol_invalid_count,
            "statistically_complete": summary.statistically_complete,
            "summary": str(output),
            "summary_sha256": summary.sha256,
        },
        0,
    )


def _handle_publish(args: argparse.Namespace) -> CleanCliResult:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    summary = load_clean_baseline_summary(args.summary)
    output = args.output_dir or (
        layout.outputs / "clean-campaigns" / summary.campaign_id / "paper"
    )
    output = _contained_directory(layout.root, output, "outputs")
    bundle = build_paper_result_bundle(
        deployment_root=layout.root,
        summary_path=args.summary,
        qualification_path=args.qualification,
        output_directory=output,
    )
    return (
        {
            "campaign_id": summary.campaign_id,
            "csv": str(bundle.csv_path),
            "latex": str(bundle.latex_path),
            "paper_claim_blockers": list(bundle.blockers),
            "paper_claim_eligible": bundle.paper_claim_eligible,
            "paper_result": str(bundle.receipt_path),
            "paper_result_sha256": bundle.receipt_sha256,
        },
        0 if bundle.paper_claim_eligible else 2,
    )


def _contained_output(root: Path, path: Path, prefix: str) -> Path:
    selected = Path(path).absolute()
    try:
        relative = selected.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "clean campaign output must remain below deployment root"
        ) from error
    if not relative.parts or relative.parts[0] != prefix:
        raise ValueError(f"clean campaign output must remain below {prefix}/")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("clean campaign output cannot traverse a symlink")
    return selected


def _contained_directory(root: Path, path: Path, prefix: str) -> Path:
    selected = Path(path).absolute()
    marker = selected / "paper_result.json"
    _contained_output(root, marker, prefix)
    if selected.is_symlink():
        raise ValueError("clean campaign output directory cannot be a symlink")
    return selected


__all__ = [
    "add_clean_baseline_subcommands",
    "handle_clean_baseline_command",
]
