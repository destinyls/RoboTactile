"""CLI registration for N0/UniVTAC protocol-alignment reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robotactile_benchmark.clean_baseline import (
    load_clean_artifact_inventory,
    load_clean_campaign_manifest,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.protocol_alignment.io import (
    load_n0_paper_reference,
    render_protocol_alignment_markdown,
    sha256_file,
    write_protocol_alignment_report,
    write_text_no_clobber,
)
from robotactile_benchmark.protocol_alignment.report import (
    build_n0_protocol_alignment_report,
)

ProtocolAlignmentCliResult = tuple[dict[str, object], int]


def add_protocol_alignment_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register one strict report command that never starts an episode."""

    parser = subparsers.add_parser(
        "n0-protocol-alignment",
        help=(
            "compare verified N0 Clean artifacts with the public UniVTAC result "
            "under the source-bound RoboTactile release contract"
        ),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)


def handle_protocol_alignment_command(
    args: argparse.Namespace,
) -> Optional[ProtocolAlignmentCliResult]:
    """Build one report or return None for another CLI subcommand."""

    if args.command != "n0-protocol-alignment":
        return None
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    manifest = load_clean_campaign_manifest(args.manifest)
    inventory = load_clean_artifact_inventory(layout.root, manifest)
    reference = load_n0_paper_reference(args.reference)
    report = build_n0_protocol_alignment_report(manifest, inventory, reference)
    output_dir = layout.outputs / "clean-campaigns" / manifest.campaign_id / "parity"
    output = _contained(
        layout.root, args.output or output_dir / "protocol_alignment.json"
    )
    markdown = _contained(
        layout.root,
        args.markdown_output or output_dir / "protocol_alignment.md",
    )
    created = write_protocol_alignment_report(output, report)
    markdown_created = write_text_no_clobber(
        markdown, render_protocol_alignment_markdown(report)
    )
    payload = {
        "campaign_id": manifest.campaign_id,
        "completed_task_count": report["completed_task_count"],
        "created": created,
        "evaluation_semantics": report["evaluation_semantics"],
        "go_for_integration_diagnostic": report["go_for_integration_diagnostic"],
        "go_for_paper_comparison": report["go_for_paper_comparison"],
        "markdown": str(markdown),
        "markdown_created": markdown_created,
        "markdown_sha256": sha256_file(markdown),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "success_count": report["success_count"],
    }
    return payload, 0 if report["go_for_paper_comparison"] is True else 2


def _contained(root: Path, path: Path) -> Path:
    selected = Path(path).absolute()
    try:
        relative = selected.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "protocol alignment output must stay below deployment root"
        ) from error
    if not relative.parts or relative.parts[0] != "outputs":
        raise ValueError("protocol alignment output must stay below outputs/")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("protocol alignment output cannot traverse a symlink")
    return selected


__all__ = [
    "add_protocol_alignment_subcommand",
    "handle_protocol_alignment_command",
]
