"""CLI integration for expert-to-live trajectory alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robotactile_benchmark.recorded.alignment import build_expert_alignment_report
from robotactile_benchmark.recorded.alignment_io import (
    load_expert_hdf5_trajectory,
    load_live_alignment_trajectory,
    write_alignment_report,
)


def add_expert_alignment_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the source-bound expert alignment command."""

    parser = subparsers.add_parser(
        "expert-alignment",
        help="align one verified live trace to recorded UniVTAC expert trajectories",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--expert-hdf5", type=Path, action="append", required=True, dest="experts"
    )
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--sample-count", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)


def handle_expert_alignment_command(
    args: argparse.Namespace,
) -> Optional[dict[str, object]]:
    """Run expert alignment or return None for another subcommand."""

    if args.command != "expert-alignment":
        return None
    live, executed = load_live_alignment_trajectory(args.artifact)
    experts = tuple(
        load_expert_hdf5_trajectory(path, args.task) for path in args.experts
    )
    report = build_expert_alignment_report(
        task_id=args.task,
        live=live,
        experts=experts,
        executed_action_count=executed,
        sample_count=args.sample_count,
    )
    output_sha256 = write_alignment_report(args.output, report)
    return {
        "diagnostic_codes": report["diagnostic_codes"],
        "evidence_level": report["evidence_level"],
        "expert_count": len(experts),
        "output": str(args.output),
        "output_sha256": output_sha256,
        "selected_expert_id": report["selected_expert_id"],
        "task_id": args.task,
    }


__all__ = ["add_expert_alignment_subcommand", "handle_expert_alignment_command"]
