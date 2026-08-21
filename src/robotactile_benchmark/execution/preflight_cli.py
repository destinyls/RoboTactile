"""CLI bridge for no-allocation live deployment preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from robotactile_benchmark.execution.preflight import (
    run_live_preflight,
    write_live_preflight_receipt,
)


def add_live_preflight_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("preflight-live")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--act-checkout", type=Path, required=True)
    parser.add_argument("--official-act-artifact-root", type=Path, required=True)
    parser.add_argument("--stats-sha256", required=True)
    parser.add_argument("--encoder-sha256", required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def handle_live_preflight_command(
    args: argparse.Namespace,
) -> Optional[Tuple[Mapping[str, object], int]]:
    if args.command != "preflight-live":
        return None
    receipt = run_live_preflight(
        args.request,
        act_checkout=args.act_checkout,
        artifact_root=args.official_act_artifact_root,
        stats_sha256=args.stats_sha256,
        encoder_sha256=args.encoder_sha256,
        isaac_python=args.isaac_python,
    )
    write_live_preflight_receipt(args.output, receipt)
    payload = {
        "evidence_level": receipt.evidence_level,
        "failed_checks": [item.check_id for item in receipt.checks if not item.passed],
        "passed": receipt.passed,
        "receipt_sha256": receipt.receipt_sha256,
        "simulator_execution_claimed": receipt.simulator_execution_claimed,
        "simulator_qualification_claimed": (receipt.simulator_qualification_claimed),
        "task_id": receipt.task_id,
    }
    return payload, 0 if receipt.passed else 2


__all__ = ["add_live_preflight_parser", "handle_live_preflight_command"]
