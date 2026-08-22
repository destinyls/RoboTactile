"""CLI bridge for no-allocation live deployment preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.preflight import (
    run_live_preflight,
    write_live_preflight_receipt,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_act_runtime_artifacts,
)


def add_live_preflight_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "preflight-live",
        help="verify a live request and resources without allocating Isaac",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--act-checkout", type=Path)
    parser.add_argument("--official-act-artifact-root", type=Path)
    parser.add_argument("--stats-sha256")
    parser.add_argument("--encoder-sha256")
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--output", type=Path)


def handle_live_preflight_command(
    args: argparse.Namespace,
) -> Optional[Tuple[Mapping[str, object], int]]:
    if args.command != "preflight-live":
        return None
    legacy = (
        args.official_act_artifact_root,
        args.stats_sha256,
        args.encoder_sha256,
    )
    needs_layout = (
        args.config is not None
        or any(value is None for value in legacy)
        or args.act_checkout is None
        or args.isaac_python is None
        or args.output is None
    )
    layout = (
        DeploymentLayout(resolve_deployment_root(args.root)) if needs_layout else None
    )
    if args.config is not None or any(value is None for value in legacy):
        assert layout is not None
        config_path = args.config or (
            layout.model_artifacts / "act/integration_config.json"
        )
        resolved = resolve_act_runtime_artifacts(config_path)
        artifact_root = resolved.artifact_root
        stats_sha256 = resolved.stats_sha256
        encoder_sha256 = resolved.encoder_sha256
    else:
        artifact_root = args.official_act_artifact_root
        stats_sha256 = args.stats_sha256
        encoder_sha256 = args.encoder_sha256
    act_checkout = args.act_checkout
    isaac_python = args.isaac_python
    output = args.output
    if layout is not None:
        act_checkout = act_checkout or layout.sources / "WorldArena"
        isaac_python = isaac_python or layout.runtime / "isaac-sim-4.5.0/python.sh"
        output = output or layout.artifacts / "preflight" / (
            f"{Path(args.request).stem}.json"
        )
    assert artifact_root is not None
    assert stats_sha256 is not None
    assert encoder_sha256 is not None
    assert act_checkout is not None
    assert isaac_python is not None
    assert output is not None
    receipt = run_live_preflight(
        args.request,
        act_checkout=act_checkout,
        artifact_root=artifact_root,
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
        isaac_python=isaac_python,
    )
    write_live_preflight_receipt(output, receipt)
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
