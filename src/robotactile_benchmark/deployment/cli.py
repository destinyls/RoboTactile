"""CLI wiring for repo-contained deployment management."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from robotactile_benchmark.deployment.doctor import (
    DEPLOYMENT_PROFILES,
    diagnose_deployment,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.deployment.migration import plan_legacy_migration


@dataclass(frozen=True)
class DeploymentCommandResult:
    payload: Mapping[str, object]
    exit_code: int = 0


def add_deployment_subcommands(subparsers: Any) -> None:
    deployment = subparsers.add_parser(
        "deployment",
        help="inspect and manage the repo-contained writable workspace",
        description="Resolve, initialize, diagnose, or plan migration of deployment state.",
    )
    actions = deployment.add_subparsers(dest="deployment_command", required=True)
    for name in ("show", "init"):
        command = actions.add_parser(
            name,
            help=(
                "show resolved paths without writing"
                if name == "show"
                else "create the canonical directory tree and receipt"
            ),
        )
        command.add_argument("--root", type=Path, help="deployment root")
        if name == "show":
            command.add_argument(
                "--json",
                action="store_true",
                help="compatibility flag; output is always canonical JSON",
            )
    doctor = actions.add_parser(
        "doctor", help="check canonical paths for one deployment profile"
    )
    doctor.add_argument("--root", type=Path, help="deployment root")
    doctor.add_argument(
        "--profile",
        choices=DEPLOYMENT_PROFILES,
        default="core",
        help="readiness profile (default: core)",
    )
    migrate = actions.add_parser(
        "migrate-plan", help="write no data; inventory a legacy deployment"
    )
    migrate.add_argument("--root", type=Path, help="new deployment root")
    migrate.add_argument(
        "--legacy-root", type=Path, required=True, help="existing legacy root"
    )


def handle_deployment_command(
    args: argparse.Namespace,
) -> Optional[DeploymentCommandResult]:
    if args.command != "deployment":
        return None
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    if args.deployment_command == "show":
        return DeploymentCommandResult(layout.to_dict())
    if args.deployment_command == "init":
        receipt = initialize_deployment_layout(layout)
        return DeploymentCommandResult(
            {
                **layout.to_dict(),
                "evidence_level": receipt.evidence_level,
                "receipt_path": str(layout.receipt_path),
                "receipt_sha256": receipt.receipt_sha256,
                "simulator_execution_claimed": False,
            }
        )
    if args.deployment_command == "migrate-plan":
        plan = plan_legacy_migration(args.legacy_root, layout)
        return DeploymentCommandResult(plan.to_dict())
    result = diagnose_deployment(layout, args.profile)
    return DeploymentCommandResult(result.to_dict(), 0 if result.passed else 2)


__all__ = [
    "DeploymentCommandResult",
    "add_deployment_subcommands",
    "handle_deployment_command",
]
