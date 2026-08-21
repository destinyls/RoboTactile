"""Unified CLI surface for registered model integrations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.integrations.registry import (
    get_model_integration,
    list_model_integrations,
    load_model_integration_config,
)


@dataclass(frozen=True)
class IntegrationCommandResult:
    payload: Mapping[str, object]
    exit_code: int = 0


def add_integration_subcommands(subparsers: Any) -> None:
    """Attach model commands without enlarging the root CLI module."""

    integration = subparsers.add_parser("integrations")
    actions = integration.add_subparsers(dest="integration_command", required=True)
    actions.add_parser("list")
    validate = actions.add_parser("validate")
    validate.add_argument("--model", choices=("act", "n0_twam"), required=True)
    validate.add_argument("--config", type=Path)
    validate.add_argument("--checkout", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--model", choices=("act", "n0_twam"), required=True)
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--official-act-artifact-root", type=Path)
    evaluate.add_argument("--stats-sha256")
    evaluate.add_argument("--encoder-sha256")


def _public_spec(integration_id: str) -> dict[str, object]:
    spec = get_model_integration(integration_id)
    pin = load_integration_lock().by_id(spec.external_pin_id)
    return {
        "display_name": spec.display_name,
        "external_commit": pin.commit_sha,
        "integration_id": spec.integration_id,
        "license_spdx": spec.license_spdx,
        "matched_no_touch": spec.capabilities.matched_no_touch,
        "release_ready": pin.release_ready,
        "stateful_commit": spec.capabilities.stateful_commit,
        "structural_absence": spec.capabilities.structural_absence,
        "supported_conditions": list(spec.capabilities.supported_conditions),
    }


def _handle_validate(args: argparse.Namespace) -> IntegrationCommandResult:
    config = load_model_integration_config(args.model, args.config)
    pin = load_integration_lock().by_id(config.spec.external_pin_id)
    if args.checkout is not None:
        verify_external_checkout(pin, args.checkout)
    return IntegrationCommandResult(
        {
            "config_valid": True,
            "external_commit": pin.commit_sha,
            "integration_id": config.integration_id,
            "license_spdx": config.spec.license_spdx,
            "release_ready": pin.release_ready,
        }
    )


def _handle_evaluate(args: argparse.Namespace) -> IntegrationCommandResult:
    if args.model == "n0_twam":
        return IntegrationCommandResult(
            {
                "backend_effects": 0,
                "integration_id": "n0_twam",
                "policy_effects": 0,
                "reason": "a registered live N0 transport must be injected via API",
                "status": "unsupported_contract",
            },
            exit_code=2,
        )
    required = {
        "official_act_artifact_root": args.official_act_artifact_root,
        "stats_sha256": args.stats_sha256,
        "encoder_sha256": args.encoder_sha256,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"ACT evaluate is missing required arguments: {missing}")
    from robotactile_benchmark.execution.loading import load_live_univtac_request
    from robotactile_benchmark.execution.official_act import (
        execute_official_act_live_run,
        official_act_live_summary,
    )

    artifact = execute_official_act_live_run(
        load_live_univtac_request(args.request),
        artifact_root=args.official_act_artifact_root,
        stats_sha256=args.stats_sha256,
        encoder_sha256=args.encoder_sha256,
    )
    return IntegrationCommandResult(official_act_live_summary(artifact))


def handle_integration_command(
    args: argparse.Namespace,
) -> Optional[IntegrationCommandResult]:
    """Handle only the commands owned by the integration subsystem."""

    if args.command == "integrations":
        if args.integration_command == "list":
            return IntegrationCommandResult(
                {
                    "integrations": [
                        _public_spec(spec.integration_id)
                        for spec in list_model_integrations()
                    ]
                }
            )
        return _handle_validate(args)
    if args.command == "evaluate":
        return _handle_evaluate(args)
    return None


__all__ = [
    "IntegrationCommandResult",
    "add_integration_subcommands",
    "handle_integration_command",
]
