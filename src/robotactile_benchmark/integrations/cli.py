"""Unified CLI surface for registered model integrations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.integrations.configuration import (
    configure_act_integration,
    configure_n0_twam_integration,
)
from robotactile_benchmark.integrations.doctor import diagnose_model_integration
from robotactile_benchmark.integrations.n0_twam.artifacts import serve_task_id
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

    integration = subparsers.add_parser(
        "integrations",
        help="inspect, configure, and diagnose ACT or N0-TWAM",
        description="Manage hash-bound external model integrations.",
    )
    actions = integration.add_subparsers(dest="integration_command", required=True)
    actions.add_parser("list", help="list pinned first-class integrations")
    validate = actions.add_parser(
        "validate",
        help="validate one integration config and optional checkout",
    )
    validate.add_argument(
        "--model", choices=("act", "n0_twam"), required=True, help="model ID"
    )
    validate.add_argument("--config", type=Path, help="integration config path")
    validate.add_argument("--checkout", type=Path, help="external source checkout")
    configure = actions.add_parser(
        "configure",
        help="generate hash-bound model configuration",
        description=(
            "Use 'configure act' or 'configure n0-twam'. The legacy "
            "'configure --model ...' syntax remains supported."
        ),
    )
    _add_legacy_configure_arguments(configure)
    configure_models = configure.add_subparsers(dest="configure_model")
    act = configure_models.add_parser(
        "act", help="configure official UniVTAC ACT artifacts"
    )
    _add_act_configure_arguments(act)
    n0 = configure_models.add_parser(
        "n0-twam",
        aliases=("n0_twam",),
        help="configure an N0-TWAM artifact bundle",
    )
    _add_n0_configure_arguments(n0)
    doctor = actions.add_parser(
        "doctor", help="check source, artifact, and transport readiness"
    )
    doctor.add_argument(
        "--model", choices=("act", "n0_twam"), required=True, help="model ID"
    )
    doctor.add_argument("--root", type=Path, help="deployment root")
    doctor.add_argument("--config", type=Path, help="integration config path")
    doctor.add_argument("--task", default="pull_out_key", help="UniVTAC task ID")

    setup = subparsers.add_parser(
        "setup",
        help="initialize and diagnose one canonical model deployment",
        description=(
            "Initialize the deployment tree, materialize canonical model config "
            "when its files exist, and run the fail-closed integration doctor."
        ),
    )
    setup.add_argument(
        "--model",
        choices=("act", "n0-twam", "n0_twam"),
        required=True,
        help="model integration to prepare",
    )
    setup.add_argument("--root", type=Path, help="deployment root")
    setup.add_argument("--device", default="cuda:0", help="inference device")
    setup.add_argument("--task", default="pull_out_key", help="ACT task ID")
    setup.add_argument(
        "--profile",
        choices=("univtac", "vision_only"),
        default="univtac",
        help="ACT observation profile",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="execute one live request through a configured model",
    )
    evaluate.add_argument(
        "--model", choices=("act", "n0_twam"), required=True, help="model ID"
    )
    evaluate.add_argument("--request", type=Path, required=True, help="request JSON")
    evaluate.add_argument("--root", type=Path, help="deployment root")
    evaluate.add_argument("--config", type=Path, help="integration config path")
    evaluate.add_argument("--official-act-artifact-root", type=Path)
    evaluate.add_argument("--stats-sha256")
    evaluate.add_argument("--encoder-sha256")
    evaluate.add_argument("--n0-source-root", type=Path)
    evaluate.add_argument("--n0-host", default="127.0.0.1")
    evaluate.add_argument("--n0-port", type=int, default=29601)


def _add_common_configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, help="deployment root")
    parser.add_argument("--device", default="cuda:0", help="inference device")
    parser.add_argument("--manifest-output", type=Path, help="artifact manifest path")
    parser.add_argument(
        "--integration-config-output", type=Path, help="integration config path"
    )


def _add_act_configure_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_configure_arguments(parser)
    parser.add_argument("--artifact-root", type=Path, help="ACT artifact root")
    parser.add_argument("--upstream-root", type=Path, help="UniVTAC checkout")
    parser.add_argument("--task", default="pull_out_key", help="frozen task ID")
    parser.add_argument(
        "--profile",
        choices=("univtac", "vision_only"),
        default="univtac",
        help="observation profile",
    )


def _add_n0_configure_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_configure_arguments(parser)
    parser.add_argument("--bundle-root", type=Path, help="N0-TWAM artifact root")
    parser.add_argument("--task", default="pull_out_key", help="UniVTAC task ID")
    parser.add_argument("--base-root", type=Path, help="pinned base model root")
    parser.add_argument("--checkpoint-root", type=Path, help="pinned post-train root")
    parser.add_argument("--serve-bundle-root", type=Path, help="assembled serve bundle")
    parser.add_argument("--serve-pool-root", type=Path, help="task serve-pool root")
    parser.add_argument("--checkpoint", type=Path, help="transformer safetensors")
    parser.add_argument("--model-config", type=Path, help="model config file")
    parser.add_argument("--train-meta", type=Path, help="training metadata file")
    parser.add_argument("--normalizer", type=Path, help="normalizer file")
    parser.add_argument("--prompt-manifest", type=Path, help="released prompts JSON")
    parser.add_argument(
        "--serve-bundle-manifest", type=Path, help="assembled bundle manifest"
    )
    parser.add_argument("--serve-info", type=Path, help="served meta/info.json")
    parser.add_argument("--serve-tasks", type=Path, help="served meta/tasks.jsonl")


def _add_legacy_configure_arguments(parser: argparse.ArgumentParser) -> None:
    """Retain the 0.4 CLI without polluting the preferred model-specific help."""

    hidden = argparse.SUPPRESS
    parser.add_argument("--model", choices=("act", "n0_twam"), help=hidden)
    parser.add_argument("--root", type=Path, help=hidden)
    parser.add_argument("--device", default="cuda:0", help=hidden)
    parser.add_argument("--artifact-root", type=Path, help=hidden)
    parser.add_argument("--upstream-root", type=Path, help=hidden)
    parser.add_argument("--task", default="pull_out_key", help=hidden)
    parser.add_argument(
        "--profile", choices=("univtac", "vision_only"), default="univtac", help=hidden
    )
    parser.add_argument("--bundle-root", type=Path, help=hidden)
    parser.add_argument("--base-root", type=Path, help=hidden)
    parser.add_argument("--checkpoint-root", type=Path, help=hidden)
    parser.add_argument("--serve-bundle-root", type=Path, help=hidden)
    parser.add_argument("--serve-pool-root", type=Path, help=hidden)
    parser.add_argument("--checkpoint", type=Path, help=hidden)
    parser.add_argument("--model-config", type=Path, help=hidden)
    parser.add_argument("--train-meta", type=Path, help=hidden)
    parser.add_argument("--normalizer", type=Path, help=hidden)
    parser.add_argument("--prompt-manifest", type=Path, help=hidden)
    parser.add_argument("--serve-bundle-manifest", type=Path, help=hidden)
    parser.add_argument("--serve-info", type=Path, help=hidden)
    parser.add_argument("--serve-tasks", type=Path, help=hidden)
    parser.add_argument("--manifest-output", type=Path, help=hidden)
    parser.add_argument("--integration-config-output", type=Path, help=hidden)


def _selected_configure_model(args: argparse.Namespace) -> Optional[str]:
    selected = getattr(args, "configure_model", None) or getattr(args, "model", None)
    return None if selected is None else str(selected).replace("-", "_")


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


def _handle_configure(args: argparse.Namespace) -> IntegrationCommandResult:
    model = _selected_configure_model(args)
    if model is None:
        return IntegrationCommandResult(
            {
                "reason": "select 'act' or 'n0-twam' after configure",
                "status": "invalid_arguments",
            },
            exit_code=2,
        )
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    model_root = layout.model_artifacts / model
    default_config_root = (
        model_root if model == "act" else model_root / "configs" / args.task
    )
    manifest_output = (
        args.manifest_output or default_config_root / "artifact_manifest.json"
    )
    config_output = (
        args.integration_config_output
        or default_config_root / "integration_config.json"
    )
    if model == "act":
        from robotactile_benchmark.policies.univtac_official_act import (
            OfficialACTProfile,
        )

        result = configure_act_integration(
            task_id=args.task,
            profile=OfficialACTProfile(args.profile),
            artifact_root=args.artifact_root or model_root,
            upstream_root=args.upstream_root or layout.sources / "UniVTAC",
            manifest_path=manifest_output,
            config_path=config_output,
            device=args.device,
        )
    else:
        bundle_root = args.bundle_root or model_root
        base_root = args.base_root or bundle_root / "base"
        checkpoint_root = args.checkpoint_root or bundle_root / "univtac-delta"
        serve_bundle_root = args.serve_bundle_root or bundle_root / "serve-bundle"
        serve_pool_root = (
            args.serve_pool_root or bundle_root / "serve-pools" / args.task
        )
        task_key = serve_task_id(args.task)
        result = configure_n0_twam_integration(
            bundle_root=bundle_root,
            task_id=args.task,
            base_root=base_root,
            checkpoint_root=checkpoint_root,
            serve_bundle_root=serve_bundle_root,
            serve_pool_root=serve_pool_root,
            checkpoint_path=(
                args.checkpoint
                or checkpoint_root / "transformer/diffusion_pytorch_model.safetensors"
            ),
            model_config_path=(
                args.model_config or checkpoint_root / "transformer/config.json"
            ),
            train_meta_path=(args.train_meta or checkpoint_root / "train_meta.json"),
            normalizer_path=(
                args.normalizer or serve_pool_root / "norm_stat_per_robot.json"
            ),
            prompt_manifest_path=(
                args.prompt_manifest or checkpoint_root / "norm/PROMPTS.json"
            ),
            serve_bundle_manifest_path=(
                args.serve_bundle_manifest
                or serve_pool_root / "serve_bundle_manifest.json"
            ),
            serve_info_path=(
                args.serve_info
                or serve_pool_root / "train" / task_key / "meta/info.json"
            ),
            serve_tasks_path=(
                args.serve_tasks
                or serve_pool_root / "train" / task_key / "meta/tasks.jsonl"
            ),
            manifest_path=manifest_output,
            config_path=config_output,
            device=args.device,
        )
    return IntegrationCommandResult(result.to_dict())


def _handle_setup(args: argparse.Namespace) -> IntegrationCommandResult:
    model = str(args.model).replace("-", "_")
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    receipt = initialize_deployment_layout(layout)
    configure_args = argparse.Namespace(
        model=model,
        configure_model=None,
        root=layout.root,
        device=args.device,
        artifact_root=None,
        upstream_root=None,
        task=args.task,
        profile=args.profile,
        bundle_root=None,
        base_root=None,
        checkpoint_root=None,
        serve_bundle_root=None,
        serve_pool_root=None,
        checkpoint=None,
        model_config=None,
        train_meta=None,
        normalizer=None,
        prompt_manifest=None,
        serve_bundle_manifest=None,
        serve_info=None,
        serve_tasks=None,
        manifest_output=None,
        integration_config_output=None,
    )
    try:
        configured = _handle_configure(configure_args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        configuration: Mapping[str, object] = {
            "detail": str(error),
            "status": "blocked",
        }
    else:
        configuration = {**configured.payload, "status": "configured"}
    diagnosed = diagnose_model_integration(
        integration_id=model,
        layout=layout,
        config_path=(
            None
            if model == "act"
            else layout.model_artifacts
            / f"n0_twam/configs/{args.task}/integration_config.json"
        ),
    )
    payload = {
        "configuration": configuration,
        "deployment_root": str(layout.root),
        "doctor": diagnosed.to_dict(),
        "evidence_level": "model_setup_diagnostic_only_v1",
        "layout_receipt_sha256": receipt.receipt_sha256,
        "live_inference_claimed": False,
        "model": model,
        "next_action": (
            "resolve failed doctor checks and rerun the same setup command"
            if not diagnosed.passed
            else "run preflight-live with a frozen request"
        ),
    }
    return IntegrationCommandResult(payload, 0 if diagnosed.passed else 2)


def _handle_doctor(args: argparse.Namespace) -> IntegrationCommandResult:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    result = diagnose_model_integration(
        integration_id=args.model,
        layout=layout,
        config_path=args.config,
        task_id=args.task,
    )
    return IntegrationCommandResult(result.to_dict(), 0 if result.passed else 2)


def _handle_evaluate(args: argparse.Namespace) -> IntegrationCommandResult:
    if args.model == "n0_twam":
        import os

        from robotactile_benchmark.execution.loading import (
            load_live_univtac_request,
        )
        from robotactile_benchmark.execution.official_act import (
            official_act_live_summary,
        )
        from robotactile_benchmark.execution.official_n0 import (
            execute_official_n0_live_run,
        )
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_n0_runtime_artifacts,
        )

        request = load_live_univtac_request(args.request)
        layout = DeploymentLayout(resolve_deployment_root(args.root))
        config_path = args.config or (
            layout.model_artifacts
            / f"n0_twam/configs/{request.task_id}/integration_config.json"
        )
        runtime = resolve_n0_runtime_artifacts(config_path)
        artifact = execute_official_n0_live_run(
            request,
            manifest=runtime.manifest,
            source_root=args.n0_source_root or layout.sources / "N0-TWAM",
            host=args.n0_host,
            port=args.n0_port,
            api_key=os.environ.get("N0_TWAM_API_KEY"),
        )
        return IntegrationCommandResult(official_act_live_summary(artifact))
    required = {
        "official_act_artifact_root": args.official_act_artifact_root,
        "stats_sha256": args.stats_sha256,
        "encoder_sha256": args.encoder_sha256,
    }
    if args.config is not None or any(value is None for value in required.values()):
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_act_runtime_artifacts,
        )

        layout = DeploymentLayout(resolve_deployment_root(args.root))
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
    assert artifact_root is not None
    assert stats_sha256 is not None
    assert encoder_sha256 is not None
    from robotactile_benchmark.execution.loading import load_live_univtac_request
    from robotactile_benchmark.execution.official_act import (
        execute_official_act_live_run,
        official_act_live_summary,
    )

    artifact = execute_official_act_live_run(
        load_live_univtac_request(args.request),
        artifact_root=artifact_root,
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
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
        if args.integration_command == "validate":
            return _handle_validate(args)
        if args.integration_command == "configure":
            return _handle_configure(args)
        return _handle_doctor(args)
    if args.command == "setup":
        return _handle_setup(args)
    if args.command == "evaluate":
        return _handle_evaluate(args)
    return None
