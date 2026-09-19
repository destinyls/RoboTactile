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
    configure_dream_tac_integration,
    configure_ftp1_policy_integration,
    configure_n0_twam_integration,
    configure_n0_vtla_integration,
)
from robotactile_benchmark.integrations.doctor import diagnose_model_integration
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    CHECKPOINT_STEP as FTP1_CHECKPOINT_STEP,
)
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    TASK_RELEASES as FTP1_TASK_RELEASES,
)
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
from robotactile_benchmark.policies.dream_tac import DreamTacGripperMapping


@dataclass(frozen=True)
class IntegrationCommandResult:
    payload: Mapping[str, object]
    exit_code: int = 0


def add_integration_subcommands(subparsers: Any) -> None:
    """Attach model commands without enlarging the root CLI module."""

    integration = subparsers.add_parser(
        "integrations",
        help="inspect, configure, and diagnose first-class model integrations",
        description="Manage hash-bound external model integrations.",
    )
    actions = integration.add_subparsers(dest="integration_command", required=True)
    actions.add_parser("list", help="list pinned first-class integrations")
    validate = actions.add_parser(
        "validate",
        help="validate one integration config and optional checkout",
    )
    validate.add_argument(
        "--model",
        choices=("act", "dream_tac", "ftp1_policy", "n0_twam", "n0_vtla"),
        required=True,
        help="model ID",
    )
    validate.add_argument("--config", type=Path, help="integration config path")
    validate.add_argument("--checkout", type=Path, help="external source checkout")
    configure = actions.add_parser(
        "configure",
        help="generate hash-bound model configuration",
        description=(
            "Use a model-specific configure subcommand. The legacy "
            "'configure --model ...' syntax remains supported."
        ),
    )
    _add_legacy_configure_arguments(configure)
    configure_models = configure.add_subparsers(dest="configure_model")
    act = configure_models.add_parser(
        "act", help="configure official UniVTAC ACT artifacts"
    )
    _add_act_configure_arguments(act)
    dream_tac = configure_models.add_parser(
        "dream-tac",
        aliases=("dream_tac",),
        help="configure a source-bound user-supplied Dream-Tac checkpoint",
    )
    _add_dream_tac_configure_arguments(dream_tac)
    ftp1 = configure_models.add_parser(
        "ftp1-policy",
        aliases=("ftp1_policy",),
        help="configure one released FTP-1 UniVTAC checkpoint",
    )
    _add_ftp1_configure_arguments(ftp1)
    n0 = configure_models.add_parser(
        "n0-twam",
        aliases=("n0_twam",),
        help="configure an N0-TWAM artifact bundle",
    )
    _add_n0_configure_arguments(n0)
    vtla = configure_models.add_parser(
        "n0-vtla",
        aliases=("n0_vtla",),
        help="configure the released N0-VTLA insert_hole checkpoint",
    )
    _add_n0_vtla_configure_arguments(vtla)
    doctor = actions.add_parser(
        "doctor", help="check source, artifact, and transport readiness"
    )
    doctor.add_argument(
        "--model",
        choices=("act", "dream_tac", "ftp1_policy", "n0_twam", "n0_vtla"),
        required=True,
        help="model ID",
    )
    doctor.add_argument("--root", type=Path, help="deployment root")
    doctor.add_argument("--config", type=Path, help="integration config path")
    doctor.add_argument("--task", default="pull_out_key", help="UniVTAC task ID")
    doctor.add_argument(
        "--profile",
        choices=("univtac", "vision_only"),
        default="univtac",
        help="ACT observation profile",
    )

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
        choices=(
            "act",
            "dream-tac",
            "dream_tac",
            "ftp1-policy",
            "ftp1_policy",
            "n0-twam",
            "n0_twam",
            "n0-vtla",
            "n0_vtla",
        ),
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
    setup.add_argument("--instruction", help="exact Dream-Tac training prompt")
    setup.add_argument(
        "--experiment-config", help="exact upstream Dream-Tac experiment config"
    )
    setup.add_argument("--control-hz", type=float, help="bound policy control rate")
    setup.add_argument(
        "--gripper-mapping",
        choices=tuple(item.value for item in DreamTacGripperMapping),
        help="explicit upstream gripper calibration",
    )
    setup.add_argument("--gripper-threshold", type=float, default=0.5)
    setup.add_argument("--gripper-qpos-min", type=float)
    setup.add_argument("--gripper-qpos-max", type=float)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="execute one live request through a configured model",
    )
    evaluate.add_argument(
        "--model",
        choices=("act", "ftp1_policy", "n0_twam", "n0_vtla"),
        required=True,
        help="model ID",
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
    evaluate.add_argument("--n0-vtla-source-root", type=Path)
    evaluate.add_argument("--n0-vtla-endpoint", default="tcp://127.0.0.1:5557")
    evaluate.add_argument("--ftp1-source-root", type=Path)
    evaluate.add_argument("--ftp1-endpoint", default="tcp://127.0.0.1:5561")


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


def _add_dream_tac_configure_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_configure_arguments(parser)
    parser.add_argument("--bundle-root", type=Path, help="Dream-Tac artifact root")
    parser.add_argument("--checkpoint-root", type=Path, help="checkpoint directory")
    parser.add_argument("--dataset-stats", type=Path, help="dataset statistics JSON")
    parser.add_argument("--t5-embeddings", type=Path, help="T5 embeddings pickle")
    parser.add_argument("--task", required=True, help="exact target task ID")
    parser.add_argument(
        "--instruction", required=True, help="exact training/serving instruction"
    )
    parser.add_argument(
        "--experiment-config", required=True, help="upstream experiment config name"
    )
    parser.add_argument(
        "--control-hz", required=True, type=float, help="bound policy control rate"
    )
    parser.add_argument(
        "--gripper-mapping",
        required=True,
        choices=tuple(item.value for item in DreamTacGripperMapping),
        help="explicit resolution of the upstream gripper convention",
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-qpos-min", type=float)
    parser.add_argument("--gripper-qpos-max", type=float)


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


def _add_ftp1_configure_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_configure_arguments(parser)
    parser.add_argument("--bundle-root", type=Path, help="FTP-1 artifact root")
    parser.add_argument("--checkpoint-root", type=Path, help="task step-19999 root")
    parser.add_argument("--task", default="insert_hole", help="released task ID")


def _add_n0_vtla_configure_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_configure_arguments(parser)
    parser.add_argument("--bundle-root", type=Path, help="N0-VTLA artifact root")
    parser.add_argument("--checkpoint-root", type=Path, help="downloaded checkpoint")
    parser.set_defaults(task="insert_hole")


def _add_legacy_configure_arguments(parser: argparse.ArgumentParser) -> None:
    """Retain the 0.4 CLI without polluting the preferred model-specific help."""

    hidden = argparse.SUPPRESS
    parser.add_argument(
        "--model",
        choices=("act", "dream_tac", "ftp1_policy", "n0_twam", "n0_vtla"),
        help=hidden,
    )
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
    parser.add_argument("--dataset-stats", type=Path, help=hidden)
    parser.add_argument("--t5-embeddings", type=Path, help=hidden)
    parser.add_argument("--instruction", help=hidden)
    parser.add_argument("--experiment-config", help=hidden)
    parser.add_argument("--control-hz", type=float, help=hidden)
    parser.add_argument(
        "--gripper-mapping",
        choices=tuple(item.value for item in DreamTacGripperMapping),
        help=hidden,
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5, help=hidden)
    parser.add_argument("--gripper-qpos-min", type=float, help=hidden)
    parser.add_argument("--gripper-qpos-max", type=float, help=hidden)
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
                "reason": "select a model after configure",
                "status": "invalid_arguments",
            },
            exit_code=2,
        )
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    model_root = layout.model_artifacts / model
    default_config_root = model_root / "configs" / args.task
    if model == "act":
        default_config_root = default_config_root / args.profile
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
    elif model == "dream_tac":
        required = {
            "control_hz": args.control_hz,
            "experiment_config": args.experiment_config,
            "gripper_mapping": args.gripper_mapping,
            "instruction": args.instruction,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "Dream-Tac configuration requires explicit " + ", ".join(missing)
            )
        bundle_root = args.bundle_root or model_root
        result = configure_dream_tac_integration(
            bundle_root=bundle_root,
            checkpoint_root=args.checkpoint_root or bundle_root / "checkpoint",
            dataset_stats_path=(
                args.dataset_stats or bundle_root / "dataset_statistics_franka.json"
            ),
            t5_embeddings_path=(
                args.t5_embeddings or bundle_root / "t5_embeddings.pkl"
            ),
            task_id=args.task,
            instruction=args.instruction,
            experiment_config=args.experiment_config,
            control_hz=args.control_hz,
            gripper_mapping=DreamTacGripperMapping(args.gripper_mapping),
            gripper_threshold=args.gripper_threshold,
            gripper_qpos_min=args.gripper_qpos_min,
            gripper_qpos_max=args.gripper_qpos_max,
            manifest_path=manifest_output,
            config_path=config_output,
            device=args.device,
        )
    elif model == "ftp1_policy":
        bundle_root = args.bundle_root or model_root
        try:
            ftp1_release = FTP1_TASK_RELEASES[args.task]
        except KeyError as error:
            raise ValueError(
                f"FTP-1 has no released UniVTAC checkpoint for {args.task}"
            ) from error
        result = configure_ftp1_policy_integration(
            bundle_root=bundle_root,
            checkpoint_root=(
                args.checkpoint_root
                or bundle_root
                / ftp1_release.checkpoint_name
                / str(FTP1_CHECKPOINT_STEP)
            ),
            task_id=args.task,
            manifest_path=manifest_output,
            config_path=config_output,
            device=args.device,
        )
    elif model == "n0_twam":
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
    else:
        if args.task != "insert_hole":
            raise ValueError("released N0-VTLA supports only task=insert_hole")
        bundle_root = args.bundle_root or model_root
        result = configure_n0_vtla_integration(
            bundle_root=bundle_root,
            checkpoint_root=args.checkpoint_root or bundle_root / "checkpoint",
            manifest_path=manifest_output,
            config_path=config_output,
            device=args.device,
        )
    return IntegrationCommandResult(result.to_dict())


def _handle_setup(args: argparse.Namespace) -> IntegrationCommandResult:
    model = str(args.model).replace("-", "_")
    task = "insert_hole" if model == "n0_vtla" else args.task
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    receipt = initialize_deployment_layout(layout)
    configure_args = argparse.Namespace(
        model=model,
        configure_model=None,
        root=layout.root,
        device=args.device,
        artifact_root=None,
        upstream_root=None,
        task=task,
        profile=args.profile,
        bundle_root=None,
        base_root=None,
        checkpoint_root=None,
        dataset_stats=None,
        t5_embeddings=None,
        instruction=args.instruction,
        experiment_config=args.experiment_config,
        control_hz=args.control_hz,
        gripper_mapping=args.gripper_mapping,
        gripper_threshold=args.gripper_threshold,
        gripper_qpos_min=args.gripper_qpos_min,
        gripper_qpos_max=args.gripper_qpos_max,
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
            layout.model_artifacts
            / (
                f"act/configs/{task}/{args.profile}/integration_config.json"
                if model == "act"
                else f"{model}/configs/{task}/integration_config.json"
            )
        ),
        profile_id=args.profile,
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
    task = "insert_hole" if args.model == "n0_vtla" else args.task
    result = diagnose_model_integration(
        integration_id=args.model,
        layout=layout,
        config_path=args.config,
        task_id=task,
        profile_id=args.profile,
    )
    return IntegrationCommandResult(result.to_dict(), 0 if result.passed else 2)


def _handle_evaluate(args: argparse.Namespace) -> IntegrationCommandResult:
    if args.model == "ftp1_policy":
        from robotactile_benchmark.execution.loading import (
            load_live_univtac_request,
        )
        from robotactile_benchmark.execution.official_act import (
            official_act_live_summary,
        )
        from robotactile_benchmark.execution.official_ftp1_policy import (
            execute_official_ftp1_live_run,
        )
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_ftp1_policy_runtime_artifacts,
        )

        request = load_live_univtac_request(args.request)
        layout = DeploymentLayout(resolve_deployment_root(args.root))
        config_path = args.config or (
            layout.model_artifacts
            / f"ftp1_policy/configs/{request.task_id}/integration_config.json"
        )
        ftp1_runtime = resolve_ftp1_policy_runtime_artifacts(config_path)
        ftp1_artifact = execute_official_ftp1_live_run(
            request,
            manifest=ftp1_runtime.manifest,
            source_root=args.ftp1_source_root or layout.sources / "ftp1-policy",
            endpoint=args.ftp1_endpoint,
        )
        return IntegrationCommandResult(official_act_live_summary(ftp1_artifact))
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
        n0_runtime = resolve_n0_runtime_artifacts(config_path)
        n0_artifact = execute_official_n0_live_run(
            request,
            manifest=n0_runtime.manifest,
            source_root=args.n0_source_root or layout.sources / "N0-TWAM",
            host=args.n0_host,
            port=args.n0_port,
            api_key=os.environ.get("N0_TWAM_API_KEY"),
        )
        return IntegrationCommandResult(official_act_live_summary(n0_artifact))
    if args.model == "n0_vtla":
        from robotactile_benchmark.execution.loading import (
            load_live_univtac_request,
        )
        from robotactile_benchmark.execution.official_act import (
            official_act_live_summary,
        )
        from robotactile_benchmark.execution.official_n0_vtla import (
            execute_official_n0_vtla_live_run,
        )
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_n0_vtla_runtime_artifacts,
        )

        request = load_live_univtac_request(args.request)
        layout = DeploymentLayout(resolve_deployment_root(args.root))
        config_path = args.config or (
            layout.model_artifacts
            / "n0_vtla/configs/insert_hole/integration_config.json"
        )
        n0_vtla_runtime = resolve_n0_vtla_runtime_artifacts(config_path)
        n0_vtla_artifact = execute_official_n0_vtla_live_run(
            request,
            manifest=n0_vtla_runtime.manifest,
            source_root=args.n0_vtla_source_root or layout.sources / "N0-VTLA",
            endpoint=args.n0_vtla_endpoint,
        )
        return IntegrationCommandResult(official_act_live_summary(n0_vtla_artifact))
    if args.model != "act":
        raise ValueError(f"unsupported evaluation model: {args.model}")
    from robotactile_benchmark.execution.loading import load_live_univtac_request
    from robotactile_benchmark.execution.official_act import (
        execute_official_act_live_run,
        official_act_live_summary,
        official_act_profile,
    )

    request = load_live_univtac_request(args.request)
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
        config_path = args.config
        if config_path is None:
            canonical = (
                layout.model_artifacts
                / "act/configs"
                / request.task_id
                / official_act_profile(request.condition).value
                / "integration_config.json"
            )
            legacy = layout.model_artifacts / "act/integration_config.json"
            config_path = (
                legacy if legacy.is_file() and not canonical.exists() else canonical
            )
        resolved = resolve_act_runtime_artifacts(config_path)
        if request.act_device_name != resolved.device:
            raise ValueError("ACT request device does not match integration config")
        manifest = resolved.manifest
        if (
            manifest.task_id != request.task_id
            or manifest.profile is not official_act_profile(request.condition)
            or manifest.checkpoint_sha256 != request.checkpoint_sha256
            or manifest.config_sha256 != request.config_sha256
        ):
            raise ValueError("ACT request does not match integration artifact manifest")
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
    artifact = execute_official_act_live_run(
        request,
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
