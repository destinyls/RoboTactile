#!/usr/bin/env python3
"""Freeze a deterministic multi-task official N0-TWAM Clean campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from robotactile_benchmark.backends.univtac_contracts import (
    load_univtac_task_registry,
)
from robotactile_benchmark.clean_baseline import (
    CleanCampaignProtocol,
    build_clean_campaign_manifest,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_UNIVTAC_SEED_DERIVATION,
    CleanCampaignSamplingSpec,
)
from robotactile_benchmark.clean_baseline.seeds import (
    derive_clean_campaign_seed,
    derive_univtac_task_seed,
    univtac_task_seed_start,
)
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.integrations.n0_twam.requests import (
    DEFAULT_N0_WATCHDOG_SECONDS_PER_ACTION,
    build_official_n0_clean_request,
    derive_n0_infrastructure_watchdog_timeout,
    write_official_n0_clean_request,
    write_official_n0_trial_set,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)

_CAMPAIGN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PROTOCOLS = ("diagnostic_v1", "pilot_v1", "paper_v1")
_SAMPLING_CONTRACTS = ("legacy_sha_v1", "univtac_official_v1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--protocol", choices=_PROTOCOLS, required=True)
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--trials-per-task", type=int, required=True)
    parser.add_argument("--sampling-contract", choices=_SAMPLING_CONTRACTS)
    parser.add_argument("--official-eval-seed", type=int, default=0)
    parser.add_argument("--replacement-reserve-per-task", type=int)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--integration-config", type=Path)
    parser.add_argument("--integration-config-label")
    parser.add_argument(
        "--wall-timeout-s",
        type=float,
        default=1800.0,
        help="minimum infrastructure-watchdog budget for each episode",
    )
    parser.add_argument(
        "--watchdog-seconds-per-action",
        type=float,
        default=DEFAULT_N0_WATCHDOG_SECONDS_PER_ACTION,
        help="conservative watchdog budget multiplied by the official action horizon",
    )
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument(
        "--initial-state-policy",
        choices=tuple(item.value for item in InitialStatePolicy),
        default=InitialStatePolicy.OFFICIAL_REPRODUCTION.value,
    )
    parser.add_argument("--request-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_campaign_seed(
    *,
    namespace: str,
    master_seed: int,
    task_id: str,
    ordinal: int,
    role: str,
) -> int:
    """Derive a non-zero signed-31-bit seed without mutable RNG state."""

    return derive_clean_campaign_seed(
        protocol_id=namespace,
        master_seed=master_seed,
        task_id=task_id,
        task_ordinal=ordinal,
        role=role,
    )


def derive_watchdog_timeout_s(
    *,
    requested_floor_s: float,
    action_horizon: int,
    seconds_per_action: float,
) -> float:
    """Freeze a non-scoring watchdog budget from the official action horizon."""

    return derive_n0_infrastructure_watchdog_timeout(
        requested_floor_s=requested_floor_s,
        action_horizon=action_horizon,
        seconds_per_action=seconds_per_action,
    )


def _validate_scope(
    protocol: str, tasks: tuple[str, ...], all_tasks: tuple[str, ...], count: int
) -> None:
    if type(count) is not int or count < 1:
        raise ValueError("trials_per_task must be a positive integer")
    if len(tasks) != len(set(tasks)) or set(tasks) - set(all_tasks):
        raise ValueError("tasks must be unique frozen UniVTAC task IDs")
    if protocol == "pilot_v1" and (tasks != all_tasks or count != 10):
        raise ValueError("pilot_v1 requires all eight tasks and exactly 10 trials")
    if protocol == "paper_v1" and (tasks != all_tasks or count != 100):
        raise ValueError("paper_v1 requires all eight tasks and exactly 100 trials")


def _sampling_plan(
    args: argparse.Namespace,
) -> tuple[int, int, CleanCampaignSamplingSpec | None]:
    """Resolve legacy generation or an explicit released-evaluator contract."""

    selected = args.sampling_contract
    if selected is None:
        if args.protocol == "paper_v1":
            raise ValueError(
                "paper_v1 requires explicit --sampling-contract univtac_official_v1"
            )
        selected = "legacy_sha_v1"
    if selected == "legacy_sha_v1":
        if args.master_seed is None or type(args.master_seed) is not int:
            raise ValueError("legacy_sha_v1 requires --master-seed")
        if args.master_seed < 0:
            raise ValueError("master_seed must be a non-negative integer")
        if args.replacement_reserve_per_task not in {None, 0}:
            raise ValueError("legacy_sha_v1 does not accept a replacement reserve")
        if args.protocol == "paper_v1":
            raise ValueError("paper_v1 requires univtac_official_v1 sampling")
        return args.master_seed, args.trials_per_task, None
    reserve = args.replacement_reserve_per_task
    if type(reserve) is not int or reserve < 0:
        raise ValueError(
            "univtac_official_v1 requires a non-negative --replacement-reserve-per-task"
        )
    if type(args.official_eval_seed) is not int or args.official_eval_seed < 0:
        raise ValueError("official_eval_seed must be a non-negative integer")
    if args.master_seed is not None and args.master_seed != args.official_eval_seed:
        raise ValueError("master_seed, when supplied, must equal official_eval_seed")
    candidate_count = args.trials_per_task + reserve
    sampling = CleanCampaignSamplingSpec(
        seed_protocol=CLEAN_UNIVTAC_SEED_DERIVATION,
        exception_handling="replace_exception_until_target_valid_v1",
        target_valid_trials_per_task=args.trials_per_task,
        candidate_trials_per_task=candidate_count,
        official_eval_seed=args.official_eval_seed,
        task_seed_start=univtac_task_seed_start(args.official_eval_seed),
        policy_seed_mode="same_as_task_seed_v1",
    )
    return args.official_eval_seed, candidate_count, sampling


def _relative(root: Path, path: Path, name: str) -> str:
    root_resolved = root.resolve(strict=True)
    candidate = path.resolve(strict=False)
    if root_resolved not in candidate.parents:
        raise ValueError(f"{name} must remain below the deployment root")
    return candidate.relative_to(root_resolved).as_posix()


def _select_integration_config(
    *,
    layout: DeploymentLayout,
    task_id: str,
    explicit_tasks: Sequence[str] | None,
    requested: Path | None,
    label: str | None,
) -> Path:
    if requested is not None and label is not None:
        raise ValueError(
            "integration_config and integration_config_label are mutually exclusive"
        )
    if requested is None:
        config_name = task_id if label is None else f"{task_id}-{label}"
        selected = (
            layout.model_artifacts
            / "n0_twam/configs"
            / config_name
            / "integration_config.json"
        )
        if label is None:
            return selected
        _relative(layout.root, selected, "integration_config")
        if selected.is_symlink() or not selected.is_file():
            raise ValueError("labeled integration_config must be a regular file")
        return selected
    if explicit_tasks is None or len(explicit_tasks) != 1:
        raise ValueError("integration_config requires exactly one explicit --task")
    selected = requested.absolute()
    _relative(layout.root, selected, "integration_config")
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("integration_config must be a regular non-symlink file")
    return selected


def generate_campaign(args: argparse.Namespace) -> dict[str, object]:
    if _CAMPAIGN_ID.fullmatch(args.campaign_id) is None:
        raise ValueError("campaign_id contains unsupported characters")
    if args.wall_timeout_s <= 0:
        raise ValueError("wall_timeout_s must be positive")
    if (
        not isinstance(args.watchdog_seconds_per_action, (int, float))
        or isinstance(args.watchdog_seconds_per_action, bool)
        or not 0.0 < float(args.watchdog_seconds_per_action) < float("inf")
    ):
        raise ValueError("watchdog_seconds_per_action must be positive and finite")
    if (
        args.integration_config_label is not None
        and _CAMPAIGN_ID.fullmatch(args.integration_config_label) is None
    ):
        raise ValueError("integration_config_label contains unsupported characters")
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    registry = load_univtac_task_registry()
    all_tasks = tuple(task.task_id for task in registry.tasks)
    tasks = all_tasks if args.tasks is None else tuple(args.tasks)
    _validate_scope(args.protocol, tasks, all_tasks, args.trials_per_task)
    campaign_seed, candidate_count, sampling = _sampling_plan(args)
    initial_state_policy = InitialStatePolicy(
        getattr(
            args,
            "initial_state_policy",
            InitialStatePolicy.OFFICIAL_REPRODUCTION.value,
        )
    )
    if args.integration_config is not None and (
        args.tasks is None or len(args.tasks) != 1
    ):
        raise ValueError("integration_config requires exactly one explicit --task")
    request_root = (
        args.request_root
        or layout.requests / "clean-campaigns" / args.campaign_id / "trials"
    ).absolute()
    artifact_root = (
        args.artifact_root
        or layout.artifacts / "live-univtac" / "clean-campaigns" / args.campaign_id
    ).absolute()
    output = (
        args.output
        or layout.requests
        / "clean-campaigns"
        / args.campaign_id
        / "campaign_manifest.json"
    ).absolute()
    for path, name in (
        (request_root, "request_root"),
        (artifact_root, "artifact_root"),
        (output, "output"),
    ):
        _relative(layout.root, path, name)

    seen_seeds: set[int] = set()
    request_paths: list[Path] = []
    task_watchdog_timeouts: dict[str, float] = {}
    for task_id in tasks:
        task = registry.task(task_id)
        watchdog_timeout_s = derive_watchdog_timeout_s(
            requested_floor_s=args.wall_timeout_s,
            action_horizon=task.action_horizon,
            seconds_per_action=args.watchdog_seconds_per_action,
        )
        task_watchdog_timeouts[task_id] = watchdog_timeout_s
        config_path = _select_integration_config(
            layout=layout,
            task_id=task_id,
            explicit_tasks=args.tasks,
            requested=args.integration_config,
            label=args.integration_config_label,
        )
        runtime = resolve_n0_runtime_artifacts(config_path)
        if runtime.manifest.task_id != task_id:
            raise ValueError("N0 integration config task identity mismatch")
        for ordinal in range(candidate_count):
            if sampling is None:
                initial_seed = derive_campaign_seed(
                    namespace=args.protocol,
                    master_seed=campaign_seed,
                    task_id=task_id,
                    ordinal=ordinal,
                    role="initial",
                )
                exogenous_seed = derive_campaign_seed(
                    namespace=args.protocol,
                    master_seed=campaign_seed,
                    task_id=task_id,
                    ordinal=ordinal,
                    role="exogenous",
                )
                if initial_seed in seen_seeds or exogenous_seed in seen_seeds:
                    raise RuntimeError("derived campaign seed collision")
                seen_seeds.update((initial_seed, exogenous_seed))
            else:
                initial_seed = derive_univtac_task_seed(
                    eval_seed=sampling.official_eval_seed,
                    candidate_ordinal=ordinal,
                )
                exogenous_seed = initial_seed
            trial_root = request_root / task_id / f"{ordinal:04d}"
            request_path = trial_root / "request.json"
            trial_set_path = trial_root / "trial_set_manifest.json"
            live_output = artifact_root / task_id / f"{ordinal:04d}"
            trial_set = write_official_n0_trial_set(
                manifest_path=trial_set_path,
                task_id=task_id,
                initial_seed=initial_seed,
                exogenous_seed=exogenous_seed,
                max_control_cycles=task.action_horizon,
                max_observation_steps=task.action_horizon + 1,
            )
            request = build_official_n0_clean_request(
                manifest=runtime.manifest,
                layout=layout,
                dataset_sha256=trial_set.sha256,
                initial_seed=initial_seed,
                exogenous_seed=exogenous_seed,
                max_control_cycles=task.action_horizon,
                max_observation_steps=task.action_horizon + 1,
                wall_timeout_s=watchdog_timeout_s,
                simulator_device=args.simulator_device,
                live_output_dir=live_output,
                initial_state_policy=initial_state_policy,
                wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
            )
            generated = write_official_n0_clean_request(request_path, request)
            if generated.request_file_sha256 != _sha256_file(request_path):
                raise RuntimeError("generated request file hash mismatch")
            request_paths.append(request_path)
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=request_paths,
        campaign_id=args.campaign_id,
        protocol_id=CleanCampaignProtocol(args.protocol),
        master_seed=campaign_seed,
        confidence_level=args.confidence_level,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        sampling=sampling,
    )
    created = write_clean_campaign_manifest(output, manifest)
    result: dict[str, object] = {
        "campaign_id": args.campaign_id,
        "campaign_manifest": str(output),
        "campaign_manifest_sha256": manifest.sha256,
        "created": created,
        "planned_trial_count": manifest.planned_trial_count,
        "protocol_id": args.protocol,
        "request_root": str(request_root),
        "initial_state_policy": initial_state_policy.value,
        "task_watchdog_timeouts_s": task_watchdog_timeouts,
        "wall_timeout_role": WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1.value,
        "watchdog_seconds_per_action": args.watchdog_seconds_per_action,
    }
    if sampling is not None:
        result.update(
            {
                "candidate_trials_per_task": sampling.candidate_trials_per_task,
                "sampling_contract": "univtac_official_v1",
                "target_valid_trials_per_task": (sampling.target_valid_trials_per_task),
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(generate_campaign(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
