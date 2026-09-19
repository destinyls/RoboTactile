#!/usr/bin/env python3
"""Generate one hash-bound official N0-TWAM clean smoke request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotactile_benchmark.backends.univtac_contracts import (
    resolve_univtac_full_horizon_budget,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--dataset-sha256",
        help="optional expected SHA-256 of the generated frozen trial-set manifest",
    )
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument(
        "--max-control-cycles",
        type=int,
        help="defaults to the selected task's frozen action horizon",
    )
    parser.add_argument(
        "--max-observation-steps",
        type=int,
        help="defaults to max-control-cycles + 1",
    )
    parser.add_argument(
        "--wall-timeout-s",
        type=float,
        default=1800.0,
        help="minimum infrastructure-watchdog budget",
    )
    parser.add_argument(
        "--watchdog-seconds-per-action",
        type=float,
        default=DEFAULT_N0_WATCHDOG_SECONDS_PER_ACTION,
    )
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument(
        "--initial-state-policy",
        choices=tuple(item.value for item in InitialStatePolicy),
        default=InitialStatePolicy.OFFICIAL_REPRODUCTION.value,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trial-set", type=Path)
    parser.add_argument("--live-output", type=Path)
    args = parser.parse_args()

    max_control_cycles, max_observation_steps = resolve_univtac_full_horizon_budget(
        args.task,
        max_control_cycles=args.max_control_cycles,
        max_observation_steps=args.max_observation_steps,
    )

    layout = DeploymentLayout(resolve_deployment_root(args.root))
    initialize_deployment_layout(layout)
    config = args.config or (
        layout.model_artifacts
        / "n0_twam"
        / "configs"
        / args.task
        / "integration_config.json"
    )
    runtime = resolve_n0_runtime_artifacts(config)
    if runtime.manifest.task_id != args.task:
        raise SystemExit("--task does not match the N0 artifact manifest")
    request_path = args.output or (
        layout.requests / "n0-twam" / args.task / "clean.json"
    )
    live_output = args.live_output or (
        layout.artifacts / "live-univtac" / "n0-twam" / args.task / "clean"
    )
    trial_set_path = args.trial_set or (
        layout.requests / "n0-twam" / args.task / "trial_set_manifest.json"
    )
    trial_set = write_official_n0_trial_set(
        manifest_path=trial_set_path,
        task_id=args.task,
        initial_seed=args.initial_seed,
        exogenous_seed=args.exogenous_seed,
        max_control_cycles=max_control_cycles,
        max_observation_steps=max_observation_steps,
    )
    if args.dataset_sha256 is not None and args.dataset_sha256 != trial_set.sha256:
        raise SystemExit(
            "--dataset-sha256 does not match the generated frozen trial-set manifest"
        )
    watchdog_timeout_s = derive_n0_infrastructure_watchdog_timeout(
        requested_floor_s=args.wall_timeout_s,
        action_horizon=max_control_cycles,
        seconds_per_action=args.watchdog_seconds_per_action,
    )
    request = build_official_n0_clean_request(
        manifest=runtime.manifest,
        layout=layout,
        dataset_sha256=trial_set.sha256,
        initial_seed=args.initial_seed,
        exogenous_seed=args.exogenous_seed,
        max_control_cycles=max_control_cycles,
        max_observation_steps=max_observation_steps,
        wall_timeout_s=watchdog_timeout_s,
        simulator_device=args.simulator_device,
        live_output_dir=live_output,
        initial_state_policy=InitialStatePolicy(args.initial_state_policy),
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
    )
    generated = write_official_n0_clean_request(request_path, request)
    print(
        json.dumps(
            {
                "live_execution_claimed": False,
                "dataset_sha256": trial_set.sha256,
                "request": str(generated.request_path),
                "request_file_sha256": generated.request_file_sha256,
                "initial_state_policy": request.initial_state_policy.value,
                "wall_timeout_role": request.wall_timeout_role.value,
                "watchdog_timeout_s": request.wall_timeout_s,
                "task_id": request.task_id,
                "trial_set_manifest": str(trial_set.manifest_path),
                "trial_manifest_sha256": generated.trial_manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
