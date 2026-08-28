"""CLI wiring for source-bound recorded N0-TWAM experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import OfficialN0Policy
from robotactile_benchmark.recorded.artifact import (
    write_recorded_experiment_artifact,
)
from robotactile_benchmark.recorded.evaluator import (
    RecordedExperimentResult,
    RecordedN0Policy,
    result_counts,
    run_recorded_n0_experiment,
)
from robotactile_benchmark.recorded.source import (
    build_recorded_rest_references,
    load_univtac_hdf5_episode,
    retarget_recorded_episode,
)
from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    load_official_n0_rpc,
)


def add_recorded_n0_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the official N0 recorded robustness command."""

    parser = subparsers.add_parser(
        "recorded-n0",
        help="run N0-TWAM on one real UniVTAC HDF5 anchor without simulation",
    )
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--integration-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--episode-id", default="univtac-lift_bottle-clean-90")
    parser.add_argument("--anchor-index", type=int, default=186)
    parser.add_argument("--release-index", type=int, default=280)
    parser.add_argument("--rest-index", type=int, default=0)
    parser.add_argument("--fault-start-index", type=int, default=17)
    parser.add_argument("--initial-seed", type=int, default=90)
    parser.add_argument("--exogenous-seed", type=int, default=20260823)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument(
        "--operator",
        action="append",
        dest="operators",
        choices=sorted(CORE_OPERATOR_IDS),
        help="repeat to select operators; default is all 14",
    )
    parser.add_argument(
        "--severity",
        action="append",
        dest="severities",
        type=int,
        choices=range(1, 6),
        help="repeat to select levels; default is level 3",
    )
    parser.add_argument(
        "--api-key-env",
        default="ROBOTACTILE_N0_API_KEY",
        help="environment variable holding an optional websocket API key",
    )


def _clean_summary(result: RecordedExperimentResult) -> dict[str, object]:
    clean = result.results[0]
    if (
        clean.expert_anchor is None
        or clean.expert_full is None
        or clean.expert_future_only is None
    ):
        raise RuntimeError("completed clean result is missing action metrics")
    return {
        "expert_anchor_h0": clean.expert_anchor.to_dict(),
        "expert_full_h0_h11": clean.expert_full.to_dict(),
        "expert_future_h1_h11": clean.expert_future_only.to_dict(),
    }


def handle_recorded_n0_command(
    args: argparse.Namespace,
) -> Optional[dict[str, object]]:
    """Execute the recorded command or return None for another subcommand."""

    if args.command != "recorded-n0":
        return None
    runtime = resolve_n0_runtime_artifacts(args.integration_config)
    manifest = runtime.manifest
    if manifest.task_id != args.task:
        raise ValueError("--task does not match the N0 integration artifact")
    pin = load_integration_lock().by_id("n0_twam")
    source_receipt = verify_external_checkout(pin, args.source_root)
    identity = PolicyIdentity(
        system_id=(
            f"n0-twam:{source_receipt.commit_sha}:{manifest.serve_task_id}:"
            f"{N0_RECORDED_CHECKPOINT_INPUT_PROFILE.profile_id}"
        ),
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )
    operators = (
        tuple(args.operators)
        if args.operators is not None
        else tuple(sorted(CORE_OPERATOR_IDS))
    )
    needs_release = "F6_history_residual_imprint" in operators
    loaded_episode = load_univtac_hdf5_episode(
        args.hdf5,
        task_id=args.task,
        episode_id=args.episode_id,
        initial_seed=args.initial_seed,
        anchor_index=(args.release_index if needs_release else args.anchor_index),
        rest_index=args.rest_index,
    )
    episode = retarget_recorded_episode(loaded_episode, args.anchor_index)
    release_episode = (
        retarget_recorded_episode(loaded_episode, args.release_index)
        if needs_release
        else None
    )
    references = build_recorded_rest_references(episode)
    api_key = os.environ.get(args.api_key_env)

    def policy_factory() -> RecordedN0Policy:
        return OfficialN0Policy(
            identity,
            lambda: OfficialN0Client(
                load_official_n0_rpc(
                    source_root=args.source_root,
                    host=args.host,
                    port=args.port,
                    api_key=api_key,
                )
            ),
            input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
        )

    result = run_recorded_n0_experiment(
        episode=episode,
        release_episode=release_episode,
        rest_references=references,
        policy_identity=identity,
        policy_factory=policy_factory,
        exogenous_seed=args.exogenous_seed,
        fault_start_index=args.fault_start_index,
        operator_ids=operators,
        severity_levels=(
            tuple(args.severities) if args.severities is not None else (3,)
        ),
    )
    receipt_sha256 = write_recorded_experiment_artifact(args.output, result)
    return {
        **result_counts(result),
        "artifact_root_sha256": receipt_sha256,
        "clean": _clean_summary(result),
        "evidence_level": result.evidence_level,
        "hdf5_sha256": result.episode.source_sha256,
        "live_execution_claimed": False,
        "output": str(args.output),
        "policy_identity_sha256": identity.sha256,
        "success_rate_claimed": False,
    }


__all__ = ["add_recorded_n0_subcommand", "handle_recorded_n0_command"]
