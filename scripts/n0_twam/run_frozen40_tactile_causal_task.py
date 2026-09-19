#!/usr/bin/env python3
"""Evaluate one five-episode frozen40 task against a remote N0 server."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.execution.contracts import N0ObservedTactileMode
from robotactile_benchmark.policies.n0_input_profile import (
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import OfficialN0Policy
from robotactile_benchmark.recorded.evaluator import RecordedN0Policy
from robotactile_benchmark.recorded.selection import select_univtac_contact_anchor
from robotactile_benchmark.recorded.source import load_univtac_hdf5_episode
from robotactile_benchmark.recorded.tactile_causal import (
    EVIDENCE_LEVEL,
    run_recorded_tactile_causal_episode,
)
from robotactile_benchmark.recorded.tactile_causal_artifact import (
    write_recorded_tactile_causal_task_artifact,
)
from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    load_official_n0_rpc,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--hdf5", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _identity(args: argparse.Namespace, *, absence: bool) -> PolicyIdentity:
    base = PolicyIdentity(
        system_id=(
            f"n0-twam:{args.source_commit}:{args.task}:"
            f"{N0_RECORDED_CHECKPOINT_INPUT_PROFILE.profile_id}"
        ),
        checkpoint_sha256=args.checkpoint_sha256,
        config_sha256=args.config_sha256,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )
    if not absence:
        return base
    return replace(
        base,
        system_id=f"{base.system_id}:observed-tactile-absent-v1",
        supports_structural_absence=True,
    )


def _policy_factory(
    args: argparse.Namespace,
    *,
    identity: PolicyIdentity,
    mode: N0ObservedTactileMode,
) -> Callable[[], RecordedN0Policy]:
    def factory() -> RecordedN0Policy:
        return OfficialN0Policy(
            identity,
            lambda: OfficialN0Client(
                load_official_n0_rpc(
                    source_root=args.source_root,
                    host=args.host,
                    port=args.port,
                    api_key=os.environ.get("ROBOTACTILE_N0_API_KEY"),
                )
            ),
            input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
            observed_tactile_mode=mode,
        )

    return factory


def _validate(args: argparse.Namespace) -> tuple[Path, ...]:
    if len(args.hdf5) != 5:
        raise ValueError("one frozen40 task requires exactly five HDF5 episodes")
    if len({str(path.absolute()) for path in args.hdf5}) != 5:
        raise ValueError("HDF5 episode paths must be unique")
    if not args.task or args.task.strip() != args.task:
        raise ValueError("task must be a non-empty canonical identifier")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1,65535]")
    if args.task_index < 0 or args.exogenous_seed < 0:
        raise ValueError("task index and exogenous seed must be non-negative")
    for name in (
        "checkpoint_sha256",
        "config_sha256",
        "split_manifest_sha256",
    ):
        if _SHA256.fullmatch(getattr(args, name)) is None:
            raise ValueError(f"{name} must be a lowercase SHA256")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source_commit must be a lowercase Git commit")
    source_root = args.source_root.resolve(strict=True)
    client = (
        source_root
        / "n0_twam/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py"
    )
    if source_root.is_symlink() or not client.is_file():
        raise ValueError("pinned N0 websocket client source is unavailable")
    paths: list[Path] = []
    for raw_path in args.hdf5:
        path = raw_path.absolute()
        if path.is_symlink() or not path.is_file() or path.suffix != ".hdf5":
            raise ValueError("each HDF5 source must be a non-symlink .hdf5 file")
        paths.append(path)
    return tuple(sorted(paths, key=lambda item: int(item.stem)))


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = _validate(args)
    clean_identity = _identity(args, absence=False)
    absence_identity = _identity(args, absence=True)
    results = []
    for episode_offset, hdf5_path in enumerate(paths):
        episode_seed = int(hdf5_path.stem)
        selection = select_univtac_contact_anchor(hdf5_path, task_id=args.task)
        episode = load_univtac_hdf5_episode(
            hdf5_path,
            task_id=args.task,
            episode_id=f"univtac-{args.task}-clean-{episode_seed}",
            initial_seed=episode_seed,
            anchor_index=selection.anchor_index,
            rest_index=selection.rest_index,
        )
        results.append(
            run_recorded_tactile_causal_episode(
                episode=episode,
                anchor_selection=selection,
                clean_identity=clean_identity,
                absence_identity=absence_identity,
                clean_policy_factory=_policy_factory(
                    args,
                    identity=clean_identity,
                    mode=N0ObservedTactileMode.REQUIRED,
                ),
                absence_policy_factory=_policy_factory(
                    args,
                    identity=absence_identity,
                    mode=N0ObservedTactileMode.ABSENT,
                ),
                exogenous_seed=(
                    args.exogenous_seed + args.task_index * 5 + episode_offset
                ),
            )
        )
    output = args.output.absolute()
    digest = write_recorded_tactile_causal_task_artifact(
        output,
        task_id=args.task,
        results=results,
        source_commit=args.source_commit,
        split_manifest_sha256=args.split_manifest_sha256,
    )
    return {
        "completed_episode_count": len(results),
        "evidence_level": EVIDENCE_LEVEL,
        "hdf5_source_sha256": [item.episode.source_sha256 for item in results],
        "output": str(output),
        "output_sha256": digest,
        "success_rate_claimed": False,
        "task_id": args.task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
