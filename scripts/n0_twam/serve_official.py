#!/usr/bin/env python3
"""Launch the pinned official server with an explicit debug-offload override."""

from __future__ import annotations

import argparse
import inspect
import os
from collections.abc import Sequence
from pathlib import Path

from n0_twam.configs import TWAM_CONFIGS  # type: ignore[import-not-found]
from n0_twam.n0_twam_server import (  # type: ignore[import-not-found]
    init_logger,
    run,
)

from robotactile_benchmark.runtime_attestation import (
    build_n0_server_runtime_attestation,
)

_SOURCE_BOUND_ARGUMENTS = (
    "deployment_root",
    "task",
    "session_id",
    "attestation",
    "qualification",
    "integration_config",
    "n0_source_root",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--debug-offload", action="store_true")
    parser.add_argument("--deployment-root", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--session-id")
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--integration-config", type=Path)
    parser.add_argument("--n0-source-root", type=Path)
    return parser


def _source_bound(args: argparse.Namespace) -> bool:
    present = tuple(getattr(args, name) is not None for name in _SOURCE_BOUND_ARGUMENTS)
    if any(present) and not all(present):
        raise ValueError("source-bound N0 server arguments must be complete")
    return all(present)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_bound = _source_bound(args)
    if source_bound and args.debug_offload:
        raise ValueError("source-bound paper serving forbids debug offload")
    config = TWAM_CONFIGS["multitask_server"]
    if args.debug_offload:
        config.enable_offload = True
    if source_bound and getattr(config, "enable_offload", False):
        raise ValueError("source-bound paper serving requires fast GPU mode")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if source_bound and rank == 0:
        loaded_module = inspect.getsourcefile(run)
        if loaded_module is None:
            raise ValueError("loaded N0 server module path is unavailable")
        build_n0_server_runtime_attestation(
            deployment_root=args.deployment_root,
            task_id=args.task,
            session_id=args.session_id,
            output_path=args.attestation,
            qualification_path=args.qualification,
            integration_config_path=args.integration_config,
            n0_source_root=args.n0_source_root,
            loaded_n0_module_path=Path(loaded_module),
            rank=rank,
            world_size=world_size,
            debug_offload=args.debug_offload,
        )
    init_logger()
    run(
        argparse.Namespace(
            config_name="multitask_server",
            port=args.port,
            save_root=args.save_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
