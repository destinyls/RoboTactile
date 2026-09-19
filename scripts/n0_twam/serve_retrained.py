#!/usr/bin/env python3
"""Serve one prepared retrained task through the upstream server's own registry."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import (
    load_retrained,
    server_overrides,
)


def configure_server(
    server: Any,
    artifact: dict[str, Any],
    task: str,
    output: Path,
    port: int,
    debug_offload: bool,
) -> str:
    """Modify only the actual server module's registry, preserving original configs."""
    name = "robotactile_retrained_10hz"
    config = copy.deepcopy(server.TWAM_CONFIGS["posttrain_server"])
    config.update(server_overrides(artifact, task))
    config.update(save_root=str(output), port=port, enable_offload=debug_offload)
    server.TWAM_CONFIGS[name] = config
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument(
        "--debug-offload",
        action="store_true",
        help="CPU offload for functional preview; no throughput claims",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1,65535]")
    artifact = load_retrained(args.artifact)
    if args.task not in artifact["tasks"]:
        parser.error("task is absent from prepared artifact")
    root = Path(artifact["source"]["root"])
    sys.path[:0] = [str(root / "n0_twam"), str(root)]
    server = importlib.import_module("n0_twam.n0_twam_server")
    loaded_file = server.__file__
    if (
        loaded_file is None
        or Path(loaded_file).resolve() != (root / "n0_twam/n0_twam_server.py").resolve()
    ):
        raise RuntimeError("N0 server imported from a different checkout")
    configs = sys.modules.get("configs")
    if (
        configs is None
        or Path(str(configs.__file__)).resolve().parent
        != (root / "n0_twam/configs").resolve()
    ):
        raise RuntimeError("upstream configs namespace came from a different checkout")
    output = args.output.absolute()
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    name = configure_server(
        server, artifact, args.task, output, args.port, args.debug_offload
    )
    if rank == 0:
        config = server.TWAM_CONFIGS[name]
        # The inherited primitive config is captured as loaded, including inference knobs.
        snapshot = json.loads(json.dumps(dict(config), default=str))
        receipt = {
            "schema_version": "robotactile-n0-retrained-server-v1",
            "artifact_sha256": artifact["artifact_sha256"],
            "task": args.task,
            "config_sha256": artifact["tasks"][args.task]["config_sha256"],
            "runtime_config_sha256": canonical_hash(snapshot),
            "runtime_config": snapshot,
            "debug_offload": args.debug_offload,
            "source_verification": artifact["source"]["verification"],
        }
        with (output / "server_receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
    server.init_logger()
    server.run(
        argparse.Namespace(config_name=name, port=args.port, save_root=str(output))
    )


if __name__ == "__main__":
    main()
