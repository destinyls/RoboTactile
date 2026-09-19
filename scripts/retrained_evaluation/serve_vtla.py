"""Serve the mixed8 checkpoint with training assets and paired episode RNG."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from robotactile_benchmark.integrations.n0_vtla.availability_server import (
    AVAILABILITY_PROTOCOLS,
    baseline_rule,
    install_availability_protocol,
)
from scripts.retrained_evaluation.group import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    binding = read_object(args.binding)
    availability_mode = binding.get("evaluation", {}).get(
        "tactile_availability_mode", "required"
    )
    if availability_mode not in {"required", *AVAILABILITY_PROTOCOLS}:
        raise ValueError("unsupported N0-VTLA tactile availability mode")
    prompt = binding["tasks"][args.task]["prompt"]
    os.environ.update(
        VTLA_ASSET_ID=binding["asset_id"],
        VTLA_DEFAULT_PROMPT=prompt,
        VTLA_ATTN_IMPL="eager",
        VTLA_PREFIX_CACHE="0",
        JAX_PLATFORMS="cpu",
    )
    source = Path(binding["source_root"])
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "retrained_vtla_server", source / "scripts/serve_zmq.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned VTLA server not found")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    config = server._config.get_config("sim_single_arm_tactile")
    if config.model.action_horizon != 50 or config.data.raw_action_dim != 8:
        raise ValueError("mixed8 VTLA checkpoint requires qpos8 / horizon50")
    policy = server._policy_config.create_trained_policy(
        config, binding["checkpoint_root"], default_prompt=prompt
    )
    views = list(config.model.tactile_image_keys)
    if not config.model.tactile_predictor_enabled or len(views) != 2:
        raise ValueError("mixed8 VTLA must have two enabled tactile views")
    worker = server.VtlaZmqServer(
        policy,
        binding["endpoint"],
        50,
        prompt,
        views,
        8,
        [value for value in server.CAM_MAP.values() if value != "cam_right_wrist"],
    )
    upstream_reset = worker._reset

    def reset() -> dict[str, Any]:
        import numpy as np

        torch = importlib.import_module("torch")

        random.seed(args.seed)
        np.random.seed(args.seed % (1 << 32))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        return cast(dict[str, Any], upstream_reset())

    worker._reset = reset
    availability_receipt: dict[str, object] = {}
    if availability_mode != "required":
        witness_path = args.receipt.with_name(f"{args.receipt.stem}.availability.jsonl")
        install_availability_protocol(
            worker,
            availability_mode,
            decoder=server._decode_image,
            witness_path=witness_path,
        )
        availability_receipt = {
            "tactile_availability_mode": availability_mode,
            "tactile_baseline_rule": baseline_rule(availability_mode),
            "availability_witness_path": str(witness_path),
        }
    write_json(
        args.receipt,
        {
            "binding_sha256": binding["binding_sha256"],
            "task": args.task,
            "seed": args.seed,
            "status": "model_loaded",
            "tactile_views": views,
            **availability_receipt,
        },
    )
    worker.run()


if __name__ == "__main__":
    main()
