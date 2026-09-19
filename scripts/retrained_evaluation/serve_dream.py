"""Run Dream-Tac's pinned HTTP implementation with train759 artifacts."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any

from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.retrained_evaluation.group import write_json

TACTILE_GATE_CONTRACT = "train759_adjacent_rgb_delta_v1"
EULER_CONTRACT = "episode_continuous_xyz_v1"


def _validated_tactile_gate(payload: object) -> float:
    if not isinstance(payload, dict):
        raise ValueError("Dream-Tac inference payload must be a mapping")
    value = payload.get("tactile_self_attn_gate")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.15 <= float(value) <= 1.0
    ):
        raise ValueError(
            "tactile_self_attn_gate must be a finite scalar in [0.15, 1.0]"
        )
    return float(value)


def _install_tactile_gate_bridge(server: Any, model: Any, torch_module: Any) -> None:
    """Inject the request gate into the upstream sampling data batch."""

    upstream_get_action = server.get_action
    upstream_generate = model.generate_samples_from_batch

    def generate_with_gate(data_batch: Any, *args: Any, **kwargs: Any) -> Any:
        gate = getattr(model, "_robotactile_tactile_self_attn_gate", None)
        if gate is None:
            raise RuntimeError("Dream-Tac tactile gate was not bound to this request")
        if not isinstance(data_batch, dict) or "video" not in data_batch:
            raise TypeError("Dream-Tac upstream data batch is missing video")
        patched = dict(data_batch)
        video = patched["video"]
        batch_size = int(video.shape[0])
        patched["tactile_self_attn_gate"] = torch_module.full(
            (batch_size,),
            gate,
            dtype=torch_module.float32,
            device=video.device,
        )
        return upstream_generate(patched, *args, **kwargs)

    def get_action_with_gate(*args: Any, **kwargs: Any) -> Any:
        payload = server.request.get_json(silent=True)
        gate = _validated_tactile_gate(payload)
        model._robotactile_tactile_self_attn_gate = gate
        try:
            return upstream_get_action(*args, **kwargs)
        finally:
            model._robotactile_tactile_self_attn_gate = None

    model.generate_samples_from_batch = generate_with_gate
    server.get_action = get_action_with_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    binding = read_object(args.binding)
    source = Path(binding["source_root"]) / "cosmos_policy"
    for name in ("dataset_stats", "t5_embeddings", "tokenizer_checkpoint"):
        if not Path(binding[name]).is_file():
            raise FileNotFoundError(f"missing training artifact: {name}")
    os.environ.update(
        FRANKA_COSMOS_CONFIG=binding["experiment_config"],
        FRANKA_COSMOS_CKPT=binding["serve_checkpoint_root"],
        FRANKA_DATASET_STATS_PATH=binding["dataset_stats"],
        FRANKA_T5_EMBEDDINGS_PATH=binding["t5_embeddings"],
        FRANKA_USE_TACTILE="1",
        FRANKA_CHUNK_SIZE="20",
    )
    sys.path[:0] = [str(source.parent), str(source)]
    os.chdir(source.parent)
    checkpoints: Any = importlib.import_module(
        "cosmos_policy._src.imaginaire.utils.checkpoint_db"
    )
    upstream_checkpoint_path = checkpoints.get_checkpoint_path

    def checkpoint_path(name: str) -> Any:
        # The public registry evaluates placeholder base paths while importing
        # unrelated experiments. The selected model is loaded solely from the
        # bound retrained DCP below; no released base weights are substituted.
        if name.startswith("/path/to/"):
            return binding["serve_checkpoint_root"]
        return upstream_checkpoint_path(name)

    checkpoints.get_checkpoint_path = checkpoint_path
    server: Any = importlib.import_module("experiments.robot.franka.franka_server")
    loader = importlib.import_module("cosmos_policy._src.predict2.utils.model_loader")

    def get_retrained_model(cfg: Any) -> tuple[Any, Any]:
        model, config = loader.load_model_from_checkpoint(
            experiment_name=cfg.config,
            s3_checkpoint_dir=cfg.ckpt_path,
            config_file=cfg.config_file,
            load_ema_to_reg=False,
            instantiate_ema=False,
            experiment_opts=[
                f"model.config.tokenizer.vae_pth={binding['tokenizer_checkpoint']}"
            ],
        )
        return model.eval().to("cuda"), config

    server.get_model = get_retrained_model
    if not server.load_model():
        raise RuntimeError("Dream-Tac model load failed; see upstream traceback")
    cosmos_utils: Any = importlib.import_module(
        "cosmos_policy.experiments.robot.cosmos_utils"
    )
    prompt = binding["tasks"][args.task]["prompt"]
    if prompt not in cosmos_utils.t5_text_embeddings_cache:
        raise ValueError(
            "Dream-Tac task prompt is absent from the bound T5 embeddings cache"
        )
    torch_module: Any = importlib.import_module("torch")
    _install_tactile_gate_bridge(server, server.model, torch_module)
    write_json(
        args.receipt,
        {
            "binding_sha256": binding["binding_sha256"],
            "task": args.task,
            "status": "model_loaded",
            "randomness_contract": "upstream_fixed_seed0_per_inference",
            "tactile_gate_contract": TACTILE_GATE_CONTRACT,
            "euler_contract": EULER_CONTRACT,
            "prompt": prompt,
        },
    )
    server.app.run(
        host="127.0.0.1", port=binding["port"], threaded=False, use_reloader=False
    )


if __name__ == "__main__":
    main()
