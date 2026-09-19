#!/usr/bin/env python3
"""Execute exactly one frozen Clean lift_can cell with model action provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from scripts.model_acceptance.episode import TracedPolicy, publish_acceptance


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def make_request(manifest: dict[str, Any], output: Path) -> tuple[Any, Any]:
    from robotactile_benchmark.backends.univtac_contracts import (
        build_univtac_backend_config,
    )
    from robotactile_benchmark.deployment.layout import DeploymentLayout
    from robotactile_benchmark.execution.official_act import (
        build_official_act_live_binding,
        make_official_act_policy_factory,
    )
    from robotactile_benchmark.integrations.act.requests import (
        build_official_act_request,
    )
    from robotactile_benchmark.integrations.runtime_config import (
        resolve_act_runtime_artifacts,
    )
    from robotactile_benchmark.trials import Condition

    model = manifest["model"]
    binding_path = Path(manifest["binding_path"])
    if model == "act":
        resolved = resolve_act_runtime_artifacts(binding_path)
        if resolved.manifest.task_id != "lift_can":
            raise ValueError("ACT artifact is not task-specific lift_can")
        horizon = build_univtac_backend_config("lift_can").task.action_horizon
        request = build_official_act_request(
            base_manifest=resolved.manifest,
            layout=DeploymentLayout(Path(manifest["node_deployment_root"])),
            condition=Condition.CLEAN,
            dataset_sha256=manifest["dataset_manifest_sha256"],
            initial_seed=91111,
            exogenous_seed=91111,
            max_control_cycles=horizon,
            max_observation_steps=horizon + 1,
            wall_timeout_s=7200,
            act_device_name=resolved.device,
            simulator_device="cuda:0",
            live_output_dir=output / "live_artifact",
        )
        official = build_official_act_live_binding(
            request,
            artifact_root=resolved.artifact_root,
            stats_sha256=resolved.stats_sha256,
            encoder_sha256=resolved.encoder_sha256,
        )
        factory = make_official_act_policy_factory(official)
    else:
        from scripts.retrained_evaluation.group import build_clean, policy_factory

        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            binding["model"] != model
            or binding["dataset_sha256"] != manifest["dataset_manifest_sha256"]
        ):
            raise ValueError("model or seeded dataset differs from frozen binding")
        receipt = Path(manifest["server_receipt"])
        if not receipt.is_file():
            raise FileNotFoundError("model-loaded server receipt is missing")
        loaded_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        if model in {"n0_vtla", "dream_tac"}:
            if (
                loaded_receipt.get("binding_sha256") != binding["binding_sha256"]
                or loaded_receipt.get("task") != "lift_can"
                or loaded_receipt.get("status") != "model_loaded"
            ):
                raise ValueError("model server receipt differs from binding")
        elif (
            model == "ftp1_policy"
            and loaded_receipt != binding["tasks"]["lift_can"]["transport_metadata"]
        ):
            raise ValueError("FTP-1 server metadata differs from binding")
        request = build_clean(binding, "lift_can", output, 91111)
        factory = partial(policy_factory, binding)
    request = replace(
        request,
        runtime_dir=output / "runtime",
        output_dir=output / "live_artifact",
    )
    return request, factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "robotactile-a800-single-clean-cell-v1"
        or manifest.get("model") not in {"act", "n0_vtla", "ftp1_policy", "dream_tac"}
        or (manifest.get("task"), manifest.get("condition"), manifest.get("seed"))
        != ("lift_can", "Clean", 91111)
        or manifest.get("expected_episode_count") != 1
    ):
        raise ValueError("unexpected or out-of-scope cell manifest")
    if file_sha256(Path(__file__)) != manifest["worker_sha256"]:
        raise ValueError("worker differs from frozen manifest")
    if file_sha256(Path(manifest["binding_path"])) != manifest["binding_file_sha256"]:
        raise ValueError("model binding changed after freezing")
    if (
        file_sha256(Path(manifest["dataset_manifest"]))
        != manifest["dataset_manifest_sha256"]
    ):
        raise ValueError("seeded dataset identity changed")
    output = manifest_path.parent / "episode"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json_once(
        output / "launch.json",
        {
            "model": manifest["model"],
            "task": "lift_can",
            "condition": "Clean",
            "seed": 91111,
            "manifest_sha256": file_sha256(manifest_path),
            "binding_file_sha256": manifest["binding_file_sha256"],
            "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "evidence_scope": "one_seed_per_host_closed_loop_diagnostic",
        },
    )
    try:
        from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
        from robotactile_benchmark.execution.live_artifacts import (
            write_live_univtac_artifact,
        )
        from robotactile_benchmark.execution.live_univtac import (
            execute_live_univtac_run,
        )
        from robotactile_benchmark.execution.request_values import (
            live_univtac_request_to_dict,
        )

        request, factory = make_request(manifest, output)
        write_json_once(output / "request.json", live_univtac_request_to_dict(request))

        def traced_factory(loaded: Any) -> TracedPolicy:
            return TracedPolicy(factory(loaded), output / "inference_trace.jsonl")

        def publish_before_close(path: Path, loaded: Any, evidence: Any) -> Any:
            exported = write_live_univtac_artifact(path, loaded, evidence)
            acceptance = publish_acceptance(output)
            write_json_once(
                output / "preclose_result.json",
                {
                    **result_to_dict(evidence.result),
                    "inference_count": acceptance["inference_count"],
                    "plan_execution_hash_links_valid": acceptance[
                        "plan_execution_hash_links_valid"
                    ],
                    "root_receipt_sha256": acceptance["root_receipt_sha256"],
                    "wall_episode_s": time.monotonic() - started,
                    "publication_stage": "before_simulator_close",
                },
            )
            return exported

        execute_live_univtac_run(
            request,
            policy_factory=traced_factory,
            artifact_exporter=publish_before_close,
        )
    except Exception as error:
        write_json_once(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "wall_episode_s": time.monotonic() - started,
            },
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
