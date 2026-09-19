"""Run a prepared retrained N0 task through the existing live benchmark core."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    build_univtac_backend_config,
)
from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.closed_loop.contracts import WallTimeoutRole
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_univtac import execute_live_univtac_run
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.n0_twam.retrained import (
    file_sha256,
    load_retrained,
    read_object,
)
from robotactile_benchmark.policies.n0_input_profile import N0_RETRAINED_INPUT_PROFILE
from robotactile_benchmark.policies.n0_official import OfficialN0Policy
from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    load_official_n0_rpc,
)
from robotactile_benchmark.trials import Condition


def make_request(
    artifact: dict[str, Any],
    *,
    task: str,
    dataset_manifest: Path,
    upstream: Path,
    runtime: Path,
    output: Path,
    seed: int,
    watchdog_s: float,
) -> LiveUniVTACRunRequest:
    """Bind task-specific norms/prompt to 10 Hz absolute-EE execution."""
    entry = artifact["tasks"][task]
    if artifact["action_per_frame"] != 4 or artifact["action_hz"] != 10:
        raise ValueError("retrained live runner requires the 10 Hz / 4-action contract")
    config = build_univtac_backend_config(task)
    return LiveUniVTACRunRequest(
        task_id=task,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.N0,
        base_system_id=f"n0-train759-vt-step{artifact['training_step']}",
        dataset_sha256=file_sha256(dataset_manifest),
        checkpoint_sha256=artifact["checkpoint_sha256"],
        config_sha256=canonical_hash(
            {
                "server_config_sha256": entry["config_sha256"],
                "input_profile_sha256": N0_RETRAINED_INPUT_PROFILE.sha256,
                "execution_contract": N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
            }
        ),
        base_system_manifest_sha256=None,
        initial_seed=seed,
        exogenous_seed=seed,
        max_control_cycles=config.task.action_horizon,
        max_observation_steps=config.task.action_horizon + 1,
        execute_action_steps=8,
        wall_timeout_s=watchdog_s,
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
        upstream_root=upstream,
        runtime_dir=runtime,
        output_dir=output,
        fault_manifest_path=None,
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=None,
        simulator_device="cuda:0",
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit=artifact["source"]["source_commit"],
        n0_normalizer_sha256=entry["normalizer_sha256"],
        n0_serve_bundle_sha256=artifact["artifact_sha256"],
        n0_prompt_manifest_sha256=canonical_hash({task: entry["prompt"]}),
        n0_action_per_frame=4,
        n0_prompt_override=entry["prompt"],
    )


def validate_server_receipt(
    artifact: dict[str, Any], task: str, receipt: dict[str, Any], port: int
) -> None:
    """Reject mismatched server identity before launching Isaac Sim."""
    from robotactile_benchmark.integrations.n0_twam.retrained import server_overrides

    config = receipt.get("runtime_config", {})
    if task not in artifact["tasks"] or not isinstance(config, dict):
        raise ValueError("retrained server receipt does not match this task/artifact")
    expected = server_overrides(artifact, task)
    if (
        receipt.get("schema_version") != "robotactile-n0-retrained-server-v1"
        or receipt.get("artifact_sha256") != artifact["artifact_sha256"]
        or receipt.get("task") != task
        or receipt.get("config_sha256") != artifact["tasks"][task]["config_sha256"]
        or receipt.get("runtime_config_sha256") != canonical_hash(config)
        or canonical_hash(expected) != artifact["tasks"][task]["config_sha256"]
        or not set(expected).issubset(config)
        or canonical_hash({key: config[key] for key in expected})
        != canonical_hash(expected)
        or config.get("port") != port
        or config.get("host") != "127.0.0.1"
        or config.get("action_delta_mode") != "none"
        or config.get("action_per_frame") != 4
        or config.get("serve_task") != task
    ):
        raise ValueError("retrained server receipt does not match this task/artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--server-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument("--watchdog-s", type=float, default=7200.0)
    parser.add_argument(
        "--capture-profile",
        choices=[p.value for p in LiveCaptureProfile],
        default="preview_v1",
    )
    args = parser.parse_args()
    artifact = load_retrained(args.artifact)
    server_receipt = read_object(args.server_receipt)
    validate_server_receipt(artifact, args.task, server_receipt, args.port)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"episode output already exists; not rerunning: {output}")
    request = make_request(
        artifact,
        task=args.task,
        dataset_manifest=args.dataset_manifest,
        upstream=args.upstream,
        runtime=args.runtime,
        output=output / "live",
        seed=args.seed,
        watchdog_s=args.watchdog_s,
    )
    # Establish RPC before expensive simulator construction. It does not infer.
    rpc = load_official_n0_rpc(
        source_root=Path(artifact["source"]["root"]), host="127.0.0.1", port=args.port
    )
    client = OfficialN0Client(rpc, action_per_frame=4)

    def policy_factory(loaded: LoadedLiveUniVTACRun) -> OfficialN0Policy:
        return OfficialN0Policy(
            loaded.policy_identity,
            lambda: client,
            input_profile=N0_RETRAINED_INPUT_PROFILE,
            action_per_frame=4,
            prompt_override=loaded.run_spec.prompt,
        )

    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    invocation = {
        "artifact_sha256": artifact["artifact_sha256"],
        "task": args.task,
        "seed": args.seed,
        "server_receipt": server_receipt,
        "input_profile": N0_RETRAINED_INPUT_PROFILE.to_dict(),
        "max_observation_steps": request.max_observation_steps,
        "action_hz": 10,
        "capture_profile": args.capture_profile,
        "source_verification": artifact["source"]["verification"],
        "dataset_manifest": str(args.dataset_manifest.absolute()),
        "dataset_manifest_sha256": request.dataset_sha256,
        "evidence_scope": "one_per_task_diagnostic_not_paper_statistics",
    }
    with (output / "invocation.json").open("x") as stream:
        json.dump(invocation, stream, indent=2, sort_keys=True)
    with (output / "request.json").open("x") as stream:
        json.dump(
            live_univtac_request_to_dict(request), stream, indent=2, sort_keys=True
        )
    try:
        result = execute_live_univtac_run(
            request,
            policy_factory=policy_factory,
            artifact_exporter=lambda p, loaded, evidence: write_live_univtac_artifact(
                p,
                loaded,
                evidence,
                capture_profile=LiveCaptureProfile(args.capture_profile),
            ),
            n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        )
        reopened = load_live_univtac_artifact(output / "live")
        if reopened.evidence.result != result.evidence.result:
            raise RuntimeError("live artifact result differs from executed result")
        summary = {
            **result_to_dict(result.evidence.result),
            "task": args.task,
            "wall_s": time.monotonic() - started,
            "artifact_sha256": artifact["artifact_sha256"],
            "live_root_receipt_sha256": reopened.root_receipt_sha256,
            "evidence_scope": invocation["evidence_scope"],
        }
        with (output / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except Exception as error:
        with (output / "failure.json").open("x") as stream:
            json.dump(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "wall_s": time.monotonic() - started,
                },
                stream,
                indent=2,
            )
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
