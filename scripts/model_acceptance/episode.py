"""Run one bounded ACT/FTP-1 runtime-acceptance episode with full plan traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from scripts.model_acceptance.probe import action_summary, array_summary


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class TracedPolicy:
    """Forward the entire policy protocol and retain every full inferred plan."""

    def __init__(self, policy: Any, trace_path: Path) -> None:
        self.policy = policy
        self.identity = policy.identity
        self.trace_path = trace_path

    def reset(self, context: Any) -> None:
        self.policy.reset(context)

    def infer(self, observation: Any) -> Any:
        start = time.monotonic()
        plan = self.policy.infer(observation)
        record = {
            "source_step_index": observation.step_index,
            "plan_source_step_index": plan.source_step_index,
            "elapsed_s": time.monotonic() - start,
            "action_spec": plan.action_spec,
            "action_plan_sha256": plan.sha256,
            "planned_actions": action_summary(plan.actions),
            "planned_action_values": plan.actions.tolist(),
            "observation_proprio": array_summary(observation.proprio),
            "observation_vision": {
                key: array_summary(value) for key, value in observation.vision.items()
            },
            "observation_tactile": {
                sensor.slot_id: None
                if sensor.payload is None
                else array_summary(sensor.payload)
                for sensor in observation.tactile
            },
        }
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return plan

    def commit(self, execution: Any) -> None:
        self.policy.commit(execution)

    def abort(self, reason_code: str) -> None:
        self.policy.abort(reason_code)

    def close(self) -> None:
        self.policy.close()


def prepare(args: argparse.Namespace, output: Path) -> tuple[Any, Any]:
    from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
    from robotactile_benchmark.execution.request_values import (
        live_univtac_request_to_dict,
    )

    if args.model == "ftp1_policy":
        from scripts.retrained_evaluation.group import build_clean, policy_factory

        binding = json.loads(args.binding.read_text())
        if binding["model"] != "ftp1_policy":
            raise ValueError("FTP-1 binding selects a different model")
        request = build_clean(binding, args.task, output, args.seed)
        factory = partial(policy_factory, binding)
    else:
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

        resolved = resolve_act_runtime_artifacts(args.binding)
        if resolved.manifest.task_id != args.task:
            raise ValueError("ACT config task differs from requested task")
        dataset = {
            "protocol": "official_univtac_seeded_runtime_acceptance_v1",
            "task": args.task,
            "initial_seed": args.seed,
            "exogenous_seed": args.seed,
            "not_frozen40_replay": True,
            "not_success_rate_evaluation": True,
        }
        dataset_bytes = canonical_json_bytes(dataset)
        (output / "dataset_identity.json").write_bytes(dataset_bytes)
        root = resolved.manifest.upstream_root.parent.parent
        request = build_official_act_request(
            base_manifest=resolved.manifest,
            layout=DeploymentLayout(root),
            condition=Condition.CLEAN,
            dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
            initial_seed=args.seed,
            exogenous_seed=args.seed,
            max_control_cycles=args.max_cycles,
            max_observation_steps=args.max_cycles + 1,
            wall_timeout_s=1800,
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
    request = replace(
        request,
        max_control_cycles=args.max_cycles,
        max_observation_steps=args.max_cycles + 1,
        wall_timeout_s=1800,
        runtime_dir=output / "runtime",
        output_dir=output / "live_artifact",
    )
    (output / "request.json").write_bytes(
        canonical_json_bytes(live_univtac_request_to_dict(request))
    )
    return request, factory


def publish_acceptance(output: Path) -> dict[str, Any]:
    """Publish critical receipts before Isaac teardown can exit the interpreter."""
    from robotactile_benchmark.closed_loop.contracts import ActionPlan
    from robotactile_benchmark.execution.live_artifacts import (
        load_live_univtac_artifact,
    )

    bundle = load_live_univtac_artifact(output / "live_artifact")
    trace_path = output / "inference_trace.jsonl"
    traces = (
        [json.loads(line) for line in trace_path.read_text().splitlines()]
        if trace_path.exists()
        else []
    )
    write_json(output / "inference_trace.json", traces)
    result = bundle.evidence.result
    executed = bundle.evidence.action_entries
    plan_by_hash = {}
    for recorded in traces:
        plan = ActionPlan(
            recorded["action_spec"],
            recorded["plan_source_step_index"],
            np.asarray(recorded["planned_action_values"], dtype=np.float32),
        )
        if (
            plan.sha256 != recorded["action_plan_sha256"]
            or plan.source_step_index != recorded["source_step_index"]
            or plan.action_spec != bundle.trial.action_spec
        ):
            raise ValueError("recorded inference plan does not match its contract/hash")
        plan_by_hash[plan.sha256] = plan
    links = bool(executed)
    for entry in executed:
        plan = plan_by_hash.get(entry.action_plan_sha256)
        links = bool(
            links
            and plan is not None
            and plan.source_step_index == entry.source_step_index
            and np.array_equal(
                plan.actions[: len(entry.executed_actions)], entry.executed_actions
            )
        )
    runtime_passed = bool(
        links
        and traces
        and bundle.has_full_trace
        and bundle.evidence.finalization.delivered_records
        and result.validation_passed is True
        and result.score_eligible
    )
    receipt = dict(
        runtime_acceptance_passed=runtime_passed,
        not_success_rate_evaluation=True,
        publication_stage="artifact_exported_before_simulator_close",
        root_receipt_sha256=bundle.root_receipt_sha256,
        inference_count=len(traces),
        executed_action_count=sum(
            entry.executed_actions.shape[0] for entry in executed
        ),
        plan_execution_hash_links_valid=links,
        observation_count=result.observation_count,
        termination=result.terminal_status.value,
        execution_status=None
        if result.execution_status is None
        else result.execution_status.value,
        score_success=result.score_success,
        score_eligible=result.score_eligible,
        validation_passed=result.validation_passed,
        failure_stage=result.failure_stage,
        failure_code=result.failure_code,
        timeout_is_task_success=False,
        success_predicate_id=bundle.run_spec.success_predicate_id,
    )
    write_json(output / "acceptance.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("act", "ftp1_policy"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="must not exist")
    parser.add_argument("--task", default="grasp_classify")
    parser.add_argument("--seed", type=int, default=110911)
    parser.add_argument("--max-cycles", type=int, default=3)
    args = parser.parse_args()
    if args.max_cycles <= 0 or args.seed < 0:
        parser.error("positive max-cycles and nonnegative seed required")
    args.binding = args.binding.absolute()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    launch = {
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "model": args.model,
        "binding": str(args.binding),
        "output": str(output),
        "task": args.task,
        "seed": args.seed,
        "max_cycles": args.max_cycles,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix_s": time.time(),
        "purpose": "bounded_runtime_acceptance_not_SR",
    }
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        launch["gpu_inventory"] = gpu.stdout.strip()
        launch["gpu_inventory_returncode"] = gpu.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        launch["gpu_inventory_error"] = str(error)
    write_json(output / "launch.json", launch)
    receipt: dict[str, Any] = {
        "runtime_acceptance_passed": False,
        "not_success_rate_evaluation": True,
    }
    try:
        from robotactile_benchmark.execution.live_artifacts import (
            write_live_univtac_artifact,
        )
        from robotactile_benchmark.execution.live_univtac import (
            execute_live_univtac_run,
        )

        request, factory = prepare(args, output)
        trace_path = output / "inference_trace.jsonl"

        def traced_factory(loaded: Any) -> TracedPolicy:
            return TracedPolicy(factory(loaded), trace_path)

        def preclose_exporter(artifact_output: Path, loaded: Any, evidence: Any) -> Any:
            exported = write_live_univtac_artifact(artifact_output, loaded, evidence)
            receipt.update(publish_acceptance(output))
            return exported

        execute_live_univtac_run(
            request, policy_factory=traced_factory, artifact_exporter=preclose_exporter
        )
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    if not (output / "acceptance.json").exists():
        write_json(output / "acceptance.json", receipt)
    elif "error" in receipt:
        write_json(output / "postclose_error.json", receipt["error"])
    sys.exit(0 if receipt["runtime_acceptance_passed"] else 1)


if __name__ == "__main__":
    main()
