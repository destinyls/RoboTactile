"""Audit saved real traces, optionally infer once through an existing service."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def policy_kind_matches(model: str, policy_kind: str) -> bool:
    """Historical live N0 requests use the policy enum value 'n0'."""
    return policy_kind in ({"n0", "n0_twam"} if model == "n0_twam" else {model})


def array_summary(value: Any) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def action_summary(value: Any) -> dict[str, object]:
    result = array_summary(value)
    shape = result["shape"]
    result["valid"] = bool(
        len(shape) == 2 and shape[0] > 0 and shape[1] == 8 and result["finite"]
    )
    return result


def records_for(bundle: Any) -> tuple[Any, ...]:
    if bundle.has_full_trace:
        return bundle.evidence.finalization.delivered_records
    if bundle.preview_trace is not None:
        return bundle.preview_trace.delivered_records
    raise ValueError("artifact retains no real observation records")


def audit(row: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    from robotactile_benchmark.execution.live_artifacts import (
        load_live_univtac_artifact,
    )

    bundle = load_live_univtac_artifact(Path(row["live_artifact"]))
    if not policy_kind_matches(row["model"], bundle.request_identity["policy_kind"]):
        raise ValueError("artifact policy kind differs from manifest model")
    binding_path = Path(row["binding_path"])
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    if row["model"] != "act":
        if binding["model"] != row["model"]:
            raise ValueError("binding model differs from manifest model")
        if (
            binding.get("checkpoint_sha256")
            != bundle.request_identity["checkpoint_sha256"]
        ):
            raise ValueError("binding checkpoint differs from artifact checkpoint")
        if Path(binding["source_root"]).resolve() != Path(row["source_root"]).resolve():
            raise ValueError("source_root differs from binding")
    else:
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_act_runtime_artifacts,
        )

        resolved = resolve_act_runtime_artifacts(binding_path)
        if (
            resolved.manifest.checkpoint_sha256
            != bundle.request_identity["checkpoint_sha256"]
        ):
            raise ValueError("ACT checkpoint differs from artifact checkpoint")
    records = records_for(bundle)
    if not records:
        raise ValueError("artifact has zero retained observations")
    observations = [record.observation for record in records]
    steps = [obs.step_index for obs in observations]
    increasing = all(b > a for a, b in zip(steps, steps[1:]))
    identities = {(obs.episode_id, obs.task, obs.seed) for obs in observations}
    observation_checks = []
    for obs in observations:
        arrays = [
            obs.proprio,
            *obs.vision.values(),
            *(sensor.payload for sensor in obs.tactile if sensor.payload is not None),
        ]
        observation_checks.append(
            bool(arrays and all(np.isfinite(x).all() for x in arrays))
        )
    actions = [
        {
            "source_step_index": entry.source_step_index,
            "action_plan_sha256": entry.action_plan_sha256,
            "planned_actions": None,
            "planned_actions_status": "not_retained_in_live_action_trace",
            "array_semantics": "executed_action_prefix",
            **action_summary(entry.executed_actions),
        }
        for entry in bundle.evidence.action_entries
    ]
    source_steps = [entry["source_step_index"] for entry in actions]
    action_order = all(b > a for a, b in zip(source_steps, source_steps[1:]))
    result = bundle.evidence.result
    timestamps = {
        slot: [
            next(s.delivery_time_s for s in obs.tactile if s.slot_id == slot)
            for obs in observations
        ]
        for slot in ("left", "right")
    }
    temporal = {
        slot: {
            "first_s": values[0],
            "last_s": values[-1],
            "strictly_increasing": all(b > a for a, b in zip(values, values[1:])),
        }
        for slot, values in timestamps.items()
    }
    arrays_valid = bool(
        actions
        and all(action["valid"] for action in actions)
        and all(observation_checks)
    )
    record_links = bool(
        len(identities) == 1
        and increasing
        and action_order
        and all(0 <= step < result.observation_count for step in source_steps)
    )
    if bundle.has_full_trace:
        record_links = record_links and len(records) == result.observation_count
    eligible = bool(result.score_eligible and result.validation_passed is True)
    passed = bool(
        arrays_valid
        and record_links
        and eligible
        and all(value["strictly_increasing"] for value in temporal.values())
    )
    return {
        "passed": passed,
        "evidence_level": "OFFLINE_REVALIDATION_OF_EXISTING_LIVE_ARTIFACT",
        "artifact_root_sha256": bundle.root_receipt_sha256,
        "binding_file_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "checkpoint_sha256": bundle.request_identity["checkpoint_sha256"],
        "historical_policy_kind": bundle.request_identity["policy_kind"],
        "task": bundle.trial.task,
        "capture_profile": bundle.capture_profile.value,
        "simulator_qualification_claimed": bundle.root_receipt.simulator_qualification_claimed,
        "has_full_trace": bundle.has_full_trace,
        "retained_observations": len(records),
        "total_observation_count": result.observation_count,
        "control_cycle_count": result.control_cycle_count,
        "executed_action_count": sum(item["shape"][0] for item in actions),
        "action_trace": actions,
        "observation_arrays_finite": all(observation_checks),
        "record_links_valid": record_links,
        "retained_step_indices": steps,
        "tactile_delivery_timing": temporal,
        "score_eligible": result.score_eligible,
        "score_success": result.score_success,
        "validation_passed": result.validation_passed,
        "validation_failure_codes": list(result.validation_failure_codes),
        "termination": result.terminal_status.value,
        "execution_status": None
        if result.execution_status is None
        else result.execution_status.value,
        "failure_stage": result.failure_stage,
        "failure_code": result.failure_code,
    }, bundle


def infer(row: dict[str, Any], bundle: Any) -> dict[str, Any]:
    from robotactile_benchmark.closed_loop.runner_checks import build_episode_context
    from robotactile_benchmark.execution.loading import (
        load_live_univtac_request,
        load_live_univtac_run,
    )

    loaded = load_live_univtac_run(load_live_univtac_request(Path(row["request_path"])))
    if loaded.trial != bundle.trial or loaded.run_spec != bundle.run_spec:
        raise ValueError("inference request does not match the captured trial/run")
    observation = records_for(bundle)[0].observation
    if observation.step_index != 0:
        raise ValueError("single inference requires the real episode-start observation")
    if row["model"] == "act":
        from robotactile_benchmark.integrations.act.factory import load_act_adapter
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_act_runtime_artifacts,
        )
        from robotactile_benchmark.policies.univtac_official_act_loading import (
            OfficialUniVTACACTLoadRequest,
        )

        resolved = resolve_act_runtime_artifacts(Path(row["binding_path"]))
        manifest = resolved.manifest
        policy = load_act_adapter(
            loaded.policy_identity,
            OfficialUniVTACACTLoadRequest(
                manifest=manifest,
                task_id=manifest.task_id,
                profile=manifest.profile,
                device_name=resolved.device,
                live=True,
            ),
        )
    else:
        from scripts.retrained_evaluation.group import policy_factory

        binding = json.loads(Path(row["binding_path"]).read_text())
        if row.get("server_receipt"):
            binding["server_receipt"] = row["server_receipt"]
        policy = policy_factory(binding, loaded)
    try:
        policy.reset(build_episode_context(loaded.trial, loaded.run_spec.prompt))
        start = time.monotonic()
        plan = policy.infer(observation)
        duration = time.monotonic() - start
        actions = action_summary(plan.actions)
        valid = bool(
            actions["valid"]
            and plan.action_spec == loaded.trial.action_spec
            and plan.source_step_index == observation.step_index
        )
        return {
            "passed": valid,
            "evidence_level": "OFFLINE_SINGLE_REAL_OBSERVATION_INFERENCE",
            "duration_s": duration,
            "action": actions,
            "action_spec": plan.action_spec,
            "action_array_semantics": "full_inferred_action_plan_not_executed",
            "action_plan_sha256": plan.sha256,
            "source_step_index": plan.source_step_index,
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
            "actions_executed_in_simulator": 0,
        }
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "infer"), required=True)
    args = parser.parse_args()
    row = json.loads(args.row.read_text())
    report = {"model": row["model"], "mode": args.mode, "paths": row, "passed": False}
    try:
        audited, bundle = audit(row)
        report["audit"] = audited
        if args.mode == "infer":
            report["inference"] = infer(row, bundle)
            report["passed"] = report["inference"]["passed"]
        else:
            report["passed"] = audited["passed"]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
