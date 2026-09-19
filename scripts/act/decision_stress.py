"""ACT diagnostic preparation/reporting with frozen cross-model fault instances."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotactile_benchmark.calibration.artifacts import load_rest_reference_artifact
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import (
    OPTICAL_DECISION_STRESS_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.integrations.act.requests import (
    build_official_act_request,
    write_official_act_request,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_act_runtime_artifacts,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import Condition

OPERATORS = (
    "F1_global_response_drift",
    "F3_persistent_surface_artifact",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def validate_reference(document: dict[str, Any]) -> None:
    """Require the actual N0 manifests, not newly sampled ACT perturbations."""
    if document["schema"] != "robotactile_decision_stress_cross_model_reference_v1":
        raise ValueError("cross-model reference schema mismatch")
    seen = set()
    for group in document["groups"]:
        task, seed, stop = group["task"], group["seed"], group["max_observation_steps"]
        if (
            task not in {"grasp_classify", "lift_can"}
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError("invalid pilot task/seed")
        if (task, seed) in seen:
            raise ValueError("duplicate pilot task/seed")
        seen.add((task, seed))
        if [f["operator_id"] for f in group["faults"]] != list(OPERATORS):
            raise ValueError("reference operator order differs from the paired pilot")
        onset = derive_early_random_onset(task=task, seed=seed, stop=stop, cap=8)
        for item in group["faults"]:
            fault = FaultManifest.from_dict(item["document"])
            if (
                fault.sha256 != item["fault_manifest_sha256"]
                or fault.operator_id != item["operator_id"]
                or fault.severity_registry != OPTICAL_DECISION_STRESS_REGISTRY_ID
                or fault.start_index != onset
                or fault.stop_index != stop
            ):
                raise ValueError("frozen fault manifest identity/window mismatch")
    if not seen:
        raise ValueError("reference has no groups")


def prepare(
    *,
    reference: Path,
    rest_root: Path,
    output: Path,
    deployment_root: Path,
    config_root: Path,
    isaac_python: Path,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    source = read(reference)
    validate_reference(source)
    if not isaac_python.is_file():
        raise FileNotFoundError(isaac_python)
    configs = {
        g["task"]: config_root / g["task"] / "univtac/integration_config.json"
        for g in source["groups"]
    }
    runtimes = {
        task: resolve_act_runtime_artifacts(path) for task, path in configs.items()
    }
    rest = {task: load_rest_reference_artifact(rest_root / task) for task in configs}
    for group in source["groups"]:
        for item in group["faults"]:
            fault = FaultManifest.from_dict(item["document"])
            if (
                operator_requires_rest_reference(
                    fault.operator_id, severity_registry=fault.severity_registry
                )
                and fault.parameters["rest_reference_sha256"]
                != rest[group["task"]].references.sha256
            ):
                raise ValueError("certified rest differs from the frozen N0 fault")
    for task, runtime in runtimes.items():
        if (
            runtime.manifest.task_id != task
            or runtime.manifest.profile.value != "univtac"
        ):
            raise ValueError(
                "ACT task/profile must match the Vision+tactile checkpoint"
            )
    write(output / "n0_reference.json", source)
    groups = []
    for group in source["groups"]:
        task, seed = group["task"], group["seed"]
        runtime = runtimes[task]
        root = output / f"seed-{seed}-{task}"
        horizon = int(group["max_observation_steps"])
        base = build_official_act_request(
            base_manifest=runtime.manifest,
            layout=DeploymentLayout(deployment_root),
            condition=Condition.CLEAN,
            dataset_sha256=group["dataset_sha256"],
            initial_seed=seed,
            exogenous_seed=seed,
            max_control_cycles=horizon - 1,
            max_observation_steps=horizon,
            wall_timeout_s=group["wall_timeout_s"],
            act_device_name=runtime.device,
            simulator_device="cuda:0",
            live_output_dir=root / "artifacts/clean",
        )
        base = replace(base, runtime_dir=root / "runtime")
        config = load_live_univtac_run(base).backend_config
        cells = []
        for item in [None, *group["faults"]]:
            label = "clean" if item is None else item["operator_id"]
            request = base
            fault_sha = None
            if item is not None:
                fault = FaultManifest.from_dict(item["document"])
                if fault.parameters["sample_period_s"] != 1 / config.sim_hz:
                    raise ValueError(
                        "ACT/N0 source clocks differ; do not silently rescale faults"
                    )
                fault_path = root / "fault_manifests" / f"{label}.json"
                write(fault_path, fault.to_dict())
                fault_sha = fault.sha256
                request = replace(
                    base,
                    condition=Condition.FAULTED,
                    fault_manifest_path=fault_path,
                    output_dir=root / "artifacts" / label,
                    rest_references_path=(
                        rest_root / task / "rest_reference.json"
                        if operator_requires_rest_reference(
                            fault.operator_id,
                            severity_registry=fault.severity_registry,
                        )
                        else None
                    ),
                )
            generated = write_official_act_request(
                root / "requests" / f"{label}.json", request
            )
            cells.append(
                {
                    "condition": label,
                    "request": str(generated.request_path),
                    "request_file_sha256": generated.request_file_sha256,
                    "artifact": str(request.output_dir),
                    "fault_manifest_sha256": fault_sha,
                }
            )
        entry = {
            "task": task,
            "seed": seed,
            "path": str(root),
            "integration_config": str(configs[task]),
            "cells": cells,
            "checkpoint_sha256": runtime.manifest.checkpoint_sha256,
            "source_period_s": 1 / config.sim_hz,
            "action_spec": config.action_spec,
            "execute_action_steps": base.execute_action_steps,
            "rest_reference_root": str(rest_root / task),
            "rest_reference_root_receipt_sha256": rest[task].root_receipt_sha256,
        }
        write(root / "group_plan.json", entry)
        groups.append(entry)
    write(
        output / "pilot_plan.json",
        {
            "schema": "robotactile_act_decision_stress_pilot_v1",
            "model": "act",
            "source_reference_sha256": hashlib.sha256(
                reference.read_bytes()
            ).hexdigest(),
            "source_n0_campaign": source["source_campaign"],
            "groups": groups,
            "isaac_python": str(isaac_python),
            "planned_episode_count": 5 * len(groups),
            "capture_profile": "preview_v1",
            "severity_registry": OPTICAL_DECISION_STRESS_REGISTRY_ID,
            "native_policy": "official UniVTAC policy_last; chunk50, temporal aggregation, execute1",
            "comparison_boundary": "identical fault manifests and seeds; native controllers and hardware differ; cross-model initial-state equality not assumed",
        },
    )


def report(output: Path) -> dict[str, Any]:
    plan = read(output / "pilot_plan.json")
    rows = []
    for group in plan["groups"]:
        clean_path = Path(group["cells"][0]["artifact"]) / "terminal_result.json"
        clean = read(clean_path) if clean_path.exists() else None
        for cell in group["cells"]:
            artifact = Path(cell["artifact"])
            row = {
                "model": "act",
                "task": group["task"],
                "seed": group["seed"],
                "condition": cell["condition"],
                "artifact": str(artifact),
                "fault_manifest_sha256": cell["fault_manifest_sha256"],
                "status": "pending",
            }
            terminal = artifact / "terminal_result.json"
            if terminal.exists():
                result = read(terminal)
                row.update(
                    status="terminal",
                    **{
                        k: result.get(k)
                        for k in (
                            "validation_passed",
                            "score_eligible",
                            "score_success",
                            "terminal_status",
                            "failure_code",
                            "failure_stage",
                            "observation_count",
                            "control_cycle_count",
                        )
                    },
                )
                trace = artifact / "action_trace.json"
                entries = read(trace)["entries"] if trace.exists() else []
                row["inference_count"] = len(entries)
                if cell["condition"] != "clean":
                    fault = read(artifact / "fault_manifest.json")
                    row["onset_index"] = fault["start_index"]
                    row["post_onset_inference_count"] = sum(
                        e["source_step_index"] >= fault["start_index"] for e in entries
                    )
                    row["paired_clean_success"] = (
                        None if clean is None else clean["score_success"]
                    )
                    row["valid_pair"] = clean is not None and (
                        result["initial_state_sha256"] == clean["initial_state_sha256"]
                        and all(
                            x.get("validation_passed") is True
                            and x.get("score_eligible") is True
                            for x in (clean, result)
                        )
                    )
            rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["condition"])].append(row)
    metrics = []
    for (task, condition), values in grouped.items():
        eligible = [
            x
            for x in values
            if x.get("validation_passed") is True and x.get("score_eligible") is True
        ]
        paired = [x for x in eligible if x.get("valid_pair") is True]
        successes = sum(x["score_success"] is True for x in eligible)
        metrics.append(
            {
                "task": task,
                "condition": condition,
                "planned": len(values),
                "completed": sum(x["status"] == "terminal" for x in values),
                "eligible": len(eligible),
                "success_count": successes,
                "success_rate": successes / len(eligible) if eligible else None,
                "paired_count": len(paired),
                "clean_success_to_fault_failure": sum(
                    x["paired_clean_success"] is True and x["score_success"] is False
                    for x in paired
                ),
                "clean_failure_to_fault_success": sum(
                    x["paired_clean_success"] is False and x["score_success"] is True
                    for x in paired
                ),
            }
        )
    return {
        "schema": "robotactile_act_decision_stress_results_v1",
        "rows": rows,
        "completed": sum(x["status"] == "terminal" for x in rows),
        "planned": len(rows),
        "metrics": metrics,
        "held_out_confirmation": False,
    }
