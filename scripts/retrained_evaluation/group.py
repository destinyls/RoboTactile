"""Prepare and execute one same-snapshot Clean / optical-fault task group."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.closed_loop.contracts import WallTimeoutRole
from robotactile_benchmark.constants import (
    DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS,
    SENSOR_SLOTS,
    operator_requires_rest_reference,
)
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
from robotactile_benchmark.execution.live_univtac import LiveUniVTACExecutionResult
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.paired_live_univtac import (
    default_paired_backend_session_factory,
    execute_paired_live_univtac_runs,
    execute_referenced_fault_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.fault_timing import (
    DEFAULT_EARLY_ONSET_MAX_INDEX,
    EARLY_RANDOM_ONSET_DERIVATION,
    EARLY_RANDOM_ONSET_MODE,
    derive_early_random_onset,
)
from robotactile_benchmark.integrations.n0_twam.retrained import (
    load_retrained,
    read_object,
)
from robotactile_benchmark.integrations.n0_twam.retrained_live import (
    make_request,
    validate_server_receipt,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.n0_vtla_execution import n0_vtla_execution_steps
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode,
    normalize_zero_shape,
    validate_availability_config,
)
from robotactile_benchmark.trials import Condition

OPERATORS = (
    "F1_global_response_drift",
    "F2_spatial_sensitivity_loss",
    "F3_persistent_surface_artifact",
    "F4_local_nonresponsive_patch",
    "F5_contact_shape_distortion",
    "F6_history_residual_imprint",
    "F7_high_load_saturation",
    "T1_fixed_source_delay",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
    "C1_sensor_identity_misrouting",
    "C2_frame_misregistration",
)
REGISTRIES = ("optical_marker_v1", "optical_marker_extreme_v1")
SUPPORTED_REGISTRIES = (*REGISTRIES, "optical_decision_stress_v1")
FAULT_WINDOW_MODES = (
    "delayed_onset_v1",
    "full_episode_v1",
    EARLY_RANDOM_ONSET_MODE,
)
AVAILABILITY_OPERATORS = ("A1_stream_absence", "A2_frame_erasure")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_clean(
    binding: dict[str, Any],
    task: str,
    output: Path,
    seed: int,
    *,
    n0_artifact: dict[str, Any] | None = None,
) -> LiveUniVTACRunRequest:
    root = Path(binding["deployment_root"])
    evaluation = binding.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    profile = evaluation.get("n0_vtla_execution_profile")
    if profile is not None:
        if binding["model"] != "n0_vtla":
            raise ValueError("N0-VTLA execution profile cannot change another model")
        n0_vtla_execution_steps(profile, task)
    if binding["model"] == "n0_twam":
        return make_request(
            (
                load_retrained(Path(binding["artifact"]))
                if n0_artifact is None
                else n0_artifact
            ),
            task=task,
            dataset_manifest=Path(binding["dataset_manifest"]),
            upstream=root / "sources/UniVTAC",
            runtime=output / "runtime",
            output=output / "artifacts/clean",
            seed=seed,
            watchdog_s=7200,
        )
    from robotactile_benchmark.backends.univtac_contracts import (
        build_univtac_backend_config,
    )

    entry = binding["tasks"][task]
    model = binding["model"]
    horizon = build_univtac_backend_config(task).task.action_horizon
    execution_steps = (
        n0_vtla_execution_steps(profile, task)
        if model == "n0_vtla"
        else {"ftp1_policy": 1, "dream_tac": 20}[model]
    )
    system_id = f"retrained-{model}-step{binding['training_step']}"
    config_identity = {"binding": binding["binding_sha256"], "task": entry}
    if profile is not None:
        system_id = f"{system_id}-{profile}"
        config_identity["n0_vtla_execution_profile"] = profile
    return LiveUniVTACRunRequest(
        task_id=task,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind(model),
        base_system_id=system_id,
        dataset_sha256=binding["dataset_sha256"],
        checkpoint_sha256=binding["checkpoint_sha256"],
        config_sha256=canonical_hash(config_identity),
        base_system_manifest_sha256=None,
        initial_seed=seed,
        exogenous_seed=seed,
        max_control_cycles=math.ceil(horizon / execution_steps)
        if profile is not None
        else horizon,
        max_observation_steps=horizon + 1,
        execute_action_steps=execution_steps,
        n0_vtla_execution_profile=profile,
        wall_timeout_s=7200,
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
        upstream_root=root / "sources/UniVTAC",
        runtime_dir=output / "runtime",
        output_dir=output / "artifacts/clean",
        fault_manifest_path=None,
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=None,
        simulator_device="cuda:0",
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit=binding["source_commit"],
        n0_normalizer_sha256=binding["normalizer_sha256"],
        n0_serve_bundle_sha256=binding["binding_sha256"],
        n0_prompt_manifest_sha256=canonical_hash({task: entry["prompt"]}),
        retrained_prompt=entry["prompt"],
        retrained_control_hz=binding["control_hz"],
        retrained_tactile_payload=binding["tactile_payload"],
    )


def prepare_group(
    binding: dict[str, Any],
    task: str,
    output: Path,
    seed: int,
    *,
    n0_artifact: dict[str, Any] | None = None,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite group: {output}")
    evaluation = binding.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    mode = TactileAvailabilityMode(
        evaluation.get("tactile_availability_mode", "required")
    )
    zero_shape = normalize_zero_shape(evaluation.get("tactile_zero_shape"))
    validate_availability_config(mode, zero_shape, binding["model"])
    operators = evaluation.get("operators", OPERATORS)
    if (
        not isinstance(operators, (list, tuple))
        or not operators
        or any(
            not isinstance(op, str) or op not in (*OPERATORS, *AVAILABILITY_OPERATORS)
            for op in operators
        )
        or len(set(operators)) != len(operators)
    ):
        raise ValueError(
            "operators must be a nonempty unique list of registered conditions"
        )
    slots = evaluation.get("sensor_slots", SENSOR_SLOTS)
    if (
        not isinstance(slots, (list, tuple))
        or not slots
        or any(not isinstance(slot, str) or slot not in SENSOR_SLOTS for slot in slots)
        or len(set(slots)) != len(slots)
    ):
        raise ValueError("sensor_slots must be a nonempty unique left/right list")
    level = evaluation.get("severity_level", 5)
    if type(level) is not int or level not in range(1, 6):
        raise ValueError("severity_level must be an integer from 1 to 5")
    window_mode = evaluation.get("fault_window_mode", "delayed_onset_v1")
    if not isinstance(window_mode, str) or window_mode not in FAULT_WINDOW_MODES:
        raise ValueError("fault_window_mode must be a supported window protocol")
    registries = evaluation.get("severity_registries", REGISTRIES)
    if (
        not isinstance(registries, (list, tuple))
        or not registries
        or any(
            not isinstance(item, str) or item not in SUPPORTED_REGISTRIES
            for item in registries
        )
        or len(set(registries)) != len(registries)
    ):
        raise ValueError(
            "severity_registries must be nonempty, unique supported registries"
        )
    capture_profile = LiveCaptureProfile(
        evaluation.get("capture_profile", LiveCaptureProfile.PREVIEW.value)
    )
    if level != 5 and any(
        registry in DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS for registry in registries
    ):
        raise ValueError("diagnostic stress registry requires level 5")
    clean = (
        build_clean(binding, task, output, seed)
        if n0_artifact is None
        else build_clean(
            binding,
            task,
            output,
            seed,
            n0_artifact=n0_artifact,
        )
    )
    if mode is not TactileAvailabilityMode.REQUIRED:
        clean = replace(
            clean,
            base_system_id=f"{clean.base_system_id}-{mode.value}",
            base_system_manifest_sha256=None,
            config_sha256=canonical_hash(
                {
                    "base_config_sha256": clean.config_sha256,
                    "tactile_availability_mode": mode.value,
                    "tactile_zero_shape": zero_shape,
                }
            ),
            tactile_availability_mode=mode,
            tactile_zero_shape=zero_shape,
        )
    loaded = load_live_univtac_run(clean)
    hz = loaded.backend_config.sim_hz / loaded.backend_config.physics_steps_per_action
    # Preserve legacy modes; new seeded onset is opt-in at the binding level.
    # Registry durations use the actual sensor rate, never the physics rate.
    chunk = clean.execute_action_steps
    start = math.ceil(max(2.0 * hz, chunk) / chunk) * chunk
    if binding["model"] == "n0_twam":
        start = 20  # cold 4, warm 8: inference boundaries 0, 4, 12, 20.
    if window_mode == "full_episode_v1":
        start = 0
    stop = clean.max_observation_steps
    if window_mode == "early_random_onset_v1":
        if "fault_start_index" in evaluation:
            raise ValueError(
                "early_random_onset_v1 does not accept fixed fault_start_index"
            )
        start = derive_early_random_onset(
            task=task,
            seed=seed,
            stop=stop,
            cap=evaluation.get("fault_onset_max_index", DEFAULT_EARLY_ONSET_MAX_INDEX),
        )
    else:
        if "fault_onset_max_index" in evaluation:
            raise ValueError("fault_onset_max_index requires early_random_onset_v1")
        start = evaluation.get("fault_start_index", start)
    if type(start) is not int or not 0 <= start < stop:
        raise ValueError("fault_start_index must be an integer in [0, task horizon)")
    if window_mode == "full_episode_v1" and start != 0:
        raise ValueError("full_episode_v1 requires fault_start_index=0")
    ref_path = binding.get("rest_references", {}).get(task)
    rest = None if ref_path is None else load_rest_reference_artifact(Path(ref_path))
    if rest is not None and rest.validation.task != task:
        raise ValueError("calibration reference must be task-bound")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "binding.json", binding)
    write_json(output / "requests/clean.json", live_univtac_request_to_dict(clean))
    ordered = ["requests/clean.json"]
    excluded: list[dict[str, str]] = []
    for registry in registries:
        # Legacy requests retain their old exclusion inventory; explicit subsets
        # only report selected conditions, never manufacture extra missing runs.
        unavailable = (
            AVAILABILITY_OPERATORS
            if "operators" not in evaluation
            else tuple(op for op in operators if op in AVAILABILITY_OPERATORS)
        )
        if mode is TactileAvailabilityMode.REQUIRED:
            for operator in unavailable:
                excluded.append(
                    {
                        "registry": registry,
                        "operator": operator,
                        "status": "unsupported_contract",
                        "reason": "native policy requires both tactile payloads; no imputation",
                    }
                )
        for operator in operators:
            if (
                operator in AVAILABILITY_OPERATORS
                and mode is TactileAvailabilityMode.REQUIRED
            ):
                continue
            requires_rest = operator_requires_rest_reference(
                operator, severity_registry=registry
            )
            if requires_rest and rest is None:
                excluded.append(
                    {
                        "registry": registry,
                        "operator": operator,
                        "status": "not_run_missing_calibration",
                        "reason": "no certified task-bound rest reference",
                    }
                )
                continue
            parameters: dict[str, object] = {"sample_period_s": 1.0 / hz}
            if operator == "A2_frame_erasure" and "a2_end_policy" in evaluation:
                parameters["a2_end_policy"] = evaluation["a2_end_policy"]
            if operator.startswith("T"):
                if window_mode == "full_episode_v1":
                    parameters["temporal_schedule"] = "full_episode_v1"
                elif window_mode == "early_random_onset_v1":
                    parameters["temporal_schedule"] = "window_to_end_v1"
            if requires_rest:
                assert rest is not None
                parameters["rest_reference_sha256"] = rest.references.sha256
            if operator == "C2_frame_misregistration":
                parameters["realization"] = "registered_pixels"
            operator_seed = (
                int(
                    canonical_hash({"task": task, "seed": seed, "operator": operator})[
                        :8
                    ],
                    16,
                )
                & 0x7FFFFFFF
            )
            fault = FaultManifest(
                operator_id=operator,
                severity_level=level,
                operator_seed=operator_seed,
                start_index=start,
                stop_index=stop,
                sensor_slots=tuple(slots),
                observability=Observability.BLIND,
                parameters=parameters,
                severity_registry=registry,
            )
            label = f"{registry}/{operator}"
            manifest_path = output / "faults" / f"{label}.json"
            write_json(manifest_path, fault.to_dict())
            request = replace(
                clean,
                condition=Condition.FAULTED,
                fault_manifest_path=manifest_path,
                rest_references_path=(
                    Path(ref_path) / "rest_reference.json" if requires_rest else None
                ),
                output_dir=output / "artifacts" / label,
            )
            load_live_univtac_run(request)
            relative = f"requests/{label}.json"
            write_json(output / relative, live_univtac_request_to_dict(request))
            ordered.append(relative)
    plan = {
        "schema": "robotactile-retrained-paired-group-v1",
        "task": task,
        "model": binding["model"],
        "seed": seed,
        "control_hz": hz,
        "ordered_requests": ordered,
        "excluded": excluded,
        "capture_profile": capture_profile.value,
        "fault_window_mode": window_mode,
        "fault_start_index": start,
        "fault_stop_index": stop,
        "evidence_scope": "one_seed_diagnostic_not_paper_statistics",
    }
    if window_mode == "early_random_onset_v1":
        plan["fault_onset_max_index"] = evaluation.get(
            "fault_onset_max_index", DEFAULT_EARLY_ONSET_MAX_INDEX
        )
        plan["fault_onset_derivation"] = EARLY_RANDOM_ONSET_DERIVATION
    if mode is not TactileAvailabilityMode.REQUIRED:
        plan.update(
            tactile_availability_mode=mode.value,
            tactile_zero_shape=zero_shape,
            baseline_policy=(
                "integration_native_baseline_v1"
                if binding["model"] != "n0_vtla"
                else "first_available_per_slot_v1"
                if mode is TactileAvailabilityMode.NATIVE_MISSING
                else "first_delivered_including_fill_v1"
            ),
            fault_representation="structural_absence",
            policy_input_adaptation=mode.value,
        )
    write_json(output / "group.json", plan)
    return output / "group.json"


def policy_factory(
    binding: dict[str, Any],
    loaded: LoadedLiveUniVTACRun,
    *,
    n0_artifact: dict[str, Any] | None = None,
    n0_server_receipt: dict[str, Any] | None = None,
) -> Any:
    model, task = binding["model"], loaded.request.task_id
    if model == "n0_twam":
        from robotactile_benchmark.policies.n0_input_profile import (
            N0_RETRAINED_INPUT_PROFILE,
        )
        from robotactile_benchmark.policies.n0_official import OfficialN0Policy
        from robotactile_benchmark.transport.n0_official import (
            OfficialN0Client,
            load_official_n0_rpc,
        )

        artifact = (
            load_retrained(Path(binding["artifact"]))
            if n0_artifact is None
            else n0_artifact
        )
        validate_server_receipt(
            artifact,
            task,
            (
                read_object(Path(binding["server_receipt"]))
                if n0_server_receipt is None
                else n0_server_receipt
            ),
            binding["port"],
        )
        return OfficialN0Policy(
            loaded.policy_identity,
            lambda: OfficialN0Client(
                load_official_n0_rpc(
                    source_root=Path(artifact["source"]["root"]),
                    host="127.0.0.1",
                    port=binding["port"],
                ),
                action_per_frame=4,
            ),
            input_profile=N0_RETRAINED_INPUT_PROFILE,
            action_per_frame=4,
            prompt_override=loaded.run_spec.prompt,
        )
    if model == "n0_vtla":
        from robotactile_benchmark.integrations.n0_vtla.transport import (
            OfficialN0VTLAClient,
        )
        from robotactile_benchmark.policies.n0_vtla import OfficialN0VTLAPolicy

        return OfficialN0VTLAPolicy(
            loaded.policy_identity,
            lambda: OfficialN0VTLAClient(binding["endpoint"], timeout_ms=600000),
            task_id=task,
            training_prompt=loaded.run_spec.prompt,
            tactile_availability_mode=loaded.request.tactile_availability_mode,
            tactile_zero_shape=loaded.request.tactile_zero_shape,
            execution_profile=loaded.request.n0_vtla_execution_profile,
        )
    if model == "ftp1_policy":
        from robotactile_benchmark.integrations.ftp1_policy.transport import (
            OfficialFTP1PolicyClient,
        )
        from robotactile_benchmark.policies.ftp1_policy import OfficialFTP1Policy

        return OfficialFTP1Policy(
            loaded.policy_identity,
            lambda: OfficialFTP1PolicyClient(
                binding["endpoint"],
                expected_metadata=binding["tasks"][task]["transport_metadata"],
                timeout_ms=600000,
                retrained_task=(
                    task,
                    loaded.run_spec.prompt,
                    binding["tasks"][task]["use_wrist"],
                ),
            ),
            task_id=task,
            training_prompt=loaded.run_spec.prompt,
            use_wrist=binding["tasks"][task]["use_wrist"],
        )
    if model == "dream_tac":
        from robotactile_benchmark.integrations.dream_tac.transport import (
            OfficialDreamTacClient,
        )
        from robotactile_benchmark.policies.dream_tac import (
            DreamTacGripperMapping,
            OfficialDreamTacPolicy,
        )

        return OfficialDreamTacPolicy(
            loaded.policy_identity,
            lambda: OfficialDreamTacClient(binding["endpoint"], timeout_s=600),
            task_id=task,
            instruction=loaded.run_spec.prompt,
            experiment_config=binding["experiment_config"],
            gripper_mapping=DreamTacGripperMapping.DIRECT_QPOS,
            gripper_qpos_min=binding["gripper_qpos_min"],
            gripper_qpos_max=binding["gripper_qpos_max"],
        )
    raise ValueError(f"unsupported retrained model: {model}")


def run_group(path: Path) -> None:
    root = path.parent
    plan, binding = read_object(path), read_object(root / "binding.json")
    capture_profile = LiveCaptureProfile(
        plan.get("capture_profile", LiveCaptureProfile.PREVIEW.value)
    )
    requests = tuple(
        load_live_univtac_request(root / name) for name in plan["ordered_requests"]
    )
    recovery = plan.get("recovery_protocol") == "historical_clean_reference_v1"
    execute_indices = (
        plan["execute_indices"] if recovery else list(range(len(requests)))
    )
    if any(
        request.output_dir is not None and request.output_dir.exists()
        for index, request in enumerate(requests)
        if index in execute_indices
    ):
        raise FileExistsError(
            "existing episode artifacts: adopt results, do not repeat group"
        )
    started = time.monotonic()

    def publish_one(index: int, execution: LiveUniVTACExecutionResult) -> None:
        request = execution.loaded.request
        assert request.output_dir is not None
        reopened = load_live_univtac_artifact(request.output_dir)
        if reopened.evidence.result != execution.evidence.result:
            raise ValueError("published episode differs from executed result")
        row = {
            **result_to_dict(reopened.evidence.result),
            "task": plan["task"],
            "model": plan["model"],
            "request": plan["ordered_requests"][index],
            "artifact": str(request.output_dir),
            "root_receipt_sha256": reopened.root_receipt_sha256,
            "elapsed_group_s": time.monotonic() - started,
        }
        if (
            reopened.evidence.finalization is not None
            and reopened.evidence.finalization.validation is not None
        ):
            row["delivery_validation_metrics"] = dict(
                reopened.evidence.finalization.validation.metrics
            )
        write_json(root / "results" / f"{index:02d}.json", row)
        print(json.dumps(row), flush=True)

    def publish_failure(evidence: dict[str, object]) -> None:
        write_json(
            root / "execution_failure.json",
            {
                **evidence,
                "ended_unix": time.time(),
                "published_result_count": len(list((root / "results").glob("*.json"))),
            },
        )

    if recovery:
        from robotactile_benchmark.backends.univtac_pairing import (
            UniVTACPairedResetReceipt,
        )
        from robotactile_benchmark.backends.univtac_reset_witness import (
            UniVTACResetReference,
        )
        from scripts.retrained_evaluation.availability_recovery import (
            adopt_control_rows,
        )

        if execute_indices != [2]:
            raise ValueError("availability recovery may execute only the missing A2")
        reference = UniVTACResetReference.from_dict(
            read_object(root / "reset_reference.json")
        )
        adopt_control_rows(root, plan)

        def publish_recovery(
            execution: LiveUniVTACExecutionResult, receipt: UniVTACPairedResetReceipt
        ) -> None:
            publish_one(2, execution)
            write_json(
                root / "recovery_receipt.json",
                {
                    "protocol": plan["recovery_protocol"],
                    "same_process_pairing_claimed": False,
                    "source_group": plan["source_group"],
                    "reference": reference.to_dict(),
                    "reset_receipt": receipt.to_dict(),
                    "executed_indices": [2],
                    "adopted_results": plan["adopted_results"],
                },
            )

        execute_referenced_fault_run(
            requests[2],
            reference,
            policy_factory=partial(policy_factory, binding),
            artifact_exporter=partial(
                write_live_univtac_artifact, capture_profile=capture_profile
            ),
            pre_close_publisher=publish_recovery,
            failure_publisher=publish_failure,
        )
        return

    execute_paired_live_univtac_runs(
        requests,
        policy_factory=partial(policy_factory, binding),
        session_factory=default_paired_backend_session_factory,
        artifact_exporter=partial(
            write_live_univtac_artifact, capture_profile=capture_profile
        ),
        post_execution_gate=publish_one,
        pre_close_publisher=lambda result: write_json(
            root / "paired_receipt.json", result.to_dict()
        ),
        require_shared_runtime_dir=True,
        failure_publisher=publish_failure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--binding", type=Path, required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run")
    run.add_argument("--group", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare_group(
                read_object(args.binding), args.task, args.output.absolute(), args.seed
            )
        )
    else:
        run_group(args.group)


if __name__ == "__main__":
    main()
