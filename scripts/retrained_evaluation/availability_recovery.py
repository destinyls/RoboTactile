"""No-clobber recovery of one unpublished A2; reuse, never rerun, valid controls."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotactile_benchmark.act_fault_campaign.reset_reference import (
    build_act_reset_reference_from_artifact,
)
from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operator_parameters import rematerialization_inputs
from scripts.retrained_evaluation.group import write_json


def prepare_recovery_group(
    binding: dict[str, Any],
    task: str,
    output: Path,
    seed: int,
    source: Path,
) -> Path:
    """Validate both historical controls, then create a new A2-only execution plan."""
    if output.exists():
        raise FileExistsError(f"recovery group exists: {output}")
    plan = read_object(source / "group.json")
    old_binding = read_object(source / "binding.json")
    if (
        plan["model"] != "n0_vtla"
        or binding["model"] != plan["model"]
        or plan["task"] != task
        or plan["seed"] != seed
        or binding["binding_sha256"] != old_binding["binding_sha256"]
        or plan.get("tactile_availability_mode") != "native_missing_v1"
    ):
        raise ValueError(
            "recovery must preserve the exact N0-VTLA native binding/task/seed"
        )
    names = plan["ordered_requests"]
    if [Path(name).stem for name in names] != [
        "clean",
        "A1_stream_absence",
        "A2_frame_erasure",
    ]:
        raise ValueError("recovery supports exactly Clean/A1 complete, A2 unpublished")
    requests = [load_live_univtac_request(source / name) for name in names]
    a2_output = requests[2].output_dir
    if (source / "results/02.json").exists() or (
        a2_output is not None and a2_output.exists()
    ):
        raise FileExistsError("A2 already has an artifact or result; never repeat it")
    adopted = []
    clean = None
    for index in (0, 1):
        row_path = source / "results" / f"{index:02d}.json"
        row = read_object(row_path)
        artifact = load_live_univtac_artifact(Path(row["artifact"]))
        loaded = load_live_univtac_run(requests[index])
        if (
            Path(row["artifact"]) != requests[index].output_dir
            or artifact.run_content_sha256 != loaded.content_sha256
            or row["root_receipt_sha256"] != artifact.root_receipt_sha256
            or any(
                row.get(key) != value
                for key, value in result_to_dict(artifact.evidence.result).items()
            )
            or artifact.evidence.result.score_eligible is not True
            or artifact.evidence.result.validation_passed is not True
        ):
            raise ValueError("historical result/artifact/request mismatch")
        adopted.append(
            {
                "index": index,
                "result": str(row_path),
                "result_file_sha256": hashlib.sha256(row_path.read_bytes()).hexdigest(),
                "root_receipt_sha256": artifact.root_receipt_sha256,
                "run_content_sha256": artifact.run_content_sha256,
            }
        )
        if index == 0:
            clean = artifact
    assert clean is not None
    # Reuse the existing generic witness builder; its historical name is ACT,
    # but it checks artifact/reset provenance, not the policy implementation.
    reference = build_act_reset_reference_from_artifact(clean)
    fault_path = requests[2].fault_manifest_path
    assert fault_path is not None
    fault = FaultManifest.from_dict(read_object(fault_path))
    parameters = rematerialization_inputs(
        fault.operator_id, fault.parameters, preserve_erasure_schedule=True
    )
    fault = replace(
        fault, parameters={**parameters, "a2_end_policy": "episode_censored_v1"}
    )
    new_fault = output / "faults/A2_frame_erasure.json"
    new_request = replace(
        requests[2],
        fault_manifest_path=new_fault,
        runtime_dir=output / "runtime",
        output_dir=output / "artifacts/A2_frame_erasure",
    )
    if load_live_univtac_run(requests[2]).trial.pair_key != reference.pair_key:
        raise ValueError("historical A2 and Clean do not share one pair key")
    write_json(new_fault, fault.to_dict())
    for index, name in enumerate(names):
        # Historical request identities retain their paths; only the missing
        # episode gets a new output and the explicitly versioned endpoint rule.
        write_json(
            output / name,
            live_univtac_request_to_dict(
                new_request if index == 2 else requests[index]
            ),
        )
    write_json(output / "binding.json", binding)
    write_json(output / "reset_reference.json", reference.to_dict())
    plan.update(
        recovery_protocol="historical_clean_reference_v1",
        a2_end_policy="episode_censored_v1",
        adopted_results=adopted,
        source_group=str(source),
        execute_indices=[2],
        evidence_scope="cross_process_reference_recovery_not_same_process_pairing",
    )
    write_json(output / "group.json", plan)
    return output / "group.json"


def adopt_control_rows(root: Path, plan: dict[str, Any]) -> None:
    """Recheck immutable source receipts and copy only result references."""
    for item in plan["adopted_results"]:
        path = Path(item["result"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["result_file_sha256"]:
            raise ValueError("historical result changed since recovery preparation")
        row = read_object(path)
        artifact = load_live_univtac_artifact(Path(row["artifact"]))
        if artifact.root_receipt_sha256 != item["root_receipt_sha256"]:
            raise ValueError("historical artifact changed since recovery preparation")
        write_json(
            root / "results" / f"{item['index']:02d}.json",
            {
                **row,
                "adoption": {**item, "executed_again": False},
            },
        )
