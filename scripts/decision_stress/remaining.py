"""Prepare missing operators while referencing, never rerunning, prior Clean."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotactile_benchmark.calibration.artifacts import load_rest_reference_artifact
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.n0_fault_campaign.generation import (
    derive_operator_template_seed,
)
from robotactile_benchmark.policies.tactile_availability import TactileAvailabilityMode
from robotactile_benchmark.trials import Condition

REGISTRY = "optical_decision_stress_v1"
NATIVE = (
    "F2_spatial_sensitivity_loss",
    "F4_local_nonresponsive_patch",
    "F5_contact_shape_distortion",
    "F6_history_residual_imprint",
    "F7_high_load_saturation",
    "T1_fixed_source_delay",
    "C1_sensor_identity_misrouting",
    "C2_frame_misregistration",
)
AVAILABILITY = ("A1_stream_absence", "A2_frame_erasure")
PREVIOUS = (
    "clean",
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fault(
    operator: str,
    *,
    task: str,
    seed: int,
    pair_key: str,
    rest_sha: str,
    stop: int = 301,
) -> FaultManifest:
    parameters: dict[str, object] = {"sample_period_s": 1 / 120}
    if operator_requires_rest_reference(operator, severity_registry=REGISTRY):
        parameters["rest_reference_sha256"] = rest_sha
    if operator.startswith("T"):
        parameters["temporal_schedule"] = "window_to_end_v1"
    if operator == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    if operator == "A2_frame_erasure":
        parameters["a2_end_policy"] = "episode_censored_v1"
    return FaultManifest(
        operator_id=operator,
        severity_level=5,
        operator_seed=derive_operator_template_seed(
            master_seed=seed,
            pair_key=pair_key,
            operator_id=operator,
        ),
        start_index=derive_early_random_onset(task=task, seed=seed, stop=stop, cap=8),
        stop_index=stop,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters=parameters,
        severity_registry=REGISTRY,
    )


def prior_group(model: str, entry: dict[str, Any]) -> dict[str, Any]:
    root = Path(entry["path"])
    if model == "act":
        return entry
    task = entry["task"]
    campaign = root / task / "fault_campaign"
    paths = list(campaign.glob("requests/*/*/clean.json"))
    if len(paths) != 1:
        raise ValueError("expected exactly one historical N0 Clean request")
    clean = load_live_univtac_request(paths[0])
    assert clean.output_dir is not None
    prepared = read(root / task / "prepared.json")
    return {
        **entry,
        "integration_config": prepared["integration_config"],
        "rest_reference_root": str(campaign / "rest_references" / task),
        "cells": [
            {
                "condition": op,
                "request": str(paths[0]),
                "artifact": str(
                    clean.output_dir
                    if op == "clean"
                    else clean.output_dir.parent / op / "s5"
                ),
            }
            for op in PREVIOUS
        ],
    }


def prepare(model: str, prior: Path, output: Path, reference: Path | None) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not (prior / "finished.json").is_file():
        raise ValueError("prior campaign must be terminal before a complement")
    old = read(prior / "pilot_plan.json")
    external = read(reference) if reference is not None else None
    if model == "act" and external is None:
        raise ValueError("ACT requires the exact exported N0 remaining-fault reference")
    if (
        external is not None
        and external["schema"] != "remaining_decision_stress_reference_v1"
    ):
        raise ValueError("remaining-fault reference schema mismatch")
    groups, exported = [], []
    for entry in old["groups"]:
        previous = prior_group(model, entry)
        task, seed = entry["task"], entry["seed"]
        base = load_live_univtac_request(Path(previous["cells"][0]["request"]))
        loaded = load_live_univtac_run(base)
        if base.condition is not Condition.CLEAN or task not in {
            "grasp_classify",
            "lift_can",
        }:
            raise ValueError("unexpected previous task/condition")
        if (
            seed not in {5, 3, 7}
            or base.initial_seed != seed
            or base.exogenous_seed != seed
        ):
            raise ValueError("seed mismatch")
        if (
            base.max_observation_steps != 301
            or base.tactile_availability_mode is not TactileAvailabilityMode.REQUIRED
        ):
            raise ValueError("previous horizon/protocol differs")
        rest_path = Path(previous["rest_reference_root"])
        rest = load_rest_reference_artifact(rest_path)
        if external is None:
            faults = [
                make_fault(
                    op,
                    task=task,
                    seed=seed,
                    pair_key=loaded.trial.pair_key,
                    rest_sha=rest.references.sha256,
                )
                for op in (*NATIVE, *AVAILABILITY)
            ]
        else:
            matched = [
                g for g in external["groups"] if (g["task"], g["seed"]) == (task, seed)
            ]
            if len(matched) != 1:
                raise ValueError("missing or duplicate cross-model reference group")
            if matched[0]["dataset_sha256"] != base.dataset_sha256:
                raise ValueError("ACT/N0 dataset identity differs")
            faults = [
                FaultManifest.from_dict(f["document"]) for f in matched[0]["faults"]
            ]
            if [f.sha256 for f in faults] != [
                f["sha256"] for f in matched[0]["faults"]
            ]:
                raise ValueError("reference fault SHA mismatch")
        if [f.operator_id for f in faults] != list((*NATIVE, *AVAILABILITY)):
            raise ValueError(
                "complement must contain exactly the ten remaining operators"
            )
        for checked in faults:
            if (
                checked.start_index
                != derive_early_random_onset(task=task, seed=seed, stop=301, cap=8)
                or checked.stop_index != 301
            ):
                raise ValueError("fault onset/horizon mismatch")
            if checked.severity_registry != REGISTRY or checked.severity_level != 5:
                raise ValueError("fault severity mismatch")
            if (
                operator_requires_rest_reference(
                    checked.operator_id, severity_registry=REGISTRY
                )
                and checked.parameters["rest_reference_sha256"]
                != rest.references.sha256
            ):
                raise ValueError("ACT/N0 certified rest identity differs")
        root = output / f"seed-{seed}-{task}"
        cells = []
        # Native Clean is referenced. The zero-fill supplementary Clean is new.
        ordered = [*faults[: len(NATIVE)], None, *faults[len(NATIVE) :]]
        for fault in ordered:
            supplementary = fault is None or fault.operator_id in AVAILABILITY
            label = "availability_clean" if fault is None else fault.operator_id
            fault_path = None
            if fault is not None:
                fault_path = root / "faults" / f"{label}.json"
                write(fault_path, fault.to_dict())
            request = replace(
                base,
                condition=Condition.CLEAN if fault is None else Condition.FAULTED,
                fault_manifest_path=fault_path,
                rest_references_path=(
                    rest_path / "rest_reference.json"
                    if fault is not None
                    and operator_requires_rest_reference(
                        fault.operator_id, severity_registry=REGISTRY
                    )
                    else None
                ),
                output_dir=root / "artifacts" / label,
                runtime_dir=root / "runtime",
                tactile_availability_mode=(
                    TactileAvailabilityMode.ZERO_FILL
                    if supplementary
                    else TactileAvailabilityMode.REQUIRED
                ),
                tactile_zero_shape=(
                    tuple(rest.references.payload_for("left").shape)
                    if supplementary
                    else None
                ),
            )
            item = load_live_univtac_run(request)
            if item.backend_config != loaded.backend_config:
                raise ValueError("new requests changed the physical simulator contract")
            request_path = root / "requests" / f"{label}.json"
            write(request_path, live_univtac_request_to_dict(request))
            cells.append(
                {
                    "condition": label,
                    "protocol": "zero_fill_v1" if supplementary else "native",
                    "request": str(request_path),
                    "request_sha256": sha(request_path),
                    "artifact": str(request.output_dir),
                    "run_content_sha256": item.content_sha256,
                    "fault_sha256": None if fault is None else fault.sha256,
                }
            )
        old_cells = [
            {
                **c,
                "terminal_file_sha256": sha(
                    Path(c["artifact"]) / "terminal_result.json"
                ),
            }
            for c in previous["cells"]
        ]
        old_clean = read(Path(old_cells[0]["artifact"]) / "terminal_result.json")
        if (
            old_clean["validation_passed"] is not True
            or old_clean["score_eligible"] is not True
        ):
            raise ValueError("historical Clean is not a valid scoring reference")
        group = {
            "task": task,
            "seed": seed,
            "path": str(root),
            "cells": cells,
            "model": model,
            "integration_config": previous["integration_config"],
            "prior_cells": old_cells,
            "prior_initial_state_sha256": old_clean["initial_state_sha256"],
            "capture_profile": "preview_v1",
            "checkpoint_sha256": base.checkpoint_sha256,
        }
        write(root / "group_plan.json", group)
        groups.append(group)
        exported.append(
            {
                "task": task,
                "seed": seed,
                "dataset_sha256": base.dataset_sha256,
                "faults": [
                    {"document": f.to_dict(), "sha256": f.sha256} for f in faults
                ],
            }
        )
    isaac = (
        old.get("isaac_python")
        or read(Path(old["groups"][0]["path"]) / "plan.json")["isaac_python"]
    )
    write(
        output / "reference.json",
        {"schema": "remaining_decision_stress_reference_v1", "groups": exported},
    )
    write(
        output / "plan.json",
        {
            "schema": "remaining_decision_stress_campaign_v1",
            "model": model,
            "prior": str(prior),
            "prior_plan_sha256": sha(prior / "pilot_plan.json"),
            "groups": groups,
            "isaac_python": isaac,
            "n0_port": 29695,
            "planned_episode_count": len(groups) * 11,
            "prior_episode_count": len(groups) * 5,
            "boundary": "3 fixed seeds; diagnostic, native controllers/hardware differ; cross-process Clean pairing checked separately",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("n0", "act"), required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    prepare(
        args.model,
        args.prior.resolve(strict=True),
        args.output.absolute(),
        args.reference,
    )


if __name__ == "__main__":
    main()
