"""Run fixed-seed three-model Clean and fourteen-fault Isaac diagnostics.

Each model/host owns a separate no-clobber campaign. Native F/T/C and
zero-fill A1/A2 have separate matched Clean controls and result directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.retrained_evaluation.campaign import run_one
from scripts.retrained_evaluation.group import (
    AVAILABILITY_OPERATORS,
    OPERATORS,
    write_json,
)

MODELS = ("dream_tac", "ftp1_policy", "n0_vtla")
TASKS = ("grasp_classify", "lift_can")
PHASES = ("native", "zero_fill_v1")
SEED = 3
REGISTRY = "optical_decision_stress_v1"
REST_ROOT = Path(
    "/share_data/xuzhuoran/lsaac_sim/RoboTactile/deployment-sm120/outputs"
    "/n0-decision-stress-20260913-v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_binding(source: dict[str, Any], phase: str) -> dict[str, Any]:
    model = source["model"]
    if model not in MODELS or phase not in PHASES:
        raise ValueError("unsupported model or phase")
    binding = dict(source)
    previous = source.get("evaluation") or {}
    evaluation: dict[str, Any] = {
        "operators": list(OPERATORS if phase == "native" else AVAILABILITY_OPERATORS),
        "severity_registries": [REGISTRY],
        "severity_level": 5,
        "fault_window_mode": "early_random_onset_v1",
        "fault_onset_max_index": 8,
        "capture_profile": "preview_v1",
    }
    if "reset_time_limit_s" in previous:
        evaluation["reset_time_limit_s"] = previous["reset_time_limit_s"]
    if phase == "zero_fill_v1":
        evaluation.update(
            tactile_availability_mode="zero_fill_v1",
            tactile_zero_shape=[240, 320, 3],
            a2_end_policy="episode_censored_v1",
        )
    binding["evaluation"] = evaluation
    binding["rest_references"] = {
        task: str(
            REST_ROOT
            / f"seed-{SEED}-{task}"
            / task
            / "fault_campaign/rest_references"
            / task
        )
        for task in TASKS
    }
    if model == "ftp1_policy":
        # The original binding hash was calculated before FTP transport metadata.
        # Regenerate that metadata after changing evaluation/rest identity.
        binding["tasks"] = {
            task: {
                key: value
                for key, value in entry.items()
                if key != "transport_metadata"
            }
            for task, entry in source["tasks"].items()
        }
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = canonical_hash(binding)
    if model == "ftp1_policy":
        for task, entry in binding["tasks"].items():
            previous_metadata = source["tasks"][task]["transport_metadata"]
            entry["transport_metadata"] = {
                **previous_metadata,
                "serve_bundle_sha256": binding["binding_sha256"],
            }
    return binding


def run(model: str, source_binding: Path, campaign: Path, code: Path) -> None:
    source = read_object(source_binding)
    if source["model"] != model:
        raise ValueError("source binding model mismatch")
    if not code.is_dir() or not (code / "src").is_dir():
        raise FileNotFoundError("source snapshot requires code/src")
    for task in TASKS:
        rest = (
            REST_ROOT
            / f"seed-{SEED}-{task}"
            / task
            / "fault_campaign/rest_references"
            / task
        )
        if not (rest / "rest_reference.json").is_file():
            raise FileNotFoundError(f"certified task-bound rest missing: {rest}")
    campaign.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "robotactile-three-model-seed3-decision-stress-v1",
        "model": model,
        "tasks": list(TASKS),
        "seed": SEED,
        "registry": REGISTRY,
        "phases": list(PHASES),
        "source_binding": str(source_binding),
        "source_binding_sha256": file_sha256(source_binding),
        "code_group_sha256": file_sha256(
            code / "scripts/retrained_evaluation/group.py"
        ),
        "code_campaign_sha256": file_sha256(
            code / "scripts/retrained_evaluation/campaign.py"
        ),
        "code_availability_sha256": file_sha256(
            code / "src/robotactile_benchmark/operators/availability.py"
        ),
        "scope": "one_seed_diagnostic_not_paper_statistics",
    }
    plan_path = campaign / "plan.json"
    if plan_path.exists():
        if read_object(plan_path) != plan:
            raise ValueError("existing campaign plan differs; choose a new path")
    else:
        write_json(plan_path, plan)
    for phase in PHASES:
        binding_path = campaign / "bindings" / f"{model}-{phase}.json"
        binding = make_binding(source, phase)
        if binding_path.exists():
            if read_object(binding_path) != binding:
                raise ValueError("existing frozen binding differs")
        else:
            write_json(binding_path, binding)
        for task in TASKS:
            phase_root = campaign / "phases" / phase
            group = phase_root / "groups" / model / task
            if group.exists():
                state = "preserved_existing_group"
            else:
                try:
                    run_one(binding_path, task, phase_root, code, code / "src", SEED)
                except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                    write_json(
                        campaign / "errors" / phase / f"{task}.json",
                        {
                            "model": model,
                            "phase": phase,
                            "task": task,
                            "status": "infrastructure_failure",
                            "error": str(error),
                        },
                    )
                    state = "infrastructure_failure"
                else:
                    state = "completed"
            print(
                json.dumps(
                    {
                        "model": model,
                        "phase": phase,
                        "task": task,
                        "state": state,
                        "group": str(group),
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.source_binding, args.campaign, args.code)


if __name__ == "__main__":
    main()
