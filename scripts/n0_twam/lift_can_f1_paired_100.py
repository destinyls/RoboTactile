"""Run 100 frozen-seed, paired N0-TWAM lift_can Clean/F1 rollouts."""

from __future__ import annotations

import argparse
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import (
    file_sha256,
    load_retrained,
    read_object,
)
from scripts.n0_twam.single_task_robustness import result_row
from scripts.retrained_evaluation.campaign import run_task_seeds
from scripts.retrained_evaluation.group import write_json

TASK = "lift_can"
FAULT = "F1_global_response_drift"
REGISTRY = "optical_decision_stress_v1"
COUNT = 100
MASTER_SEED = 20260914
EXPECTED_CHECKPOINT = "4be88d4d98217c3f48ba807e2b095f6cc6c6d9bfe0828916aa6dd9ec7e04647d"
EXPECTED_DATASET = "365bf1d241ab37ae659c1817f48eae892a340f477edb9f5cd99e11d22c1daf32"
EXPECTED_REQUESTS = (
    "requests/clean.json",
    f"requests/{REGISTRY}/{FAULT}.json",
)


def select_seeds(master_seed: int = MASTER_SEED) -> list[int]:
    """Freeze 100 unique, held-out-range simulator seeds before execution."""
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    return random.Random(master_seed).sample(range(1_000_000, 1_000_000_000), COUNT)


def prepare(
    source: Path,
    campaign: Path,
    isaac_python: Path,
    artifact_override: Path | None = None,
) -> Path:
    if campaign.exists():
        raise FileExistsError(campaign)
    binding = deepcopy(read_object(source))
    if (
        binding.get("model") != "n0_twam"
        or TASK not in binding.get("tasks", {})
        or binding.get("checkpoint_sha256") != EXPECTED_CHECKPOINT
        or binding.get("dataset_sha256") != EXPECTED_DATASET
        or binding.get("training_step") != 10000
    ):
        raise ValueError("source binding is not the frozen train759 N0 step10000")
    if not isaac_python.is_file():
        raise FileNotFoundError(isaac_python)
    artifact_sha256 = None
    if artifact_override is not None:
        artifact = load_retrained(artifact_override)
        if (
            artifact.get("checkpoint_sha256") != EXPECTED_CHECKPOINT
            or artifact.get("training_step") != 10000
        ):
            raise ValueError("replacement artifact is not the frozen checkpoint")
        binding["artifact"] = str(artifact_override.resolve(strict=True))
        artifact_sha256 = artifact["artifact_sha256"]
    binding["isaac_python"] = str(isaac_python.resolve())
    binding["rest_references"] = {}
    binding["evaluation"] = {
        "operators": [FAULT],
        "severity_registries": [REGISTRY],
        "severity_level": 5,
        "fault_window_mode": "early_random_onset_v1",
        "fault_onset_max_index": 8,
        "capture_profile": "metrics_only_v1",
        "measure_n0_rest": True,
        "sensor_slots": ["left", "right"],
    }
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = canonical_hash(binding)
    seeds = select_seeds()
    campaign.mkdir(parents=True, exist_ok=False)
    write_json(campaign / "binding.json", binding)
    write_json(
        campaign / "plan.json",
        {
            "schema": "n0-lift-can-clean-f1-paired100-v1",
            "task": TASK,
            "model": "n0_twam",
            "seed_master": MASTER_SEED,
            "seed_selection": "python_random_sample_without_replacement_v1",
            "seeds": seeds,
            "planned_per_condition": COUNT,
            "planned_rollouts": 2 * COUNT,
            "conditions": ["clean", FAULT],
            "binding_sha256": binding["binding_sha256"],
            "source_binding_sha256": file_sha256(source),
            "checkpoint_sha256": EXPECTED_CHECKPOINT,
            "model_artifact_sha256": artifact_sha256,
            "dataset_sha256": EXPECTED_DATASET,
            "severity_registry": REGISTRY,
            "severity_level": 5,
            "fault_window_mode": "early_random_onset_v1",
            "fault_onset_max_index": 8,
            "capture_profile": "metrics_only_v1",
            "created_unix": time.time(),
        },
    )
    return campaign / "binding.json"


def seed_group(campaign: Path, seed: int) -> Path:
    return (
        campaign
        / "tasks"
        / TASK
        / "seeds"
        / f"seed-{seed:03d}"
        / "groups"
        / "n0_twam"
        / TASK
    )


def inspect_seed(campaign: Path, seed: int) -> dict[str, Any]:
    group = seed_group(campaign, seed)
    result: dict[str, Any] = {
        "seed": seed,
        "group": str(group),
        "paired_reset_exact": False,
        "conditions": {},
        "pair_valid": False,
    }
    plan_path, receipt_path = group / "group.json", group / "paired_receipt.json"
    if not plan_path.is_file():
        result["status"] = "not_started"
        return result
    group_plan = read_object(plan_path)
    if tuple(group_plan.get("ordered_requests", ())) != EXPECTED_REQUESTS:
        result["status"] = "request_mismatch"
        return result
    if receipt_path.is_file():
        receipt = read_object(receipt_path)
        reset = receipt.get("reset_receipt", {})
        result["paired_reset_exact"] = bool(
            isinstance(reset, dict)
            and reset.get("initial_seed") == seed
            and reset.get("all_exact") is True
            and len(reset.get("witnesses", ())) == 2
            and len(receipt.get("executions", ())) == 2
        )
        result["paired_receipt_sha256"] = file_sha256(receipt_path)
    for index, condition in enumerate(("clean", FAULT)):
        path = group / "results" / f"{index:02d}.json"
        if not path.is_file():
            result["conditions"][condition] = {"status": "missing"}
            continue
        raw = read_object(path)
        row = result_row(raw, condition)
        row["result_sha256"] = file_sha256(path)
        row["delivery_validation_metrics"] = raw.get("delivery_validation_metrics")
        result["conditions"][condition] = row
    result["pair_valid"] = result["paired_reset_exact"] and all(
        result["conditions"][condition].get("valid_episode_count") == 1
        for condition in ("clean", FAULT)
    )
    result["status"] = "valid_pair" if result["pair_valid"] else "incomplete_or_invalid"
    return result


def wilson(success: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    center = (success / total + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * math.sqrt(
            success / total * (1 - success / total) / total
            + z * z / (4 * total * total)
        )
        / (1 + z * z / total)
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(campaign: Path) -> dict[str, Any]:
    plan = read_object(campaign / "plan.json")
    seeds = plan["seeds"]
    if len(seeds) != COUNT or len(set(seeds)) != COUNT:
        raise ValueError("campaign has no frozen 100-seed sample")
    rows = [inspect_seed(campaign, seed) for seed in seeds]
    pairs = [row for row in rows if row["pair_valid"]]
    counts: dict[str, dict[str, Any]] = {}
    for condition in ("clean", FAULT):
        observed = [
            row["conditions"][condition]
            for row in rows
            if row["conditions"].get(condition, {}).get("valid_episode_count") == 1
        ]
        success = sum(item["success_count"] for item in observed)
        counts[condition] = {
            "valid": len(observed),
            "success": success,
            "success_rate": success / len(observed) if observed else None,
            "wilson95": wilson(success, len(observed)),
        }
    n11 = n10 = n01 = n00 = 0
    differences: list[int] = []
    for row in pairs:
        clean = row["conditions"]["clean"]["success_count"]
        fault = row["conditions"][FAULT]["success_count"]
        differences.append(fault - clean)
        if clean and fault:
            n11 += 1
        elif clean:
            n10 += 1
        elif fault:
            n01 += 1
        else:
            n00 += 1
    paired_n = len(pairs)
    interval = None
    if paired_n:
        rng = random.Random(plan["seed_master"] ^ 0xF1)
        draws = sorted(
            sum(differences[rng.randrange(paired_n)] for _ in range(paired_n))
            / paired_n
            for _ in range(5000)
        )
        interval = [draws[124], draws[4874]]
    discordant = n10 + n01
    mcnemar_p = min(
        1.0,
        2
        * sum(
            math.comb(discordant, k) * 0.5**discordant for k in range(min(n10, n01) + 1)
        ),
    )
    return {
        "schema": "n0-lift-can-clean-f1-paired100-summary-v1",
        "task": TASK,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "seed_master": plan["seed_master"],
        "planned_per_condition": COUNT,
        "completed_valid_pairs": paired_n,
        "missing_or_invalid_pairs": COUNT - paired_n,
        "conditions": counts,
        "paired_contingency": {
            "both_success": n11,
            "clean_only_success": n10,
            "f1_only_success": n01,
            "both_failure": n00,
        },
        "matched_clean_success_rate": (n11 + n10) / paired_n if paired_n else None,
        "matched_f1_success_rate": (n11 + n01) / paired_n if paired_n else None,
        "paired_f1_minus_clean": (n01 - n10) / paired_n if paired_n else None,
        "paired_difference_bootstrap95": interval,
        "mcnemar_exact_two_sided_p": mcnemar_p if paired_n else None,
        "episodes": rows,
        "evidence_scope": (
            "100_seed_paired_simulator_diagnostic"
            if paired_n == COUNT
            else "incomplete_paired_simulator_diagnostic"
        ),
    }


def run(campaign: Path, code: Path) -> None:
    plan = read_object(campaign / "plan.json")
    binding_path = campaign / "binding.json"
    binding = read_object(binding_path)
    if binding["binding_sha256"] != plan["binding_sha256"]:
        raise ValueError("campaign binding changed after preparation")
    source_files = (
        "scripts/n0_twam/lift_can_f1_paired_100.py",
        "scripts/retrained_evaluation/campaign.py",
        "scripts/retrained_evaluation/group.py",
        "src/robotactile_benchmark/severity.py",
    )
    write_json(
        campaign / "started.json",
        {
            "started_unix": time.time(),
            "code": str(code),
            "requested_rollouts": 2 * COUNT,
            "source_file_sha256": {
                name: file_sha256(code / name) for name in source_files
            },
        },
    )

    def after_seed(seed: int, _group: Path) -> None:
        row = inspect_seed(campaign, seed)
        write_json(campaign / "progress" / f"seed-{seed:03d}.json", row)
        print({"seed": seed, "status": row["status"]}, flush=True)

    try:
        run_task_seeds(
            binding_path,
            TASK,
            campaign / "tasks" / TASK,
            code,
            code / "src",
            plan["seeds"],
            after_seed,
        )
    finally:
        write_json(campaign / "summary.json", summarize(campaign))
    write_json(
        campaign / "finished.json",
        {
            "finished_unix": time.time(),
            "completed_valid_pairs": read_object(campaign / "summary.json")[
                "completed_valid_pairs"
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--binding", type=Path, required=True)
    prep.add_argument("--campaign", type=Path, required=True)
    prep.add_argument("--isaac-python", type=Path, required=True)
    prep.add_argument("--artifact-override", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--campaign", type=Path, required=True)
    execute.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare(
                args.binding.absolute(),
                args.campaign.absolute(),
                args.isaac_python.absolute(),
                args.artifact_override.absolute() if args.artifact_override else None,
            )
        )
    else:
        run(args.campaign.absolute(), args.code.absolute())


if __name__ == "__main__":
    main()
