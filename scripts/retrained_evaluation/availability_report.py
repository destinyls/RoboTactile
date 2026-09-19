"""Report one availability protocol without merging old Clean runs or imputing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONDITIONS = ("clean", "A1_stream_absence", "A2_frame_erasure")
MODES = ("required", "native_missing_v1", "zero_fill_v1")
MODELS = ("n0_vtla", "n0_twam", "ftp1_policy", "dream_tac")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def result_fields(path: Path) -> dict[str, Any]:
    result = read_object(path)
    eligible = (
        result.get("score_eligible") is True
        and result.get("validation_passed") is True
        and result.get("failure_stage") is None
        and result.get("terminal_status") in {"success", "timeout", "early_stop"}
    )
    observations = result.get("observation_count")
    return {
        "status": "completed" if eligible else "invalid_execution",
        "published": True,
        "eligible": eligible,
        "success": result.get("score_success") is True if eligible else None,
        "terminal_status": result.get("terminal_status"),
        "failure_stage": result.get("failure_stage"),
        "failure_code": result.get("failure_code"),
        "validation_failure_codes": result.get("validation_failure_codes", []),
        "control_cycle_count": result.get("control_cycle_count"),
        "observation_count": observations,
        "executed_action_steps": (
            observations - 1
            if type(observations) is int and observations >= 1
            else None
        ),
        "source_file": str(path),
        "adoption": result.get("adoption"),
        "delivery_validation_metrics": result.get("delivery_validation_metrics"),
    }


def summarize_availability(
    campaign: Path,
    *,
    model: str,
    task: str,
    seeds: tuple[int, ...],
    mode: str,
) -> dict[str, Any]:
    """Summarize the explicitly planned seed set, including absent groups."""
    if mode not in MODES or model not in MODELS:
        raise ValueError("unknown availability mode or model")
    if not task or task in {".", ".."} or Path(task).name != task:
        raise ValueError("task must be a single directory name")
    if (
        not isinstance(seeds, tuple)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "seeds must be a nonempty tuple of unique nonnegative integers"
        )
    episodes: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for seed in seeds:
        directory = campaign / "seeds" / f"seed-{seed:03d}" / "groups" / model / task
        group_path = directory / "group.json"
        by_condition: dict[str, Path] = {}
        unsupported: set[str] = set()
        if group_path.exists():
            plan = read_object(group_path)
            if (
                plan.get("model") != model
                or plan.get("task") != task
                or type(plan.get("seed")) is not int
                or plan["seed"] != seed
                or plan.get("tactile_availability_mode", "required") != mode
            ):
                raise ValueError(f"group identity/protocol mismatch: {group_path}")
            ordered = plan.get("ordered_requests")
            if not isinstance(ordered, list) or any(
                not isinstance(name, str) or not name for name in ordered
            ):
                raise ValueError(f"invalid ordered_requests: {group_path}")
            if len(set(ordered)) != len(ordered):
                raise ValueError(f"duplicate requests: {group_path}")
            for index, name in enumerate(ordered):
                condition = Path(name).stem
                if condition not in CONDITIONS:
                    ignored.append(
                        {"seed": seed, "condition": condition, "request": name}
                    )
                    continue
                if condition in by_condition:
                    raise ValueError(f"duplicate condition {condition}: {group_path}")
                by_condition[condition] = directory / "results" / f"{index:02d}.json"
            excluded = plan.get("excluded", [])
            if not isinstance(excluded, list):
                raise ValueError(f"invalid excluded list: {group_path}")
            for entry in excluded:
                if not isinstance(entry, dict):
                    raise ValueError(f"invalid excluded entry: {group_path}")
                excluded_condition = entry.get("operator")
                if (
                    isinstance(excluded_condition, str)
                    and excluded_condition in CONDITIONS
                    and entry.get("status") == "unsupported_contract"
                ):
                    if excluded_condition in by_condition:
                        raise ValueError(
                            f"condition both scheduled and unsupported: {group_path}"
                        )
                    unsupported.add(excluded_condition)
        for condition in CONDITIONS:
            source = by_condition.get(condition)
            row: dict[str, Any] = {
                "seed": seed,
                "condition": condition,
                "mode": mode,
                "status": "missing",
                "published": False,
                "eligible": False,
                "success": None,
                "supported": source is not None,
                "support_unknown": source is None and condition not in unsupported,
                "group_file": str(group_path),
                "recovery_protocol": plan.get("recovery_protocol")
                if group_path.exists()
                else None,
            }
            if condition in unsupported:
                row["status"] = "unsupported_contract"
            elif source is not None:
                row["source_file"] = str(source)
                if source.is_file():
                    row.update(result_fields(source))
            episodes.append(row)
    cells = []
    for condition in CONDITIONS:
        selected = [row for row in episodes if row["condition"] == condition]
        published = sum(row["published"] for row in selected)
        eligible = sum(row["eligible"] for row in selected)
        success = sum(row["success"] is True for row in selected)
        cells.append(
            {
                "condition": condition,
                "planned": len(seeds),
                "published": published,
                "eligible": eligible,
                "success": success,
                "success_rate": success / eligible if eligible else None,
                "invalid": sum(
                    row["status"] == "invalid_execution" for row in selected
                ),
                "unsupported": sum(
                    row["status"] == "unsupported_contract" for row in selected
                ),
                "missing": sum(row["status"] == "missing" for row in selected),
                "coverage": published / len(seeds),
                "support_coverage": sum(row["supported"] for row in selected)
                / len(seeds),
                "support_unknown": sum(row["support_unknown"] for row in selected),
            }
        )
    planned = len(CONDITIONS) * len(seeds)
    return {
        "campaign": str(campaign),
        "model": model,
        "task": task,
        "seeds": list(seeds),
        "tactile_availability_mode": mode,
        "planned_episode_count": planned,
        "completed_execution_count": sum(row["published"] for row in episodes),
        "eligible_episode_count": sum(row["eligible"] for row in episodes),
        "coverage": sum(row["published"] for row in episodes) / planned,
        "support_coverage": sum(row["supported"] for row in episodes) / planned,
        "support_unknown_episode_count": sum(
            row["support_unknown"] for row in episodes
        ),
        "cells": cells,
        "episodes": episodes,
        "ignored_conditions": ignored,
        "evidence_scope": "availability_protocol_diagnostic_not_final_paper_statistics",
        "counting_notes": {
            "completed_execution_count": "Published result rows, including validator-rejected executions.",
            "coverage": "Published results divided by the full planned three-condition matrix.",
            "support_coverage": "Explicitly scheduled conditions divided by the full planned matrix; missing groups have unknown support.",
            "success_rate": "Successes divided only by valid eligible episodes; unsupported and missing are never model failures.",
            "executed_action_steps": "Reported observation_count minus the initial observation; no action arrays are inspected.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scripts.retrained_evaluation.group import write_json

    write_json(
        args.output,
        summarize_availability(
            args.campaign,
            model=args.model,
            task=args.task,
            seeds=tuple(args.seeds),
            mode=args.mode,
        ),
    )


if __name__ == "__main__":
    main()
