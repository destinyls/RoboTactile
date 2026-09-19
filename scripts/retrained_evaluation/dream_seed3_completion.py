"""Complete Dream-Tac's missing seed-3 groups without repeating prior episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.retrained_evaluation.campaign import run_one
from scripts.retrained_evaluation.group import write_json
from scripts.retrained_evaluation.seed3_three_models import make_binding

SCHEDULE = (
    ("native", "lift_can"),
    ("zero_fill_v1", "grasp_classify"),
    ("zero_fill_v1", "lift_can"),
)


def extend_dream_binding(dream: dict[str, Any], vtla: dict[str, Any]) -> dict[str, Any]:
    if dream["model"] != "dream_tac" or vtla["model"] != "n0_vtla":
        raise ValueError("expected Dream-Tac and N0-VTLA source bindings")
    if "lift_can" in dream["tasks"]:
        raise ValueError("source binding already declares lift_can")
    prompt = vtla["tasks"]["lift_can"]["prompt"]
    if prompt != "Rotate a lying can so it stands upright":
        raise ValueError("lift_can prompt differs from frozen Dream T5 cache key")
    enriched = dict(dream)
    enriched["tasks"] = {**dream["tasks"], "lift_can": {"prompt": prompt}}
    return enriched


def run(dream_binding: Path, vtla_binding: Path, campaign: Path, code: Path) -> None:
    source = extend_dream_binding(read_object(dream_binding), read_object(vtla_binding))
    campaign.mkdir(parents=True, exist_ok=False)
    write_json(
        campaign / "plan.json",
        {
            "schema": "robotactile-dream-seed3-completion-v1",
            "seed": 3,
            "schedule": [list(item) for item in SCHEDULE],
            "dream_source_binding": str(dream_binding),
            "vtla_prompt_source_binding": str(vtla_binding),
            "lift_can_prompt": source["tasks"]["lift_can"]["prompt"],
            "prior_native_grasp_campaign": str(campaign.parent / "dream-a800"),
            "scope": "missing_groups_only_no_prior_episode_repetition",
        },
    )
    for phase in ("native", "zero_fill_v1"):
        binding_path = campaign / "bindings" / f"dream_tac-{phase}.json"
        write_json(binding_path, make_binding(source, phase))
    for phase, task in SCHEDULE:
        group = campaign / "phases" / phase / "groups" / "dream_tac" / task
        binding_path = campaign / "bindings" / f"dream_tac-{phase}.json"
        try:
            run_one(
                binding_path, task, campaign / "phases" / phase, code, code / "src", 3
            )
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            write_json(
                campaign / "errors" / phase / f"{task}.json",
                {
                    "phase": phase,
                    "task": task,
                    "status": "infrastructure_failure",
                    "error": str(error),
                },
            )
            status = "infrastructure_failure"
        else:
            status = "completed"
        print(
            json.dumps(
                {
                    "model": "dream_tac",
                    "phase": phase,
                    "task": task,
                    "status": status,
                    "group": str(group),
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-binding", type=Path, required=True)
    parser.add_argument("--vtla-binding", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    run(args.dream_binding, args.vtla_binding, args.campaign, args.code)


if __name__ == "__main__":
    main()
