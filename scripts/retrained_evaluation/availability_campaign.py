"""Prepare or run a no-clobber A1/A2 supplement, separately for each protocol."""

from __future__ import annotations

import argparse
import signal
from copy import deepcopy
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode,
    normalize_zero_shape,
    validate_availability_config,
)
from scripts.retrained_evaluation.campaign import run_one
from scripts.retrained_evaluation.group import (
    AVAILABILITY_OPERATORS,
    prepare_group,
    write_json,
)


def prepare_supplement(
    source: dict[str, Any],
    output: Path,
    *,
    task: str,
    seeds: tuple[int, ...],
    mode: str,
    zero_shape: object = None,
    a2_end_policy: str | None = None,
    recovery_group: Path | None = None,
) -> Path:
    """Materialize a separate binding and requests; never launch during prepare."""
    selected = TactileAvailabilityMode(mode)
    shape = normalize_zero_shape(zero_shape)
    if selected is TactileAvailabilityMode.REQUIRED:
        raise ValueError("supplement requires an explicit availability protocol")
    validate_availability_config(selected, shape, source["model"])
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be nonempty, unique nonnegative integers")
    if task not in source["tasks"]:
        raise ValueError("task is not in the checkpoint binding")
    if output.exists():
        raise FileExistsError(f"supplement already exists: {output}")
    recovery_groups: dict[str, object] = {}
    if recovery_group is not None:
        recovered = read_object(recovery_group / "group.json")
        if (
            selected is not TactileAvailabilityMode.NATIVE_MISSING
            or recovered["task"] != task
            or recovered["seed"] not in seeds
            or a2_end_policy != "episode_censored_v1"
        ):
            raise ValueError(
                "recovery source must match native task/seed and explicit A2 endpoint policy"
            )
        recovery_groups[str(recovered["seed"])] = {
            "group": str(recovery_group),
            "binding": str(recovery_group / "binding.json"),
        }
    binding = deepcopy(source)
    evaluation = binding.setdefault("evaluation", {})
    evaluation.update(
        tactile_availability_mode=selected.value,
        operators=list(AVAILABILITY_OPERATORS),
        fault_window_mode="full_episode_v1",
        fault_start_index=0,
        severity_registries=["optical_marker_extreme_v1"],
        severity_level=5,
        sensor_slots=["left", "right"],
        capture_profile="paper_full_v1",
    )
    if a2_end_policy is not None:
        if a2_end_policy != "episode_censored_v1":
            raise ValueError("unsupported A2 endpoint policy")
        evaluation["a2_end_policy"] = a2_end_policy
    if shape is not None:
        evaluation["tactile_zero_shape"] = list(shape)
    else:
        evaluation.pop("tactile_zero_shape", None)
    binding["parent_binding_sha256"] = source["binding_sha256"]
    binding["binding_sha256"] = canonical_hash(
        {k: v for k, v in binding.items() if k != "binding_sha256"}
    )
    # Dry group plans prove the exact condition matrix without touching runtimes.
    for seed in seeds:
        prepare_group(binding, task, output / "prepared" / f"seed-{seed:03d}", seed)
    path = output / "binding.json"
    write_json(path, binding)
    write_json(
        output / "supplement.json",
        {
            "model": binding["model"],
            "task": task,
            "seeds": list(seeds),
            "mode": selected.value,
            "planned_episode_count": len(seeds) * 3,
            "conditions": ["clean", *AVAILABILITY_OPERATORS],
            "parent_binding_sha256": source["binding_sha256"],
            "status": "prepared_not_executed",
            "clean_control": "new_protocol_control_not_overwrite_of_historical_clean",
            "evidence_scope": "diagnostic_not_paper_statistics",
            "recovery_groups": recovery_groups,
            "a2_end_policy": a2_end_policy or "legacy_strict",
        },
    )
    return path


def run_supplement(output: Path, *, code: Path, package: Path) -> None:
    """Serially use the existing runtime supervisor, without automatic retries."""
    from robotactile_benchmark.visualization.live_artifact import (
        export_live_artifact_visualization,
    )
    from scripts.retrained_evaluation.availability_report import summarize_availability

    plan = read_object(output / "supplement.json")
    binding_path = output / "binding.json"
    binding = read_object(binding_path)
    # An explicit receipt prevents duplicate starts, including after interruption.
    write_json(
        output / "started.json",
        {
            "status": "started_not_completed",
            "mode": plan["mode"],
            "code": str(code),
            "package": str(package),
        },
    )
    campaign = output / "campaign"
    for seed in plan["seeds"]:
        recovery = plan.get("recovery_groups", {}).get(str(seed))
        if recovery is None:
            run_one(
                binding_path,
                plan["task"],
                campaign / "seeds" / f"seed-{seed:03d}",
                code,
                package,
                seed,
            )
        else:
            run_one(
                Path(recovery["binding"]),
                plan["task"],
                campaign / "seeds" / f"seed-{seed:03d}",
                code,
                package,
                seed,
                recovery_group=Path(recovery["group"]),
            )
        group = (
            campaign
            / "seeds"
            / f"seed-{seed:03d}"
            / "groups"
            / plan["model"]
            / plan["task"]
        )
        spec = read_object(group / "group.json")
        for index, request in enumerate(spec["ordered_requests"]):
            result_path = group / "results" / f"{index:02d}.json"
            if not result_path.exists():
                raise RuntimeError(f"missing executed condition: {result_path}")
            result = read_object(result_path)
            label = Path(request).stem
            export = output / "videos" / f"seed-{seed:03d}" / label
            # Visualizations derive only from new live artifacts, including failure.
            exported = export_live_artifact_visualization(
                Path(result["artifact"]),
                export,
                fps=int(spec["control_hz"]),
                stride=1,
                video=True,
            )
            write_json(export.parent / f"{label}.json", exported.to_cli_dict())
        summary = summarize_availability(
            campaign,
            model=plan["model"],
            task=plan["task"],
            seeds=tuple(plan["seeds"]),
            mode=plan["mode"],
        )
        write_json(output / "reports" / f"after-seed-{seed:03d}.json", summary)
    write_json(
        output / "completed.json",
        {
            "status": "execution_finished",
            "model": binding["model"],
            "mode": plan["mode"],
            "summary": summary,
            "all_conditions_score_valid_claimed": False,
        },
    )


def main() -> None:
    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit("stopped; preserve artifacts and do not restart automatically")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--binding", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    prepare.add_argument(
        "--mode", choices=["native_missing_v1", "zero_fill_v1"], required=True
    )
    prepare.add_argument("--zero-shape", type=int, nargs=3)
    prepare.add_argument("--a2-end-policy", choices=["episode_censored_v1"])
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--code", type=Path, required=True)
    run.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_supplement(
            read_object(args.binding),
            args.output.absolute(),
            task=args.task,
            seeds=tuple(args.seeds),
            mode=args.mode,
            zero_shape=args.zero_shape,
            a2_end_policy=args.a2_end_policy,
        )
    else:
        run_supplement(
            args.output.absolute(),
            code=args.code.absolute(),
            package=args.package.absolute(),
        )


if __name__ == "__main__":
    main()
