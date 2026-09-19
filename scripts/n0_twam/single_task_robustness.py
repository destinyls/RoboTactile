"""Measured N0 rest calibration and a native single-task robustness delivery."""

from __future__ import annotations

import argparse
import csv
import fcntl
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.backends.univtac_rest_calibration import (
    N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
)
from robotactile_benchmark.calibration import (
    build_rest_reference_from_live_artifact,
    write_rest_reference_artifact,
)
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import array_sha256
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExportReceipt,
    default_live_backend_factory,
    execute_live_univtac_run,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from robotactile_benchmark.policies.n0_input_profile import N0_RETRAINED_INPUT_PROFILE
from robotactile_benchmark.rest_references import ReferenceSplit
from scripts.retrained_evaluation.group import (
    FAULT_WINDOW_MODES,
    build_clean,
    policy_factory,
    write_json,
)


def calibrate(binding_path: Path, task: str, output: Path) -> None:
    """Calibration is not a scored episode; model actions are never applied."""
    binding = read_object(binding_path)
    if binding["model"] != "n0_twam":
        raise ValueError("this calibration is N0-TWAM only")
    if output.exists():
        raise FileExistsError(output)
    request = replace(
        build_clean(binding, task, output, 910001),
        max_control_cycles=3,
        max_observation_steps=21,
    )
    write_json(output / "request.json", live_univtac_request_to_dict(request))
    execute_live_univtac_run(
        request,
        backend_factory=partial(
            default_live_backend_factory,
            calibration_contract=N0_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
            n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        ),
        policy_factory=partial(policy_factory, binding),
        artifact_exporter=partial(_publish_calibration, task=task, output=output),
        n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    )


def _publish_calibration(
    artifact_path: Path,
    loaded: LoadedLiveUniVTACRun,
    evidence: ClosedLoopExecutionEvidence,
    *,
    task: str,
    output: Path,
) -> LiveArtifactExportReceipt:
    """Publish every calibration product before Isaac's potentially fatal close."""
    if loaded.trial.task != task:
        raise ValueError("calibration source must match the requested task")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            artifact_path,
            output / "rest",
            output / "calibration_receipt.json",
        )
    ):
        raise FileExistsError("refusing to overwrite calibration products")
    live_receipt = write_live_univtac_artifact(
        artifact_path, loaded, evidence, capture_profile=LiveCaptureProfile.PAPER_FULL
    )
    artifact = load_live_univtac_artifact(artifact_path)
    if artifact.evidence.result.failure_stage is not None:
        raise RuntimeError(
            "calibration has an infrastructure failure; no reference published"
        )
    references, validation = build_rest_reference_from_live_artifact(
        artifact, ReferenceSplit.CALIBRATION, 5
    )
    exported = write_rest_reference_artifact(output / "rest", references, validation)
    write_json(
        output / "calibration_receipt.json",
        {
            "task": task,
            "source_live_artifact": str(artifact_path),
            "source_root_sha256": artifact.root_receipt_sha256,
            "input_profile": N0_RETRAINED_INPUT_PROFILE.to_dict(),
            "control_hz": 10,
            "rest_reference_sha256": exported.rest_reference_sha256,
            "qualified_start_index": validation.qualified_start_index,
            "qualified_stop_index": validation.qualified_stop_index,
            "model_actions_applied": False,
            "scored_episode": False,
        },
    )
    return live_receipt


def result_row(result: dict[str, Any], condition: str) -> dict[str, Any]:
    """Never convert infrastructure or fault-validation failure into model SR."""
    valid = (
        result.get("score_eligible") is True
        and result.get("failure_stage") is None
        and result.get("terminal_status") in {"success", "timeout", "early_stop"}
        and result.get("validation_passed") is True
    )
    success = result.get("score_success") is True
    return {
        "condition": condition,
        "status": "completed" if valid else "invalid_execution",
        "valid_episode_count": int(valid),
        "success_count": int(success) if valid else None,
        "success_rate": float(success) if valid else None,
        "terminal_status": result.get("terminal_status"),
        "failure_stage": result.get("failure_stage"),
        "failure_code": result.get("failure_code"),
        "observation_count": result.get("observation_count"),
        "control_cycle_count": result.get("control_cycle_count"),
        "artifact": result.get("artifact"),
    }


def exposure_metrics(artifact: LoadedLiveUniVTACArtifact) -> dict[str, object]:
    """Separate scheduled coverage, routed faults, and actual payload changes."""
    finalization = artifact.evidence.finalization
    if finalization is None:
        return {}
    delivered = finalization.delivered_records
    manifest = artifact.fault_manifest
    scheduled = {
        record.observation.step_index
        for record in delivered
        if manifest is not None and manifest.active(record.observation.step_index)
    }
    active = {
        record.observation.step_index
        for record in delivered
        if any(record.provenance_for(slot).active_fault_ids for slot in SENSOR_SLOTS)
    }
    changed = {
        faulted.observation.step_index
        for clean, faulted in zip(finalization.clean_records, delivered)
        if any(
            array_sha256(clean.observation.sensor(slot).payload)
            != array_sha256(faulted.observation.sensor(slot).payload)
            for slot in SENSOR_SLOTS
        )
    }
    queries = artifact.evidence.action_entries
    active_queries = sum(entry.source_step_index in active for entry in queries)
    count = len(delivered)
    return {
        "scheduled_fault_frames": len(scheduled),
        "active_fault_frames": len(active),
        "changed_payload_frames": len(changed),
        "changed_frames_at_action_query": sum(
            entry.source_step_index in changed for entry in queries
        ),
        "action_query_count": len(queries),
        "active_fault_action_queries": active_queries,
        "scheduled_fault_fraction": len(scheduled) / count if count else None,
        "active_fault_fraction": len(active) / count if count else None,
        "changed_payload_fraction": len(changed) / count if count else None,
        "active_fault_action_query_fraction": (
            active_queries / len(queries) if queries else None
        ),
        "full_episode_window_observed": (
            count > 0 and len(scheduled) == count if manifest is not None else None
        ),
        "fault_start_index": manifest.start_index if manifest is not None else None,
        "fault_stop_index": manifest.stop_index if manifest is not None else None,
    }


def report(group: Path, output: Path) -> None:
    """Export truthful 15-row accounting and actual rendered videos at 10 FPS."""
    from robotactile_benchmark.visualization.live_artifact import (
        export_live_artifact_visualization,
    )

    if output.exists():
        raise FileExistsError(output)
    plan = read_object(group / "group.json")
    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for index, relative in enumerate(plan["ordered_requests"]):
        condition = Path(relative).stem
        result_path = group / "results" / f"{index:02d}.json"
        if not result_path.is_file():
            rows.append({"condition": condition, "status": "not_executed"})
            continue
        row = result_row(read_object(result_path), condition)
        rows.append(row)
        artifact = Path(row["artifact"])
        loaded = load_live_univtac_artifact(artifact)
        row.update(exposure_metrics(loaded))
        # Even invalid executions may contain useful source-bound diagnostic video.
        try:
            visual = export_live_artifact_visualization(
                artifact,
                output / "videos" / condition,
                fps=int(plan["control_hz"]),
                stride=1,
                video=True,
            )
            visuals.append({**visual.to_cli_dict(), "condition": condition})
        except (ValueError, RuntimeError) as exc:
            visuals.append({"condition": condition, "visualization_error": str(exc)})
    rows.extend({"condition": item["operator"], **item} for item in plan["excluded"])
    summary = {
        "schema": "robotactile-n0-single-task-robustness-v1",
        "task": plan["task"],
        "model": "n0_twam",
        "seed": plan["seed"],
        "fault_window_mode": plan.get("fault_window_mode", "delayed_onset_v1"),
        "evidence_scope": "one_episode_per_condition_diagnostic_not_paper_statistics",
        "fault_conditions_expected": 14,
        "completed_episode_count": sum(r["status"] == "completed" for r in rows),
        "conditions": rows,
        "visualizations": visuals,
        "source_group": str(group),
    }
    write_json(output / "summary.json", summary)
    fields = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _render_summary(
        rows,
        plan["task"],
        output,
        plan.get("fault_window_mode", "delayed_onset_v1"),
        seed=plan["seed"],
    )


def _render_summary(
    rows: list[dict[str, Any]],
    task: str,
    output: Path,
    window_mode: str = "delayed_onset_v1",
    seed: int = 0,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.new("RGB", (1250, 125 + 38 * len(rows)), "white")
    draw = ImageDraw.Draw(panel)
    try:
        draw.font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        draw.font = ImageFont.load_default()
    draw.text(
        (25, 15), f"N0-TWAM step10000 | {task} | Clean + 14 factors", fill="black"
    )
    draw.text(
        (25, 42),
        f"{window_mode} | S5 optical / T2 schedule-specific | seed {seed} | n=1 diagnostic",
        fill="black",
    )
    draw.text(
        (25, 77),
        "Condition                                     Outcome / eligibility                  SR",
        fill="black",
    )
    for index, row in enumerate(rows):
        y = 115 + index * 38
        valid = row.get("valid_episode_count") == 1
        status = str(row.get("terminal_status") if valid else row["status"])
        color = (
            "#187846"
            if valid and row["success_count"]
            else "#a63935"
            if valid
            else "#777777"
        )
        draw.text((25, y), str(row["condition"]), fill="black")
        draw.text((575, y), status, fill=color)
        draw.text(
            (1030, y),
            f"{int(100 * row['success_rate'])}% (n=1)" if valid else "N/A",
            fill=color,
        )
    panel.save(output / "metrics.png")


def single_task_evaluation(
    window_mode: str = "early_random_onset_v1",
) -> dict[str, object]:
    """Use an early seed-bound onset for newly created evaluation campaigns."""
    if window_mode not in FAULT_WINDOW_MODES:
        raise ValueError("unsupported fault window protocol")
    evaluation: dict[str, object] = {
        "capture_profile": "paper_full_v1",
        "severity_registries": ["optical_marker_extreme_v1"],
        "fault_window_mode": window_mode,
        "measure_n0_rest": True,
    }
    if window_mode == "early_random_onset_v1":
        evaluation["fault_onset_max_index"] = 8
    else:
        evaluation["fault_start_index"] = 0 if window_mode == "full_episode_v1" else 20
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--binding", type=Path, required=True)
    cal.add_argument("--task", required=True)
    cal.add_argument("--output", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--group", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--binding", type=Path, required=True)
    run.add_argument("--task", default="grasp_classify")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--code", type=Path, required=True)
    run.add_argument("--package", type=Path, required=True)
    run.add_argument(
        "--fault-window", choices=FAULT_WINDOW_MODES, default="early_random_onset_v1"
    )
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.binding, args.task, args.output)
    elif args.command == "report":
        report(args.group, args.output)
    else:
        from scripts.retrained_evaluation.campaign import run_one

        binding = read_object(args.binding)
        if binding["model"] != "n0_twam":
            raise ValueError("single-task execution accepts N0-TWAM only")
        binding["evaluation"] = single_task_evaluation(args.fault_window)
        write_json(args.campaign / "binding.json", binding)
        with (args.campaign / "gpu.lock").open("x") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = time.time()
            try:
                run_one(
                    args.campaign / "binding.json",
                    args.task,
                    args.campaign,
                    args.code,
                    args.package,
                    0,
                )
                report(
                    args.campaign / "groups/n0_twam" / args.task,
                    args.campaign / "visualizations",
                )
            except Exception as exc:
                write_json(
                    args.campaign / "failure.json",
                    {
                        "exception": type(exc).__name__,
                        "message": str(exc),
                        "started_unix": started,
                        "finished_unix": time.time(),
                    },
                )
                raise


if __name__ == "__main__":
    main()
