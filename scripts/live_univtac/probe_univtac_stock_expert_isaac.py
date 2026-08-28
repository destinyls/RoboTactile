#!/usr/bin/env python3
"""Run one UniVTAC task's native scripted expert after an official reset."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    EARLY_STOP_NONE_IS_FALSE_TASK_IDS,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    N0_UNIVTAC_ANTIALIASING_MODE,
    launch_univtac_runtime,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import Array
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.integrations.n0_twam.dynamic_contract import (
    tactile_depth_summary,
)
from robotactile_benchmark.integrations.n0_twam.dynamic_probe_io import (
    STREAM_NAMES,
    model_streams,
    raw_depths,
    write_dynamic_probe_bundle,
)
from scripts.live_univtac.capture_observation_parity_isaac import (
    _absolute_directory,
    _output_path,
    _sha256_file,
    _streams,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rendering-mode",
        choices=("balanced", "performance", "quality"),
        default="balanced",
    )
    parser.add_argument(
        "--antialiasing-mode",
        choices=(N0_UNIVTAC_ANTIALIASING_MODE,),
        default=N0_UNIVTAC_ANTIALIASING_MODE,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("initial_seed", "exogenous_seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _emit_stage(stage: str) -> None:
    """Flush the last entered probe stage before native Isaac shutdown."""

    print(
        json.dumps(
            {
                "event": "robotactile_stock_expert_probe_stage",
                "stage": stage,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def _emit_failure(stage: str, error: BaseException) -> None:
    """Publish Python failure evidence before closing the native runtime."""

    print(
        json.dumps(
            {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "event": "robotactile_stock_expert_probe_failure",
                "stage": stage,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def _strict_predicate(task: Any, name: str, *, task_id: str | None = None) -> bool:
    predicate = getattr(task, name, None)
    if not callable(predicate):
        raise RuntimeError(f"pinned UniVTAC task lacks {name}")
    value = predicate()
    if (
        name == "check_early_stop"
        and value is None
        and task_id in EARLY_STOP_NONE_IS_FALSE_TASK_IDS
    ):
        return False
    if isinstance(value, np.bool_):
        return bool(value)
    if type(value) is not bool:
        raise RuntimeError(f"pinned UniVTAC {name} returned a non-boolean value")
    return cast(bool, value)


def _expert_panel(before: Mapping[str, Array], after: Mapping[str, Array]) -> bytes:
    """Render reset/final model pixels without generating or altering content."""

    from PIL import Image, ImageDraw

    rows = (("Official reset", before), ("Stock scripted expert final", after))
    width, height, label = 320, 240, 24
    canvas = Image.new("RGB", (width * 4, (height + label) * 2), (16, 18, 22))
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_name, streams) in enumerate(rows):
        y = row_index * (height + label)
        for column_index, name in enumerate(STREAM_NAMES):
            x = column_index * width
            image = Image.fromarray(streams[name])
            image.thumbnail((width, height))
            canvas.paste(image, (x + (width - image.width) // 2, y + label))
            draw.text((x + 4, y + 4), f"{row_name} | {name}", fill="white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _depth_summary(
    depths: Mapping[str, Array], *, far_plane_mm: float, threshold_mm: float
) -> dict[str, object]:
    return {
        name: tactile_depth_summary(
            depths[name],
            far_plane_mm=far_plane_mm,
            contact_threshold_mm=threshold_mm,
        )
        for name in ("tactile_a", "tactile_b")
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one diagnostic-only native scripted expert probe."""

    _validate_args(args)
    upstream = _absolute_directory(args.upstream_root, "upstream_root")
    runtime_dir = _absolute_directory(args.runtime_dir, "runtime_dir")
    output = _output_path(args.output_dir)
    config = build_univtac_backend_config(args.task, action_spec=EE8_ACTION_SPEC)
    launcher_args = production_univtac_launcher_args()
    launcher_args["rendering_mode"] = args.rendering_mode
    _emit_stage("runtime_launch")
    runtime = launch_univtac_runtime(
        config,
        upstream_root=upstream,
        runtime_dir=runtime_dir,
        initial_seed=args.initial_seed,
        launcher_args=launcher_args,
        device=args.device,
        antialiasing_mode=args.antialiasing_mode,
    )
    backend = UniVTACIsaacBackend(config, runtime)
    stage = "backend_reset"
    try:
        _emit_stage(stage)
        context = PolicyEpisodeContext(
            episode_id=f"stock-expert-{args.task}-{args.initial_seed}",
            task=args.task,
            initial_seed=args.initial_seed,
            exogenous_seed=args.exogenous_seed,
            instruction=config.task.prompt,
            action_spec=EE8_ACTION_SPEC,
        )
        reset_started = time.monotonic()
        reset_receipt = backend.reset(context)
        reset_duration_s = time.monotonic() - reset_started
        stage = "reset_observation_capture"
        _emit_stage(stage)
        reset_record = backend.observe()
        reset_joint = backend.initial_canonical_joint9
        if reset_joint is None:
            raise RuntimeError("backend did not expose reset canonical joint9")
        reset_raw = backend._fresh_raw()
        reset_streams = _streams(reset_record)
        reset_model = model_streams(reset_streams)
        reset_depths = raw_depths(reset_raw, config)
        reset_diagnostics = capture_task_diagnostics(runtime.task, args.task)
        initial_success = _strict_predicate(runtime.task, "check_success")
        initial_early_stop = _strict_predicate(
            runtime.task,
            "check_early_stop",
            task_id=args.task,
        )

        task = cast(Any, runtime.task)
        original_mode = task.mode
        native_step_before = int(task.step_count)
        execution_error: dict[str, str] | None = None
        expert_started = time.monotonic()
        stage = "stock_expert_execution"
        _emit_stage(stage)
        try:
            task.mode = "eval_test"
            task.play_once()
        except Exception as error:  # diagnostic evidence must retain task failures
            execution_error = {
                "message": str(error),
                "type": type(error).__name__,
            }
        finally:
            task.mode = original_mode
        expert_duration_s = time.monotonic() - expert_started
        stage = "final_observation_capture"
        _emit_stage(stage)
        task._update_render()

        final_raw = backend._fresh_raw()
        final_conversion = backend._convert(final_raw, 1)
        final_record = final_conversion.record
        final_streams = _streams(final_record)
        final_model = model_streams(final_streams)
        final_depths = raw_depths(final_raw, config)
        final_diagnostics = capture_task_diagnostics(runtime.task, args.task)
        predicate_success = _strict_predicate(runtime.task, "check_success")
        early_stop = _strict_predicate(
            runtime.task,
            "check_early_stop",
            task_id=args.task,
        )
        plan_success = bool(task.plan_success)
        expert_success = (
            execution_error is None
            and plan_success
            and predicate_success
            and not early_stop
        )
        source_manifest = (
            Path(__file__).resolve().parents[2] / "release/source_manifest.sha256"
        )
        stage = "bundle_write"
        _emit_stage(stage)
        return cast(
            dict[str, object],
            write_dynamic_probe_bundle(
                output,
                domains={
                    "reset_raw_sensor_pixels": reset_streams,
                    "reset_checkpoint_color_sensor_pixels": reset_model,
                    "stock_expert_final_raw_sensor_pixels": final_streams,
                    "stock_expert_final_checkpoint_color_sensor_pixels": final_model,
                },
                states={
                    "reset_ee8": reset_record.observation.proprio,
                    "reset_joint9": reset_joint,
                    "stock_expert_final_ee8": final_record.observation.proprio,
                    "stock_expert_final_joint9": final_conversion.canonical_joint9,
                },
                panel=_expert_panel(reset_model, final_model),
                evidence_level="univtac_stock_scripted_expert_probe_v1",
                metadata={
                    "antialiasing_mode": args.antialiasing_mode,
                    "config_sha256": config.sha256,
                    "controller_contract": {
                        "action_source": "task_specific__play_once",
                        "dense_execution": "BaseTask.move_to_take_dense_action",
                        "diagnostic_only": True,
                        "mode": "eval_test",
                        "uses_curobo_position_and_velocity_waypoints": True,
                    },
                    "exogenous_seed": args.exogenous_seed,
                    "initial_seed": args.initial_seed,
                    "renderer_mode": args.rendering_mode,
                    "reset": {
                        "duration_s": reset_duration_s,
                        "initial_early_stop": initial_early_stop,
                        "initial_success": initial_success,
                        "receipt_sha256": reset_receipt.sha256,
                        "task_diagnostics": reset_diagnostics,
                    },
                    "source_manifest_sha256": _sha256_file(source_manifest),
                    "stock_expert": {
                        "duration_s": expert_duration_s,
                        "early_stop": early_stop,
                        "execution_error": execution_error,
                        "expert_success": expert_success,
                        "native_step_after": int(task.step_count),
                        "native_step_before": native_step_before,
                        "native_step_delta": int(task.step_count) - native_step_before,
                        "plan_success": plan_success,
                        "predicate_success": predicate_success,
                        "task_diagnostics": final_diagnostics,
                    },
                    "tactile_depth": {
                        "reset": _depth_summary(
                            reset_depths,
                            far_plane_mm=config.aliases.far_plane_mm,
                            threshold_mm=config.phase_tracker.on_threshold_mm,
                        ),
                        "stock_expert_final": _depth_summary(
                            final_depths,
                            far_plane_mm=config.aliases.far_plane_mm,
                            threshold_mm=config.phase_tracker.on_threshold_mm,
                        ),
                    },
                    "task_id": args.task,
                },
            ),
        )
    except BaseException as error:
        _emit_failure(stage, error)
        raise
    finally:
        backend.close()


def main() -> int:
    args = _parser().parse_args()
    document = run(args)
    stock_expert = cast(dict[str, object], document["stock_expert"])
    print(
        json.dumps(
            {
                "content_sha256": document["content_sha256"],
                "expert_success": stock_expert["expert_success"],
                "output_dir": str(args.output_dir.absolute()),
                "task_id": document["task_id"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
