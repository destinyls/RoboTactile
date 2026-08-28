#!/usr/bin/env python3
"""Run one source-bound, simulator-only N0 dynamic contract probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import NoReturn

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    N0_UNIVTAC_ANTIALIASING_MODE,
    launch_univtac_runtime,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.execution.contracts import production_univtac_launcher_args
from robotactile_benchmark.integrations.n0_twam.dynamic_contract import (
    cadence_gate,
    joint_state_error,
    registered_image_metrics,
    select_nearest_state_match,
    stream_freshness,
    tactile_depth_summary,
)
from robotactile_benchmark.integrations.n0_twam.dynamic_probe_io import (
    STREAM_NAMES,
    camera_freshness,
    camera_runtime_state,
    comparison_panel,
    model_streams,
    raw_depths,
    write_dynamic_probe_bundle,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity import (
    action_tracking_summary,
    summarize_image_stream,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.recorded.alignment_io import (
    load_expert_hdf5_frame,
    load_expert_hdf5_trajectory,
)
from scripts.live_univtac.capture_observation_parity_isaac import (
    _absolute_directory,
    _camera_contract,
    _output_path,
    _renderer_settings,
    _sha256_file,
    _streams,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--initial-seed", type=int, default=90)
    parser.add_argument("--exogenous-seed", type=int, default=20260825)
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
    parser.add_argument("--static-render-count", type=int, default=4)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("initial_seed", "exogenous_seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(args.static_render_count, bool)
        or not isinstance(args.static_render_count, int)
        or not 2 <= args.static_render_count <= 16
    ):
        raise ValueError("static_render_count must be in [2,16]")


def _regular_file(path: Path, name: str) -> Path:
    selected = path.absolute()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    return selected


def _emit_stage(stage: str, **facts: object) -> None:
    """Publish a flush-safe breadcrumb before native Isaac shutdown can run."""

    payload = {
        "event": "robotactile_n0_dynamic_probe_stage",
        "stage": stage,
        **facts,
    }
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def _raise_probe_failure(stage: str, error: BaseException) -> NoReturn:
    """Preserve the original traceback and reject clean-looking early exits."""

    _emit_stage(
        "failed",
        error_message=str(error),
        error_type=type(error).__name__,
        failed_stage=stage,
    )
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()
    if isinstance(error, SystemExit):
        raise RuntimeError(
            f"dynamic probe aborted via SystemExit during {stage}: {error.code!r}"
        ) from error
    raise error


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    upstream = _absolute_directory(args.upstream_root, "upstream_root")
    runtime_dir = _absolute_directory(args.runtime_dir, "runtime_dir")
    hdf5 = _regular_file(args.hdf5, "HDF5")
    output = _output_path(args.output_dir)
    expert = load_expert_hdf5_trajectory(hdf5, args.task)
    config = build_univtac_backend_config(args.task, action_spec=EE8_ACTION_SPEC)
    launcher_args = production_univtac_launcher_args()
    launcher_args["rendering_mode"] = args.rendering_mode
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
    stage = "runtime_launched"
    _emit_stage(stage)
    try:
        context = PolicyEpisodeContext(
            episode_id=f"n0-dynamic-probe-{args.task}-{args.initial_seed}",
            task=args.task,
            initial_seed=args.initial_seed,
            exogenous_seed=args.exogenous_seed,
            instruction=config.task.prompt,
            action_spec=EE8_ACTION_SPEC,
        )
        stage = "reset"
        _emit_stage(stage)
        reset_started = time.monotonic()
        reset_receipt = backend.reset(context)
        reset_duration_s = time.monotonic() - reset_started
        stage = "initial_observation"
        _emit_stage(stage, reset_duration_s=reset_duration_s)
        initial = backend.observe()
        static_records = [initial]
        update_render = getattr(runtime.task, "_update_render", None)
        if not callable(update_render):
            raise RuntimeError("pinned UniVTAC task lacks _update_render")
        stage = "static_render_capture"
        _emit_stage(stage, static_render_count=args.static_render_count)
        for _ in range(args.static_render_count - 1):
            update_render()
            static_records.append(backend._convert(backend._fresh_raw(), 0).record)
        stage = "pre_action_sensor_capture"
        _emit_stage(stage)
        before_record = static_records[-1]
        before_streams = _streams(before_record)
        before_model = model_streams(before_streams)
        before_raw = backend._fresh_raw()
        before_depths = raw_depths(before_raw, config)
        before_camera = camera_runtime_state(runtime.task)
        before_joint = backend.initial_canonical_joint9
        if before_joint is None:
            raise RuntimeError("backend did not expose initial canonical joint9")
        stage = "expert_state_match"
        _emit_stage(stage)
        match = select_nearest_state_match(
            expert.states, before_record.observation.proprio
        )
        reference = load_expert_hdf5_frame(hdf5, match.index)
        if reference.source_sha256 != expert.source_sha256:
            raise RuntimeError("HDF5 changed between trajectory and frame loading")
        target = np.array(
            expert.states[match.successor_index]
            if match.passed
            else before_record.observation.proprio,
            dtype=np.float32,
            copy=True,
        )
        target_source = (
            "matched_hdf5_successor" if match.passed else "live_hold_fallback"
        )
        stage = "single_action_execution"
        _emit_stage(
            stage,
            matched_hdf5_index=match.index,
            state_match_passed=match.passed,
            target_source=target_source,
        )
        action_started = time.monotonic()
        batch = backend.execute(target[None, :])
        action_duration_s = time.monotonic() - action_started
        if len(batch.transitions) != 1:
            raise RuntimeError("dynamic probe must produce exactly one transition")
        transition = batch.transitions[0]
        stage = "post_action_sensor_capture"
        _emit_stage(stage, action_duration_s=action_duration_s)
        after_record = transition.clean_record
        after_streams = _streams(after_record)
        after_model = model_streams(after_streams)
        after_depths = raw_depths(backend._fresh_raw(), config)
        after_camera = camera_runtime_state(runtime.task)
        after_joint = backend.latest_canonical_joint9
        if after_joint is None:
            raise RuntimeError("backend did not expose post-action canonical joint9")
        stage = "metric_computation"
        _emit_stage(stage)
        static_raw = {
            name: np.stack([_streams(record)[name] for record in static_records])
            for name in STREAM_NAMES
        }
        static_model = {
            name: np.stack(
                [model_streams(_streams(record))[name] for record in static_records]
            )
            for name in STREAM_NAMES
        }
        reference_streams = reference.streams()
        pixel_metrics = {
            name: registered_image_metrics(reference_streams[name], before_model[name])
            for name in STREAM_NAMES
        }
        depth_summaries = {
            domain: {
                name: tactile_depth_summary(
                    depths[name],
                    far_plane_mm=config.aliases.far_plane_mm,
                    contact_threshold_mm=config.phase_tracker.on_threshold_mm,
                )
                for name in ("tactile_a", "tactile_b")
            }
            for domain, depths in (
                ("hdf5", reference.depths()),
                ("live_pre", before_depths),
                ("live_post", after_depths),
            )
        }
        stage = "bundle_write"
        _emit_stage(stage, output_dir=str(output))
        document = write_dynamic_probe_bundle(
            output,
            domains={
                "hdf5_checkpoint": reference_streams,
                "live_pre_raw": before_streams,
                "live_pre_model": before_model,
                "live_post_raw": after_streams,
                "live_post_model": after_model,
                "live_static_raw": static_raw,
                "live_static_model": static_model,
            },
            states={
                "hdf5_matched_ee8": reference.state,
                "hdf5_matched_joint9": reference.joint9,
                "live_pre_ee8": before_record.observation.proprio,
                "live_pre_joint9": before_joint,
                "issued_target_ee8": target,
                "live_post_ee8": after_record.observation.proprio,
                "live_post_joint9": after_joint,
            },
            panel=comparison_panel(reference_streams, before_model, after_model),
            metadata={
                "action": {
                    "duration_s": action_duration_s,
                    "signal": transition.signal.value,
                    "target_source": target_source,
                    "tracking": action_tracking_summary(
                        target[None, :], after_record.observation.proprio[None, :]
                    ),
                },
                "antialiasing_mode": args.antialiasing_mode,
                "cadence_gate": cadence_gate(
                    transition.diagnostics,
                    sim_hz=config.sim_hz,
                    decimation=config.decimation,
                    physics_steps_per_action=config.physics_steps_per_action,
                ),
                "camera_contract": _camera_contract(runtime.task),
                "camera_runtime": {
                    "freshness": camera_freshness(before_camera, after_camera),
                    "post": after_camera,
                    "pre": before_camera,
                },
                "comparison_scope": "robot_state_matched_not_object_registered",
                "config_sha256": config.sha256,
                "hdf5": {
                    "matched_native_step": reference.native_step,
                    "path": str(hdf5),
                    "sha256": expert.source_sha256,
                },
                "initial_seed": args.initial_seed,
                "exogenous_seed": args.exogenous_seed,
                "input_profile": N0_LIVE_UNIVTAC_INPUT_PROFILE.to_dict(),
                "joint_state_match": joint_state_error(reference.joint9, before_joint),
                "pixel_metrics_diagnostic_only": pixel_metrics,
                "pixel_registered_ground_truth": False,
                "renderer_settings": _renderer_settings(),
                "rendering_mode": args.rendering_mode,
                "reset": {
                    "duration_s": reset_duration_s,
                    "receipt_sha256": reset_receipt.sha256,
                    "task_diagnostics": reset_receipt.diagnostics.get("task"),
                },
                "source_manifest_sha256": _sha256_file(
                    Path(__file__).resolve().parents[2]
                    / "release"
                    / "source_manifest.sha256"
                ),
                "state_match": match.to_dict(),
                "stream_freshness": stream_freshness(before_streams, after_streams),
                "static_stream_statistics": {
                    domain: {
                        name: summarize_image_stream(streams[name]).to_dict()
                        for name in STREAM_NAMES
                    }
                    for domain, streams in (
                        ("raw", static_raw),
                        ("model", static_model),
                    )
                },
                "tactile_depth": depth_summaries,
                "task_id": args.task,
                "transition_diagnostics": dict(transition.diagnostics),
                "upstream_task_source_sha256": runtime.handshake.task_source_sha256,
            },
        )
        stage = "completed"
        _emit_stage(stage, content_sha256=document["content_sha256"])
        return document
    except BaseException as error:
        _raise_probe_failure(stage, error)
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    document = run(_parser().parse_args(argv))
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
