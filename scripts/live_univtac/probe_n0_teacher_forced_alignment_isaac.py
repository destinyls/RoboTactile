#!/usr/bin/env python3
"""Replay one expert qpos checkpoint and capture simulator/train parity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

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
from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import Array
from robotactile_benchmark.execution.contracts import production_univtac_launcher_args
from robotactile_benchmark.integrations.n0_twam.dynamic_contract import (
    joint_state_error,
    registered_image_metrics,
    select_nearest_state_match,
    state_match_at_index,
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
    summarize_image_stream,
)
from robotactile_benchmark.integrations.n0_twam.preprocessing_reference import (
    resize_cv2_area,
)
from robotactile_benchmark.integrations.n0_twam.teacher_forced_placement import (
    apply_lift_bottle_hdf5_initial_placement,
)
from robotactile_benchmark.integrations.n0_twam.teacher_forced_replay import (
    native_step,
    replay_official_qpos_sequence,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.recorded.alignment_io import (
    load_expert_hdf5_actor_poses,
    load_expert_hdf5_frame,
    load_expert_hdf5_joint9_trajectory,
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
from scripts.live_univtac.probe_n0_dynamic_contract_isaac import (
    _emit_stage,
    _raise_probe_failure,
    _regular_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--hdf5-index", type=int, default=0)
    parser.add_argument("--replay-stride", type=int, default=1)
    parser.add_argument(
        "--actor-placement",
        choices=("reset", "hdf5_initial"),
        default="reset",
        help="explicit diagnostic-only task-actor placement contract",
    )
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
    for name in ("initial_seed", "exogenous_seed", "hdf5_index"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(args.replay_stride, bool)
        or not isinstance(args.replay_stride, int)
        or args.replay_stride < 1
    ):
        raise ValueError("replay_stride must be a positive integer")
    if args.hdf5_index % args.replay_stride != 0:
        raise ValueError("hdf5_index must lie on the official replay stride")
    if args.actor_placement == "hdf5_initial" and args.task != "lift_bottle":
        raise ValueError("hdf5_initial actor placement currently requires lift_bottle")
    if (
        isinstance(args.static_render_count, bool)
        or not isinstance(args.static_render_count, int)
        or not 2 <= args.static_render_count <= 16
    ):
        raise ValueError("static_render_count must be in [2,16]")


def _n0_server_pixel_streams(
    streams: dict[str, Array],
) -> dict[str, Array]:
    """Mirror the pinned N0 server's pre-normalization INTER_AREA resize."""

    missing = sorted(set(STREAM_NAMES).difference(streams))
    if missing:
        raise ValueError(f"missing N0 image stream: {missing[0]}")
    result: dict[str, Array] = {}
    for name in STREAM_NAMES:
        values = np.asarray(streams[name])
        if values.ndim not in (3, 4):
            raise ValueError(f"{name} must be uint8 HWC or FHWC")
        batched = values[None, ...] if values.ndim == 3 else values
        size = 128 if name in ("tactile_a", "tactile_b") else 256
        resized = resize_cv2_area(batched, width=size, height=size)
        result[name] = resized[0] if values.ndim == 3 else resized
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    upstream = _absolute_directory(args.upstream_root, "upstream_root")
    runtime_dir = _absolute_directory(args.runtime_dir, "runtime_dir")
    hdf5 = _regular_file(args.hdf5, "HDF5")
    output = _output_path(args.output_dir)
    expert = load_expert_hdf5_trajectory(hdf5, args.task)
    joint9_trajectory, joint_source_sha256 = load_expert_hdf5_joint9_trajectory(hdf5)
    reference = load_expert_hdf5_frame(hdf5, args.hdf5_index)
    actor_poses: dict[str, Array] | None = None
    actor_source_sha256: str | None = None
    if args.actor_placement == "hdf5_initial":
        actor_poses, actor_source_sha256 = load_expert_hdf5_actor_poses(
            hdf5,
            ("bottle", "wall"),
        )
    if (
        reference.source_sha256 != expert.source_sha256
        or joint_source_sha256 != expert.source_sha256
        or actor_source_sha256 not in (None, expert.source_sha256)
        or len(joint9_trajectory) != expert.count
    ):
        raise RuntimeError("HDF5 changed while loading teacher-forced reference")
    if args.hdf5_index >= expert.count - 1:
        raise ValueError("hdf5_index must select a row with a successor")
    config = build_univtac_backend_config(args.task, action_spec=EE8_ACTION_SPEC)
    launcher_args = production_univtac_launcher_args()
    launcher_args["rendering_mode"] = args.rendering_mode
    stage = "runtime_launch"
    _emit_stage(stage)
    try:
        runtime = launch_univtac_runtime(
            config,
            upstream_root=upstream,
            runtime_dir=runtime_dir,
            initial_seed=args.initial_seed,
            launcher_args=launcher_args,
            device=args.device,
            antialiasing_mode=args.antialiasing_mode,
        )
    except BaseException as error:
        _raise_probe_failure(stage, error)
    backend = UniVTACIsaacBackend(config, runtime)
    stage = "runtime_launched"
    _emit_stage(stage)
    try:
        context = PolicyEpisodeContext(
            episode_id=f"n0-teacher-forced-{args.task}-{args.initial_seed}",
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
        reset_record = backend.observe()
        reset_joint = backend.initial_canonical_joint9
        if reset_joint is None:
            raise RuntimeError("backend did not expose reset canonical joint9")
        reset_raw = backend._fresh_raw()
        reset_depths = raw_depths(reset_raw, config)
        reset_streams = _streams(reset_record)
        reset_model = model_streams(reset_streams)
        reset_camera = camera_runtime_state(runtime.task)
        reset_task = capture_task_diagnostics(runtime.task, args.task)

        actor_placement: dict[str, object] = {
            "applied": False,
            "mode": "reset",
        }
        aligned_record = None
        aligned_streams = None
        aligned_model = None
        aligned_depths = None
        if actor_poses is not None:
            stage = "hdf5_initial_actor_placement"
            _emit_stage(stage)
            placement_witness = apply_lift_bottle_hdf5_initial_placement(
                runtime.task,
                actor_poses,
            )
            aligned_raw = backend._fresh_raw()
            aligned_record = backend._convert(aligned_raw, 0).record
            aligned_streams = _streams(aligned_record)
            aligned_model = model_streams(aligned_streams)
            aligned_depths = raw_depths(aligned_raw, config)
            actor_placement = {
                "applied": True,
                "hdf5_source_sha256": actor_source_sha256,
                "mode": "hdf5_initial",
                "witness": placement_witness,
            }

        stage = "official_qpos_replay"
        native_before = native_step(runtime.task)
        _emit_stage(
            stage,
            hdf5_index=args.hdf5_index,
            native_step_before=native_before,
            replay_stride=args.replay_stride,
        )
        replay = replay_official_qpos_sequence(
            task=runtime.task,
            backend=backend,
            joint9_trajectory=joint9_trajectory,
            expert_states=expert.states,
            target_index=args.hdf5_index,
            stride=args.replay_stride,
            physics_steps_per_target=config.physics_steps_per_action,
        )
        replay_conversion = replay.target_conversion
        replay_raw = replay.target_raw
        native_after = replay.native_step_after
        previous_conversion = replay.previous_conversion
        previous_index = replay.previous_index
        previous_streams = (
            None
            if previous_conversion is None
            else _streams(previous_conversion.record)
        )
        previous_model = (
            None if previous_streams is None else model_streams(previous_streams)
        )
        update_render = getattr(runtime.task, "_update_render", None)
        if not callable(update_render):
            raise RuntimeError("pinned UniVTAC task lacks _update_render")

        stage = "replayed_observation"
        _emit_stage(stage, native_step_after=native_after)
        replay_record = replay_conversion.record
        replay_joint = replay_conversion.canonical_joint9
        replay_depths = raw_depths(replay_raw, config)
        replay_streams = _streams(replay_record)
        replay_model = model_streams(replay_streams)
        replay_camera = camera_runtime_state(runtime.task)
        replay_task = capture_task_diagnostics(runtime.task, args.task)
        static_records = [replay_record]
        for _ in range(args.static_render_count - 1):
            update_render()
            static_records.append(backend._convert(backend._fresh_raw(), 0).record)
        static_model = {
            name: np.stack(
                [model_streams(_streams(record))[name] for record in static_records]
            )
            for name in STREAM_NAMES
        }

        stage = "metric_computation"
        _emit_stage(stage)
        target_match = state_match_at_index(
            expert.states, replay_record.observation.proprio, args.hdf5_index
        )
        nearest_match = select_nearest_state_match(
            expert.states, replay_record.observation.proprio
        )
        reference_streams = reference.streams()
        reference_server_pixels = _n0_server_pixel_streams(reference_streams)
        reset_server_pixels = _n0_server_pixel_streams(reset_model)
        replay_server_pixels = _n0_server_pixel_streams(replay_model)
        static_server_pixels = _n0_server_pixel_streams(static_model)
        checkpoint_color_pixel_metrics = {
            name: registered_image_metrics(reference_streams[name], replay_model[name])
            for name in STREAM_NAMES
        }
        server_pixel_metrics = {
            name: registered_image_metrics(
                reference_server_pixels[name],
                replay_server_pixels[name],
                border_px=3,
            )
            for name in STREAM_NAMES
        }
        previous_checkpoint_color_pixel_metrics = (
            None
            if previous_model is None
            else {
                name: registered_image_metrics(
                    reference_streams[name], previous_model[name]
                )
                for name in STREAM_NAMES
            }
        )
        previous_server_pixels = (
            None if previous_model is None else _n0_server_pixel_streams(previous_model)
        )
        previous_server_pixel_metrics = (
            None
            if previous_server_pixels is None
            else {
                name: registered_image_metrics(
                    reference_server_pixels[name],
                    previous_server_pixels[name],
                    border_px=3,
                )
                for name in STREAM_NAMES
            }
        )
        depth_domains = [
            ("hdf5", reference.depths()),
            ("live_reset", reset_depths),
            ("live_replayed", replay_depths),
        ]
        if aligned_depths is not None:
            depth_domains.append(("live_actor_aligned", aligned_depths))
        tactile_depth = {
            domain: {
                name: tactile_depth_summary(
                    depths[name],
                    far_plane_mm=config.aliases.far_plane_mm,
                    contact_threshold_mm=config.phase_tracker.on_threshold_mm,
                )
                for name in ("tactile_a", "tactile_b")
            }
            for domain, depths in depth_domains
        }

        stage = "bundle_write"
        _emit_stage(stage, output_dir=str(output))
        domains = {
            "hdf5_checkpoint_color_sensor_pixels": reference_streams,
            "hdf5_n0_server_pixels": reference_server_pixels,
            "live_reset_raw_sensor_pixels": reset_streams,
            "live_reset_checkpoint_color_sensor_pixels": reset_model,
            "live_reset_n0_server_pixels": reset_server_pixels,
            "live_replayed_raw_sensor_pixels": replay_streams,
            "live_replayed_checkpoint_color_sensor_pixels": replay_model,
            "live_replayed_n0_server_pixels": replay_server_pixels,
            "live_replayed_static_checkpoint_color_sensor_pixels": static_model,
            "live_replayed_static_n0_server_pixels": static_server_pixels,
        }
        states = {
            "hdf5_target_ee8": reference.state,
            "hdf5_target_joint9": reference.joint9,
            "live_reset_ee8": reset_record.observation.proprio,
            "live_reset_joint9": reset_joint,
            "live_replayed_ee8": replay_record.observation.proprio,
            "live_replayed_joint9": replay_joint,
        }
        if (
            aligned_record is not None
            and aligned_streams is not None
            and aligned_model is not None
        ):
            domains["live_actor_aligned_raw_sensor_pixels"] = aligned_streams
            domains["live_actor_aligned_checkpoint_color_sensor_pixels"] = aligned_model
            domains["live_actor_aligned_n0_server_pixels"] = _n0_server_pixel_streams(
                aligned_model
            )
            states["live_actor_aligned_ee8"] = aligned_record.observation.proprio
            states["live_actor_aligned_joint9"] = reset_joint
        if (
            previous_streams is not None
            and previous_model is not None
            and previous_conversion is not None
        ):
            domains["live_replayed_previous_raw_sensor_pixels"] = previous_streams
            domains["live_replayed_previous_checkpoint_color_sensor_pixels"] = (
                previous_model
            )
            if previous_server_pixels is not None:
                domains["live_replayed_previous_n0_server_pixels"] = (
                    previous_server_pixels
                )
            states["live_replayed_previous_ee8"] = (
                previous_conversion.record.observation.proprio
            )
            states["live_replayed_previous_joint9"] = (
                previous_conversion.canonical_joint9
            )
        document = cast(
            dict[str, object],
            write_dynamic_probe_bundle(
                output,
                domains=domains,
                states=states,
                panel=comparison_panel(
                    reference_streams,
                    aligned_model if aligned_model is not None else reset_model,
                    replay_model,
                ),
                evidence_level=(
                    "univtac_n0_simulator_hdf5_actor_placed_teacher_forced_alignment_v1"
                    if actor_poses is not None
                    else "univtac_n0_simulator_timestamped_teacher_forced_alignment_v3"
                ),
                metadata={
                    "antialiasing_mode": args.antialiasing_mode,
                    "actor_placement": actor_placement,
                    "camera_contract": _camera_contract(runtime.task),
                    "camera_runtime": {
                        "freshness": camera_freshness(reset_camera, replay_camera),
                        "replayed": replay_camera,
                        "reset": reset_camera,
                    },
                    "comparison_scope": (
                        "timestamped_teacher_forced_robot_qpos_and_actor_dynamics"
                    ),
                    "config_sha256": config.sha256,
                    "hdf5": {
                        "index": args.hdf5_index,
                        "native_step": reference.native_step,
                        "path": str(hdf5),
                        "pixel_provenance": (
                            "legacy_hdf5_jpeg_pil_rgb_decode_training_proxy"
                        ),
                        "pixel_provenance_limitation": (
                            "not_the_exact_lerobot_h264_frame_or_training_latent"
                        ),
                        "sha256": expert.source_sha256,
                    },
                    "initial_seed": args.initial_seed,
                    "exogenous_seed": args.exogenous_seed,
                    "input_profile": N0_LIVE_UNIVTAC_INPUT_PROFILE.to_dict(),
                    "joint_state_match_target": joint_state_error(
                        reference.joint9, replay_joint
                    ),
                    "n0_server_pixel_contract": {
                        "camera_shape_hwc": [256, 256, 3],
                        "diagnostic_mirror_only": True,
                        "interpolation": "cv2.INTER_AREA",
                        "normalization_after_resize": "float32_div127.5_minus1",
                        "production_resize_location": "pinned_n0_server",
                        "tactile_shape_hwc": [128, 128, 3],
                    },
                    "pixel_metrics_checkpoint_color_sensor_resolution_diagnostic_only": (
                        checkpoint_color_pixel_metrics
                    ),
                    "pixel_metrics_n0_server_pixel_domain_diagnostic_only": (
                        server_pixel_metrics
                    ),
                    "previous_replay_checkpoint_color_pixel_metrics_diagnostic_only": (
                        previous_checkpoint_color_pixel_metrics
                    ),
                    "previous_replay_n0_server_pixel_metrics_diagnostic_only": (
                        previous_server_pixel_metrics
                    ),
                    "pixel_registered_ground_truth": False,
                    "renderer_settings": _renderer_settings(),
                    "rendering_mode": args.rendering_mode,
                    "reset": {
                        "duration_s": reset_duration_s,
                        "receipt_sha256": reset_receipt.sha256,
                        "task_diagnostics": reset_task,
                    },
                    "robot_state_registered": target_match.passed,
                    "source_manifest_sha256": _sha256_file(
                        Path(__file__).resolve().parents[2]
                        / "release"
                        / "source_manifest.sha256"
                    ),
                    "state_match_nearest": nearest_match.to_dict(),
                    "state_match_target": target_match.to_dict(),
                    "static_render_count": args.static_render_count,
                    "static_checkpoint_color_stream_statistics": {
                        name: summarize_image_stream(stream).to_dict()
                        for name, stream in static_model.items()
                    },
                    "tactile_depth": tactile_depth,
                    "task_id": args.task,
                    "teacher_forced_replay": {
                        "action_count": len(replay.indices),
                        "action_type": "qpos",
                        "duration_s": replay.duration_s,
                        "execution_success": replay.execution_success,
                        "force": True,
                        "native_step_after": native_after,
                        "native_step_before": native_before,
                        "native_step_delta": native_after - native_before,
                        "plan_success": replay.plan_success,
                        "previous_hdf5_index": previous_index,
                        "qpos8": [float(value) for value in reference.joint9[:8]],
                        "replay_indices": list(replay.indices),
                        "replay_stride": args.replay_stride,
                        "physics_steps_per_target": config.physics_steps_per_action,
                        "replay_trace": list(replay.trace),
                        "returned_success": replay.returned_success,
                        "task_diagnostics": replay_task,
                    },
                    "upstream_task_source_sha256": runtime.handshake.task_source_sha256,
                },
            ),
        )
        stage = "completed"
        _emit_stage(stage, content_sha256=document["content_sha256"])
        return document
    except BaseException as error:
        _raise_probe_failure(stage, error)
        raise RuntimeError("unreachable after probe failure") from error
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    document = run(_parser().parse_args(argv))
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
