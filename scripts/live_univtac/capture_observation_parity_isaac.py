#!/usr/bin/env python3
"""Capture static UniVTAC pixels at the raw and N0 model boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    UNIVTAC_ANTIALIASING_MODE,
    UNIVTAC_RENDERING_MODE,
    launch_univtac_runtime,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity import (
    summarize_image_stream,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
    prepare_n0_image,
)

_RENDERER_KEYS = (
    "/app/hydraEngine/waitIdle",
    "/app/renderer/waitIdle",
    "/omni/replicator/asyncRendering",
    "/rtx-transient/dldenoiser/enabled",
    "/rtx-transient/dlssg/enabled",
    "/rtx/ambientOcclusion/enabled",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "/rtx/indirectDiffuse/enabled",
    "/rtx/post/aa/op",
    "/rtx/post/dlss/execMode",
    "/rtx/raytracing/cached/enabled",
    "/rtx/reflections/enabled",
    "/rtx/translucency/enabled",
)
_STREAM_NAMES = ("top", "wrist_l", "tactile_a", "tactile_b")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rendering-mode",
        choices=("balanced", "performance", "quality"),
        default=UNIVTAC_RENDERING_MODE,
    )
    parser.add_argument(
        "--antialiasing-mode",
        choices=("Off", "FXAA", "DLSS", "TAA", "DLAA"),
        default=UNIVTAC_ANTIALIASING_MODE,
    )
    parser.add_argument("--static-render-count", type=int, default=8)
    return parser


def _absolute_directory(path: Path, name: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise ValueError(f"{name} must be a non-symlink existing directory")
    return absolute


def _output_path(path: Path) -> Path:
    absolute = path.absolute()
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError("output directory must not already exist")
    if not absolute.parent.is_dir() or absolute.parent.is_symlink():
        raise ValueError("output parent must be a non-symlink existing directory")
    return absolute


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("initial_seed", "exogenous_seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(args.static_render_count, bool)
        or not isinstance(args.static_render_count, int)
        or not 1 <= args.static_render_count <= 64
    ):
        raise ValueError("static_render_count must be in [1, 64]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _renderer_settings() -> dict[str, object]:
    import carb  # type: ignore[import-not-found]

    settings = carb.settings.get_settings()
    result: dict[str, object] = {}
    for key in _RENDERER_KEYS:
        value = settings.get(key)
        result[key] = (
            value
            if value is None or isinstance(value, (bool, int, float, str))
            else repr(value)
        )
    return result


def _plain(value: Any) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    try:
        return [_plain(item) for item in value]
    except TypeError:
        return repr(value)


def _camera_contract(task: Any) -> dict[str, object]:
    """Read configured and effective USD camera facts from the live stage."""

    import omni.usd  # type: ignore[import-not-found]
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac stage is unavailable")
    result: dict[str, object] = {}
    for camera_cfg in task.cfg.cameras:
        configured_path = str(camera_cfg.prim_path)
        prim_path = configured_path.replace("env_.*", "env_0")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"live camera prim is missing: {prim_path}")
        camera = UsdGeom.Camera(prim)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        offset = getattr(camera_cfg, "offset", None)
        result[str(camera_cfg.name)] = {
            "configured": {
                "data_types": list(camera_cfg.data_types),
                "height": int(camera_cfg.height),
                "offset": None
                if offset is None
                else {
                    "convention": str(offset.convention),
                    "position_m": _plain(offset.pos),
                    "quaternion_wxyz": _plain(offset.rot),
                },
                "prim_path_expression": configured_path,
                "sensor_update_hz": 1.0 / float(camera_cfg.update_period),
                "update_period_s": float(camera_cfg.update_period),
                "width": int(camera_cfg.width),
            },
            "effective_usd": {
                "clipping_range": _plain(camera.GetClippingRangeAttr().Get()),
                "focal_length": _plain(camera.GetFocalLengthAttr().Get()),
                "focus_distance": _plain(camera.GetFocusDistanceAttr().Get()),
                "horizontal_aperture": _plain(camera.GetHorizontalApertureAttr().Get()),
                "prim_path": prim_path,
                "projection": _plain(camera.GetProjectionAttr().Get()),
                "vertical_aperture": _plain(camera.GetVerticalApertureAttr().Get()),
                "world_transform": [
                    [float(matrix[row][column]) for column in range(4)]
                    for row in range(4)
                ],
            },
        }
    return result


def _streams(record: Any) -> dict[str, Array]:
    observation = record.observation
    return {
        "top": np.asarray(observation.vision["top"]),
        "wrist_l": np.asarray(observation.vision["wrist_l"]),
        "tactile_a": np.asarray(observation.sensor("left").payload),
        "tactile_b": np.asarray(observation.sensor("right").payload),
    }


def _save_array(path: Path, value: Array) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "dtype": str(value.dtype),
        "path": str(path),
        "sha256": _sha256_file(path),
        "shape": list(value.shape),
    }


def _write_bundle(
    output: Path,
    *,
    raw_frames: dict[str, Array],
    model_frames: dict[str, Array],
    metadata: dict[str, object],
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays: dict[str, dict[str, dict[str, object]]] = {
            "raw": {},
            "model": {},
        }
        for domain, streams in (("raw", raw_frames), ("model", model_frames)):
            for name in _STREAM_NAMES:
                relative = Path("arrays") / domain / f"{name}.npy"
                arrays[domain][name] = _save_array(temporary / relative, streams[name])
                arrays[domain][name]["path"] = str(relative)
        document = {
            "arrays": arrays,
            "evidence_level": "univtac_static_observation_parity_probe_v1",
            **metadata,
        }
        document["content_sha256"] = canonical_hash(document)
        serialized = (
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        manifest = temporary / "capture.json"
        with manifest.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, output)
        return document
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    upstream_root = _absolute_directory(args.upstream_root, "upstream_root")
    runtime_dir = _absolute_directory(args.runtime_dir, "runtime_dir")
    output = _output_path(args.output_dir)
    config = build_univtac_backend_config(args.task, action_spec=EE8_ACTION_SPEC)
    launcher_args = production_univtac_launcher_args()
    launcher_args["rendering_mode"] = args.rendering_mode
    runtime = launch_univtac_runtime(
        config,
        upstream_root=upstream_root,
        runtime_dir=runtime_dir,
        initial_seed=args.initial_seed,
        launcher_args=launcher_args,
        device=args.device,
        antialiasing_mode=args.antialiasing_mode,
    )
    backend = UniVTACIsaacBackend(config, runtime)
    try:
        context = PolicyEpisodeContext(
            episode_id=(
                f"observation-parity-{args.task}-{args.initial_seed}-"
                f"{args.rendering_mode}"
            ),
            task=args.task,
            initial_seed=args.initial_seed,
            exogenous_seed=args.exogenous_seed,
            instruction=config.task.prompt,
            action_spec=EE8_ACTION_SPEC,
        )
        started = time.monotonic()
        receipt = backend.reset(context)
        records = [backend.observe()]
        update_render = getattr(runtime.task, "_update_render", None)
        if not callable(update_render):
            raise RuntimeError("pinned UniVTAC task lacks _update_render")
        for _ in range(args.static_render_count - 1):
            update_render()
            records.append(backend._convert(backend._fresh_raw(), 0).record)
        raw_frames = {
            name: np.stack([_streams(record)[name] for record in records])
            for name in _STREAM_NAMES
        }
        model_frames = {
            name: np.stack(
                [
                    prepare_n0_image(
                        frame,
                        profile=N0_LIVE_UNIVTAC_INPUT_PROFILE,
                        name=name,
                    )
                    for frame in raw_frames[name]
                ]
            )
            for name in _STREAM_NAMES
        }
        return _write_bundle(
            output,
            raw_frames=raw_frames,
            model_frames=model_frames,
            metadata={
                "config_sha256": config.sha256,
                "camera_contract": _camera_contract(runtime.task),
                "antialiasing_mode": args.antialiasing_mode,
                "input_profile": N0_LIVE_UNIVTAC_INPUT_PROFILE.to_dict(),
                "initial_seed": args.initial_seed,
                "exogenous_seed": args.exogenous_seed,
                "native_step_id": backend.initial_native_step_id,
                "renderer_settings": _renderer_settings(),
                "rendering_mode": args.rendering_mode,
                "reset_duration_s": time.monotonic() - started,
                "reset_receipt_sha256": receipt.sha256,
                "source_manifest_sha256": _sha256_file(
                    Path(__file__).resolve().parents[2]
                    / "release"
                    / "source_manifest.sha256"
                ),
                "static_render_count": args.static_render_count,
                "stream_statistics": {
                    domain: {
                        name: summarize_image_stream(streams[name]).to_dict()
                        for name in _STREAM_NAMES
                    }
                    for domain, streams in (
                        ("raw", raw_frames),
                        ("model", model_frames),
                    )
                },
                "task_id": args.task,
                "temporal_contract": {
                    "action_execution_contract": receipt.diagnostics[
                        "action_execution_contract"
                    ],
                    "camera_delivery_hz": receipt.diagnostics["camera_delivery_hz"],
                    "decimation": receipt.diagnostics["decimation"],
                    "native_step_contract": receipt.diagnostics["native_step_contract"],
                    "physics_steps_per_action": receipt.diagnostics[
                        "physics_steps_per_action"
                    ],
                    "sim_hz": receipt.diagnostics["sim_hz"],
                },
                "upstream_task_source_sha256": runtime.handshake.task_source_sha256,
            },
        )
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    document = run(_parser().parse_args(argv))
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
