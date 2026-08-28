#!/usr/bin/env python3
"""Freeze one immutable N0-TWAM x UniVTAC Clean experiment identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_factory import (
    N0_UNIVTAC_ANTIALIASING_MODE,
)
from robotactile_benchmark.backends.univtac_registry import build_config
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import (
    KEYFRAMES_PER_FRAME,
    NATIVE_ACTION_SHAPE,
    SLOTS_PER_FRAME,
    n0_training_prompt,
)

_CAMERA_ASSETS = (
    "envs/_base_task.py",
    "envs/sensors/camera.py",
    (
        "third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/"
        "Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd"
    ),
    (
        "third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/"
        "Franka/GelSight_Mini/Gripper/WristCamera.usd"
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-config", type=Path, required=True)
    parser.add_argument("--univtac-root", type=Path, required=True)
    parser.add_argument("--n0-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--planned-live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--parity-seed", type=int, default=90)
    parser.add_argument("--clean-seed", type=int, default=100)
    parser.add_argument(
        "--camera-asset-relative",
        action="append",
        dest="camera_assets",
        default=None,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    selected = path.absolute()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    return selected


def _file_receipt(path: Path, name: str) -> dict[str, object]:
    selected = _regular_file(path, name)
    return {
        "path": str(selected),
        "sha256": _sha256_file(selected),
        "size_bytes": selected.stat().st_size,
    }


def _checkout_receipt(integration_id: str, root: Path) -> dict[str, object]:
    lock = load_integration_lock()
    receipt = verify_external_checkout(lock.by_id(integration_id), root.absolute())
    return {
        "commit_sha": receipt.commit_sha,
        "integration_id": receipt.integration_id,
        "lock_sha256": receipt.lock_sha256,
        "path": str(root.absolute()),
        "repository_url": receipt.repository_url,
    }


def _camera_receipts(root: Path, relative_paths: tuple[str, ...]) -> dict[str, object]:
    upstream = root.absolute()
    result: dict[str, object] = {}
    for value in relative_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or value in result:
            raise ValueError("camera asset relative path is invalid or duplicated")
        selected = (upstream / relative).absolute()
        if upstream not in selected.parents:
            raise ValueError("camera asset escapes the UniVTAC checkout")
        result[value] = _file_receipt(selected, f"camera asset {value}")
    if not result:
        raise ValueError("at least one camera asset is required")
    return result


def _write_once(path: Path, document: object) -> str:
    output = path.absolute()
    payload = canonical_json_bytes(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("experiment lock output cannot be a symlink")
    if output.exists():
        if not output.is_file() or output.read_bytes() != payload:
            raise FileExistsError("experiment lock already differs")
        return _sha256_file(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(output)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("parity_seed", "clean_seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if args.parity_seed == args.clean_seed:
        raise ValueError("parity_seed and clean_seed must be distinct")
    planned = args.planned_live_artifact.absolute()
    if planned.exists() or planned.is_symlink():
        raise FileExistsError("planned live artifact already exists")


def _prompt_contract(
    task_id: str, checkpoint_prompts: object, simulator_instruction: str
) -> dict[str, object]:
    policy_prompt = n0_training_prompt(task_id)
    if (
        not isinstance(checkpoint_prompts, dict)
        or checkpoint_prompts.get(task_id) != policy_prompt
    ):
        raise ValueError("checkpoint and N0 policy prompts disagree")
    return {
        "checkpoint_and_policy_match": True,
        "policy_training_prompt": policy_prompt,
        "simulator_task_instruction": simulator_instruction,
        "simulator_instruction_is_model_input": False,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    """Validate all inputs and publish one content-addressed lock."""

    _validate_args(args)
    repository_root = Path(__file__).resolve().parents[2]
    source_manifest = repository_root / "release" / "source_manifest.sha256"
    task_registry = repository_root / "configs" / "univtac" / "tasks_v1.json"
    wheel = _regular_file(args.wheel, "RoboTactile wheel")
    hdf5 = _regular_file(args.hdf5, "parity HDF5")
    converter = _regular_file(args.converter, "checkpoint converter")
    runtime = resolve_n0_runtime_artifacts(args.integration_config.absolute())
    manifest = runtime.manifest
    if manifest.task_id != args.task:
        raise ValueError("N0 artifact task does not match the experiment task")
    expected_converter = (
        manifest.checkpoint_root / "norm" / "convert_univtac_single_rot6d.py"
    )
    if converter.resolve(strict=True) != expected_converter.resolve(strict=True):
        raise ValueError("converter is not the released checkpoint converter")
    config = build_config(args.task, action_spec=EE8_ACTION_SPEC)
    prompt = n0_training_prompt(args.task)
    prompts = json.loads(manifest.prompt_manifest_path.read_text(encoding="utf-8"))
    prompt_contract = _prompt_contract(args.task, prompts, config.task.prompt)
    camera_assets = tuple(args.camera_assets or _CAMERA_ASSETS)
    components = {
        "camera_assets": _camera_receipts(args.univtac_root, camera_assets),
        "checkpoint": {
            "action_mode": manifest.action_mode,
            "path": str(manifest.checkpoint_path),
            "repository": manifest.checkpoint_repository,
            "revision": manifest.checkpoint_revision,
            "sha256": manifest.checkpoint_sha256,
        },
        "converter": _file_receipt(converter, "checkpoint converter"),
        "hdf5_parity_episode": _file_receipt(hdf5, "parity HDF5"),
        "artifact_manifest": _file_receipt(
            runtime.manifest_path, "N0 artifact manifest"
        ),
        "integration_config": _file_receipt(
            args.integration_config, "N0 integration config"
        ),
        "normalizer": {
            "path": str(manifest.normalizer_path),
            "sha256": manifest.normalizer_sha256,
        },
        "robotactile_source_manifest": _file_receipt(
            source_manifest, "source manifest"
        ),
        "robotactile_wheel": _file_receipt(wheel, "RoboTactile wheel"),
        "serve_bundle": {
            "path": str(manifest.serve_bundle_manifest_path),
            "sha256": manifest.serve_bundle_sha256,
        },
        "task_registry": _file_receipt(task_registry, "task registry"),
    }
    trial_identity = {
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "clean_seed": args.clean_seed,
        "model": "n0_twam_univtac_delta",
        "prompt": prompt,
        "task_id": args.task,
    }
    document: dict[str, object] = {
        "components": components,
        "evidence_level": "n0_univtac_experiment_preregistration_v1",
        "external_sources": {
            "n0_twam": _checkout_receipt("n0_twam", args.n0_root),
            "univtac": _checkout_receipt("univtac", args.univtac_root),
        },
        "model_contract": {
            "checkpoint_action_mode": "delta",
            "cold_execute_frame_indices": [1],
            "input_profile": N0_LIVE_UNIVTAC_INPUT_PROFILE.to_dict(),
            "keyframes_per_frame": KEYFRAMES_PER_FRAME,
            "native_action_shape": list(NATIVE_ACTION_SHAPE),
            "prompt": prompt_contract,
            "served_action_mode": "absolute_ee8",
            "slots_per_frame": SLOTS_PER_FRAME,
            "tactile_keys": ["tactile_a", "tactile_b"],
            "vision_keys": ["top", "wrist_l"],
            "warm_execute_frame_indices": [0, 1],
        },
        "planned_live_artifact": {
            "absent_at_lock_time": True,
            "path": str(args.planned_live_artifact.absolute()),
        },
        "schema_version": "robotactile-n0-univtac-experiment-lock-v1",
        "simulator_contract": {
            "action_execution_contract": config.action_execution_contract,
            "antialiasing_mode": N0_UNIVTAC_ANTIALIASING_MODE,
            "decimation": config.decimation,
            "endpoint_render_hz": (config.sim_hz / config.physics_steps_per_action),
            "endpoint_render_policy": ("one_endpoint_render_no_intermediate_render_v1"),
            "native_step_contract": config.native_step_contract,
            "physics_steps_per_action": config.fixed_physics_steps_per_action,
            "sim_hz": config.sim_hz,
        },
        "trial": {
            **trial_identity,
            "parity_seed": args.parity_seed,
            "trial_id": f"n0-clean-{args.task}-seed{args.clean_seed}",
            "trial_identity_sha256": canonical_hash(trial_identity),
        },
    }
    document["content_sha256"] = canonical_hash(document)
    output_sha256 = _write_once(args.output, document)
    return document, output_sha256


def main(argv: list[str] | None = None) -> int:
    document, output_sha256 = run(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "content_sha256": document["content_sha256"],
                "output_sha256": output_sha256,
                "trial": document["trial"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
