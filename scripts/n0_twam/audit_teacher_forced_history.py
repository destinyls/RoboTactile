#!/usr/bin/env python3
"""Compare predicted-history and expert-history N0 warm inference on real HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import Array, ObservationRecord
from robotactile_benchmark.integrations.n0_twam.train_serve_gap import (
    array_parity_metrics,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
    prepare_n0_image,
)
from robotactile_benchmark.policies.n0_official import (
    NATIVE_ACTION_SHAPE,
    ee8_to_state20,
    n0_training_prompt,
    native_to_ee8_actions,
)
from robotactile_benchmark.recorded.n0_temporal_diagnostics import (
    build_cold_expert_native_action,
    causal_phase_truth_table,
    horizon_action_metrics,
    select_warm_keyframe_indices,
)
from robotactile_benchmark.recorded.source import load_univtac_hdf5_episode
from robotactile_benchmark.transport.n0_official import load_official_n0_rpc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="lift_bottle")
    parser.add_argument("--episode-id", default="univtac-lift_bottle-clean-90")
    parser.add_argument("--cold-anchor", type=int, default=174)
    parser.add_argument("--warm-anchor", type=int, default=186)
    parser.add_argument("--rest-index", type=int, default=0)
    parser.add_argument("--initial-seed", type=int, default=90)
    parser.add_argument("--exogenous-seed", type=int, default=20260823)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument("--api-key-env", default="ROBOTACTILE_N0_API_KEY")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, document: object) -> str:
    payload = canonical_json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("diagnostic output cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("diagnostic output already differs")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return completed.stdout.strip()


def _source_binding(root: Path) -> dict[str, object]:
    server = root / "n0_twam/n0_twam_server.py"
    if root.is_symlink() or not root.is_dir() or not server.is_file():
        raise ValueError("N0 source root is unavailable")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    diff = subprocess.run(
        ("git", "-C", str(root), "diff", "--binary", "--", "n0_twam"),
        check=True,
        capture_output=True,
        timeout=30.0,
    ).stdout
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty_tracked": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "server_sha256": _sha256_file(server),
    }


def _wire(
    record: ObservationRecord,
) -> tuple[dict[str, Array], dict[str, Array], Array]:
    profile = N0_RECORDED_CHECKPOINT_INPUT_PROFILE
    vision = {
        f"observation.images.{key}": prepare_n0_image(
            record.vision[key], profile=profile, name=f"vision.{key}"
        )
        for key in ("top", "wrist_l")
    }
    tactile: dict[str, Array] = {}
    for slot_id, key in (("left", "tactile_a"), ("right", "tactile_b")):
        payload = record.sensor(slot_id).payload
        if payload is None:
            raise ValueError("teacher-forced N0 diagnostic requires tactile")
        tactile[f"observation.images.{key}"] = prepare_n0_image(
            payload, profile=profile, name=f"tactile.{slot_id}"
        )
    return vision, tactile, ee8_to_state20(record.proprio)


def _action(response: Mapping[str, object], operation: str) -> Array:
    if "action" not in response or set(response) - {"action", "server_timing"}:
        raise RuntimeError(f"{operation} response fields mismatch")
    action = np.asarray(response["action"], dtype=np.float32)
    if action.shape != NATIVE_ACTION_SHAPE or not np.isfinite(action).all():
        raise ValueError(f"{operation} action must be finite {NATIVE_ACTION_SHAPE}")
    return cast(Array, np.ascontiguousarray(action))


def _timing_only(response: Mapping[str, object], operation: str) -> None:
    if set(response) - {"server_timing"}:
        raise RuntimeError(f"{operation} response fields mismatch")


def _inference_payload(record: ObservationRecord, prompt: str) -> dict[str, object]:
    vision, tactile, state = _wire(record)
    return {
        "obs": vision,
        "tactile": tactile,
        "current_state": state.tolist(),
        "prompt": prompt,
    }


def _commit_payload(
    records: tuple[ObservationRecord, ...],
    native_action: Array,
    anchor_state: Array,
    prompt: str,
) -> dict[str, object]:
    wired = tuple(_wire(record) for record in records)
    return {
        "obs": tuple(item[0] for item in wired),
        "tactile": tuple(item[1] for item in wired),
        "state": native_action,
        "current_state": anchor_state.tolist(),
        "action_anchor_state": anchor_state.tolist(),
        "state_action_format": "absolute",
        "compute_kv_cache": True,
        "imagine": False,
        "prompt": prompt,
    }


def _run_session(
    rpc: Any,
    *,
    cold_record: ObservationRecord,
    warm_record: ObservationRecord,
    keyframes: tuple[ObservationRecord, ...],
    commit_action: Optional[Array],
    prompt: str,
    seed: int,
) -> tuple[Array, Array]:
    _timing_only(rpc.infer({"reset": True, "prompt": prompt, "seed": seed}), "reset")
    cold_native = _action(
        rpc.infer(_inference_payload(cold_record, prompt)), "cold infer"
    )
    selected_action = cold_native if commit_action is None else commit_action
    anchor_state = ee8_to_state20(cold_record.proprio)
    _timing_only(
        rpc.infer(_commit_payload(keyframes, selected_action, anchor_state, prompt)),
        "commit",
    )
    warm_native = _action(
        rpc.infer(_inference_payload(warm_record, prompt)), "warm infer"
    )
    return cold_native, warm_native


def _rows(records: tuple[ObservationRecord, ...], start: int, stop: int) -> Array:
    return cast(
        Array,
        np.stack([record.proprio for record in records[start:stop]]).astype(
            np.float32, copy=False
        ),
    )


def _phase_table() -> list[dict[str, object]]:
    return [asdict(row) for row in causal_phase_truth_table(0)]


def run(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    if args.cold_anchor < 12 or args.warm_anchor <= args.cold_anchor:
        raise ValueError("anchors do not cover a cold expert history")
    keyframe_indices = select_warm_keyframe_indices(
        anchor_index=args.cold_anchor,
        target_index=args.warm_anchor,
    )
    stop = args.warm_anchor + 24
    episode = load_univtac_hdf5_episode(
        args.hdf5.absolute(),
        task_id=args.task,
        episode_id=args.episode_id,
        initial_seed=args.initial_seed,
        anchor_index=args.warm_anchor,
        rest_index=args.rest_index,
        stop_index=stop,
    )
    records = tuple(item.observation for item in episode.records)
    expert_all = _rows(records, 0, stop)
    expert_native = build_cold_expert_native_action(
        expert_all,
        anchor_index=args.cold_anchor,
        ee8_to_state20=ee8_to_state20,
    )
    cold_target = _rows(records, args.cold_anchor, args.cold_anchor + 12)
    warm_target = _rows(records, args.warm_anchor, args.warm_anchor + 24)
    keyframes = tuple(records[index] for index in keyframe_indices)
    prompt = n0_training_prompt(args.task)
    rpc = load_official_n0_rpc(
        source_root=args.source_root.absolute(),
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
    )
    try:
        predicted_cold, predicted_warm = _run_session(
            rpc,
            cold_record=records[args.cold_anchor],
            warm_record=records[args.warm_anchor],
            keyframes=keyframes,
            commit_action=None,
            prompt=prompt,
            seed=args.exogenous_seed,
        )
        teacher_cold, teacher_warm = _run_session(
            rpc,
            cold_record=records[args.cold_anchor],
            warm_record=records[args.warm_anchor],
            keyframes=keyframes,
            commit_action=expert_native,
            prompt=prompt,
            seed=args.exogenous_seed,
        )
        metadata = dict(rpc.get_server_metadata())
    finally:
        rpc.close()
    predicted_cold_ee8 = native_to_ee8_actions(predicted_cold, cold_chunk=True)
    predicted_warm_ee8 = native_to_ee8_actions(predicted_warm, cold_chunk=False)
    teacher_warm_ee8 = native_to_ee8_actions(teacher_warm, cold_chunk=False)
    report = {
        "evidence_level": "recorded_model_teacher_forced_history_n0_v1",
        "limitations": [
            "Predicted actions are not executed in Isaac Sim or on a robot.",
            "This diagnostic cannot report task success or a success rate.",
            "Expert forcing isolates action-history error but retains serve KV semantics.",
        ],
        "metrics": {
            "cold_repeat_native": array_parity_metrics(
                predicted_cold, teacher_cold
            ).to_dict(),
            "predicted_history_warm_vs_expert": asdict(
                horizon_action_metrics(predicted_warm_ee8, warm_target)
            ),
            "teacher_history_warm_vs_expert": asdict(
                horizon_action_metrics(teacher_warm_ee8, warm_target)
            ),
            "teacher_vs_predicted_warm": asdict(
                horizon_action_metrics(teacher_warm_ee8, predicted_warm_ee8)
            ),
            "cold_vs_expert": asdict(
                horizon_action_metrics(predicted_cold_ee8, cold_target)
            ),
        },
        "preprocessing_profile": N0_RECORDED_CHECKPOINT_INPUT_PROFILE.to_dict(),
        "schema_version": "robotactile-n0-teacher-forced-history-v1",
        "server_metadata": json.loads(json.dumps(metadata, default=str)),
        "source": {
            "cold_anchor": args.cold_anchor,
            "episode_id": args.episode_id,
            "exogenous_seed": args.exogenous_seed,
            "hdf5_path": str(episode.source_path),
            "hdf5_sha256": episode.source_sha256,
            "initial_seed": args.initial_seed,
            "keyframe_indices": list(keyframe_indices),
            "n0_source": _source_binding(args.source_root.absolute()),
            "task_id": args.task,
            "warm_anchor": args.warm_anchor,
        },
        "temporal_contract": {
            "causal_phase_truth_table": _phase_table(),
            "cold_emitted_horizons": 12,
            "native_action_shape": list(NATIVE_ACTION_SHAPE),
            "warm_emitted_horizons": 24,
        },
    }
    return report, _write_once(args.output.absolute(), report)


def main() -> int:
    report, digest = run(_parser().parse_args())
    print(
        json.dumps(
            {"output_sha256": digest, "schema_version": report["schema_version"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
