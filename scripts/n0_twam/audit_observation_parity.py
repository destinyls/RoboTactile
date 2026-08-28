#!/usr/bin/env python3
"""Audit released N0 training pixels against a real live UniVTAC artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import SIM_HZ
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact
from robotactile_benchmark.integrations.n0_twam.observation_parity import (
    action_tracking_summary,
    color_order_witness,
    summarize_image_stream,
    temporal_contract_summary,
)

_STREAM_FIELDS = {
    "top": ("observation/head/rgb",),
    "wrist_l": ("observation/wrist/rgb",),
    "tactile_a": (
        "tactile/left_tactile/rgb",
        "tactile/left_gsmini/rgb",
    ),
    "tactile_b": (
        "tactile/right_tactile/rgb",
        "tactile/right_gsmini/rgb",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path)
    parser.add_argument("--teacher-forced-report", type=Path)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--training-start", type=int, default=0)
    parser.add_argument("--live-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=13)
    parser.add_argument(
        "--applied-channel-transform",
        choices=("identity", "reverse_rgb"),
        required=True,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    return absolute


def _write_once(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path = path.absolute()
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


def _converter_fps(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(item, ast.Name) and item.id == "FPS" for item in targets):
            continue
        value_node = node.value
        if isinstance(value_node, ast.Constant) and type(value_node.value) is int:
            values.append(value_node.value)
    if len(values) != 1 or values[0] < 1:
        raise ValueError("checkpoint converter must define one positive integer FPS")
    return values[0]


def _decode(value: object) -> Array:
    from PIL import Image

    if not isinstance(value, (bytes, np.bytes_)):
        raise TypeError("HDF5 RGB payload must contain bytes")
    with Image.open(io.BytesIO(bytes(value))) as image:
        return cast(Array, np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8))


def _training_frames(
    path: Path, start: int, count: int
) -> tuple[dict[str, Array], Array, dict[str, str], Array]:
    h5py = importlib.import_module("h5py")

    result: dict[str, Array] = {}
    selected_fields: dict[str, str] = {}
    with h5py.File(path, "r") as root:
        total = int(root["step"].shape[0])
        if start < 0 or count < 1 or start + count > total:
            raise ValueError("training frame selection is outside the HDF5 episode")
        for name, candidates in _STREAM_FIELDS.items():
            available = [field for field in candidates if field in root]
            if len(available) != 1:
                raise ValueError(f"training stream {name} has ambiguous/missing field")
            field = available[0]
            selected_fields[name] = field
            result[name] = np.stack(
                [_decode(root[field][index]) for index in range(start, start + count)]
            )
        steps = np.asarray(root["step"][:], dtype=np.int64)
        ee = np.asarray(root["embodiment/ee"][:], dtype=np.float32)
        joint = np.asarray(root["embodiment/joint"][:], dtype=np.float32)
    expert = np.concatenate((ee, joint[:, 7:8]), axis=1).astype(np.float32)
    return result, steps, selected_fields, expert


def _live_frames(
    artifact: Any, start: int, count: int
) -> tuple[dict[str, Array], Array]:
    finalization = artifact.evidence.finalization
    if finalization is None:
        raise ValueError("live artifact has no delivered observation trace")
    records = finalization.clean_records
    if start < 0 or count < 1 or start + count > len(records):
        raise ValueError("live frame selection is outside the artifact")
    selected = records[start : start + count]
    streams = {
        "top": np.stack([item.observation.vision["top"] for item in selected]),
        "wrist_l": np.stack([item.observation.vision["wrist_l"] for item in selected]),
        "tactile_a": np.stack(
            [cast(Array, item.observation.sensor("left").payload) for item in selected]
        ),
        "tactile_b": np.stack(
            [cast(Array, item.observation.sensor("right").payload) for item in selected]
        ),
    }
    proprio = np.stack(
        [item.observation.proprio for item in finalization.clean_records]
    ).astype(np.float32, copy=False)
    return streams, proprio


def _native_steps(artifact: Any) -> Optional[Array]:
    transitions = artifact.evidence.transition_entries
    if not transitions:
        return None
    initial = artifact.evidence.initial_diagnostics.get("native_step_id")
    if type(initial) is not int:
        return None
    return np.asarray(
        (initial, *(item.native_step_id for item in transitions)), dtype=np.int64
    )


def _live_decimation(artifact: Any) -> int:
    value = artifact.evidence.initial_diagnostics.get("decimation", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("live artifact decimation must be a positive integer")
    return value


def _executed_actions(artifact: Any) -> Array:
    entries = artifact.evidence.action_entries
    if not entries:
        raise ValueError("live artifact has no executed actions")
    return np.concatenate([item.executed_actions for item in entries]).astype(
        np.float32, copy=False
    )


def _teacher_summary(path: Optional[Path]) -> Optional[dict[str, object]]:
    if path is None:
        return None
    source = _regular_file(path, "teacher-forced report")
    document = json.loads(source.read_text(encoding="utf-8"))
    metrics = document["metrics"]
    return {
        "cold_vs_expert": metrics["cold_vs_expert"]["aggregate"],
        "evidence_level": document["evidence_level"],
        "path": str(source),
        "predicted_history_warm_vs_expert": metrics["predicted_history_warm_vs_expert"][
            "aggregate"
        ],
        "sha256": _sha256_file(source),
        "teacher_history_warm_vs_expert": metrics["teacher_history_warm_vs_expert"][
            "aggregate"
        ],
    }


def _verdicts(
    training: dict[str, Array],
    live: dict[str, Array],
    *,
    applied_transform: str,
    temporal: dict[str, object],
    tracking: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    witnesses = {
        name: color_order_witness(training[name], live[name]).to_dict()
        for name in ("top", "wrist_l")
    }
    preferred = {item["preferred_transform"] for item in witnesses.values()}
    color_pass = preferred == {applied_transform}
    ratios = {
        name: (
            summarize_image_stream(live[name]).spatial_gradient_mean
            / max(summarize_image_stream(training[name]).spatial_gradient_mean, 1e-8)
        )
        for name in ("top", "wrist_l")
    }
    renderer_pass = max(ratios.values()) <= 3.0
    live_ratio = temporal["live_to_training_physical_action_hz_ratio"]
    if live_ratio is None:
        temporal_pass = None
    elif isinstance(live_ratio, (int, float)) and not isinstance(live_ratio, bool):
        temporal_pass = 0.8 <= float(live_ratio) <= 1.25
    else:
        raise TypeError("live cadence ratio must be numeric or None")
    translation = cast(dict[str, float], tracking["translation_l2_m"])
    rotation = cast(dict[str, float], tracking["rotation_geodesic_deg"])
    tracking_pass = translation["p95"] <= 1e-3 and rotation["p95"] <= 1.0
    codes = []
    if not color_pass:
        codes.append("live_rgb_transform_contradicts_direct_array_witness")
    if not renderer_pass:
        codes.append("live_renderer_high_frequency_out_of_distribution")
    if temporal_pass is False:
        codes.append("live_action_endpoint_cadence_mismatch")
    for name in ("tactile_a", "tactile_b"):
        reference = summarize_image_stream(training[name])
        candidate = summarize_image_stream(live[name])
        if reference.unique_frame_count > 1 and candidate.unique_frame_count == 1:
            codes.append(f"live_{name}_stream_static")
    return (
        {
            "action_tracking": {"passed": tracking_pass},
            "camera_color_order": {
                "applied_transform": applied_transform,
                "passed": color_pass,
                "witnesses": witnesses,
            },
            "camera_renderer_frequency": {
                "live_to_training_spatial_gradient_ratio": ratios,
                "passed": renderer_pass,
                "threshold": "maximum ratio <= 3.0",
            },
            "temporal_native_cadence": {"passed": temporal_pass},
        },
        codes,
    )


def _model_visible(value: Array, transform: str) -> Array:
    return value if transform == "identity" else np.ascontiguousarray(value[..., ::-1])


def _panel_bytes(
    training: dict[str, Array], live: dict[str, Array], transform: str
) -> bytes:
    from PIL import Image, ImageDraw

    columns = ("top", "wrist_l", "tactile_a", "tactile_b")
    rows = (
        ("Training HDF5", training),
        ("Live raw RGB", live),
        (
            f"Live model-facing ({transform})",
            {key: _model_visible(value, transform) for key, value in live.items()},
        ),
    )
    cell_width, cell_height, label_height = 320, 240, 24
    canvas = Image.new(
        "RGB", (cell_width * len(columns), (cell_height + label_height) * len(rows))
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_name, streams) in enumerate(rows):
        y = row_index * (cell_height + label_height)
        for column_index, name in enumerate(columns):
            x = column_index * cell_width
            source = Image.fromarray(streams[name][0])
            source.thumbnail((cell_width, cell_height))
            offset = (x + (cell_width - source.width) // 2, y + label_height)
            canvas.paste(source, offset)
            draw.text((x + 4, y + 4), f"{row_name} | {name}", fill="white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def run(args: argparse.Namespace) -> tuple[dict[str, object], str, Optional[str]]:
    if args.training_seed < 0:
        raise ValueError("training seed must be non-negative")
    hdf5 = _regular_file(args.hdf5, "HDF5")
    converter = _regular_file(args.checkpoint_converter, "checkpoint converter")
    artifact = load_live_univtac_artifact(args.live_artifact.absolute())
    training, training_steps, fields, expert = _training_frames(
        hdf5, args.training_start, args.frame_count
    )
    live, observed = _live_frames(artifact, args.live_start, args.frame_count)
    actions = _executed_actions(artifact)
    if observed.shape[0] != actions.shape[0] + 1:
        raise ValueError("live action and observation counts are not endpoint aligned")
    tracking = action_tracking_summary(actions, observed[1:])
    temporal = temporal_contract_summary(
        training_steps,
        checkpoint_fps=_converter_fps(converter),
        sim_hz=SIM_HZ,
        live_native_steps=_native_steps(artifact),
        live_physics_steps_per_native_step=_live_decimation(artifact),
    )
    verdicts, codes = _verdicts(
        training,
        live,
        applied_transform=args.applied_channel_transform,
        temporal=temporal,
        tracking=tracking,
    )
    compared = min(actions.shape[0], expert.shape[0])
    report: dict[str, object] = {
        "action_tracking": tracking,
        "diagnostic_codes": codes,
        "evidence_level": "n0_training_vs_live_observation_diagnostic_v1",
        "limitations": [
            "This diagnostic is not a task-success result.",
            "Pixel registration is only valid when training and live seeds/phases match.",
            "Checkpoint metadata does not cryptographically bind the raw training corpus or renderer settings.",
        ],
        "live": {
            "artifact_root": str(args.live_artifact.absolute()),
            "initial_seed": artifact.trial.initial_seed,
            "root_receipt_sha256": artifact.root_receipt_sha256,
            "selected_start": args.live_start,
            "streams": {
                name: summarize_image_stream(value).to_dict()
                for name, value in live.items()
            },
        },
        "model_on_training_distribution": _teacher_summary(args.teacher_forced_report),
        "schema_version": "robotactile-n0-observation-parity-v1",
        "temporal_contract": temporal,
        "training": {
            "checkpoint_converter": str(converter),
            "checkpoint_converter_sha256": _sha256_file(converter),
            "hdf5_path": str(hdf5),
            "hdf5_sha256": _sha256_file(hdf5),
            "selected_fields": fields,
            "selected_start": args.training_start,
            "training_seed": args.training_seed,
            "streams": {
                name: summarize_image_stream(value).to_dict()
                for name, value in training.items()
            },
        },
        "trajectory_scale": {
            "comparable_same_seed_and_start": (
                args.training_seed == artifact.trial.initial_seed
                and args.training_start == 0
                and args.live_start == 0
            ),
            "compared_rows": compared,
            "expert_translation_path_m": float(
                np.linalg.norm(np.diff(expert[:compared, :3], axis=0), axis=1).sum()
            ),
            "model_target_translation_path_m": float(
                np.linalg.norm(np.diff(actions[:compared, :3], axis=0), axis=1).sum()
            ),
        },
        "verdicts": verdicts,
    }
    report["report_content_sha256"] = canonical_hash(report)
    payload = canonical_json_bytes(report)
    report_sha256 = _write_once(args.output, payload)
    panel_sha256 = None
    if args.panel_output is not None:
        panel_sha256 = _write_once(
            args.panel_output,
            _panel_bytes(training, live, args.applied_channel_transform),
        )
    return report, report_sha256, panel_sha256


def main() -> None:
    report, report_sha256, panel_sha256 = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "diagnostic_codes": report["diagnostic_codes"],
                "panel_sha256": panel_sha256,
                "report_sha256": report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
