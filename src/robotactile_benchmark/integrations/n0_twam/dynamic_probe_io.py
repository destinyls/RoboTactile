"""Isaac-facing capture and atomic output helpers for the dynamic probe."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash, thaw_value
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
    prepare_n0_image,
)

STREAM_NAMES = ("top", "wrist_l", "tactile_a", "tactile_b")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_value(value: Any) -> object:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def camera_runtime_state(task: Any) -> dict[str, object]:
    """Capture IsaacLab camera counters and update timestamps."""

    manager = getattr(task, "_camera_manager", None)
    cameras = getattr(manager, "cameras", None)
    if not isinstance(cameras, Mapping):
        raise RuntimeError("live UniVTAC camera manager is unavailable")
    result: dict[str, object] = {}
    for name, camera in sorted(cameras.items()):
        result[str(name)] = {
            "frame": _host_value(camera.frame),
            "is_outdated": _host_value(camera._is_outdated),
            "timestamp_s": _host_value(camera._timestamp),
            "timestamp_last_update_s": _host_value(camera._timestamp_last_update),
            "update_period_s": float(camera.cfg.update_period),
        }
    return result


def _first_number(value: object) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError("camera counter must contain one finite value")
    return float(array[0])


def camera_freshness(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    """Prove that every configured camera advanced after the action."""

    if set(before) != set(after):
        raise ValueError("camera runtime inventories differ")
    result: dict[str, object] = {}
    for name in sorted(before):
        first = before[name]
        second = after[name]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise TypeError("camera runtime states must be mappings")
        frame_delta = _first_number(second["frame"]) - _first_number(first["frame"])
        timestamp_delta = _first_number(
            second["timestamp_last_update_s"]
        ) - _first_number(first["timestamp_last_update_s"])
        result[name] = {
            "frame_delta": frame_delta,
            "passed": frame_delta >= 1.0 and timestamp_delta > 0.0,
            "timestamp_last_update_delta_s": timestamp_delta,
        }
    return result


def raw_depths(raw: Mapping[str, Any], config: Any) -> dict[str, Array]:
    """Extract host float32 depth arrays from the two live tactile sensors."""

    tactile = raw["tactile"]
    if not isinstance(tactile, Mapping):
        raise TypeError("raw tactile observation must be a mapping")
    result = {}
    for stream, source in (
        ("tactile_a", config.aliases.left_tactile),
        ("tactile_b", config.aliases.right_tactile),
    ):
        leaf = tactile[source]
        if not isinstance(leaf, Mapping):
            raise TypeError("raw tactile sensor must be a mapping")
        value = leaf["depth"]
        detached = getattr(value, "detach", None)
        if callable(detached):
            value = detached()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        depth = np.asarray(value, dtype=np.float32)
        if depth.shape != config.tactile_depth_shape or not np.isfinite(depth).all():
            raise ValueError("raw tactile depth shape/value mismatch")
        result[stream] = np.ascontiguousarray(depth)
    return result


def model_streams(streams: Mapping[str, Array]) -> dict[str, Array]:
    """Apply the single released live-to-checkpoint color transform."""

    return {
        name: prepare_n0_image(
            streams[name], profile=N0_LIVE_UNIVTAC_INPUT_PROFILE, name=name
        )
        for name in STREAM_NAMES
    }


def comparison_panel(
    reference: Mapping[str, Array],
    before: Mapping[str, Array],
    after: Mapping[str, Array],
) -> bytes:
    """Render HDF5/live-pre/live-post without synthesizing any pixels."""

    from PIL import Image, ImageDraw

    rows = (
        ("HDF5 matched", reference),
        ("Live pre (model)", before),
        ("Live post (model)", after),
    )
    width, height, label = 320, 240, 24
    canvas = Image.new("RGB", (width * 4, (height + label) * 3), (16, 18, 22))
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


def write_dynamic_probe_bundle(
    output: Path,
    *,
    domains: Mapping[str, Mapping[str, Array]],
    states: Mapping[str, Array],
    panel: bytes,
    metadata: Mapping[str, object],
    evidence_level: str = "univtac_n0_simulator_dynamic_contract_probe_v1",
) -> dict[str, object]:
    """Atomically create one immutable probe directory and receipt."""

    if not isinstance(evidence_level, str) or not evidence_level.strip():
        raise ValueError("evidence_level must be a non-empty string")
    if output.exists() or output.is_symlink():
        raise FileExistsError("dynamic probe output already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("dynamic probe output parent must be a real directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        normalized_metadata = thaw_value(metadata)
        if not isinstance(normalized_metadata, dict):
            raise TypeError("dynamic probe metadata must normalize to a mapping")
        domain_entries: dict[str, dict[str, object]] = {}
        state_entries: dict[str, object] = {}
        for domain, streams in domains.items():
            domain_entries[domain] = {}
            for name, value in streams.items():
                relative = Path("arrays") / domain / f"{name}.npy"
                entry = _save_array(temporary / relative, value)
                entry["path"] = str(relative)
                domain_entries[domain][name] = entry
        for name, value in states.items():
            relative = Path("arrays") / "states" / f"{name}.npy"
            entry = _save_array(temporary / relative, value)
            entry["path"] = str(relative)
            state_entries[name] = entry
        panel_path = temporary / "comparison_panel.png"
        with panel_path.open("xb") as stream:
            stream.write(panel)
            stream.flush()
            os.fsync(stream.fileno())
        document = {
            "arrays": {"domains": domain_entries, "states": state_entries},
            "evidence_level": evidence_level,
            "panel": {"path": panel_path.name, "sha256": _sha256_file(panel_path)},
            **normalized_metadata,
        }
        document["content_sha256"] = canonical_hash(document)
        payload = (
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        with (temporary / "probe.json").open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, output)
        return document
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "STREAM_NAMES",
    "camera_freshness",
    "camera_runtime_state",
    "comparison_panel",
    "model_streams",
    "raw_depths",
    "write_dynamic_probe_bundle",
]
