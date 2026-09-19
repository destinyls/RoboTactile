"""Paper-facing panels and videos from verified live UniVTAC traces."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np

from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.execution import (
    LoadedLiveUniVTACArtifact,
    load_live_univtac_artifact,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
    N0InputProfile,
    N0SourceColorDomain,
    prepare_n0_image,
)

_TILE_WIDTH = 320
_TILE_HEIGHT = 240
_LABEL_HEIGHT = 24
_DIFFERENCE_SCALE = 3.0
_EVIDENCE_LEVEL = "derived_live_artifact_visualization_v1"
_RETRAINED_DISPLAY_PROFILE = N0InputProfile(
    profile_id="retrained-univtac-numeric-rgb-display-v1",
    source_color_domain=N0SourceColorDomain.RETRAINED_NUMERIC_RGB,
)


def _display_profile(artifact: LoadedLiveUniVTACArtifact) -> N0InputProfile:
    identity = getattr(artifact, "request_identity", {})
    if (
        identity.get("n0_action_per_frame") == 4
        or identity.get("retrained_control_hz") is not None
    ):
        return _RETRAINED_DISPLAY_PROFILE
    return N0_LIVE_UNIVTAC_INPUT_PROFILE


@dataclass(frozen=True)
class LiveVisualizationResult:
    """Published visualization paths and their source binding."""

    output: Path
    preview: Path
    video: Optional[Path]
    receipt: Path
    source_root_sha256: str
    rendered_frame_count: int

    def to_cli_dict(self) -> dict[str, object]:
        return {
            "evidence_level": _EVIDENCE_LEVEL,
            "output": str(self.output),
            "preview": str(self.preview),
            "receipt": str(self.receipt),
            "rendered_frame_count": self.rendered_frame_count,
            "source_live_artifact_root_sha256": self.source_root_sha256,
            "video": None if self.video is None else str(self.video),
        }


def export_live_artifact_visualization(
    artifact_path: Path,
    output: Path,
    *,
    fps: int = 20,
    stride: int = 1,
    max_frames: Optional[int] = None,
    video: bool = False,
    ffmpeg: str = "ffmpeg",
) -> LiveVisualizationResult:
    """Strictly load one artifact and export a source-bound visual bundle."""

    artifact = load_live_univtac_artifact(Path(artifact_path))
    return _export_loaded_visualization(
        artifact,
        Path(output),
        fps=fps,
        stride=stride,
        max_frames=max_frames,
        video=video,
        ffmpeg=ffmpeg,
    )


def _export_loaded_visualization(
    artifact: LoadedLiveUniVTACArtifact,
    output: Path,
    *,
    fps: int,
    stride: int,
    max_frames: Optional[int],
    video: bool,
    ffmpeg: str,
) -> LiveVisualizationResult:
    _validate_options(fps, stride, max_frames, ffmpeg)
    finalization = artifact.evidence.finalization
    if finalization is not None:
        clean = finalization.clean_records
        delivered = finalization.delivered_records
        source_indices = tuple(range(len(clean)))
    elif artifact.preview_trace is not None:
        clean = artifact.preview_trace.clean_records
        delivered = artifact.preview_trace.delivered_records
        source_indices = artifact.preview_trace.selected_indices
    else:
        raise ValueError(
            "metrics_only_v1 capture has no frames; use preview_v1 or paper_full_v1"
        )
    if not clean or len(clean) != len(delivered):
        raise ValueError("live artifact clean/delivered traces are not aligned")

    selected = tuple(range(0, len(clean), stride))
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("visualization selection contains no frames")

    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"visualization output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("visualization output parent must be a real directory")

    pillow = _load_pillow()
    ffmpeg_path = _resolve_ffmpeg(ffmpeg) if video else None
    with tempfile.TemporaryDirectory(prefix=".visualize-", dir=output.parent) as tmp:
        staging = Path(tmp)
        preview_index = _preview_index(clean, delivered)
        preview_path = staging / "preview.png"
        _render_panel(
            pillow,
            clean[preview_index],
            delivered[preview_index],
            artifact,
        ).save(preview_path, format="PNG", optimize=False)

        video_path: Optional[Path] = None
        if ffmpeg_path is not None:
            video_path = staging / "preview.mp4"
            _render_video(
                pillow,
                clean,
                delivered,
                artifact,
                selected,
                video_path,
                fps,
                ffmpeg_path,
            )

        members = {"preview.png": _sha256_file(preview_path)}
        if video_path is not None:
            members["preview.mp4"] = _sha256_file(video_path)
        receipt_document = {
            "condition": artifact.trial.condition.value,
            "capture_profile": artifact.capture_profile.value,
            "difference_scale": _DIFFERENCE_SCALE,
            "display_color_profile": _display_profile(artifact).to_dict(),
            "evidence_level": _EVIDENCE_LEVEL,
            "fault_manifest_sha256": (
                None
                if artifact.fault_manifest is None
                else artifact.fault_manifest.sha256
            ),
            "fps": fps,
            "members": members,
            "preview_trace_offset": preview_index,
            "preview_source_trace_index": source_indices[preview_index],
            "preview_step_index": delivered[preview_index].observation.step_index,
            "rendered_frame_count": len(selected),
            "selected_step_indices": [
                delivered[index].observation.step_index for index in selected
            ],
            "selected_source_trace_indices": [
                source_indices[index] for index in selected
            ],
            "simulator_qualification_claimed": False,
            "source_live_artifact_root_sha256": artifact.root_receipt_sha256,
            "stride": stride,
            "task": artifact.trial.task,
            "video_exported": video_path is not None,
        }
        receipt_path = staging / "visualization_receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt_document))
        os.rename(staging, output)

    return LiveVisualizationResult(
        output=output,
        preview=output / "preview.png",
        video=(output / "preview.mp4") if video else None,
        receipt=output / "visualization_receipt.json",
        source_root_sha256=artifact.root_receipt_sha256,
        rendered_frame_count=len(selected),
    )


def _validate_options(
    fps: int, stride: int, max_frames: Optional[int], ffmpeg: str
) -> None:
    for value, name in ((fps, "fps"), (stride, "stride")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_frames is not None and (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError("max_frames must be a positive integer or None")
    if not isinstance(ffmpeg, str) or not ffmpeg:
        raise ValueError("ffmpeg must be a non-empty executable name or path")


def _load_pillow() -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "visualization requires 'pip install robotactile-benchmark[visualization]'"
        ) from error
    return Image, ImageDraw, ImageFont, ImageOps


def _resolve_ffmpeg(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"ffmpeg executable is unavailable: {value}")
    return resolved


def _preview_index(
    clean: Sequence[EvaluationRecord], delivered: Sequence[EvaluationRecord]
) -> int:
    contact_phases = {"contact_onset", "sustained_contact", "release"}
    for index, record in enumerate(delivered):
        if any(item.active_fault_ids for item in record.provenance) and any(
            item.phase.value in contact_phases for item in record.provenance
        ):
            return index
    for index, record in enumerate(clean):
        if any(item.phase.value in contact_phases for item in record.provenance):
            return index
    return len(clean) // 2


def _render_panel(
    pillow: Any,
    clean: EvaluationRecord,
    delivered: EvaluationRecord,
    artifact: LoadedLiveUniVTACArtifact,
) -> Any:
    Image, ImageDraw, ImageFont, ImageOps = pillow
    canvas = Image.new("RGB", (_TILE_WIDTH * 3, _TILE_HEIGHT * 3), (20, 22, 26))
    font = ImageFont.load_default()
    profile = _display_profile(artifact)
    vision = delivered.observation.vision
    _paste_tile(
        canvas,
        Image,
        ImageDraw,
        ImageOps,
        font,
        vision["top"],
        0,
        0,
        "Top RGB",
        profile,
    )
    _paste_tile(
        canvas,
        Image,
        ImageDraw,
        ImageOps,
        font,
        vision["wrist_l"],
        1,
        0,
        "Wrist RGB",
        profile,
    )
    _paste_metadata(canvas, ImageDraw, font, delivered, artifact)

    clean_sensors = {sensor.slot_id: sensor for sensor in clean.observation.tactile}
    delivered_sensors = {
        sensor.slot_id: sensor for sensor in delivered.observation.tactile
    }
    for row, slot_id in enumerate(("left", "right"), start=1):
        clean_payload = clean_sensors[slot_id].payload
        delivered_payload = delivered_sensors[slot_id].payload
        _paste_tile(
            canvas,
            Image,
            ImageDraw,
            ImageOps,
            font,
            clean_payload,
            0,
            row,
            f"Clean tactile: {slot_id}",
            profile,
        )
        _paste_tile(
            canvas,
            Image,
            ImageDraw,
            ImageOps,
            font,
            delivered_payload,
            1,
            row,
            f"Delivered tactile: {slot_id}",
            profile,
        )
        difference = _absolute_difference(clean_payload, delivered_payload)
        _paste_tile(
            canvas,
            Image,
            ImageDraw,
            ImageOps,
            font,
            difference,
            2,
            row,
            f"Absolute difference x{_DIFFERENCE_SCALE:g}: {slot_id}",
            profile,
        )
    return canvas


def _paste_tile(
    canvas: Any,
    image_module: Any,
    draw_module: Any,
    image_ops: Any,
    font: Any,
    array: Optional[Array],
    column: int,
    row: int,
    label: str,
    profile: N0InputProfile = N0_LIVE_UNIVTAC_INPUT_PROFILE,
) -> None:
    x = column * _TILE_WIDTH
    y = row * _TILE_HEIGHT
    draw = draw_module.Draw(canvas)
    draw.rectangle((x, y, x + _TILE_WIDTH - 1, y + _TILE_HEIGHT - 1), fill=(8, 9, 11))
    if array is None:
        draw.text((x + 12, y + 104), "PAYLOAD ABSENT", fill=(220, 80, 80), font=font)
    else:
        frame = _uint8_rgb(array, profile=profile)
        image = image_module.fromarray(frame)
        fitted = image_ops.contain(
            image,
            (_TILE_WIDTH, _TILE_HEIGHT - _LABEL_HEIGHT),
            method=image_module.Resampling.BILINEAR,
        )
        left = x + (_TILE_WIDTH - fitted.width) // 2
        top = y + _LABEL_HEIGHT + (_TILE_HEIGHT - _LABEL_HEIGHT - fitted.height) // 2
        canvas.paste(fitted, (left, top))
    draw.rectangle(
        (x, y, x + _TILE_WIDTH - 1, y + _LABEL_HEIGHT - 1), fill=(28, 31, 37)
    )
    draw.text((x + 7, y + 7), label, fill=(240, 242, 245), font=font)


def _paste_metadata(
    canvas: Any,
    draw_module: Any,
    font: Any,
    delivered: EvaluationRecord,
    artifact: LoadedLiveUniVTACArtifact,
) -> None:
    x = _TILE_WIDTH * 2
    draw = draw_module.Draw(canvas)
    draw.rectangle((x, 0, x + _TILE_WIDTH - 1, _TILE_HEIGHT - 1), fill=(18, 21, 26))
    phases = ", ".join(
        f"{item.slot_id}={item.phase.value}" for item in delivered.provenance
    )
    faults = sorted(
        {
            fault_id
            for item in delivered.provenance
            for fault_id in item.active_fault_ids
        }
    )
    lines = (
        "RoboTactile live trace",
        f"task: {artifact.trial.task}",
        f"condition: {artifact.trial.condition.value}",
        f"episode: {delivered.observation.episode_id[:20]}",
        f"step: {delivered.observation.step_index}",
        f"phase: {phases}",
        f"fault: {', '.join(faults) if faults else 'inactive'}",
        "diff is visualization-only",
    )
    for index, line in enumerate(lines):
        draw.text((x + 10, 12 + index * 25), line, fill=(235, 237, 240), font=font)


def _absolute_difference(
    clean: Optional[Array],
    delivered: Optional[Array],
) -> Optional[Array]:
    if clean is None or delivered is None:
        return None
    # Compute in stored channels; the tile applies the display transform once.
    clean_rgb: Array = _uint8_rgb(clean, profile=_RETRAINED_DISPLAY_PROFILE).astype(
        np.int16
    )
    delivered_rgb: Array = _uint8_rgb(
        delivered, profile=_RETRAINED_DISPLAY_PROFILE
    ).astype(np.int16)
    if clean_rgb.shape != delivered_rgb.shape:
        raise ValueError("clean and delivered tactile shapes disagree")
    difference = np.abs(clean_rgb - delivered_rgb).astype(np.float32)
    return cast(Array, np.clip(difference * _DIFFERENCE_SCALE, 0, 255).astype(np.uint8))


def _uint8_rgb(
    value: Array, *, profile: N0InputProfile = N0_LIVE_UNIVTAC_INPUT_PROFILE
) -> Array:
    """Convert UniVTAC numeric camera channels to display/PIL RGB."""

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("visualization arrays must have exact HWC RGB shape")
    if array.dtype != np.uint8:
        raise TypeError("visualization arrays must use uint8")
    return prepare_n0_image(
        array,
        profile=profile,
        name="UniVTAC visualization frame",
    )


def _render_video(
    pillow: Any,
    clean: Sequence[EvaluationRecord],
    delivered: Sequence[EvaluationRecord],
    artifact: LoadedLiveUniVTACArtifact,
    selected: Tuple[int, ...],
    output: Path,
    fps: int,
    ffmpeg: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="frames-", dir=output.parent) as tmp:
        frames = Path(tmp)
        for frame_index, trace_index in enumerate(selected):
            path = frames / f"frame-{frame_index:06d}.png"
            _render_panel(
                pillow, clean[trace_index], delivered[trace_index], artifact
            ).save(path, format="PNG", optimize=False)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise RuntimeError(f"ffmpeg video export failed: {detail}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
