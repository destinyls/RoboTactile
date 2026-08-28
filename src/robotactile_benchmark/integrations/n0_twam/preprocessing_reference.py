"""Released N0-TWAM UniVTAC preprocessing reference operators.

The released checkpoint converter first decodes legacy UniVTAC JPEG payloads
with PIL, stores them as H264/yuv420p videos, and the pinned latent encoder then
uses FFmpeg's ``area`` scaler.  This module keeps that chain explicit for
diagnostics without adding heavyweight runtime dependencies to the main wheel.
"""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.contracts import Array


@dataclass(frozen=True)
class CheckpointVideoContract:
    """Documented video parameters shipped with the UniVTAC checkpoint."""

    source_fps: int = 30
    target_fps: int = 10
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = 30
    gop: int = 2

    def __post_init__(self) -> None:
        if self.source_fps <= 0 or self.target_fps <= 0:
            raise ValueError("video frame rates must be positive")
        if self.source_fps % self.target_fps != 0:
            raise ValueError("source FPS must be divisible by target FPS")
        if not self.codec or not self.pixel_format:
            raise ValueError("codec and pixel format must be non-empty")
        if not 0 <= self.crf <= 51 or self.gop <= 0:
            raise ValueError("invalid H264 CRF/GOP contract")

    def to_dict(self) -> dict[str, object]:
        """Return the immutable contract as JSON-safe values."""

        return {
            "codec": self.codec,
            "crf": self.crf,
            "gop": self.gop,
            "pixel_format": self.pixel_format,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
        }


def _encoded_bytes(value: object) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        payload = bytes(value)
    else:
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype != np.uint8:
            raise TypeError("encoded image payload must be bytes or a uint8 vector")
        payload = array.tobytes()
    if not payload:
        raise ValueError("encoded image payload cannot be empty")
    return payload


def _frozen_rgb(value: object) -> Array:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("RGB frame must be uint8 HWC with three channels")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def decode_checkpoint_jpeg(value: object) -> Array:
    """Decode exactly like the converter distributed with the checkpoint."""

    from PIL import Image

    with Image.open(io.BytesIO(_encoded_bytes(value))) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return _frozen_rgb(decoded)


def decode_live_numeric_proxy(value: object) -> Array:
    """Recover the legacy cv2 numeric channel order used before JPEG storage.

    UniVTAC writes simulator RGB arrays directly with ``cv2.imencode``.  A
    cv2 decode therefore approximates the original numerical channel order,
    while PIL exposes the color-interpreted RGB order used by checkpoint
    conversion.  JPEG loss means this is a proxy, not an exact live frame.
    """

    import cv2  # type: ignore[import-not-found]

    encoded = np.frombuffer(_encoded_bytes(value), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV failed to decode UniVTAC JPEG payload")
    return _frozen_rgb(decoded)


def reverse_rgb_channels(frames: object) -> Array:
    """Return a contiguous R/B-reversed copy of uint8 RGB frames."""

    array = np.asarray(frames)
    if array.ndim not in (3, 4) or array.shape[-1] != 3:
        raise ValueError("RGB input must be HWC or FHWC")
    if array.dtype != np.uint8:
        raise ValueError("RGB input must use uint8")
    result = np.ascontiguousarray(array[..., ::-1])
    result.setflags(write=False)
    return result


def _validated_frames(frames: object) -> Array:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("video frames must be uint8 FHWC")
    if array.shape[0] <= 0:
        raise ValueError("video frame sequence cannot be empty")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def build_h264_encode_command(
    ffmpeg: Path,
    output: Path,
    *,
    width: int,
    height: int,
    contract: CheckpointVideoContract,
) -> Tuple[str, ...]:
    """Build the shell-free H264 proxy command."""

    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    return (
        str(ffmpeg),
        "-n",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(contract.source_fps),
        "-i",
        "pipe:0",
        "-an",
        "-vcodec",
        contract.codec,
        "-pix_fmt",
        contract.pixel_format,
        "-crf",
        str(contract.crf),
        "-g",
        str(contract.gop),
        str(output),
    )


def build_training_decode_command(
    ffmpeg: Path,
    source: Path,
    *,
    width: int,
    height: int,
    contract: CheckpointVideoContract,
) -> Tuple[str, ...]:
    """Build the pinned encoder's decode/downsample/area-scale command."""

    if width <= 0 or height <= 0:
        raise ValueError("target dimensions must be positive")
    return (
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={contract.target_fps},scale={width}:{height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    )


def build_area_resize_command(
    ffmpeg: Path,
    *,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
) -> Tuple[str, ...]:
    """Build an FFmpeg raw RGB ``area`` resize without temporal resampling."""

    if min(input_width, input_height, output_width, output_height) <= 0:
        raise ValueError("resize dimensions must be positive")
    return (
        str(ffmpeg),
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{input_width}x{input_height}",
        "-i",
        "pipe:0",
        "-vf",
        f"scale={output_width}:{output_height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    )


def _run(command: Tuple[str, ...], *, stdin: bytes | None) -> bytes:
    completed = subprocess.run(
        command,
        check=False,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"FFmpeg failed with code {completed.returncode}: {error}")
    return completed.stdout


def encode_h264_proxy(
    frames: object,
    output: Path,
    ffmpeg: Path,
    contract: CheckpointVideoContract,
) -> None:
    """Materialize one no-clobber H264 proxy clip."""

    values = _validated_frames(frames)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"proxy video already exists: {output}")
    height, width = (int(values.shape[1]), int(values.shape[2]))
    command = build_h264_encode_command(
        ffmpeg,
        output,
        width=width,
        height=height,
        contract=contract,
    )
    _run(command, stdin=values.tobytes(order="C"))
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("FFmpeg did not create the H264 proxy")


def _reshape_rgb(raw: bytes, *, width: int, height: int) -> Array:
    frame_bytes = width * height * 3
    if not raw or len(raw) % frame_bytes != 0:
        raise RuntimeError("unexpected FFmpeg raw RGB output size")
    result = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    result.setflags(write=False)
    return result


def decode_h264_training_pixels(
    source: Path,
    ffmpeg: Path,
    *,
    width: int,
    height: int,
    expected_frames: Optional[int],
    contract: CheckpointVideoContract,
) -> Array:
    """Decode H264 through the released latent encoder's spatial pipeline."""

    raw = _run(
        build_training_decode_command(
            ffmpeg,
            source,
            width=width,
            height=height,
            contract=contract,
        ),
        stdin=None,
    )
    result = _reshape_rgb(raw, width=width, height=height)
    if expected_frames is not None and result.shape[0] != expected_frames:
        raise RuntimeError(
            f"expected {expected_frames} decoded frames, got {result.shape[0]}"
        )
    return result


def resize_ffmpeg_area(
    frames: object,
    ffmpeg: Path,
    *,
    width: int,
    height: int,
) -> Array:
    """Resize uint8 FHWC frames with FFmpeg/libswscale area."""

    values = _validated_frames(frames)
    input_height, input_width = int(values.shape[1]), int(values.shape[2])
    raw = _run(
        build_area_resize_command(
            ffmpeg,
            input_width=input_width,
            input_height=input_height,
            output_width=width,
            output_height=height,
        ),
        stdin=values.tobytes(order="C"),
    )
    result = _reshape_rgb(raw, width=width, height=height)
    if result.shape[0] != values.shape[0]:
        raise RuntimeError("area resize changed the frame count")
    return result


def resize_cv2_area(frames: object, *, width: int, height: int) -> Array:
    """Resize with the low-overhead OpenCV approximation to FFmpeg area."""

    import cv2

    values = _validated_frames(frames)
    resized = np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in values
        ]
    )
    return _validated_frames(resized)


def ffmpeg_version(ffmpeg: Path) -> str:
    """Return the first version line for provenance."""

    raw = _run((str(ffmpeg), "-version"), stdin=None)
    return raw.decode("utf-8", errors="replace").splitlines()[0]


__all__ = [
    "CheckpointVideoContract",
    "build_area_resize_command",
    "build_h264_encode_command",
    "build_training_decode_command",
    "decode_checkpoint_jpeg",
    "decode_h264_training_pixels",
    "decode_live_numeric_proxy",
    "encode_h264_proxy",
    "ffmpeg_version",
    "resize_cv2_area",
    "resize_ffmpeg_area",
    "reverse_rgb_channels",
]
