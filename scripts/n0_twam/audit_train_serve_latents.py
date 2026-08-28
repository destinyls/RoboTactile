#!/usr/bin/env python3
"""Decompose released N0-TWAM checkpoint-versus-live preprocessing gaps."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.integrations.n0_twam.latent_reference import (
    offline_latent_bundle,
    pixel_domain_comparisons,
    streaming_latent_bundle,
    tensor_comparison,
)
from robotactile_benchmark.integrations.n0_twam.preprocessing_reference import (
    CheckpointVideoContract,
    decode_checkpoint_jpeg,
    decode_h264_training_pixels,
    decode_live_numeric_proxy,
    encode_h264_proxy,
    ffmpeg_version,
    resize_cv2_area,
    resize_ffmpeg_area,
)
from robotactile_benchmark.integrations.n0_twam.train_serve_gap import (
    sampled_source_indices,
    temporal_chunk_slices,
)

_FIELDS = {
    "top": "observation/head/rgb",
    "wrist_l": "observation/wrist/rgb",
    "tactile_a": "tactile/left_gsmini/rgb",
    "tactile_b": "tactile/right_gsmini/rgb",
}
_TARGET_SIZES = {
    "top": (256, 256),
    "wrist_l": (256, 256),
    "tactile_a": (128, 128),
    "tactile_b": (128, 128),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--checkpoint-converter", type=Path, required=True)
    parser.add_argument("--rgb-encoder-source", type=Path, required=True)
    parser.add_argument("--tactile-encoder-source", type=Path, required=True)
    parser.add_argument("--n0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-index", type=int, default=186)
    parser.add_argument("--keyframe-count", type=int, default=9)
    parser.add_argument("--source-stride", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
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
        raise ValueError("output cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("output already exists with different bytes")
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


def _load_decoded_domain(
    path: Path,
    indices: Optional[tuple[int, ...]],
    decoder: Callable[[object], np.ndarray],
) -> dict[str, np.ndarray]:
    import h5py

    before = path.stat()
    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as root:
        count = int(root[_FIELDS["top"]].shape[0])
        if count <= 0:
            raise ValueError("HDF5 episode cannot be empty")
        selected = tuple(range(count)) if indices is None else indices
        for name, field in _FIELDS.items():
            if field not in root or selected[-1] >= int(root[field].shape[0]):
                raise ValueError(f"missing or short HDF5 field: {field}")
            payloads = [root[field][index] for index in selected]
            result[name] = np.stack([decoder(value) for value in payloads])
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError("HDF5 source changed while it was being decoded")
    return result


def _bilinear(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(frames)).float().permute(0, 3, 1, 2)
    resized = functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )
    return resized.permute(0, 2, 3, 1).contiguous().numpy()


def _resize_domain(
    frames: dict[str, np.ndarray], *, mode: str, ffmpeg: Path
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, values in frames.items():
        height, width = _TARGET_SIZES[name]
        if mode == "bilinear":
            result[name] = _bilinear(values, (height, width))
        elif mode == "cv2_area":
            result[name] = resize_cv2_area(values, width=width, height=height)
        elif mode == "ffmpeg_area":
            result[name] = resize_ffmpeg_area(
                values, ffmpeg, width=width, height=height
            )
        else:
            raise ValueError(f"unsupported resize mode: {mode}")
    return result


def _training_proxy_pixels(
    frames: dict[str, np.ndarray],
    *,
    ffmpeg: Path,
    contract: CheckpointVideoContract,
    selected_source_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="robotactile-n0-codec-") as directory:
        root = Path(directory)
        for name, values in frames.items():
            video = root / f"{name}.mp4"
            encode_h264_proxy(values, video, ffmpeg, contract)
            height, width = _TARGET_SIZES[name]
            decoded = decode_h264_training_pixels(
                video,
                ffmpeg,
                width=width,
                height=height,
                expected_frames=None,
                contract=contract,
            )
            frame_ids = [
                min(
                    len(values) - 1,
                    int(round(index * contract.source_fps / contract.target_fps)),
                )
                for index in range(decoded.shape[0])
            ]
            while decoded.shape[0] > 1 and decoded.shape[0] % 4 != 1:
                decoded = decoded[:-1]
                frame_ids.pop()
            lookup = {frame_id: index for index, frame_id in enumerate(frame_ids)}
            missing = [
                index for index in selected_source_indices if index not in lookup
            ]
            if missing:
                raise RuntimeError(
                    f"training proxy is missing source frames: {missing}"
                )
            result[name] = decoded[
                np.asarray([lookup[index] for index in selected_source_indices])
            ]
    return result


def _runtime_config(args: argparse.Namespace) -> Any:
    from types import SimpleNamespace

    import torch

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    return SimpleNamespace(
        chunk_slices=temporal_chunk_slices(args.keyframe_count),
        device=torch.device(args.device),
        dtype=dtype,
    )


def _source_binding(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source binding must be a regular non-symlink file: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def _implementation_binding(value: object) -> dict[str, str]:
    location = inspect.getsourcefile(value)
    if location is None:
        raise ValueError("diagnostic implementation has no source file")
    return _source_binding(Path(location).resolve())


def run(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    import torch
    from n0_twam.models.utils import load_vae

    hdf5 = args.hdf5.absolute()
    base_model = args.base_model.absolute()
    ffmpeg = args.ffmpeg.absolute()
    output = args.output.absolute()
    if hdf5.is_symlink() or not hdf5.is_file():
        raise ValueError("HDF5 input must be a non-symlink regular file")
    if not (base_model / "vae" / "config.json").is_file():
        raise ValueError("base model must contain a VAE directory")
    if ffmpeg.is_symlink() or not ffmpeg.is_file():
        raise ValueError("FFmpeg must be a regular non-symlink file")
    if len(args.n0_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.n0_commit
    ):
        raise ValueError("N0 commit must contain 40 hexadecimal characters")

    selected_indices = sampled_source_indices(
        anchor_index=args.anchor_index,
        keyframe_count=args.keyframe_count,
        source_stride=args.source_stride,
    )
    checkpoint_frames = _load_decoded_domain(hdf5, None, decode_checkpoint_jpeg)
    episode_indices = tuple(range(checkpoint_frames["top"].shape[0]))
    live_proxy_selected = _load_decoded_domain(
        hdf5, selected_indices, decode_live_numeric_proxy
    )
    checkpoint_selected = {
        name: values[np.asarray(selected_indices)]
        for name, values in checkpoint_frames.items()
    }
    contract = CheckpointVideoContract()
    reference_pixels = _training_proxy_pixels(
        checkpoint_frames,
        ffmpeg=ffmpeg,
        contract=contract,
        selected_source_indices=selected_indices,
    )
    pixel_domains = {
        "current_live_proxy_bilinear": _resize_domain(
            live_proxy_selected, mode="bilinear", ffmpeg=ffmpeg
        ),
        "color_aligned_bilinear": _resize_domain(
            checkpoint_selected, mode="bilinear", ffmpeg=ffmpeg
        ),
        "resize_aligned_only_ffmpeg_area": _resize_domain(
            live_proxy_selected, mode="ffmpeg_area", ffmpeg=ffmpeg
        ),
        "color_aligned_cv2_area": _resize_domain(
            checkpoint_selected, mode="cv2_area", ffmpeg=ffmpeg
        ),
        "color_aligned_ffmpeg_area_no_h264": _resize_domain(
            checkpoint_selected, mode="ffmpeg_area", ffmpeg=ffmpeg
        ),
    }

    cfg = _runtime_config(args)
    vae = load_vae(base_model / "vae", torch_dtype=cfg.dtype, torch_device=cfg.device)
    vae.eval()
    latent_comparisons: dict[str, object] = {}
    streaming_comparisons: dict[str, object] = {}
    with torch.no_grad():
        reference_latents = offline_latent_bundle(reference_pixels, vae, cfg)
        for domain, pixels in pixel_domains.items():
            candidate = offline_latent_bundle(pixels, vae, cfg)
            latent_comparisons[domain] = {
                name: tensor_comparison(reference_latents[name], candidate[name])
                for name in reference_latents
            }
            if domain in {
                "current_live_proxy_bilinear",
                "color_aligned_cv2_area",
                "color_aligned_ffmpeg_area_no_h264",
            }:
                streaming = streaming_latent_bundle(pixels, vae, cfg)
                streaming_comparisons[domain] = {
                    name: tensor_comparison(candidate[name], streaming[name])
                    for name in candidate
                }

    source_bindings = {
        "audit_script": _source_binding(Path(__file__).resolve()),
        "checkpoint_converter": _source_binding(args.checkpoint_converter.absolute()),
        "latent_reference": _implementation_binding(offline_latent_bundle),
        "preprocessing_reference": _implementation_binding(decode_checkpoint_jpeg),
        "rgb_encoder": _source_binding(args.rgb_encoder_source.absolute()),
        "tactile_encoder": _source_binding(args.tactile_encoder_source.absolute()),
        "train_serve_gap": _implementation_binding(sampled_source_indices),
    }
    report = {
        "base_model": {
            "path": str(base_model),
            "vae_config_sha256": _sha256_file(base_model / "vae" / "config.json"),
        },
        "contract": {
            "checkpoint_chain": [
                "legacy_hdf5_jpeg",
                "pil_rgb_decode",
                "h264_yuv420p_crf30_gop2",
                "ffmpeg_fps10_area_resize",
                "offline_wan_vae",
            ],
            "current_live_chain": [
                "simulator_numeric_rgb",
                "torch_bilinear_resize",
                "streaming_wan_vae",
            ],
            "video": contract.to_dict(),
        },
        "evidence_level": "recorded_checkpoint_preprocessing_proxy_v2",
        "limitations": [
            "No released LeRobot MP4 or precomputed training latent was present.",
            "The H264 clip recreates documented parameters but not the training binary.",
            "The cv2 JPEG decode is a lossy proxy for the original live simulator array.",
            "The mini-clip VAE reference does not reconstruct full-episode temporal state.",
            "The checkpoint metadata does not cryptographically bind an encoder commit.",
        ],
        "latent_reference_comparisons": latent_comparisons,
        "pixel_reference_comparisons": pixel_domain_comparisons(
            reference_pixels, pixel_domains, tuple(_FIELDS)
        ),
        "runtime": {
            "device": str(cfg.device),
            "dtype": args.dtype,
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_sha256": _sha256_file(ffmpeg),
            "ffmpeg_version": ffmpeg_version(ffmpeg),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        "schema_version": "robotactile-n0-checkpoint-input-gap-v2",
        "source": {
            "anchor_index": args.anchor_index,
            "chunk_slices": [list(value) for value in cfg.chunk_slices],
            "codec_source_indices": [episode_indices[0], episode_indices[-1]],
            "hdf5_path": str(hdf5),
            "hdf5_sha256": _sha256_file(hdf5),
            "keyframe_count": args.keyframe_count,
            "n0_commit": args.n0_commit,
            "selected_indices": list(selected_indices),
            "source_bindings": source_bindings,
            "source_stride": args.source_stride,
        },
        "streaming_vs_offline_same_pixels": streaming_comparisons,
    }
    return report, _write_once(output, report)


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
