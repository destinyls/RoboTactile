"""Optional GPU helpers for N0-TWAM offline/streaming VAE diagnostics."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from robotactile_benchmark.contracts import Array
from robotactile_benchmark.integrations.n0_twam.train_serve_gap import (
    array_parity_metrics,
    freeze_float32,
)


def _normalize_latents(latents: Any, vae: Any) -> Any:
    import torch  # type: ignore[import-not-found]

    mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(
        1, -1, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, device=latents.device).view(
        1, -1, 1, 1, 1
    )
    return ((latents.float() - mean) / std).to(latents.dtype)


def _video_tensor(pixels: Mapping[str, Array]) -> Any:
    import torch

    cameras = [
        torch.from_numpy(np.asarray(pixels[name])).permute(3, 0, 1, 2).float()
        for name in ("top", "wrist_l")
    ]
    return torch.stack(cameras)


def _tactile_tensors(pixels: Mapping[str, Array]) -> tuple[Any, Any]:
    import torch

    tactile = torch.stack(
        [
            torch.from_numpy(np.asarray(pixels[name])).permute(3, 0, 1, 2).float()
            for name in ("tactile_a", "tactile_b")
        ]
    )
    first = tactile[:, :, 0:1]
    return (tactile - first) / 255.0, tactile / 127.5 - 1.0


def _offline_latent(values: Any, vae: Any, cfg: Any) -> Any:
    posterior = vae.encode(values.to(cfg.device, dtype=cfg.dtype)).latent_dist
    return _normalize_latents(posterior.mean, vae)


def offline_latent_bundle(
    pixels: Mapping[str, Array], vae: Any, cfg: Any
) -> dict[str, Any]:
    """Encode one already-resized multimodal pixel domain with offline VAE."""

    import torch

    video = _offline_latent(_video_tensor(pixels) / 127.5 - 1.0, vae, cfg)
    video = torch.cat(video.split(1, dim=0), dim=-1)
    tactile_global, tactile_local = _tactile_tensors(pixels)
    return {
        "tactile_global": _offline_latent(tactile_global, vae, cfg).unsqueeze(0),
        "tactile_local": _offline_latent(tactile_local, vae, cfg).unsqueeze(0),
        "video": video,
    }


def _streaming_latent(values: Any, vae: Any, cfg: Any) -> Any:
    import torch
    from n0_twam.models.utils import (  # type: ignore[import-not-found]
        WanVAEStreamingWrapper,
    )

    wrapper = WanVAEStreamingWrapper(vae)
    chunks = []
    tensor = values.to(cfg.device, dtype=cfg.dtype)
    for start, stop in cfg.chunk_slices:
        encoded = wrapper.encode_chunk(tensor[:, :, start:stop])
        mean, _ = torch.chunk(encoded, 2, dim=1)
        chunks.append(_normalize_latents(mean, vae))
    return torch.cat(chunks, dim=2)


def streaming_latent_bundle(
    pixels: Mapping[str, Array], vae: Any, cfg: Any
) -> dict[str, Any]:
    """Encode the same pixel domain with the deployed cold/warm VAE chunks."""

    import torch

    video = _streaming_latent(_video_tensor(pixels) / 127.5 - 1.0, vae, cfg)
    video = torch.cat(video.split(1, dim=0), dim=-1)
    tactile_global, tactile_local = _tactile_tensors(pixels)
    return {
        "tactile_global": _streaming_latent(tactile_global, vae, cfg).unsqueeze(0),
        "tactile_local": _streaming_latent(tactile_local, vae, cfg).unsqueeze(0),
        "video": video,
    }


def tensor_comparison(reference: Any, candidate: Any) -> dict[str, object]:
    """Compare and content-bind two torch-like tensors."""

    reference_array = freeze_float32(reference.detach().cpu().float().numpy())
    candidate_array = freeze_float32(candidate.detach().cpu().float().numpy())
    return {
        "candidate_array_sha256": hashlib.sha256(
            candidate_array.tobytes(order="C")
        ).hexdigest(),
        "metrics": array_parity_metrics(reference_array, candidate_array).to_dict(),
        "reference_array_sha256": hashlib.sha256(
            reference_array.tobytes(order="C")
        ).hexdigest(),
    }


def pixel_domain_comparisons(
    reference: Mapping[str, Array],
    domains: Mapping[str, Mapping[str, Array]],
    stream_names: Sequence[str],
) -> dict[str, object]:
    """Compare candidate pixel domains against one checkpoint proxy."""

    return {
        domain: {
            name: array_parity_metrics(reference[name], candidate[name]).to_dict()
            for name in stream_names
        }
        for domain, candidate in domains.items()
    }


__all__ = [
    "offline_latent_bundle",
    "pixel_domain_comparisons",
    "streaming_latent_bundle",
    "tensor_comparison",
]
