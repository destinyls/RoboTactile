"""Exact grouped-Flash compatibility for N0-TWAM on vendor Torch 2.5 HIP.

The Lingchu runtime cannot lower PyTorch 2.5 FlexAttention for the dynamic
sequence lengths produced by UniVTAC.  This module preserves the official
boolean visibility relation by partitioning queries that share the same
allowed key/value set, then evaluates each partition with the vendor fused
FlashAttention kernel.  It is installed externally; the official checkout is
not edited.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]


@dataclass(frozen=True)
class GroupedAttentionMask:
    """Disjoint query groups and their exact allowed key/value indices."""

    shape: tuple[int, int, int, int]
    groups: tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class SelfMaskDescriptor:
    """Inputs required to materialize the official self-attention relation."""

    seq_ids: torch.Tensor
    frame_ids: torch.Tensor
    noise_ids: torch.Tensor
    window_size: int


@dataclass(frozen=True)
class CrossMaskDescriptor:
    """Inputs required to materialize the official text cross-attention relation."""

    query_seq_ids: torch.Tensor
    query_modality_ids: torch.Tensor
    cond_seq_ids: torch.Tensor


class _FlexAttentionInstance(Protocol):
    """Minimal writable surface of the official FlexAttnFunc instance."""

    is_cross: bool
    block_mask: object | None


def uses_vendor_torch25(torch_version: str, hip_version: str | None) -> bool:
    """Return whether the pinned grouped-Flash workaround is required."""
    release = torch_version.split("+", maxsplit=1)[0]
    version_parts = release.split(".")
    return (
        hip_version is not None
        and len(version_parts) >= 2
        and version_parts[:2] == ["2", "5"]
    )


def _query_token_limit() -> int:
    """Read the explicit fail-closed query-length bound."""
    raw_limit = os.environ.get(
        "ROBOTACTILE_GROUPED_FLASH_MAX_QUERY_TOKENS",
        "32768",
    )
    if not raw_limit or not raw_limit.isdecimal():
        raise ValueError(
            "ROBOTACTILE_GROUPED_FLASH_MAX_QUERY_TOKENS must be a positive "
            f"base-10 integer, got {raw_limit!r}"
        )
    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError("grouped-Flash query-token limit must be positive")
    return limit


def _get_mask_mod(
    seq_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    noise_ids: torch.Tensor,
    modality_ids: torch.Tensor,
    window_size: int,
) -> SelfMaskDescriptor:
    """Capture the official self-mask inputs without dense materialization."""
    del modality_ids
    return SelfMaskDescriptor(seq_ids, frame_ids, noise_ids, window_size)


def _get_cross_mask_mod(
    query_seq_ids: torch.Tensor,
    query_modality_ids: torch.Tensor,
    cond_seq_ids: torch.Tensor,
    cond_type_ids: torch.Tensor,
) -> CrossMaskDescriptor:
    """Capture the official cross-mask inputs without dense materialization."""
    del cond_type_ids
    return CrossMaskDescriptor(query_seq_ids, query_modality_ids, cond_seq_ids)


def _create_grouped_self_mask(
    descriptor: SelfMaskDescriptor,
) -> GroupedAttentionMask:
    """Partition queries by the exact official causal/window visibility set."""
    import torch

    seq_ids = descriptor.seq_ids
    frame_ids = descriptor.frame_ids
    noise_ids = descriptor.noise_ids
    if not (seq_ids.shape == frame_ids.shape == noise_ids.shape):
        raise ValueError("self-attention identifier tensors must have equal shape")
    if seq_ids.ndim != 1:
        raise ValueError("self-attention identifier tensors must be one-dimensional")
    if seq_ids.numel() > _query_token_limit():
        raise ValueError(
            "grouped-Flash query length exceeds its measured safety bound: "
            f"query_length={seq_ids.numel()}, limit={_query_token_limit()}"
        )

    descriptor_tensor = torch.stack((seq_ids, frame_ids, noise_ids), dim=1)
    valid = (seq_ids >= 0) & ((noise_ids == 0) | (noise_ids == 1))
    unique_descriptors = torch.unique(
        descriptor_tensor[valid],
        dim=0,
        sorted=True,
    )
    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for query_seq_id, query_frame_id, query_noise_id in unique_descriptors.unbind(
        dim=0
    ):
        query_match = (
            (seq_ids == query_seq_id)
            & (frame_ids == query_frame_id)
            & (noise_ids == query_noise_id)
        )
        same_sequence = (seq_ids == query_seq_id) & (seq_ids >= 0)
        within_window = (query_frame_id - frame_ids).abs() <= descriptor.window_size
        clean_to_clean = (
            (query_noise_id == 1) & (noise_ids == 1) & (frame_ids <= query_frame_id)
        )
        noise_to_clean = (
            (query_noise_id == 0) & (noise_ids == 1) & (frame_ids < query_frame_id)
        )
        noise_to_noise = (
            (query_noise_id == 0) & (noise_ids == 0) & (frame_ids == query_frame_id)
        )
        key_value_match = (
            (clean_to_clean | noise_to_clean | noise_to_noise)
            & same_sequence
            & within_window
        )
        query_indices = torch.nonzero(query_match, as_tuple=False).flatten().long()
        key_value_indices = (
            torch.nonzero(key_value_match, as_tuple=False).flatten().long()
        )
        if query_indices.numel() and key_value_indices.numel():
            groups.append((query_indices, key_value_indices))
    sequence_length = seq_ids.numel()
    return GroupedAttentionMask(
        shape=(1, 1, sequence_length, sequence_length),
        groups=tuple(groups),
    )


def _create_grouped_cross_mask(
    descriptor: CrossMaskDescriptor,
) -> GroupedAttentionMask:
    """Partition non-tactile queries by their exact text sequence."""
    import torch

    query_seq_ids = descriptor.query_seq_ids
    query_modality_ids = descriptor.query_modality_ids
    cond_seq_ids = descriptor.cond_seq_ids
    if query_seq_ids.shape != query_modality_ids.shape:
        raise ValueError("cross-attention query identifier shapes must match")
    if query_seq_ids.ndim != 1 or cond_seq_ids.ndim != 1:
        raise ValueError("cross-attention identifiers must be one-dimensional")
    if query_seq_ids.numel() > _query_token_limit():
        raise ValueError(
            "grouped-Flash cross-query length exceeds its measured safety bound: "
            f"query_length={query_seq_ids.numel()}, limit={_query_token_limit()}"
        )

    valid_query = (query_seq_ids >= 0) & (query_modality_ids != 2)
    unique_sequences = torch.unique(query_seq_ids[valid_query], sorted=True)
    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sequence_id in unique_sequences.unbind(dim=0):
        query_match = (
            (query_seq_ids == sequence_id)
            & (query_seq_ids >= 0)
            & (query_modality_ids != 2)
        )
        key_value_match = (cond_seq_ids == sequence_id) & (cond_seq_ids >= 0)
        query_indices = torch.nonzero(query_match, as_tuple=False).flatten().long()
        key_value_indices = (
            torch.nonzero(key_value_match, as_tuple=False).flatten().long()
        )
        if query_indices.numel() and key_value_indices.numel():
            groups.append((query_indices, key_value_indices))
    return GroupedAttentionMask(
        shape=(1, 1, query_seq_ids.numel(), cond_seq_ids.numel()),
        groups=tuple(groups),
    )


def _create_grouped_block_mask(
    mask_mod: object,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_value_length: int,
    *,
    device: str | torch.device,
    block_size: int | tuple[int, int] = 128,
    _compile: bool = False,
) -> GroupedAttentionMask:
    """Replace compiled BlockMask creation with exact grouped descriptors."""
    del device, block_size, _compile
    if batch_size != 1 or num_heads != 1:
        raise ValueError("official N0-TWAM mask construction requires B=H=1")
    if isinstance(mask_mod, SelfMaskDescriptor):
        mask = _create_grouped_self_mask(mask_mod)
    elif isinstance(mask_mod, CrossMaskDescriptor):
        mask = _create_grouped_cross_mask(mask_mod)
    else:
        raise TypeError(
            f"unsupported grouped mask descriptor: {type(mask_mod).__name__}"
        )
    if mask.shape[-2:] != (query_length, key_value_length):
        raise ValueError(
            "grouped mask length differs from the official request: "
            f"mask={mask.shape[-2:]}, request={(query_length, key_value_length)}"
        )
    return mask


def _run_grouped_flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: GroupedAttentionMask,
    flash_attention: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Evaluate every exact visibility group with fused FlashAttention."""
    import torch

    output = torch.zeros_like(query)
    query_groups: list[torch.Tensor] = []
    output_groups: list[torch.Tensor] = []
    for query_indices, key_value_indices in mask.groups:
        if not query_indices.numel() or not key_value_indices.numel():
            continue
        grouped_query = query.index_select(2, query_indices).transpose(1, 2)
        grouped_key = key.index_select(2, key_value_indices).transpose(1, 2)
        grouped_value = value.index_select(2, key_value_indices).transpose(1, 2)
        grouped_output = (
            flash_attention(
                grouped_query.contiguous(),
                grouped_key.contiguous(),
                grouped_value.contiguous(),
                dropout_p=0.0,
                causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        query_groups.append(query_indices)
        output_groups.append(grouped_output)
    if not output_groups:
        return output + (query.sum() + key.sum() + value.sum()) * 0
    return torch.index_copy(
        output,
        2,
        torch.cat(query_groups),
        torch.cat(output_groups, dim=2),
    )


def _grouped_init(instance: object, is_cross: bool = False) -> None:
    """Initialize the official attention module for grouped Flash execution."""
    import torch

    torch.nn.Module.__init__(instance)
    typed_instance = cast(_FlexAttentionInstance, instance)
    typed_instance.is_cross = is_cross
    typed_instance.block_mask = None


def _grouped_forward(
    instance: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: object | None = None,
) -> torch.Tensor:
    """Drop-in forward preserving official tensor layout and mask semantics."""
    import torch
    import torch.nn.functional as functional  # type: ignore[import-not-found]
    from einops import rearrange  # type: ignore[import-not-found]
    from models import model as official_model  # type: ignore[import-not-found]

    target_dtype = torch.bfloat16 if dtype is None else dtype
    if target_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported grouped-Flash dtype: {target_dtype}")
    query_varlen = rearrange(query[0], "s n d -> 1 n s d")
    key_varlen = rearrange(key[0], "s n d -> 1 n s d")
    value_varlen = rearrange(value[0], "s n d -> 1 n s d")
    query_varlen = query_varlen.to(target_dtype)
    key_varlen = key_varlen.to(target_dtype)
    value_varlen = value_varlen.to(target_dtype)

    block_mask = getattr(instance, "block_mask", None)
    if block_mask is None:
        output = functional.scaled_dot_product_attention(
            query_varlen,
            key_varlen,
            value_varlen,
        )
    else:
        if not isinstance(block_mask, GroupedAttentionMask):
            raise TypeError(
                "vendor grouped-Flash requires GroupedAttentionMask, got "
                f"{type(block_mask).__name__}"
            )
        if block_mask.shape[-2:] != (
            query_varlen.shape[2],
            key_varlen.shape[2],
        ):
            raise ValueError("grouped-Flash mask and query/key lengths differ")
        flash_attention = cast(
            Callable[..., torch.Tensor],
            official_model.flash_attn_func,
        )
        output = _run_grouped_flash(
            query_varlen,
            key_varlen,
            value_varlen,
            block_mask,
            flash_attention,
        )
    return cast(torch.Tensor, rearrange(output, "b n s d -> b s n d"))


def install_vendor_flex25_compat() -> str:
    """Patch only the trainer's source-bound model import and report provenance."""
    if os.environ.get("ROBOTACTILE_ENABLE_FLEX25_COMPAT") != "1":
        raise RuntimeError("vendor FlexAttention compatibility layer is not enabled")
    if os.environ.get("N0_FLEX_ATTENTION_BACKEND") != "grouped_flash_attn":
        raise RuntimeError(
            "N0_FLEX_ATTENTION_BACKEND must be grouped_flash_attn for this runtime"
        )
    if os.environ.get("N0_MOT_CROSS_ATTENTION_BACKEND") != "flash_attn":
        raise RuntimeError(
            "N0_MOT_CROSS_ATTENTION_BACKEND must be flash_attn for this runtime"
        )

    import torch
    from models import model as official_model

    if not uses_vendor_torch25(torch.__version__, torch.version.hip):
        raise RuntimeError(
            "grouped-Flash compatibility is restricted to PyTorch 2.5 HIP; "
            f"found torch={torch.__version__!r}, hip={torch.version.hip!r}"
        )
    expected_model_path = (
        Path(os.environ["N0_EXPECTED_REPO"]) / "n0_twam" / "models" / "model.py"
    ).resolve(strict=True)
    actual_model_path = Path(official_model.__file__).resolve(strict=True)
    if actual_model_path != expected_model_path:
        raise RuntimeError(
            "trainer model import is not source-bound to the official checkout: "
            f"{actual_model_path}"
        )
    if official_model.flash_attn_func is None:
        raise RuntimeError("vendor grouped-Flash requires flash_attn_func")
    _query_token_limit()

    flex_attention = official_model.FlexAttnFunc
    flex_attention.__init__ = _grouped_init
    flex_attention.forward = _grouped_forward
    flex_attention._get_mask_mod = staticmethod(_get_mask_mod)
    flex_attention._get_cross_mask_mod = staticmethod(_get_cross_mask_mod)
    flex_attention.compiled_create_block_mask = staticmethod(_create_grouped_block_mask)
    return f"torch25_hip_grouped_flash_v1_limit{_query_token_limit()}"
