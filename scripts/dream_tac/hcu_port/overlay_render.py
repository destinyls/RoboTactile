"""Exact preimage-to-output renderers for the Dream-Tac HCU overlay."""

from __future__ import annotations

import hashlib
from typing import Final

from .overlay_contract import (
    ATTENTION_RELATIVE_PATH,
    DISTRIBUTED_RELATIVE_PATH,
    EXPERIMENT_CONFIG_RELATIVE_PATH,
    IMAGINAIRE_CONFIG_RELATIVE_PATH,
    ITER_SPEED_RELATIVE_PATH,
    TARGET_RELATIVE_PATH,
    SourcePatchSpec,
)

_VERSION_IMPORT: Final[str] = "from packaging.version import Version\n"
_VERSIONED_ROPE_IMPORT: Final[str] = """if Version(te.__version__) >= Version("2.8.0"):
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
else:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb
"""
_COMPATIBLE_ROPE_IMPORT: Final[str] = """try:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
"""
_RMS_NORM_CLASS_END: Final[str] = """        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
"""
_RMS_NORM_CLASS_WITH_HIP_FACTORY: Final[str] = """        return output * self.weight


def _compatible_rms_norm(dim: int, eps: float) -> torch.nn.Module:
    if torch.version.hip is not None:
        return RMSNorm(dim, eps=eps)
    return te.pytorch.RMSNorm(dim, eps=eps)


# ---------------------- Feed Forward Network -----------------------
"""
_TE_RMS_NORM_CALL: Final[str] = "te.pytorch.RMSNorm("
_COMPATIBLE_RMS_NORM_CALL: Final[str] = "_compatible_rms_norm("
_CONFIG_BASE_ANCHOR: Final[str] = (
    'BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")\n\n\n'
)
_CONFIG_BASE_HELPER: Final[
    str
] = """BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


def _dream_tac_base_checkpoint() -> str | None:
    if os.environ.get("ROBOTACTILE_HCU_CONFIG_PROBE_ONLY") == "1":
        return None
    checkpoint = os.environ.get("DREAM_TAC_BASE_CHECKPOINT")
    if not checkpoint:
        raise ValueError(
            "DREAM_TAC_BASE_CHECKPOINT is required outside HCU config-probe mode"
        )
    return get_checkpoint_path(checkpoint)


"""
_CONFIG_INLINE_PLACEHOLDER: Final[str] = (
    'load_path=get_checkpoint_path("/path/to/Cosmos-Predict2-2B-Video2World/'
    'model-480p-16fps.pt"),'
)
_CONFIG_MULTILINE_PLACEHOLDER: Final[str] = """load_path=get_checkpoint_path(
                "/path/to/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
            ),"""
_CONFIG_CHECKPOINT_CALL: Final[str] = "load_path=_dream_tac_base_checkpoint(),"
_LOAD_PATH_STR_ANNOTATION: Final[str] = '    load_path: str = ""\n'
_LOAD_PATH_OPTIONAL_ANNOTATION: Final[str] = '    load_path: str | None = ""\n'
_DISTRIBUTED_NVIDIA_AFFINITY: Final[str] = """    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except pynvml.NVMLError as e:
        log.warning(f"Failed to set device affinity: {e}")
"""
_DISTRIBUTED_HCU_AFFINITY_GUARD: Final[str] = """    # Set GPU affinity.
    hcu_training = os.getenv("ROBOTACTILE_HCU_TRAINING") == "1"
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if hcu_training:
        log.info("HCU training: skipping NVIDIA NVML CPU-affinity setup")
    else:
        pynvml.nvmlInit()
        try:
            device = Device(local_rank)
            os.sched_setaffinity(0, device.get_cpu_affinity())
        except pynvml.NVMLError as e:
            log.warning(f"Failed to set device affinity: {e}")
"""
_DISTRIBUTED_CUDART_TUNING: Final[
    str
] = """    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device.
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
"""
_DISTRIBUTED_HCU_CUDART_GUARD: Final[
    str
] = """    # Increase the L2 fetch granularity for faster speed.
    if hcu_training:
        log.info("HCU training: skipping CUDA runtime L2 fetch-granularity tuning")
    else:
        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device.
        p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
"""
_ATTENTION_LEGACY_BACKEND: Final[str] = (
    "            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if "
    "compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION\n"
)
_ATTENTION_HIP_BACKEND: Final[str] = """            BEST_SDPA_BACKEND = (
                SDPBackend.FLASH_ATTENTION
                if compute_cap >= 80 or torch.version.hip is not None
                else SDPBackend.EFFICIENT_ATTENTION
            )
"""
_ITER_SPEED_CONSOLE_SCALAR: Final[
    str
] = """        if isinstance(v, Tensor) and v.numel() == 1:
            parts.append(f"{key}={v.item():.4f}")
"""
_ITER_SPEED_FINITE_CONSOLE_SCALAR: Final[str] = """        if (
            isinstance(v, Tensor)
            and v.numel() == 1
            and bool(torch.isfinite(v).item())
        ):
            parts.append(f"{key}={v.item():.4f}")
"""
_ITER_SPEED_WANDB_SCALAR: Final[
    str
] = """        if isinstance(v, Tensor) and v.numel() == 1:
            try:
                out[f"train/loss_components/{key}"] = v.item()
            except (ValueError, RuntimeError):
                pass  # skip nan/inf if needed
"""
_ITER_SPEED_FINITE_WANDB_SCALAR: Final[str] = """        if (
            isinstance(v, Tensor)
            and v.numel() == 1
            and bool(torch.isfinite(v).item())
        ):
            out[f"train/loss_components/{key}"] = v.item()
"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _patch_minimal_v4_dit(spec: SourcePatchSpec, original: bytes) -> bytes:
    if _sha256_bytes(original) != spec.preimage_sha256:
        raise ValueError("minimal_v4_dit.py preimage SHA256 mismatch")
    source = original.decode("utf-8")
    if source.count(_VERSION_IMPORT) != 1:
        raise ValueError("expected exactly one packaging.version import")
    if source.count(_VERSIONED_ROPE_IMPORT) != 1:
        raise ValueError("expected exactly one version-selected RoPE import block")
    if source.count(_RMS_NORM_CLASS_END) != 1:
        raise ValueError("expected exactly one native RMSNorm class boundary")
    if source.count(_TE_RMS_NORM_CALL) != 4:
        raise ValueError("expected exactly four TransformerEngine RMSNorm calls")
    patched = source.replace(_VERSION_IMPORT, "", 1).replace(
        _VERSIONED_ROPE_IMPORT, _COMPATIBLE_ROPE_IMPORT, 1
    )
    patched = patched.replace(
        _TE_RMS_NORM_CALL,
        _COMPATIBLE_RMS_NORM_CALL,
    ).replace(
        _RMS_NORM_CLASS_END,
        _RMS_NORM_CLASS_WITH_HIP_FACTORY,
        1,
    )
    encoded = patched.encode("utf-8")
    if _sha256_bytes(encoded) != spec.patched_sha256:
        raise RuntimeError("generated minimal_v4_dit.py patched SHA256 mismatch")
    return encoded


def _patch_experiment_config(spec: SourcePatchSpec, original: bytes) -> bytes:
    if _sha256_bytes(original) != spec.preimage_sha256:
        raise ValueError("experiment config preimage SHA256 mismatch")
    source = original.decode("utf-8")
    expected = (
        (_CONFIG_BASE_ANCHOR, "BASE_DATASETS_DIR anchor"),
        (_CONFIG_INLINE_PLACEHOLDER, "inline base-checkpoint placeholder"),
        (_CONFIG_MULTILINE_PLACEHOLDER, "multiline base-checkpoint placeholder"),
    )
    for snippet, label in expected:
        if source.count(snippet) != 1:
            raise ValueError(f"expected exactly one {label}")
    patched = source.replace(_CONFIG_BASE_ANCHOR, _CONFIG_BASE_HELPER, 1)
    patched = patched.replace(
        _CONFIG_INLINE_PLACEHOLDER, _CONFIG_CHECKPOINT_CALL, 1
    ).replace(_CONFIG_MULTILINE_PLACEHOLDER, _CONFIG_CHECKPOINT_CALL, 1)
    encoded = patched.encode("utf-8")
    if _sha256_bytes(encoded) != spec.patched_sha256:
        raise RuntimeError("generated experiment config patched SHA256 mismatch")
    return encoded


def _patch_imaginaire_config(spec: SourcePatchSpec, original: bytes) -> bytes:
    if _sha256_bytes(original) != spec.preimage_sha256:
        raise ValueError("imaginaire config preimage SHA256 mismatch")
    source = original.decode("utf-8")
    if source.count(_LOAD_PATH_STR_ANNOTATION) != 1:
        raise ValueError("expected exactly one checkpoint load_path str annotation")
    patched = source.replace(
        _LOAD_PATH_STR_ANNOTATION, _LOAD_PATH_OPTIONAL_ANNOTATION, 1
    ).encode("utf-8")
    if _sha256_bytes(patched) != spec.patched_sha256:
        raise RuntimeError("generated imaginaire config patched SHA256 mismatch")
    return patched


def _patch_distributed(spec: SourcePatchSpec, original: bytes) -> bytes:
    if _sha256_bytes(original) != spec.preimage_sha256:
        raise ValueError("distributed.py preimage SHA256 mismatch")
    source = original.decode("utf-8")
    expected = (
        (_DISTRIBUTED_NVIDIA_AFFINITY, "NVML affinity block"),
        (_DISTRIBUTED_CUDART_TUNING, "libcudart tuning block"),
    )
    for snippet, label in expected:
        if source.count(snippet) != 1:
            raise ValueError(f"expected exactly one {label}")
    patched = source.replace(
        _DISTRIBUTED_NVIDIA_AFFINITY, _DISTRIBUTED_HCU_AFFINITY_GUARD, 1
    ).replace(_DISTRIBUTED_CUDART_TUNING, _DISTRIBUTED_HCU_CUDART_GUARD, 1)
    encoded = patched.encode("utf-8")
    if _sha256_bytes(encoded) != spec.patched_sha256:
        raise RuntimeError("generated distributed.py patched SHA256 mismatch")
    return encoded


def _patch_attention(spec: SourcePatchSpec, original: bytes) -> bytes:
    if _sha256_bytes(original) != spec.preimage_sha256:
        raise ValueError("attention.py preimage SHA256 mismatch")
    source = original.decode("utf-8")
    if source.count(_ATTENTION_LEGACY_BACKEND) != 1:
        raise ValueError("expected exactly one legacy SDPA fallback backend")
    patched = source.replace(
        _ATTENTION_LEGACY_BACKEND,
        _ATTENTION_HIP_BACKEND,
        1,
    ).encode("utf-8")
    if _sha256_bytes(patched) != spec.patched_sha256:
        raise RuntimeError("generated attention.py patched SHA256 mismatch")
    return patched


def _patch_iter_speed(spec: SourcePatchSpec, original: bytes) -> bytes:
    if _sha256_bytes(original) != spec.preimage_sha256:
        raise ValueError("iter_speed.py preimage SHA256 mismatch")
    source = original.decode("utf-8")
    expected = (
        (_ITER_SPEED_CONSOLE_SCALAR, "console scalar formatter"),
        (_ITER_SPEED_WANDB_SCALAR, "Weights & Biases scalar formatter"),
    )
    for snippet, label in expected:
        if source.count(snippet) != 1:
            raise ValueError(f"expected exactly one {label}")
    patched = source.replace(
        _ITER_SPEED_CONSOLE_SCALAR,
        _ITER_SPEED_FINITE_CONSOLE_SCALAR,
        1,
    ).replace(
        _ITER_SPEED_WANDB_SCALAR,
        _ITER_SPEED_FINITE_WANDB_SCALAR,
        1,
    )
    encoded = patched.encode("utf-8")
    if _sha256_bytes(encoded) != spec.patched_sha256:
        raise RuntimeError("generated iter_speed.py patched SHA256 mismatch")
    return encoded


def render_patch(spec: SourcePatchSpec, original: bytes) -> bytes:
    """Render the one exact transformation registered for ``spec``."""

    if spec.relative_path == TARGET_RELATIVE_PATH:
        return _patch_minimal_v4_dit(spec, original)
    if spec.relative_path == EXPERIMENT_CONFIG_RELATIVE_PATH:
        return _patch_experiment_config(spec, original)
    if spec.relative_path == IMAGINAIRE_CONFIG_RELATIVE_PATH:
        return _patch_imaginaire_config(spec, original)
    if spec.relative_path == DISTRIBUTED_RELATIVE_PATH:
        return _patch_distributed(spec, original)
    if spec.relative_path == ATTENTION_RELATIVE_PATH:
        return _patch_attention(spec, original)
    if spec.relative_path == ITER_SPEED_RELATIVE_PATH:
        return _patch_iter_speed(spec, original)
    raise RuntimeError(f"no patch renderer registered for {spec.relative_path}")


__all__ = ["render_patch"]
