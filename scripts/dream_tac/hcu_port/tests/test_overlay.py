"""Tests for the commit- and digest-bound Dream-Tac HCU overlay."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.dream_tac.hcu_port import overlay
from scripts.dream_tac.hcu_port.overlay import OverlayConfig, SourcePatchSpec
from scripts.dream_tac.hcu_port.receipt import verify_receipt

_MINIMAL_ORIGINAL = b"""from packaging.version import Version

if Version(te.__version__) >= Version("2.8.0"):
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
else:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb


class RMSNorm(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
q_norm = te.pytorch.RMSNorm(128, eps=1e-6)
k_norm = te.pytorch.RMSNorm(128, eps=1e-6)
k_img_norm = te.pytorch.RMSNorm(128, eps=1e-6)
t_embedding_norm = te.pytorch.RMSNorm(2048, eps=1e-6)
"""
_MINIMAL_PATCHED = b"""
try:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb


class RMSNorm(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _compatible_rms_norm(dim: int, eps: float) -> torch.nn.Module:
    if torch.version.hip is not None:
        return RMSNorm(dim, eps=eps)
    return te.pytorch.RMSNorm(dim, eps=eps)


# ---------------------- Feed Forward Network -----------------------
q_norm = _compatible_rms_norm(128, eps=1e-6)
k_norm = _compatible_rms_norm(128, eps=1e-6)
k_img_norm = _compatible_rms_norm(128, eps=1e-6)
t_embedding_norm = _compatible_rms_norm(2048, eps=1e-6)
"""
_CONFIG_ORIGINAL = b"""import os

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


first = dict(
    load_path=get_checkpoint_path("/path/to/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
)
second = dict(
    load_path=get_checkpoint_path(
                "/path/to/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
            ),
)
"""
_CONFIG_PATCHED = b"""import os

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


def _dream_tac_base_checkpoint() -> str | None:
    if os.environ.get("ROBOTACTILE_HCU_CONFIG_PROBE_ONLY") == "1":
        return None
    checkpoint = os.environ.get("DREAM_TAC_BASE_CHECKPOINT")
    if not checkpoint:
        raise ValueError(
            "DREAM_TAC_BASE_CHECKPOINT is required outside HCU config-probe mode"
        )
    return get_checkpoint_path(checkpoint)


first = dict(
    load_path=_dream_tac_base_checkpoint(),
)
second = dict(
    load_path=_dream_tac_base_checkpoint(),
)
"""
_IMAGINAIRE_CONFIG_ORIGINAL = b"""from typing import Optional


class CheckpointConfig:
    load_path: str = ""
"""
_IMAGINAIRE_CONFIG_PATCHED = b"""from typing import Optional


class CheckpointConfig:
    load_path: str | None = ""
"""
_DISTRIBUTED_ORIGINAL = b"""def init():
    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except pynvml.NVMLError as e:
        log.warning(f"Failed to set device affinity: {e}")

    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device.
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
"""
_DISTRIBUTED_PATCHED = b"""def init():
    # Set GPU affinity.
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

    # Increase the L2 fetch granularity for faster speed.
    if hcu_training:
        log.info("HCU training: skipping CUDA runtime L2 fetch-granularity tuning")
    else:
        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device.
        p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
"""
_ATTENTION_ORIGINAL = (
    b"def choose_backend(compute_cap):\n"
    b"            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if "
    b"compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION\n"
)
_ATTENTION_PATCHED = b"""def choose_backend(compute_cap):
            BEST_SDPA_BACKEND = (
                SDPBackend.FLASH_ATTENTION
                if compute_cap >= 80 or torch.version.hip is not None
                else SDPBackend.EFFICIENT_ATTENTION
            )
"""
_ITER_SPEED_ORIGINAL = b"""import torch
from torch import Tensor


def _format_loss_components(output_batch: dict) -> str:
    parts = []
    for key in LOSS_KEYS_ORDER:
        v = output_batch[key]
        if isinstance(v, Tensor) and v.numel() == 1:
            parts.append(f"{key}={v.item():.4f}")
    return " | ".join(parts) if parts else ""


def _loss_components_for_wandb(output_batch: dict) -> dict:
    out = {}
    for key in LOSS_KEYS_ORDER:
        v = output_batch[key]
        if isinstance(v, Tensor) and v.numel() == 1:
            try:
                out[f"train/loss_components/{key}"] = v.item()
            except (ValueError, RuntimeError):
                pass  # skip nan/inf if needed
    return out
"""
_ITER_SPEED_PATCHED = b"""import torch
from torch import Tensor


def _format_loss_components(output_batch: dict) -> str:
    parts = []
    for key in LOSS_KEYS_ORDER:
        v = output_batch[key]
        if (
            isinstance(v, Tensor)
            and v.numel() == 1
            and bool(torch.isfinite(v).item())
        ):
            parts.append(f"{key}={v.item():.4f}")
    return " | ".join(parts) if parts else ""


def _loss_components_for_wandb(output_batch: dict) -> dict:
    out = {}
    for key in LOSS_KEYS_ORDER:
        v = output_batch[key]
        if (
            isinstance(v, Tensor)
            and v.numel() == 1
            and bool(torch.isfinite(v).item())
        ):
            out[f"train/loss_components/{key}"] = v.item()
    return out
"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _prepare_checkouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, str]:
    source = tmp_path / "pinned"
    source.mkdir()
    files = {
        overlay.TARGET_RELATIVE_PATH: _MINIMAL_ORIGINAL,
        overlay.EXPERIMENT_CONFIG_RELATIVE_PATH: _CONFIG_ORIGINAL,
        overlay.IMAGINAIRE_CONFIG_RELATIVE_PATH: _IMAGINAIRE_CONFIG_ORIGINAL,
        overlay.DISTRIBUTED_RELATIVE_PATH: _DISTRIBUTED_ORIGINAL,
        overlay.ATTENTION_RELATIVE_PATH: _ATTENTION_ORIGINAL,
        overlay.ITER_SPEED_RELATIVE_PATH: _ITER_SPEED_ORIGINAL,
    }
    for relative_path, payload in files.items():
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "RoboTactile Test")
    _git(source, "config", "user.email", "robotactile-test@example.invalid")
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", str(source), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    minimal_original_sha = _sha256(_MINIMAL_ORIGINAL)
    minimal_patched_sha = _sha256(_MINIMAL_PATCHED)
    config_original_sha = _sha256(_CONFIG_ORIGINAL)
    config_patched_sha = _sha256(_CONFIG_PATCHED)
    imaginaire_original_sha = _sha256(_IMAGINAIRE_CONFIG_ORIGINAL)
    imaginaire_patched_sha = _sha256(_IMAGINAIRE_CONFIG_PATCHED)
    distributed_original_sha = _sha256(_DISTRIBUTED_ORIGINAL)
    distributed_patched_sha = _sha256(_DISTRIBUTED_PATCHED)
    attention_original_sha = _sha256(_ATTENTION_ORIGINAL)
    attention_patched_sha = _sha256(_ATTENTION_PATCHED)
    iter_speed_original_sha = _sha256(_ITER_SPEED_ORIGINAL)
    iter_speed_patched_sha = _sha256(_ITER_SPEED_PATCHED)
    monkeypatch.setattr(overlay, "PINNED_COMMIT", commit)
    monkeypatch.setattr(overlay, "ORIGINAL_FILE_SHA256", minimal_original_sha)
    monkeypatch.setattr(overlay, "PATCHED_FILE_SHA256", minimal_patched_sha)
    monkeypatch.setattr(
        overlay,
        "ORIGINAL_EXPERIMENT_CONFIG_SHA256",
        config_original_sha,
    )
    monkeypatch.setattr(
        overlay,
        "PATCHED_EXPERIMENT_CONFIG_SHA256",
        config_patched_sha,
    )
    monkeypatch.setattr(
        overlay,
        "ORIGINAL_IMAGINAIRE_CONFIG_SHA256",
        imaginaire_original_sha,
    )
    monkeypatch.setattr(
        overlay,
        "PATCHED_IMAGINAIRE_CONFIG_SHA256",
        imaginaire_patched_sha,
    )
    monkeypatch.setattr(
        overlay,
        "ORIGINAL_DISTRIBUTED_SHA256",
        distributed_original_sha,
    )
    monkeypatch.setattr(
        overlay,
        "PATCHED_DISTRIBUTED_SHA256",
        distributed_patched_sha,
    )
    monkeypatch.setattr(
        overlay,
        "ORIGINAL_ATTENTION_SHA256",
        attention_original_sha,
    )
    monkeypatch.setattr(
        overlay,
        "PATCHED_ATTENTION_SHA256",
        attention_patched_sha,
    )
    monkeypatch.setattr(
        overlay,
        "ORIGINAL_ITER_SPEED_SHA256",
        iter_speed_original_sha,
    )
    monkeypatch.setattr(
        overlay,
        "PATCHED_ITER_SPEED_SHA256",
        iter_speed_patched_sha,
    )
    monkeypatch.setattr(
        overlay,
        "SOURCE_PATCH_SPECS",
        (
            SourcePatchSpec(
                relative_path=overlay.TARGET_RELATIVE_PATH,
                preimage_sha256=minimal_original_sha,
                patched_sha256=minimal_patched_sha,
                semantic_change="test RoPE and HIP RMSNorm patch",
            ),
            SourcePatchSpec(
                relative_path=overlay.EXPERIMENT_CONFIG_RELATIVE_PATH,
                preimage_sha256=config_original_sha,
                patched_sha256=config_patched_sha,
                semantic_change="test checkpoint patch",
            ),
            SourcePatchSpec(
                relative_path=overlay.IMAGINAIRE_CONFIG_RELATIVE_PATH,
                preimage_sha256=imaginaire_original_sha,
                patched_sha256=imaginaire_patched_sha,
                semantic_change="test optional checkpoint patch",
            ),
            SourcePatchSpec(
                relative_path=overlay.DISTRIBUTED_RELATIVE_PATH,
                preimage_sha256=distributed_original_sha,
                patched_sha256=distributed_patched_sha,
                semantic_change="test HCU distributed guard",
            ),
            SourcePatchSpec(
                relative_path=overlay.ATTENTION_RELATIVE_PATH,
                preimage_sha256=attention_original_sha,
                patched_sha256=attention_patched_sha,
                semantic_change="test HIP SDPA fallback",
            ),
            SourcePatchSpec(
                relative_path=overlay.ITER_SPEED_RELATIVE_PATH,
                preimage_sha256=iter_speed_original_sha,
                patched_sha256=iter_speed_patched_sha,
                semantic_change="test finite scalar metric filtering",
            ),
        ),
    )
    return source, target, commit


def _config(source: Path, target: Path, receipt: Path) -> OverlayConfig:
    return OverlayConfig(
        pinned_source_checkout=source,
        target_checkout=target,
        receipt=receipt,
    )


def test_apply_overlay_patches_only_isolated_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, commit = _prepare_checkouts(tmp_path, monkeypatch)
    receipt_path = tmp_path / "receipts" / "overlay.json"

    receipt = overlay.apply_overlay(_config(source, target, receipt_path))

    verify_receipt(receipt)
    assert receipt["status"] == "applied"
    assert receipt["pinned_commit"] == commit
    assert receipt["claim_boundary"] == (
        "source_compatibility_overlay_only_not_training_success"
    )
    assert receipt["training_launch_performed"] is False
    assert receipt["training_success_claimed"] is False
    assert (source / overlay.TARGET_RELATIVE_PATH).read_bytes() == _MINIMAL_ORIGINAL
    assert (source / overlay.EXPERIMENT_CONFIG_RELATIVE_PATH).read_bytes() == (
        _CONFIG_ORIGINAL
    )
    assert (source / overlay.IMAGINAIRE_CONFIG_RELATIVE_PATH).read_bytes() == (
        _IMAGINAIRE_CONFIG_ORIGINAL
    )
    assert (source / overlay.DISTRIBUTED_RELATIVE_PATH).read_bytes() == (
        _DISTRIBUTED_ORIGINAL
    )
    assert (source / overlay.ATTENTION_RELATIVE_PATH).read_bytes() == (
        _ATTENTION_ORIGINAL
    )
    assert (source / overlay.ITER_SPEED_RELATIVE_PATH).read_bytes() == (
        _ITER_SPEED_ORIGINAL
    )
    assert (target / overlay.TARGET_RELATIVE_PATH).read_bytes() == _MINIMAL_PATCHED
    assert (target / overlay.EXPERIMENT_CONFIG_RELATIVE_PATH).read_bytes() == (
        _CONFIG_PATCHED
    )
    assert (target / overlay.IMAGINAIRE_CONFIG_RELATIVE_PATH).read_bytes() == (
        _IMAGINAIRE_CONFIG_PATCHED
    )
    assert (target / overlay.DISTRIBUTED_RELATIVE_PATH).read_bytes() == (
        _DISTRIBUTED_PATCHED
    )
    assert (target / overlay.ATTENTION_RELATIVE_PATH).read_bytes() == (
        _ATTENTION_PATCHED
    )
    assert (target / overlay.ITER_SPEED_RELATIVE_PATH).read_bytes() == (
        _ITER_SPEED_PATCHED
    )
    assert _git(source, "status", "--porcelain=v1") == ""
    changed = _git(target, "status", "--porcelain=v1").splitlines()
    assert len(changed) == 6
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    manifest = receipt["overlay_manifest"]
    assert isinstance(manifest, dict)
    patches = manifest["source_patches"]
    assert isinstance(patches, list)
    assert len(patches) == 6


def test_existing_receipt_is_refused_before_any_checkout_write(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "missing"

    with pytest.raises(FileExistsError, match="refusing to overwrite receipt"):
        overlay.apply_overlay(_config(missing, missing, receipt))


def test_target_must_be_distinct_from_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, _ = _prepare_checkouts(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="must be distinct"):
        overlay.apply_overlay(_config(source, source, tmp_path / "receipt.json"))
    assert _git(source, "status", "--porcelain=v1") == ""


def test_receipt_must_not_modify_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _ = _prepare_checkouts(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="must not be written inside"):
        overlay.apply_overlay(_config(source, target, source / "overlay.json"))
    assert _git(source, "status", "--porcelain=v1") == ""
    assert _git(target, "status", "--porcelain=v1") == ""


def test_dirty_target_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _ = _prepare_checkouts(tmp_path, monkeypatch)
    (target / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target_checkout must be clean"):
        overlay.apply_overlay(_config(source, target, tmp_path / "receipt.json"))
    assert (target / overlay.TARGET_RELATIVE_PATH).read_bytes() == _MINIMAL_ORIGINAL


def test_wrong_commit_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _ = _prepare_checkouts(tmp_path, monkeypatch)
    monkeypatch.setattr(overlay, "PINNED_COMMIT", "0" * 40)

    with pytest.raises(ValueError, match="commit mismatch"):
        overlay.apply_overlay(_config(source, target, tmp_path / "receipt.json"))


def test_hardlinked_target_file_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _ = _prepare_checkouts(tmp_path, monkeypatch)
    source_file = source / overlay.TARGET_RELATIVE_PATH
    target_file = target / overlay.TARGET_RELATIVE_PATH
    target_file.unlink()
    os.link(source_file, target_file)
    assert _git(target, "status", "--porcelain=v1") == ""

    with pytest.raises(ValueError, match="must not share an inode"):
        overlay.apply_overlay(_config(source, target, tmp_path / "receipt.json"))
    assert source_file.read_bytes() == _MINIMAL_ORIGINAL


def test_receipt_failure_rolls_back_all_source_patches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _ = _prepare_checkouts(tmp_path, monkeypatch)

    def fail_receipt(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic receipt failure")

    monkeypatch.setattr(overlay, "write_or_verify_receipt", fail_receipt)
    with pytest.raises(RuntimeError, match="synthetic receipt failure"):
        overlay.apply_overlay(_config(source, target, tmp_path / "receipt.json"))

    assert (target / overlay.TARGET_RELATIVE_PATH).read_bytes() == _MINIMAL_ORIGINAL
    assert (target / overlay.EXPERIMENT_CONFIG_RELATIVE_PATH).read_bytes() == (
        _CONFIG_ORIGINAL
    )
    assert (target / overlay.IMAGINAIRE_CONFIG_RELATIVE_PATH).read_bytes() == (
        _IMAGINAIRE_CONFIG_ORIGINAL
    )
    assert (target / overlay.DISTRIBUTED_RELATIVE_PATH).read_bytes() == (
        _DISTRIBUTED_ORIGINAL
    )
    assert (target / overlay.ATTENTION_RELATIVE_PATH).read_bytes() == (
        _ATTENTION_ORIGINAL
    )
    assert (target / overlay.ITER_SPEED_RELATIVE_PATH).read_bytes() == (
        _ITER_SPEED_ORIGINAL
    )
    assert _git(target, "status", "--porcelain=v1") == ""
