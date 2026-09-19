"""Run a bounded BF16 CASA forward/backward micro on one HCU device."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import warnings
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from .overlay_contract import PINNED_COMMIT
from .receipt import signed_receipt, write_or_verify_receipt

CASA_MICRO_PROTOCOL: Final[str] = "dream_tac_hcu_casa_bf16_micro_v1"
CLAIM_BOUNDARY: Final[str] = "casa_random_tensor_forward_backward_only_not_training"
CASA_SOURCE_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/_src/predict2/networks/tactile_self_attn_chunked.py"
)
CASA_SOURCE_SHA256: Final[str] = (
    "58fdb49296a5731d2ca43c611b1d218e6399466e26c84c80fc7a90d378f2945d"
)
CASA_FUNCTION_NAME: Final[str] = "self_attention_with_tactile_outer_bias_chunked"
BACKEND: Final[str] = "flashbias_sdpa"
SEED: Final[int] = 0
QKV_SHAPE: Final[tuple[int, int, int, int]] = (2, 64, 4, 16)
OUTPUT_SHAPE: Final[tuple[int, int, int]] = (2, 64, 64)
CasaExecutor = Callable[[Path, int], tuple[dict[str, object], bool]]


@dataclass(frozen=True)
class CasaMicroConfig:
    """Explicit source, output, and device for one CASA correctness micro."""

    dream_tac_checkout: Path
    output: Path
    device_index: int

    def __post_init__(self) -> None:
        if not self.dream_tac_checkout.is_absolute():
            raise ValueError("dream_tac_checkout must be an absolute path")
        if not self.output.is_absolute():
            raise ValueError("output must be an absolute path")
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed in {checkout}: {detail}")
    return completed.stdout.strip()


def _validate_source(checkout: Path) -> tuple[Path, dict[str, object]]:
    try:
        resolved = checkout.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Dream-Tac checkout does not exist: {checkout}") from error
    if not resolved.is_dir():
        raise ValueError(f"Dream-Tac checkout is not a directory: {resolved}")
    top_level = Path(_git(resolved, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != resolved:
        raise ValueError("dream_tac_checkout must name the Git checkout root")
    commit = _git(resolved, "rev-parse", "HEAD")
    if commit != PINNED_COMMIT:
        raise ValueError(
            f"Dream-Tac commit mismatch: expected {PINNED_COMMIT}, found {commit}"
        )
    source_file = resolved / CASA_SOURCE_RELATIVE_PATH
    if source_file.is_symlink() or not source_file.is_file():
        raise ValueError(
            f"CASA source must be a regular non-symlink file: {source_file}"
        )
    digest = _sha256_file(source_file)
    if digest != CASA_SOURCE_SHA256:
        raise ValueError(
            f"CASA source SHA256 mismatch: expected {CASA_SOURCE_SHA256}, found {digest}"
        )
    dirty = _git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return source_file, {
        "status": "passed",
        "checkout": str(resolved),
        "pinned_commit": commit,
        "source_relative_path": CASA_SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": digest,
        "function": CASA_FUNCTION_NAME,
        "checkout_clean": not dirty,
        "checkout_dirty_entries": dirty,
    }


def _load_casa_function(source_file: Path) -> Callable[..., Any]:
    module_name = "_robotactile_pinned_dream_tac_casa"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CASA source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, CASA_FUNCTION_NAME, None)
    if not callable(function):
        raise ImportError(f"CASA function is missing: {CASA_FUNCTION_NAME}")
    return cast(Callable[..., Any], function)


def _gradient_check(torch_module: Any, tensor: Any) -> dict[str, object]:
    gradient = tensor.grad
    present = gradient is not None
    if not present:
        return {
            "present": False,
            "finite": False,
            "nonzero": False,
            "l2_norm": None,
        }
    finite = bool(torch_module.isfinite(gradient).all().item())
    nonzero = bool(torch_module.count_nonzero(gradient).item() > 0)
    return {
        "present": True,
        "finite": finite,
        "nonzero": nonzero,
        "l2_norm": float(gradient.float().norm().item()),
    }


def _set_backend(
    environ: MutableMapping[str, str],
) -> tuple[str, str | None]:
    key = "COSMOS_TACTILE_SELF_ATTN_BACKEND"
    previous = environ.get(key)
    environ[key] = BACKEND
    return key, previous


def _restore_backend(
    environ: MutableMapping[str, str], key: str, previous: str | None
) -> None:
    if previous is None:
        environ.pop(key, None)
    else:
        environ[key] = previous


def _execute_casa(
    source_file: Path, device_index: int
) -> tuple[dict[str, object], bool]:
    torch_module: Any = importlib.import_module("torch")
    if not getattr(torch_module.version, "hip", None):
        raise RuntimeError("torch.version.hip is empty; an HCU/HIP build is required")
    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("PyTorch HCU/HIP device API is unavailable")
    device_count = int(torch_module.cuda.device_count())
    if device_index >= device_count:
        raise ValueError(
            f"device_index {device_index} is outside visible device count {device_count}"
        )
    torch_module.manual_seed(SEED)
    torch_module.cuda.manual_seed_all(SEED)
    torch_module.cuda.set_device(device_index)
    device = torch_module.device("cuda", device_index)
    dtype = torch_module.bfloat16
    batch, sequence, heads, head_dim = QKV_SHAPE

    def random_leaf(shape: tuple[int, ...], *, positive: bool = False) -> Any:
        value = torch_module.rand(shape, device=device, dtype=dtype)
        value = value.add(0.5) if positive else value.mul(2).sub(1)
        return value.requires_grad_(True)

    q = random_leaf(QKV_SHAPE)
    k = random_leaf(QKV_SHAPE)
    v = random_leaf(QKV_SHAPE)
    a = random_leaf((sequence,))
    b = random_leaf((sequence,))
    gamma = random_leaf((batch,), positive=True)
    projection = torch_module.nn.Linear(
        heads * head_dim,
        heads * head_dim,
        bias=True,
        device=device,
        dtype=dtype,
    )
    dropout = torch_module.nn.Identity()
    casa = _load_casa_function(source_file)
    backend_key, previous_backend = _set_backend(os.environ)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = casa(
                q,
                k,
                v,
                a,
                b,
                gamma,
                16,
                projection,
                dropout,
            )
            loss = output.float().square().mean()
            loss.backward()
            torch_module.cuda.synchronize(device)
    finally:
        _restore_backend(os.environ, backend_key, previous_backend)
    output_finite = bool(torch_module.isfinite(output).all().item())
    output_nonzero = bool(torch_module.count_nonzero(output).item() > 0)
    loss_finite = bool(torch_module.isfinite(loss).item())
    loss_nonzero = bool(loss.item() != 0.0)
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
        "gamma": gamma,
        "projection_weight": projection.weight,
        "projection_bias": projection.bias,
    }
    gradient_checks = {
        name: _gradient_check(torch_module, tensor) for name, tensor in tensors.items()
    }
    gradients_passed = all(
        bool(check["present"] and check["finite"] and check["nonzero"])
        for check in gradient_checks.values()
    )
    shape_passed = tuple(int(value) for value in output.shape) == OUTPUT_SHAPE
    dtype_passed = output.dtype == dtype
    passed = (
        shape_passed
        and dtype_passed
        and output_finite
        and output_nonzero
        and loss_finite
        and loss_nonzero
        and gradients_passed
    )
    warning_entries = [
        {
            "category": item.category.__name__,
            "message": str(item.message)[:1000],
        }
        for item in caught[:50]
    ]
    return {
        "status": "passed" if passed else "failed",
        "seed": SEED,
        "requested_backend": BACKEND,
        "kernel_selection_observed": False,
        "fused_kernel_or_performance_parity_claimed": False,
        "device": {
            "index": device_index,
            "name": str(torch_module.cuda.get_device_name(device_index)),
            "visible_device_count": device_count,
            "torch_version": str(torch_module.__version__),
            "hip_version": str(torch_module.version.hip),
        },
        "input": {
            "qkv_shape": list(QKV_SHAPE),
            "dtype": str(dtype),
            "chunk_q": 16,
        },
        "output": {
            "shape": [int(value) for value in output.shape],
            "dtype": str(output.dtype),
            "shape_expected": list(OUTPUT_SHAPE),
            "shape_passed": shape_passed,
            "dtype_passed": dtype_passed,
            "finite": output_finite,
            "nonzero": output_nonzero,
        },
        "loss": {
            "value": float(loss.item()),
            "finite": loss_finite,
            "nonzero": loss_nonzero,
            "backward_completed": True,
        },
        "gradient_checks": gradient_checks,
        "all_required_gradients_passed": gradients_passed,
        "warnings": warning_entries,
        "warning_count": len(caught),
        "warnings_are_nonfatal_for_correctness": True,
    }, passed


def _error_payload(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)[:2000]}


def build_casa_micro_receipt(
    config: CasaMicroConfig,
    *,
    executor: CasaExecutor = _execute_casa,
) -> dict[str, object]:
    """Run the source-bound micro and return a signed evidence receipt."""

    try:
        source_file, source_identity = _validate_source(config.dream_tac_checkout)
    except Exception as error:
        source_identity = {
            "status": "failed",
            "requested_checkout": str(config.dream_tac_checkout),
            "error": _error_payload(error),
        }
        execution: dict[str, object] = {
            "status": "not_run",
            "reason": "source identity validation failed",
        }
        passed = False
    else:
        try:
            execution, passed = executor(source_file, config.device_index)
        except Exception as error:
            execution = {"status": "failed", "error": _error_payload(error)}
            passed = False
    return signed_receipt(
        {
            "schema_version": 1,
            "protocol_id": CASA_MICRO_PROTOCOL,
            "overall_status": "passed" if passed else "failed",
            "claim_boundary": CLAIM_BOUNDARY,
            "training_launch_performed": False,
            "training_success_claimed": False,
            "fused_kernel_or_performance_parity_claimed": False,
            "source_identity": source_identity,
            "execution": execution,
        }
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-tac-checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-index", default=0, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = CasaMicroConfig(
        dream_tac_checkout=args.dream_tac_checkout,
        output=args.output,
        device_index=args.device_index,
    )
    receipt = build_casa_micro_receipt(config)
    disposition = write_or_verify_receipt(config.output, receipt)
    print(
        json.dumps(
            {
                "claim_boundary": receipt["claim_boundary"],
                "output": str(config.output),
                "overall_status": receipt["overall_status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "write_disposition": disposition,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
