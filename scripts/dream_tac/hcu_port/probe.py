"""Probe Dream-Tac HCU prerequisites without launching or claiming training."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal

from .receipt import signed_receipt, write_or_verify_receipt

HCU_PROBE_PROTOCOL: Final[str] = "dream_tac_hcu_compatibility_probe_v1"
CLAIM_BOUNDARY: Final[str] = "compatibility_probe_only_not_training_success"
DEFAULT_REQUIRED_MODULES: Final[tuple[str, ...]] = (
    "cosmos_policy",
    "flash_attn",
    "transformer_engine",
    "natten",
    "xformers",
    "h5py",
    "cv2",
    "transformers",
)
DTK_ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "DTK_VERSION",
    "DTK_ROOT",
    "DTK_HOME",
    "ROCM_VERSION",
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HIP_HOME",
)
PhaseStatus = Literal["passed", "failed", "not_run", "not_requested"]
Importer = Callable[[str], ModuleType]


@dataclass(frozen=True)
class ProbeConfig:
    """Validated inputs for one non-training HCU compatibility probe."""

    output: Path
    device_index: int
    matrix_size: int
    required_modules: tuple[str, ...]
    casa_module: str | None

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        if not 2 <= self.matrix_size <= 4096:
            raise ValueError("matrix_size must be in [2, 4096]")
        if not self.required_modules:
            raise ValueError("at least one required module must be checked")
        names = (
            *self.required_modules,
            *(() if self.casa_module is None else (self.casa_module,)),
        )
        if any(not name or name.strip() != name for name in names):
            raise ValueError("module names must be non-empty and trimmed")
        if len(set(self.required_modules)) != len(self.required_modules):
            raise ValueError("required module names must be unique")


def _error_payload(error: Exception) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error)[:1000],
    }


def _module_version(module_name: str, module: ModuleType) -> str | None:
    direct = getattr(module, "__version__", None)
    if direct is not None:
        return str(direct)
    package = module_name.split(".", maxsplit=1)[0]
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _import_inventory(
    module_names: Sequence[str], *, importer: Importer
) -> tuple[dict[str, object], bool]:
    inventory: list[dict[str, object]] = []
    passed = True
    for name in module_names:
        try:
            module = importer(name)
            inventory.append(
                {
                    "module": name,
                    "status": "passed",
                    "version": _module_version(name, module),
                    "file": (
                        None
                        if getattr(module, "__file__", None) is None
                        else str(module.__file__)
                    ),
                }
            )
        except Exception as error:
            passed = False
            inventory.append(
                {
                    "module": name,
                    "status": "failed",
                    "error": _error_payload(error),
                }
            )
    return {
        "status": "passed" if passed else "failed",
        "modules": inventory,
    }, passed


def _device_payload(torch_module: Any, device_index: int) -> dict[str, object]:
    properties = torch_module.cuda.get_device_properties(device_index)
    return {
        "index": device_index,
        "name": str(torch_module.cuda.get_device_name(device_index)),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "major": int(properties.major),
        "minor": int(properties.minor),
        "gcn_arch_name": (
            None
            if getattr(properties, "gcnArchName", None) is None
            else str(properties.gcnArchName)
        ),
        "bf16_reported_supported": bool(torch_module.cuda.is_bf16_supported()),
    }


def _runtime_phase(
    *,
    importer: Importer,
    environ: Mapping[str, str],
    device_index: int,
) -> tuple[dict[str, object], Any | None, bool]:
    try:
        torch_module = importer("torch")
        torch_any: Any = torch_module
        hip_version = getattr(torch_any.version, "hip", None)
        cuda_version = getattr(torch_any.version, "cuda", None)
        if not hip_version:
            raise RuntimeError(
                "torch.version.hip is empty; this is not an HCU/HIP build"
            )
        if not bool(torch_any.cuda.is_available()):
            raise RuntimeError("PyTorch HIP device API is unavailable")
        device_count = int(torch_any.cuda.device_count())
        if device_index >= device_count:
            raise RuntimeError(
                f"device_index {device_index} is outside visible device count {device_count}"
            )
        torch_any.cuda.set_device(device_index)
        build_config = str(torch_any.__config__.show())
        markers = [
            line.strip()
            for line in build_config.splitlines()
            if any(token in line.upper() for token in ("DTK", "HIP", "ROCM", "DAS"))
        ]
        dtk_environment = {
            key: environ[key] for key in DTK_ENVIRONMENT_KEYS if environ.get(key)
        }
        payload: dict[str, object] = {
            "status": "passed",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": str(torch_any.__version__),
            "torch_git_version": (
                None
                if getattr(torch_any.version, "git_version", None) is None
                else str(torch_any.version.git_version)
            ),
            "hip_version": str(hip_version),
            "cuda_version": None if cuda_version is None else str(cuda_version),
            "dtk": {
                "environment": dtk_environment,
                "torch_build_markers": markers,
                "identity_reported": bool(dtk_environment or markers),
            },
            "visible_device_count": device_count,
            "device": _device_payload(torch_any, device_index),
        }
        return payload, torch_any, True
    except Exception as error:
        return (
            {
                "status": "failed",
                "error": _error_payload(error),
            },
            None,
            False,
        )


def _bf16_phase(
    torch_module: Any, *, device_index: int, matrix_size: int
) -> tuple[dict[str, object], bool]:
    try:
        torch_module.manual_seed(0)
        torch_module.cuda.manual_seed_all(0)
        device = torch_module.device("cuda", device_index)
        left = torch_module.randn(
            (matrix_size, matrix_size),
            device=device,
            dtype=torch_module.bfloat16,
            requires_grad=True,
        )
        right = torch_module.randn(
            (matrix_size, matrix_size),
            device=device,
            dtype=torch_module.bfloat16,
            requires_grad=True,
        )
        product = left @ right
        loss = product.float().square().mean()
        loss.backward()
        torch_module.cuda.synchronize(device)
        gradients_present = left.grad is not None and right.grad is not None
        finite = bool(torch_module.isfinite(loss).item())
        if gradients_present:
            finite = finite and bool(torch_module.isfinite(left.grad).all().item())
            finite = finite and bool(torch_module.isfinite(right.grad).all().item())
        passed = gradients_present and finite
        return {
            "status": "passed" if passed else "failed",
            "device_index": device_index,
            "matrix_shape": [matrix_size, matrix_size],
            "input_dtype": str(left.dtype),
            "output_dtype": str(product.dtype),
            "gradients_present": gradients_present,
            "all_finite": finite,
            "forward_completed": True,
            "backward_completed": True,
        }, passed
    except Exception as error:
        return {
            "status": "failed",
            "device_index": device_index,
            "matrix_shape": [matrix_size, matrix_size],
            "error": _error_payload(error),
        }, False


def build_probe_receipt(
    config: ProbeConfig,
    *,
    importer: Importer = importlib.import_module,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, object]:
    """Execute bounded compatibility phases and return a signed JSON receipt."""

    runtime, torch_module, runtime_passed = _runtime_phase(
        importer=importer,
        environ=environ,
        device_index=config.device_index,
    )
    required, required_passed = _import_inventory(
        config.required_modules, importer=importer
    )
    if runtime_passed and torch_module is not None:
        bf16, bf16_passed = _bf16_phase(
            torch_module,
            device_index=config.device_index,
            matrix_size=config.matrix_size,
        )
    else:
        bf16 = {
            "status": "not_run",
            "reason": "runtime phase did not establish an HCU/HIP device",
        }
        bf16_passed = False
    if config.casa_module is None:
        casa: dict[str, object] = {
            "status": "not_requested",
            "module": None,
        }
        casa_passed = True
    else:
        casa, casa_passed = _import_inventory((config.casa_module,), importer=importer)
    passed = runtime_passed and required_passed and bf16_passed and casa_passed
    overall = (
        "probe_passed_with_casa_import"
        if passed and config.casa_module is not None
        else "core_probe_passed_casa_not_checked"
        if passed
        else "probe_failed"
    )
    return signed_receipt(
        {
            "schema_version": 1,
            "protocol_id": HCU_PROBE_PROTOCOL,
            "overall_status": overall,
            "claim_boundary": CLAIM_BOUNDARY,
            "training_launch_performed": False,
            "training_success_claimed": False,
            "config": {
                "device_index": config.device_index,
                "matrix_size": config.matrix_size,
                "required_modules": list(config.required_modules),
                "casa_module": config.casa_module,
            },
            "phases": {
                "runtime": runtime,
                "required_imports": required,
                "bf16_forward_backward": bf16,
                "optional_casa_import": casa,
            },
        }
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument(
        "--required-module",
        action="append",
        dest="required_modules",
        help="exact required import; repeat to replace the default module set",
    )
    parser.add_argument("--casa-module", help="optional CASA backend import path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    required = (
        DEFAULT_REQUIRED_MODULES
        if args.required_modules is None
        else tuple(args.required_modules)
    )
    config = ProbeConfig(
        output=args.output,
        device_index=args.device_index,
        matrix_size=args.matrix_size,
        required_modules=required,
        casa_module=args.casa_module,
    )
    receipt = build_probe_receipt(config)
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
    return 0 if receipt["overall_status"] != "probe_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_REQUIRED_MODULES",
    "HCU_PROBE_PROTOCOL",
    "ProbeConfig",
    "build_probe_receipt",
    "main",
]
