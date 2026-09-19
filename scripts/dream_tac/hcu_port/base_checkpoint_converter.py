"""Convert one pinned Dream-Tac flat base checkpoint to model-only DCP.

The converter is deliberately strict: it verifies source/config/input identity,
rejects unexpected or implicitly missing tensors, performs a DCP round trip,
and publishes the completed tree with a no-clobber rename. It does not train a
model and does not claim HCU training success.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from scripts.dream_tac.training.identity import (
    canonical_json_sha256,
    signed_payload,
    write_or_verify_json,
)
from scripts.dream_tac.training.training_request import (
    DONOR_EXPERIMENT,
    sha256_regular_file,
)

from .base_checkpoint_contract import validate_flat_contract
from .overlay_contract import PINNED_COMMIT
from .receipt import signed_receipt, write_or_verify_receipt

CONVERTER_PROTOCOL: Final[str] = "dream_tac_flat_pt_to_model_dcp_v1"
CONVERTER_RECEIPT_NAME: Final[str] = "converter_receipt.json"
CONVERTER_MANIFEST_NAME: Final[str] = "converter_manifest.json"
DCP_LOAD_RELATIVE_PATH: Final[Path] = Path("iter_000000000/model")
CONFIG_ROUTER_RELATIVE_PATH: Final[Path] = Path("cosmos_policy/config/config.py")
EXPERIMENT_CONFIG_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py"
)
CLAIM_BOUNDARY: Final[str] = (
    "model_only_checkpoint_format_conversion_not_training_or_model_quality"
)
_CRITICAL_SOURCE_FILES: Final[tuple[Path, ...]] = (
    CONFIG_ROUTER_RELATIVE_PATH,
    Path("cosmos_policy/_src/predict2/utils/model_loader.py"),
    Path("cosmos_policy/_src/predict2/checkpointer/dcp.py"),
)
_SHA256_LENGTH: Final[int] = 64
_SELECTED_TENSOR_LIMIT: Final[int] = 8


@dataclass(frozen=True)
class TensorDescriptor:
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def to_dict(self) -> dict[str, object]:
        return {"shape": list(self.shape), "dtype": self.dtype, "numel": self.numel}


@dataclass(frozen=True)
class ConversionConfig:
    dream_tac_checkout: Path
    experiment_config: Path
    experiment_config_sha256: str
    input_checkpoint: Path
    input_checkpoint_sha256: str
    tokenizer_checkpoint: Path
    tokenizer_checkpoint_sha256: str
    output_root: Path
    allowed_missing_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, path in (
            ("dream_tac_checkout", self.dream_tac_checkout),
            ("experiment_config", self.experiment_config),
            ("input_checkpoint", self.input_checkpoint),
            ("tokenizer_checkpoint", self.tokenizer_checkpoint),
            ("output_root", self.output_root),
        ):
            if not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
        for name, digest in (
            ("experiment_config_sha256", self.experiment_config_sha256),
            ("input_checkpoint_sha256", self.input_checkpoint_sha256),
            ("tokenizer_checkpoint_sha256", self.tokenizer_checkpoint_sha256),
        ):
            if len(digest) != _SHA256_LENGTH or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA256")
            if digest == "0" * _SHA256_LENGTH:
                raise ValueError(f"{name} cannot be an all-zero placeholder")
        if len(set(self.allowed_missing_keys)) != len(self.allowed_missing_keys):
            raise ValueError("allowed_missing_keys must be unique")
        if any(not key or key.strip() != key for key in self.allowed_missing_keys):
            raise ValueError("allowed_missing_keys must contain exact nonempty keys")


class ConverterBackend(Protocol):
    def load_flat_checkpoint(self, path: Path) -> object: ...

    def describe_tensor(self, value: object) -> TensorDescriptor | None: ...

    def create_model(self, config: ConversionConfig) -> tuple[object, object]: ...

    def load_model(
        self, model: object, upstream_config: object, path: Path
    ) -> object: ...

    def model_state_dict(
        self, model: object, *, load_ema_to_reg: bool
    ) -> dict[str, object]: ...

    def load_ema_to_reg(self, upstream_config: object) -> bool: ...

    def save_dcp(self, state: Mapping[str, object], root: Path) -> None: ...

    def empty_like(self, value: object) -> object: ...

    def load_dcp(self, state: dict[str, object], root: Path) -> None: ...

    def tensor_equal(self, left: object, right: object) -> bool: ...


def _normalize_state(
    payload: object, backend: ConverterBackend
) -> tuple[str, dict[str, object], dict[str, TensorDescriptor]]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("flat checkpoint must contain a nonempty mapping")
    wrappers = [
        name
        for name in ("model", "state_dict")
        if isinstance(payload.get(name), Mapping)
    ]
    if len(wrappers) > 1:
        raise ValueError("checkpoint has ambiguous model and state_dict wrappers")
    format_name = wrappers[0] if wrappers else "plain"
    selected = payload[format_name] if wrappers else payload
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError("normalized checkpoint state must be a nonempty mapping")
    state: dict[str, object] = {}
    descriptors: dict[str, TensorDescriptor] = {}
    for key, value in selected.items():
        if not isinstance(key, str) or not key or key.strip() != key:
            raise ValueError("checkpoint tensor keys must be exact nonempty strings")
        descriptor = backend.describe_tensor(value)
        if descriptor is None:
            raise ValueError(f"checkpoint value is not a tensor: {key}")
        state[key] = value
        descriptors[key] = descriptor
    return format_name, state, descriptors


def _descriptors(
    state: Mapping[str, object], backend: ConverterBackend, *, label: str
) -> dict[str, TensorDescriptor]:
    result: dict[str, TensorDescriptor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} has an invalid tensor key")
        descriptor = backend.describe_tensor(value)
        if descriptor is None:
            raise ValueError(f"{label} value is not a tensor: {key}")
        result[key] = descriptor
    if not result:
        raise ValueError(f"{label} cannot be empty")
    return result


def _selected_keys(keys: Sequence[str]) -> tuple[str, ...]:
    ordered = sorted(keys)
    if len(ordered) <= _SELECTED_TENSOR_LIMIT:
        return tuple(ordered)
    indices = {
        round(index * (len(ordered) - 1) / (_SELECTED_TENSOR_LIMIT - 1))
        for index in range(_SELECTED_TENSOR_LIMIT)
    }
    return tuple(ordered[index] for index in sorted(indices))


def _assert_selected_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
    keys: Sequence[str],
    backend: ConverterBackend,
    *,
    label: str,
) -> None:
    if any(not backend.tensor_equal(left[key], right[key]) for key in keys):
        raise ValueError(f"selected tensor equality failed: {label}")


def _state_identity(descriptors: Mapping[str, TensorDescriptor]) -> dict[str, object]:
    inventory = [
        {"key": key, **descriptors[key].to_dict()} for key in sorted(descriptors)
    ]
    return {
        "key_count": len(inventory),
        "tensor_inventory_sha256": canonical_json_sha256(inventory),
        "dtype_counts": dict(
            sorted(Counter(item["dtype"] for item in inventory).items())
        ),
    }


def _file_manifest(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"DCP output contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "relative_path": path.relative_to(root.parent.parent).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_regular_file(path),
                }
            )
    if not files:
        raise ValueError("DCP save produced no regular files")
    return files


def convert_checkpoint(
    config: ConversionConfig, *, backend: ConverterBackend | None = None
) -> tuple[Path, dict[str, object]]:
    """Perform one strict conversion and publish a content-bound output tree."""

    from .base_checkpoint_converter_runtime import verify_conversion_inputs

    source_identity = verify_conversion_inputs(
        config,
        pinned_commit=PINNED_COMMIT,
        critical_source_files=_CRITICAL_SOURCE_FILES,
    )
    if backend is None:
        from .base_checkpoint_converter_runtime import TorchUpstreamBackend

        active_backend: ConverterBackend = TorchUpstreamBackend(
            config.dream_tac_checkout
        )
    else:
        active_backend = backend
    format_name, flat_state, flat_descriptors = _normalize_state(
        active_backend.load_flat_checkpoint(config.input_checkpoint), active_backend
    )
    model, upstream_config = active_backend.create_model(config)
    expected_flat = active_backend.model_state_dict(model, load_ema_to_reg=False)
    expected_descriptors = _descriptors(
        expected_flat, active_backend, label="model template"
    )
    ignored_backend_metadata = validate_flat_contract(
        flat_descriptors, expected_descriptors, config.allowed_missing_keys
    )
    model = active_backend.load_model(model, upstream_config, config.input_checkpoint)
    loaded_flat = active_backend.model_state_dict(model, load_ema_to_reg=False)
    loaded_descriptors = _descriptors(loaded_flat, active_backend, label="loaded model")
    if loaded_descriptors != expected_descriptors:
        raise ValueError("loaded model tensor inventory changed unexpectedly")
    comparable_flat_state = {
        key: value
        for key, value in flat_state.items()
        if key not in ignored_backend_metadata
    }
    flat_selected = _selected_keys(tuple(comparable_flat_state))
    _assert_selected_equal(
        comparable_flat_state,
        loaded_flat,
        flat_selected,
        active_backend,
        label="flat loader",
    )
    source_config_load_ema = active_backend.load_ema_to_reg(upstream_config)
    dcp_state = active_backend.model_state_dict(model, load_ema_to_reg=False)
    dcp_descriptors = _descriptors(dcp_state, active_backend, label="DCP model")

    output = config.output_root
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        model_root = staging / DCP_LOAD_RELATIVE_PATH
        model_root.parent.mkdir(parents=True)
        active_backend.save_dcp(dcp_state, model_root)
        roundtrip = {
            key: active_backend.empty_like(value) for key, value in dcp_state.items()
        }
        active_backend.load_dcp(roundtrip, model_root)
        roundtrip_descriptors = _descriptors(
            roundtrip, active_backend, label="DCP roundtrip"
        )
        if roundtrip_descriptors != dcp_descriptors:
            raise ValueError("DCP roundtrip key/shape/dtype inventory mismatch")
        roundtrip_selected = _selected_keys(tuple(dcp_state))
        _assert_selected_equal(
            dcp_state,
            roundtrip,
            roundtrip_selected,
            active_backend,
            label="DCP roundtrip",
        )
        files = _file_manifest(model_root)
        manifest = signed_payload(
            {
                "schema_version": 1,
                "protocol_id": CONVERTER_PROTOCOL,
                "dcp_model_relative_path": DCP_LOAD_RELATIVE_PATH.as_posix(),
                "files": files,
            },
            field="converter_manifest_sha256",
        )
        write_or_verify_json(staging / CONVERTER_MANIFEST_NAME, manifest)
        receipt = signed_receipt(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol_id": CONVERTER_PROTOCOL,
                "claim_boundary": CLAIM_BOUNDARY,
                "training_launch_performed": False,
                "training_success_claimed": False,
                "input_checkpoint": {
                    "path": str(config.input_checkpoint),
                    "sha256": config.input_checkpoint_sha256,
                    "top_level_format": format_name,
                    **_state_identity(flat_descriptors),
                },
                "tokenizer_checkpoint": {
                    "path": str(config.tokenizer_checkpoint),
                    "sha256": config.tokenizer_checkpoint_sha256,
                },
                "tokenizer_load_mean_std": False,
                "source": source_identity,
                "experiment": {
                    "name": DONOR_EXPERIMENT,
                    "config_path": str(config.experiment_config),
                    "config_sha256": config.experiment_config_sha256,
                },
                "allowed_missing_keys": list(config.allowed_missing_keys),
                "ignored_source_backend_metadata": {
                    "reason": "transformer_engine_fp8_extra_state_absent_from_hcu_backend",
                    "keys": list(ignored_backend_metadata),
                    **_state_identity(
                        {key: flat_descriptors[key] for key in ignored_backend_metadata}
                    ),
                },
                "source_config_load_ema_to_reg": source_config_load_ema,
                "dcp_model_wrapper_load_ema_to_reg": False,
                "dcp_iteration_root": str(output / DCP_LOAD_RELATIVE_PATH.parent),
                "dcp_model_root": str(output / DCP_LOAD_RELATIVE_PATH),
                "dcp_state": _state_identity(dcp_descriptors),
                "converter_manifest_sha256": manifest["converter_manifest_sha256"],
                "dcp_manifest_sha256": canonical_json_sha256(files),
                "roundtrip_status": "passed",
                "roundtrip_selected_keys": list(roundtrip_selected),
            }
        )
        write_or_verify_receipt(staging / CONVERTER_RECEIPT_NAME, receipt)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output root: {output}")
        os.rename(staging, output)
        return output / CONVERTER_RECEIPT_NAME, receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    from .base_checkpoint_converter_runtime import cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_ROUTER_RELATIVE_PATH",
    "CONVERTER_MANIFEST_NAME",
    "CONVERTER_PROTOCOL",
    "CONVERTER_RECEIPT_NAME",
    "ConversionConfig",
    "ConverterBackend",
    "DCP_LOAD_RELATIVE_PATH",
    "EXPERIMENT_CONFIG_RELATIVE_PATH",
    "TensorDescriptor",
    "convert_checkpoint",
    "main",
]
