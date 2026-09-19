"""Artifact and runtime-root validators for the HCU optimizer-step preflight."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from scripts.dream_tac.training.identity import load_json_object
from scripts.dream_tac.training.training_request import (
    BoundFile,
    sha256_regular_file,
    verify_bound_file,
)

from .base_checkpoint_converter import (
    CLAIM_BOUNDARY as CONVERTER_CLAIM_BOUNDARY,
)
from .base_checkpoint_converter import (
    CONVERTER_MANIFEST_NAME,
    CONVERTER_PROTOCOL,
    DCP_LOAD_RELATIVE_PATH,
)
from .optimizer_step_request import HcuOptimizerStepRequest
from .overlay_contract import PINNED_COMMIT
from .receipt import canonical_json_sha256, verify_receipt

_ZERO_SHA256: Final[str] = "0" * 64
_PLACEHOLDER_FRAGMENTS: Final[tuple[str, ...]] = (
    "/absolute/",
    "/path/to/",
    "<",
    ">",
    "changeme",
    "placeholder",
    "replace-me",
    "replace_me",
)
_LOG_REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    "Resuming ckpt",
    "Loaded checkpoint",
    "Starting training",
    "Done with training",
)
_INVALID_NUMERIC_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9_])(nan|[+-]?inf(?:inity)?)(?![a-z0-9_])", re.IGNORECASE
)


def _mapping_field(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _reject_placeholder_path(path: Path, name: str) -> None:
    normalized = str(path).casefold()
    if any(fragment in normalized for fragment in _PLACEHOLDER_FRAGMENTS):
        raise ValueError(f"{name} contains a placeholder path")


def _sha256_value(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == _ZERO_SHA256
    ):
        raise ValueError(f"{name} must be a nonzero lowercase SHA256")
    return value


def _validate_nested_sha256_fields(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{label}.{key}"
            if key == "sha256" or key.endswith("_sha256"):
                _sha256_value(item, child)
            else:
                _validate_nested_sha256_fields(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_nested_sha256_fields(item, f"{label}[{index}]")


def verify_nonempty_bound_file(artifact: BoundFile, name: str) -> Path:
    """Verify one non-placeholder, non-symlink, nonempty bound file."""

    _reject_placeholder_path(artifact.path, name)
    _sha256_value(artifact.sha256, f"{name}.sha256")
    if artifact.path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = verify_bound_file(artifact, name)
    if path.stat().st_size == 0:
        raise ValueError(f"{name} must not be empty")
    return path


def load_bound_receipt(
    artifact: BoundFile, name: str
) -> tuple[Path, dict[str, object]]:
    """Verify one bound receipt's bytes, signature, and nested SHA fields."""

    path = verify_nonempty_bound_file(artifact, name)
    receipt = load_json_object(path)
    verify_receipt(receipt)
    _validate_nested_sha256_fields(receipt, name)
    return path, receipt


def validate_request_paths(request: HcuOptimizerStepRequest) -> tuple[Path, ...]:
    """Validate placeholder-free paths and real ordered Python roots."""

    paths = {
        "dream_tac_root": request.dream_tac_root,
        "python_executable": request.python_executable,
        "materialization_root": request.materialization_root,
        "base_dcp_root": request.base_dcp_root,
        "output_root": request.output_root,
    }
    for name, path in paths.items():
        _reject_placeholder_path(path, name)
    for name, digest in (
        ("source_manifest_sha256", request.source_manifest_sha256),
        (
            "materialization_receipt_sha256",
            request.materialization_receipt_sha256,
        ),
        ("t5_cache_sha256", request.t5_cache_sha256),
    ):
        _sha256_value(digest, name)
    runtime_roots: list[Path] = []
    for index, root in enumerate(request.runtime_pythonpath_roots):
        name = f"runtime_pythonpath_roots[{index}]"
        _reject_placeholder_path(root, name)
        if root.is_symlink():
            raise ValueError(f"{name} must not be a symlink")
        try:
            resolved = root.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"{name} does not exist") from error
        if not resolved.is_dir():
            raise ValueError(f"{name} must be a directory")
        runtime_roots.append(resolved)
    return tuple(runtime_roots)


def _validate_manifest_file(
    conversion_root: Path,
    model_root: Path,
    entry: Mapping[str, object],
) -> dict[str, object]:
    if set(entry) != {"relative_path", "size_bytes", "sha256"}:
        raise ValueError("DCP manifest file entry fields mismatch")
    relative_value = entry["relative_path"]
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError("DCP manifest relative_path is invalid")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("DCP manifest relative_path is unsafe")
    candidate = conversion_root / relative
    current = conversion_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("DCP manifest path traverses a symlink")
    resolved = candidate.resolve(strict=True)
    resolved_model = model_root.resolve(strict=True)
    if resolved_model not in resolved.parents:
        raise ValueError("DCP manifest file is outside the model root")
    size = entry["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("DCP manifest file size is invalid")
    if resolved.stat().st_size != size:
        raise ValueError("DCP manifest file size mismatch")
    expected_sha = _sha256_value(entry["sha256"], "DCP manifest file SHA256")
    if sha256_regular_file(resolved) != expected_sha:
        raise ValueError("DCP manifest file SHA256 mismatch")
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": size,
        "sha256": expected_sha,
    }


def validate_base_dcp(request: HcuOptimizerStepRequest) -> dict[str, object]:
    """Bind one converter receipt to the exact loadable DCP model tree."""

    if request.base_dcp_root.suffix.casefold() == ".pt":
        raise ValueError("base_dcp_root cannot be a flat .pt checkpoint")
    _reject_placeholder_path(request.base_dcp_root, "base_dcp_root")
    if request.base_dcp_root.is_symlink():
        raise ValueError("base_dcp_root must not be a symlink")
    iteration_root = request.base_dcp_root.resolve(strict=True)
    if not iteration_root.is_dir():
        raise ValueError("base_dcp_root must be a DCP iteration directory")
    model_root = iteration_root / "model"
    if model_root.is_symlink() or not model_root.resolve(strict=True).is_dir():
        raise ValueError("base_dcp_root/model must be a real DCP directory")
    receipt_path, receipt = load_bound_receipt(
        request.base_dcp_receipt, "base_dcp_receipt"
    )
    expected = {
        "status": "complete",
        "protocol_id": CONVERTER_PROTOCOL,
        "claim_boundary": CONVERTER_CLAIM_BOUNDARY,
        "training_launch_performed": False,
        "training_success_claimed": False,
        "dcp_model_wrapper_load_ema_to_reg": False,
        "tokenizer_load_mean_std": False,
        "roundtrip_status": "passed",
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError("Dream-Tac base DCP converter receipt contract mismatch")
    if not isinstance(receipt.get("source_config_load_ema_to_reg"), bool):
        raise ValueError("DCP converter source EMA setting is invalid")
    input_checkpoint = _mapping_field(receipt, "input_checkpoint")
    if input_checkpoint.get("sha256") != request.base_checkpoint.sha256:
        raise ValueError("DCP converter input does not match base_checkpoint")
    input_path = input_checkpoint.get("path")
    if not isinstance(input_path, str) or Path(input_path).resolve(
        strict=True
    ) != request.base_checkpoint.path.resolve(strict=True):
        raise ValueError("DCP converter input path does not match base_checkpoint")
    if input_checkpoint.get("top_level_format") not in {"plain", "model", "state_dict"}:
        raise ValueError("DCP converter input format is invalid")
    tokenizer = _mapping_field(receipt, "tokenizer_checkpoint")
    if tokenizer.get("sha256") != request.tokenizer_checkpoint.sha256:
        raise ValueError("DCP converter tokenizer does not match request")
    tokenizer_path = tokenizer.get("path")
    if not isinstance(tokenizer_path, str) or Path(tokenizer_path).resolve(
        strict=True
    ) != request.tokenizer_checkpoint.path.resolve(strict=True):
        raise ValueError("DCP converter tokenizer path does not match request")
    if receipt.get("dcp_iteration_root") != str(iteration_root):
        raise ValueError("DCP converter iteration root does not match request")
    if receipt.get("dcp_model_root") != str(model_root.resolve(strict=True)):
        raise ValueError("DCP converter model root does not match request")
    source = _mapping_field(receipt, "source")
    if source.get("commit") != PINNED_COMMIT:
        raise ValueError("DCP converter source commit mismatch")
    selected = receipt.get("roundtrip_selected_keys")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(key, str) or not key for key in selected)
    ):
        raise ValueError("DCP converter roundtrip key evidence is invalid")
    manifest_path = receipt_path.parent / CONVERTER_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("DCP converter manifest is missing or unsafe")
    manifest = load_json_object(manifest_path)
    unsigned_manifest = dict(manifest)
    claimed_manifest_sha = _sha256_value(
        unsigned_manifest.pop("converter_manifest_sha256", None),
        "converter_manifest_sha256",
    )
    if canonical_json_sha256(unsigned_manifest) != claimed_manifest_sha:
        raise ValueError("DCP converter manifest digest is invalid")
    if receipt.get("converter_manifest_sha256") != claimed_manifest_sha:
        raise ValueError("DCP converter receipt does not bind the manifest")
    if (
        manifest.get("protocol_id") != CONVERTER_PROTOCOL
        or manifest.get("dcp_model_relative_path") != DCP_LOAD_RELATIVE_PATH.as_posix()
    ):
        raise ValueError("DCP converter manifest contract mismatch")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("DCP converter manifest has no files")
    files: list[dict[str, object]] = []
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("DCP converter manifest file entry is invalid")
        files.append(_validate_manifest_file(receipt_path.parent, model_root, raw))
    relative_paths = [str(entry["relative_path"]) for entry in files]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("DCP converter manifest contains duplicate files")
    actual_paths = {
        path.relative_to(receipt_path.parent).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(relative_paths):
        raise ValueError("DCP converter manifest does not cover the exact model tree")
    dcp_manifest_sha = _sha256_value(
        receipt.get("dcp_manifest_sha256"), "dcp_manifest_sha256"
    )
    if canonical_json_sha256(files) != dcp_manifest_sha:
        raise ValueError("DCP model file manifest digest mismatch")
    return {
        "iteration_root": str(iteration_root),
        "model_root": str(model_root.resolve(strict=True)),
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "bound_receipt_sha256": request.base_dcp_receipt.sha256,
        "converter_manifest_path": str(manifest_path),
        "converter_manifest_sha256": claimed_manifest_sha,
        "dcp_manifest_sha256": dcp_manifest_sha,
        "file_count": len(files),
        "input_checkpoint_sha256": request.base_checkpoint.sha256,
        "roundtrip_status": "passed",
    }


def checkpoint_evidence(
    request: HcuOptimizerStepRequest, log_path: Path
) -> dict[str, object]:
    """Validate one-step logs and the complete saved DCP state."""

    log = log_path.read_text(encoding="utf-8", errors="replace")
    markers = {marker: marker in log for marker in _LOG_REQUIRED_MARKERS}
    trained_from_scratch = "Training from scratch." in log
    invalid_numeric_token = _INVALID_NUMERIC_TOKEN.search(log)
    checkpoint = request.job_root / "checkpoints" / "iter_000000001"
    latest = checkpoint.parent / "latest_checkpoint.txt"
    components: dict[str, list[dict[str, object]]] = {}
    for name in ("model", "optim", "scheduler", "trainer"):
        root = checkpoint / name
        files: list[dict[str, object]] = []
        if root.is_dir() and not root.is_symlink():
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"DCP checkpoint contains a symlink: {path}")
                if path.is_file():
                    files.append(
                        {
                            "path": path.relative_to(checkpoint).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256_regular_file(path),
                        }
                    )
        components[name] = files
    latest_value = (
        latest.read_text(encoding="utf-8").strip()
        if latest.is_file() and not latest.is_symlink()
        else None
    )
    passed = (
        all(markers.values())
        and not trained_from_scratch
        and invalid_numeric_token is None
        and latest_value == checkpoint.name
        and all(components.values())
    )
    return {
        "status": "passed" if passed else "failed",
        "required_log_markers": markers,
        "trained_from_scratch": trained_from_scratch,
        "invalid_numeric_token": (
            None if invalid_numeric_token is None else invalid_numeric_token.group(0)
        ),
        "checkpoint_root": str(checkpoint),
        "latest_marker": str(latest),
        "latest_marker_value": latest_value,
        "components": components,
    }


__all__ = [
    "checkpoint_evidence",
    "load_bound_receipt",
    "validate_base_dcp",
    "validate_request_paths",
    "verify_nonempty_bound_file",
]
