"""Fail-closed artifact gate for one Dream-Tac HCU optimizer-step attempt.

The gate is read-only except for its deterministic no-clobber JSON receipt. It
does not probe an accelerator, deserialize checkpoints, download artifacts, or
launch training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scripts.dream_tac.training.training_preflight import _data_artifacts
from scripts.dream_tac.training.training_request import (
    DreamTacTrainingRequest,
    sha256_regular_file,
)

from .casa_micro import (
    CASA_MICRO_PROTOCOL,
    CASA_SOURCE_RELATIVE_PATH,
    CASA_SOURCE_SHA256,
)
from .casa_micro import CLAIM_BOUNDARY as CASA_CLAIM_BOUNDARY
from .optimizer_step_request import HcuOptimizerStepRequest
from .optimizer_step_validators import (
    load_bound_receipt as _load_bound_receipt,
)
from .optimizer_step_validators import (
    validate_base_dcp as _validate_base_dcp,
)
from .optimizer_step_validators import (
    validate_request_paths as _validate_request_paths,
)
from .optimizer_step_validators import (
    verify_nonempty_bound_file as _verify_nonempty_bound_file,
)
from .overlay import overlay_manifest
from .overlay_contract import (
    CLAIM_BOUNDARY as OVERLAY_CLAIM_BOUNDARY,
)
from .overlay_contract import OVERLAY_PROTOCOL, PINNED_COMMIT, SOURCE_PATCH_SPECS
from .receipt import (
    canonical_json_sha256,
    signed_receipt,
    write_or_verify_receipt,
)

PREFLIGHT_RECEIPT_NAME: Final[str] = "optimizer_step_preflight_receipt.json"
PREFLIGHT_PROTOCOL: Final[str] = "dream_tac_hcu_optimizer_step_preflight_v1"
CLAIM_BOUNDARY: Final[str] = (
    "artifact_and_prior_probe_identity_only_not_optimizer_step_or_training_success"
)
_GRADIENT_NAMES: Final[frozenset[str]] = frozenset(
    {"q", "k", "v", "a", "b", "gamma", "projection_weight", "projection_bias"}
)


def _mapping_field(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


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
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.rstrip()


def _training_data_adapter(
    request: HcuOptimizerStepRequest,
) -> DreamTacTrainingRequest:
    """Adapt only fields consumed by the canonical train759/T5 validator."""

    return DreamTacTrainingRequest(
        phase="p2_micro",
        run_name=request.run_name,
        dream_tac_root=request.dream_tac_root,
        python_executable=request.python_executable,
        materialization_root=request.materialization_root,
        output_root=request.output_root,
        source_manifest_sha256=request.source_manifest_sha256,
        materialization_receipt_sha256=request.materialization_receipt_sha256,
        t5_cache_sha256=request.t5_cache_sha256,
        base_checkpoint=request.base_checkpoint,
        resume_checkpoint=None,
        cuda_devices=(request.hcu_device,),
        nproc_per_node=1,
        master_port=request.master_port,
        max_iter=1,
        save_iter=1,
        batch_size=1,
        num_workers=request.num_workers,
    )


def _validate_overlay(request: HcuOptimizerStepRequest) -> dict[str, object]:
    receipt_path, receipt = _load_bound_receipt(
        request.overlay_receipt, "overlay_receipt"
    )
    expected_manifest = overlay_manifest()
    expected = {
        "status": "applied",
        "protocol_id": OVERLAY_PROTOCOL,
        "claim_boundary": OVERLAY_CLAIM_BOUNDARY,
        "training_launch_performed": False,
        "training_success_claimed": False,
        "pinned_commit": PINNED_COMMIT,
        "overlay_manifest_sha256": canonical_json_sha256(expected_manifest),
        "overlay_manifest": expected_manifest,
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError("Dream-Tac HCU overlay receipt contract mismatch")
    target_value = receipt.get("target_checkout")
    if not isinstance(target_value, str) or not target_value:
        raise ValueError("overlay receipt has no target checkout")
    root = request.dream_tac_root.resolve(strict=True)
    if Path(target_value).resolve(strict=True) != root:
        raise ValueError("overlay receipt target does not match dream_tac_root")
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise ValueError("dream_tac_root must name the HCU overlay checkout root")
    if _git(root, "rev-parse", "HEAD") != PINNED_COMMIT:
        raise ValueError("HCU overlay checkout commit mismatch")
    dirty_lines = tuple(
        line
        for line in _git(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    expected_dirty_paths = {
        spec.relative_path.as_posix() for spec in SOURCE_PATCH_SPECS
    }
    actual_dirty_paths: set[str] = set()
    for line in dirty_lines:
        if len(line) < 4 or line[:2] not in {" M", "M "}:
            raise ValueError("HCU overlay checkout has an unexpected dirty entry")
        actual_dirty_paths.add(line[3:])
    if actual_dirty_paths != expected_dirty_paths:
        raise ValueError("HCU overlay checkout dirty paths do not match the overlay")
    patched_files: list[dict[str, str]] = []
    for spec in SOURCE_PATCH_SPECS:
        path = root / spec.relative_path
        digest = sha256_regular_file(path)
        if digest != spec.patched_sha256:
            raise ValueError(
                f"HCU overlay source SHA256 mismatch: {spec.relative_path}"
            )
        patched_files.append(
            {
                "path": str(path),
                "relative_path": spec.relative_path.as_posix(),
                "sha256": digest,
            }
        )
    return {
        "path": str(receipt_path),
        "sha256": request.overlay_receipt.sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "overlay_manifest_sha256": receipt["overlay_manifest_sha256"],
        "receipt_target_checkout": target_value,
        "runtime_checkout": str(root),
        "runtime_checkout_commit": PINNED_COMMIT,
        "runtime_checkout_dirty_entries": list(dirty_lines),
        "patched_files": patched_files,
    }


def _validate_gradient_checks(execution: Mapping[str, object]) -> None:
    checks = _mapping_field(execution, "gradient_checks")
    if set(checks) != _GRADIENT_NAMES:
        raise ValueError("CASA receipt gradient inventory mismatch")
    for name in sorted(_GRADIENT_NAMES):
        check = _mapping_field(checks, name)
        if any(
            check.get(field) is not True for field in ("present", "finite", "nonzero")
        ):
            raise ValueError(f"CASA receipt gradient check failed: {name}")


def _validate_casa(request: HcuOptimizerStepRequest) -> dict[str, object]:
    receipt_path, receipt = _load_bound_receipt(request.casa_receipt, "casa_receipt")
    expected = {
        "protocol_id": CASA_MICRO_PROTOCOL,
        "overall_status": "passed",
        "claim_boundary": CASA_CLAIM_BOUNDARY,
        "training_launch_performed": False,
        "training_success_claimed": False,
        "fused_kernel_or_performance_parity_claimed": False,
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError("Dream-Tac HCU CASA receipt contract mismatch")
    source = _mapping_field(receipt, "source_identity")
    source_expected = {
        "status": "passed",
        "pinned_commit": PINNED_COMMIT,
        "source_relative_path": CASA_SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": CASA_SOURCE_SHA256,
    }
    if any(source.get(name) != value for name, value in source_expected.items()):
        raise ValueError("Dream-Tac HCU CASA source identity mismatch")
    checkout_value = source.get("checkout")
    if not isinstance(checkout_value, str):
        raise ValueError("CASA receipt has no source checkout")
    root = request.dream_tac_root.resolve(strict=True)
    if Path(checkout_value).resolve(strict=True) != root:
        raise ValueError("CASA checkout does not match dream_tac_root")
    casa_source = root / CASA_SOURCE_RELATIVE_PATH
    if sha256_regular_file(casa_source) != CASA_SOURCE_SHA256:
        raise ValueError("current CASA source SHA256 mismatch")
    execution = _mapping_field(receipt, "execution")
    if (
        execution.get("status") != "passed"
        or execution.get("all_required_gradients_passed") is not True
    ):
        raise ValueError("Dream-Tac HCU CASA execution did not pass")
    device = _mapping_field(execution, "device")
    if device.get("index") != request.hcu_device or not device.get("hip_version"):
        raise ValueError("CASA receipt HCU device identity mismatch")
    output = _mapping_field(execution, "output")
    if any(
        output.get(field) is not True
        for field in ("shape_passed", "dtype_passed", "finite", "nonzero")
    ):
        raise ValueError("CASA receipt output checks did not pass")
    loss = _mapping_field(execution, "loss")
    if any(
        loss.get(field) is not True
        for field in ("finite", "nonzero", "backward_completed")
    ):
        raise ValueError("CASA receipt loss/backward checks did not pass")
    _validate_gradient_checks(execution)
    return {
        "path": str(receipt_path),
        "sha256": request.casa_receipt.sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "checkout": str(root),
        "source_sha256": CASA_SOURCE_SHA256,
        "device": dict(device),
    }


def build_optimizer_step_preflight_receipt(
    request: HcuOptimizerStepRequest,
) -> dict[str, object]:
    """Verify formal inputs without launching the requested optimizer step."""

    runtime_roots = _validate_request_paths(request)
    hcu_environment_script = _verify_nonempty_bound_file(
        request.hcu_environment_script, "hcu_environment_script"
    )
    base = _verify_nonempty_bound_file(request.base_checkpoint, "base_checkpoint")
    tokenizer = _verify_nonempty_bound_file(
        request.tokenizer_checkpoint, "tokenizer_checkpoint"
    )
    base_dcp = _validate_base_dcp(request)
    data = _data_artifacts(_training_data_adapter(request))
    overlay = _validate_overlay(request)
    casa = _validate_casa(request)
    return signed_receipt(
        {
            "schema_version": 1,
            "status": "passed",
            "protocol_id": PREFLIGHT_PROTOCOL,
            "claim_boundary": CLAIM_BOUNDARY,
            "request_sha256": request.request_sha256,
            "optimizer_step_launched": False,
            "training_success_claimed": False,
            "runtime_pythonpath_roots": [str(path) for path in runtime_roots],
            "hcu_environment_script": {
                "path": str(hcu_environment_script),
                "sha256": request.hcu_environment_script.sha256,
            },
            "cosmos_base_checkpoint": {
                "path": str(base),
                "sha256": request.base_checkpoint.sha256,
            },
            "cosmos_tokenizer_checkpoint": {
                "path": str(tokenizer),
                "sha256": request.tokenizer_checkpoint.sha256,
            },
            "cosmos_base_dcp": base_dcp,
            "train759_and_t5": data,
            "source_overlay": overlay,
            "casa_micro": casa,
        }
    )


def write_optimizer_step_preflight_receipt(
    request: HcuOptimizerStepRequest,
) -> tuple[Path, dict[str, object]]:
    """Create one exact receipt, or verify an identical existing receipt."""

    receipt = build_optimizer_step_preflight_receipt(request)
    path = request.receipt_root / PREFLIGHT_RECEIPT_NAME
    write_or_verify_receipt(path, receipt)
    return path, receipt


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    request = HcuOptimizerStepRequest.load(args.request)
    path, receipt = write_optimizer_step_preflight_receipt(request)
    print(
        json.dumps(
            {
                "claim_boundary": receipt["claim_boundary"],
                "output": str(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "status": receipt["status"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_BOUNDARY",
    "PREFLIGHT_PROTOCOL",
    "PREFLIGHT_RECEIPT_NAME",
    "build_optimizer_step_preflight_receipt",
    "main",
    "write_optimizer_step_preflight_receipt",
]
