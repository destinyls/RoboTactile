"""Contract tests for the HCU optimizer-step artifact preflight."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from scripts.dream_tac.hcu_port import optimizer_step_preflight as preflight
from scripts.dream_tac.hcu_port.base_checkpoint_converter import (
    CLAIM_BOUNDARY as CONVERTER_CLAIM_BOUNDARY,
)
from scripts.dream_tac.hcu_port.base_checkpoint_converter import (
    CONVERTER_MANIFEST_NAME,
    CONVERTER_PROTOCOL,
    CONVERTER_RECEIPT_NAME,
    DCP_LOAD_RELATIVE_PATH,
)
from scripts.dream_tac.hcu_port.casa_micro import (
    CASA_MICRO_PROTOCOL,
    CASA_SOURCE_RELATIVE_PATH,
    CASA_SOURCE_SHA256,
)
from scripts.dream_tac.hcu_port.casa_micro import (
    CLAIM_BOUNDARY as CASA_CLAIM_BOUNDARY,
)
from scripts.dream_tac.hcu_port.optimizer_step_preflight import (
    CLAIM_BOUNDARY,
    PREFLIGHT_PROTOCOL,
    build_optimizer_step_preflight_receipt,
)
from scripts.dream_tac.hcu_port.optimizer_step_request import (
    ACCELERATOR_CONTRACT,
    PHASE,
    REQUEST_SCHEMA,
    HcuOptimizerStepRequest,
)
from scripts.dream_tac.hcu_port.overlay import overlay_manifest
from scripts.dream_tac.hcu_port.overlay_contract import (
    CLAIM_BOUNDARY as OVERLAY_CLAIM_BOUNDARY,
)
from scripts.dream_tac.hcu_port.overlay_contract import (
    OVERLAY_PROTOCOL,
    PINNED_COMMIT,
    SOURCE_PATCH_SPECS,
)
from scripts.dream_tac.hcu_port.receipt import (
    canonical_json_sha256,
    signed_receipt,
    verify_receipt,
)
from scripts.dream_tac.training.identity import signed_payload
from scripts.dream_tac.training.training_request import (
    DreamTacTrainingRequest,
    sha256_regular_file,
)


@dataclass(frozen=True)
class _Fixture:
    request: HcuOptimizerStepRequest
    request_payload: dict[str, object]
    runtime_roots: tuple[Path, ...]
    source_root: Path
    casa_source: Path
    inputs: tuple[Path, ...]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gradient_checks() -> dict[str, dict[str, object]]:
    names = ("q", "k", "v", "a", "b", "gamma", "projection_weight", "projection_bias")
    return {
        name: {"present": True, "finite": True, "nonzero": True, "l2_norm": 1.0}
        for name in names
    }


def _make_fixture(tmp_path: Path) -> _Fixture:
    source_root = tmp_path / "source" / "Dream-Tac-HCU"
    patched_paths: list[Path] = []
    for spec in SOURCE_PATCH_SPECS:
        path = source_root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"patched:{spec.relative_path}".encode())
        patched_paths.append(path)
    casa_source = source_root / CASA_SOURCE_RELATIVE_PATH
    casa_source.parent.mkdir(parents=True, exist_ok=True)
    casa_source.write_bytes(b"casa-source")

    manifest = overlay_manifest()
    overlay_receipt = signed_receipt(
        {
            "schema_version": 1,
            "status": "applied",
            "protocol_id": OVERLAY_PROTOCOL,
            "claim_boundary": OVERLAY_CLAIM_BOUNDARY,
            "training_launch_performed": False,
            "training_success_claimed": False,
            "pinned_source_checkout": str(tmp_path / "source" / "Dream-Tac"),
            "target_checkout": str(source_root),
            "pinned_commit": PINNED_COMMIT,
            "overlay_manifest_sha256": canonical_json_sha256(manifest),
            "overlay_manifest": manifest,
        }
    )
    overlay_path = tmp_path / "prior_receipts" / "overlay.json"
    _write_json(overlay_path, overlay_receipt)

    casa_receipt = signed_receipt(
        {
            "schema_version": 1,
            "protocol_id": CASA_MICRO_PROTOCOL,
            "overall_status": "passed",
            "claim_boundary": CASA_CLAIM_BOUNDARY,
            "training_launch_performed": False,
            "training_success_claimed": False,
            "fused_kernel_or_performance_parity_claimed": False,
            "source_identity": {
                "status": "passed",
                "checkout": str(source_root),
                "pinned_commit": PINNED_COMMIT,
                "source_relative_path": CASA_SOURCE_RELATIVE_PATH.as_posix(),
                "source_sha256": CASA_SOURCE_SHA256,
            },
            "execution": {
                "status": "passed",
                "device": {"index": 0, "name": "HCU", "hip_version": "6.3"},
                "output": {
                    "shape_passed": True,
                    "dtype_passed": True,
                    "finite": True,
                    "nonzero": True,
                },
                "loss": {
                    "finite": True,
                    "nonzero": True,
                    "backward_completed": True,
                },
                "gradient_checks": _gradient_checks(),
                "all_required_gradients_passed": True,
            },
        }
    )
    casa_path = tmp_path / "prior_receipts" / "casa.json"
    _write_json(casa_path, casa_receipt)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    base = artifacts / "model-480p-16fps.pt"
    tokenizer = artifacts / "tokenizer.pth"
    base.write_bytes(b"cosmos-base")
    tokenizer.write_bytes(b"cosmos-tokenizer")
    conversion_root = artifacts / "base-dcp-conversion"
    base_dcp_root = conversion_root / DCP_LOAD_RELATIVE_PATH.parent
    dcp_model_root = conversion_root / DCP_LOAD_RELATIVE_PATH
    dcp_model_root.mkdir(parents=True)
    dcp_payload = dcp_model_root / ".metadata"
    dcp_payload.write_bytes(b"dcp-metadata")
    dcp_files = [
        {
            "relative_path": dcp_payload.relative_to(conversion_root).as_posix(),
            "size_bytes": dcp_payload.stat().st_size,
            "sha256": sha256_regular_file(dcp_payload),
        }
    ]
    dcp_manifest = signed_payload(
        {
            "schema_version": 1,
            "protocol_id": CONVERTER_PROTOCOL,
            "dcp_model_relative_path": DCP_LOAD_RELATIVE_PATH.as_posix(),
            "files": dcp_files,
        },
        field="converter_manifest_sha256",
    )
    dcp_manifest_path = conversion_root / CONVERTER_MANIFEST_NAME
    _write_json(dcp_manifest_path, dcp_manifest)
    dcp_receipt = signed_receipt(
        {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": CONVERTER_PROTOCOL,
            "claim_boundary": CONVERTER_CLAIM_BOUNDARY,
            "training_launch_performed": False,
            "training_success_claimed": False,
            "input_checkpoint": {
                "path": str(base),
                "sha256": sha256_regular_file(base),
                "top_level_format": "plain",
                "key_count": 1,
                "tensor_inventory_sha256": "8" * 64,
                "dtype_counts": {"torch.bfloat16": 1},
            },
            "tokenizer_checkpoint": {
                "path": str(tokenizer),
                "sha256": sha256_regular_file(tokenizer),
            },
            "tokenizer_load_mean_std": False,
            "source": {"commit": PINNED_COMMIT, "critical_files": []},
            "experiment": {
                "name": "cosmos_predict2_2b_480p_franka_cut_banana_20260321",
                "config_path": str(source_root / "cosmos_policy/config/config.py"),
                "config_sha256": "9" * 64,
            },
            "allowed_missing_keys": [],
            "source_config_load_ema_to_reg": True,
            "dcp_model_wrapper_load_ema_to_reg": False,
            "dcp_iteration_root": str(base_dcp_root),
            "dcp_model_root": str(dcp_model_root),
            "dcp_state": {
                "key_count": 1,
                "tensor_inventory_sha256": "a" * 64,
                "dtype_counts": {"torch.bfloat16": 1},
            },
            "converter_manifest_sha256": dcp_manifest["converter_manifest_sha256"],
            "dcp_manifest_sha256": canonical_json_sha256(dcp_files),
            "roundtrip_status": "passed",
            "roundtrip_selected_keys": ["model.weight"],
        }
    )
    dcp_receipt_path = conversion_root / CONVERTER_RECEIPT_NAME
    _write_json(dcp_receipt_path, dcp_receipt)
    materialization = tmp_path / "materialized"
    materialization.mkdir()
    runtime_roots = tuple(
        tmp_path / "runtime" / name
        for name in ("probe-site", "site-cosmos", "site-natten", "site-xformers")
    )
    for root in runtime_roots:
        root.mkdir(parents=True)
    hcu_environment_script = tmp_path / "runtime" / "hcu-env.sh"
    hcu_environment_script.write_text(
        "export HCU_RUNTIME_BOOTSTRAPPED=1\n", encoding="utf-8"
    )
    request_payload: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "accelerator_contract": ACCELERATOR_CONTRACT,
        "dream_tac_commit": PINNED_COMMIT,
        "donor_experiment": "cosmos_predict2_2b_480p_franka_cut_banana_20260321",
        "phase": PHASE,
        "run_name": "dream-tac-hcu-step-test",
        "dream_tac_root": str(source_root),
        "python_executable": str(tmp_path / "runtime" / "bin" / "python"),
        "hcu_environment_script": {
            "path": str(hcu_environment_script),
            "sha256": sha256_regular_file(hcu_environment_script),
        },
        "runtime_pythonpath_roots": [str(path) for path in runtime_roots],
        "materialization_root": str(materialization),
        "output_root": str(tmp_path / "outputs"),
        "source_manifest_sha256": "1" * 64,
        "materialization_receipt_sha256": "2" * 64,
        "t5_cache_sha256": "3" * 64,
        "base_checkpoint": {
            "path": str(base),
            "sha256": sha256_regular_file(base),
        },
        "base_dcp_root": str(base_dcp_root),
        "base_dcp_receipt": {
            "path": str(dcp_receipt_path),
            "sha256": sha256_regular_file(dcp_receipt_path),
        },
        "tokenizer_checkpoint": {
            "path": str(tokenizer),
            "sha256": sha256_regular_file(tokenizer),
        },
        "overlay_receipt": {
            "path": str(overlay_path),
            "sha256": sha256_regular_file(overlay_path),
        },
        "casa_receipt": {
            "path": str(casa_path),
            "sha256": sha256_regular_file(casa_path),
        },
        "hcu_device": 0,
        "master_port": 12341,
        "nproc_per_node": 1,
        "max_iter": 1,
        "save_iter": 1,
        "batch_size": 1,
        "num_workers": 0,
        "resume_checkpoint": None,
    }
    return _Fixture(
        request=HcuOptimizerStepRequest.from_dict(request_payload),
        request_payload=request_payload,
        runtime_roots=runtime_roots,
        source_root=source_root,
        casa_source=casa_source,
        inputs=(
            base,
            tokenizer,
            dcp_payload,
            dcp_manifest_path,
            dcp_receipt_path,
            overlay_path,
            casa_path,
            hcu_environment_script,
            *patched_paths,
            casa_source,
        ),
    )


def _install_stubs(monkeypatch: pytest.MonkeyPatch, fixture: _Fixture) -> None:
    real_sha = sha256_regular_file
    expected_hashes: dict[Path, str] = {
        (fixture.source_root / spec.relative_path).resolve(): spec.patched_sha256
        for spec in SOURCE_PATCH_SPECS
    }
    expected_hashes[fixture.casa_source.resolve()] = CASA_SOURCE_SHA256

    def source_aware_sha(path: Path) -> str:
        expected = expected_hashes.get(path.resolve())
        if expected is not None:
            return expected
        return cast(str, real_sha(path))

    def data_gate(request: DreamTacTrainingRequest) -> dict[str, object]:
        assert request.materialization_root == fixture.request.materialization_root
        assert request.source_manifest_sha256 == "1" * 64
        assert request.materialization_receipt_sha256 == "2" * 64
        assert request.t5_cache_sha256 == "3" * 64
        return {
            "dataset_root": str(request.materialization_root / "dataset"),
            "dataset_statistics_sha256": "4" * 64,
            "materialization_receipt_sha256": "2" * 64,
            "materialized_episode_count": 759,
            "source_manifest_sha256": "1" * 64,
            "t5_cache_sha256": "3" * 64,
            "t5_request_sha256": "5" * 64,
            "t5_receipt_sha256": "6" * 64,
        }

    def git(checkout: Path, *arguments: str) -> str:
        assert checkout.resolve() == fixture.source_root.resolve()
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(fixture.source_root)
        if arguments == ("rev-parse", "HEAD"):
            return cast(str, PINNED_COMMIT)
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "\n".join(
                f" M {spec.relative_path.as_posix()}" for spec in SOURCE_PATCH_SPECS
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(preflight, "sha256_regular_file", source_aware_sha)
    monkeypatch.setattr(preflight, "_data_artifacts", data_gate)
    monkeypatch.setattr(preflight, "_git", git)


def test_preflight_binds_all_inputs_without_mutating_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    _install_stubs(monkeypatch, fixture)
    assert len(SOURCE_PATCH_SPECS) == 6
    before = {path: sha256_regular_file(path) for path in fixture.inputs}

    receipt = build_optimizer_step_preflight_receipt(fixture.request)

    verify_receipt(receipt)
    assert receipt["status"] == "passed"
    assert receipt["protocol_id"] == PREFLIGHT_PROTOCOL
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY
    assert receipt["optimizer_step_launched"] is False
    assert receipt["training_success_claimed"] is False
    assert receipt["runtime_pythonpath_roots"] == [
        str(path) for path in fixture.runtime_roots
    ]
    data = cast(dict[str, object], receipt["train759_and_t5"])
    assert data["materialized_episode_count"] == 759
    base_dcp = cast(dict[str, object], receipt["cosmos_base_dcp"])
    assert base_dcp["iteration_root"] == str(fixture.request.base_dcp_root)
    assert base_dcp["roundtrip_status"] == "passed"
    overlay = cast(dict[str, object], receipt["source_overlay"])
    assert overlay["receipt_target_checkout"] == str(fixture.source_root)
    assert overlay["runtime_checkout"] == str(fixture.source_root)
    assert before == {path: sha256_regular_file(path) for path in fixture.inputs}
