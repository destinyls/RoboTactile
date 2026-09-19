"""No-allocation live deployment preflight tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotactile_benchmark.cli import _parser
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution import (
    LivePolicyKind,
    LivePreflightCheck,
    LivePreflightError,
    LivePreflightReceipt,
    LiveUniVTACRunRequest,
    load_live_preflight_receipt,
    run_live_preflight,
    write_live_preflight_receipt,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.preflight import _LIVE_ISAAC_MODULES
from robotactile_benchmark.execution.preflight_contracts import (
    LIVE_PREFLIGHT_SEMANTIC_VERSION,
)
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.integrations.provenance import ExternalCheckoutReceipt
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition

_TACTILE_CONFIG_SHA256 = (
    "acdab30e50fa7280918804c4a196f75a6db6e854a3c7a86a0ddd84791c533397"
)


def test_live_isaac_preflight_covers_full_univtac_native_stack() -> None:
    assert set(_LIVE_ISAAC_MODULES) == {
        "curobo",
        "isaaclab.app",
        "numpy",
        "robotactile_benchmark",
        "tacex",
        "tacex_assets",
        "tacex_tasks",
        "tacex_uipc",
        "torch_scatter",
        "typing_extensions",
        "uipc",
    }


class _HostProbe:
    def platform(self) -> dict[str, str]:
        return {"machine": "x86_64", "system": "Linux"}

    def nvidia(self) -> dict[str, str]:
        return {"gpu_inventory_sha256": "1" * 64, "nvidia_smi_exit": "0"}

    def isaac_python(self, executable: Path) -> dict[str, str]:
        assert executable.name == "python.sh"
        return {
            "executable_sha256": "2" * 64,
            "probe_sha256": "3" * 64,
            "python_version": "3.10.0",
        }


def _request_file(root: Path) -> Path:
    request = LiveUniVTACRunRequest(
        task_id="pull_out_key",
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.ACT,
        base_system_id="official-act-pull-out-key-touch",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256=_TACTILE_CONFIG_SHA256,
        base_system_manifest_sha256=None,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=2,
        max_observation_steps=3,
        execute_action_steps=1,
        wall_timeout_s=30.0,
        upstream_root=root / "UniVTAC",
        runtime_dir=root / "runtime",
        output_dir=root / "output",
        fault_manifest_path=None,
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cuda:0",
        simulator_device="cuda:0",
        launcher_args=production_univtac_launcher_args(),
    )
    path = root / "request.json"
    path.write_bytes(canonical_json_bytes(live_univtac_request_to_dict(request)))
    return path


def _checkout(pin: object, root: Path) -> ExternalCheckoutReceipt:
    return ExternalCheckoutReceipt(
        integration_id=pin.integration_id,  # type: ignore[attr-defined]
        repository_url=pin.repository_url,  # type: ignore[attr-defined]
        commit_sha=pin.commit_sha,  # type: ignore[attr-defined]
        lock_sha256="4" * 64,
    )


def _artifact(manifest: object) -> dict[str, str]:
    assert manifest.profile is OfficialACTProfile.UNIVTAC  # type: ignore[attr-defined]
    return {
        "checkpoint_sha256": manifest.checkpoint_sha256,  # type: ignore[attr-defined]
        "config_sha256": manifest.config_sha256,  # type: ignore[attr-defined]
    }


def _release_ready(lock: object) -> LivePreflightCheck:
    assert lock.by_id("univtac").commit_sha  # type: ignore[attr-defined]
    return LivePreflightCheck(
        "external_release_readiness",
        True,
        {"univtac": "true"},
    )


def test_preflight_is_deterministic_no_allocation_and_strictly_reloadable(
    tmp_path: Path,
) -> None:
    arguments = {
        "artifact_root": tmp_path / "artifacts",
        "stats_sha256": "c" * 64,
        "encoder_sha256": "d" * 64,
        "isaac_python": tmp_path / "isaac-sim" / "python.sh",
        "host_probe": _HostProbe(),
        "checkout_verifier": _checkout,
        "artifact_validator": _artifact,
        "release_gate": _release_ready,
    }
    first = run_live_preflight(_request_file(tmp_path), **arguments)
    second = run_live_preflight(tmp_path / "request.json", **arguments)
    output = tmp_path / "preflight.json"

    assert first == second
    assert first.passed is True
    assert first.simulator_execution_claimed is False
    assert first.simulator_qualification_claimed is False
    assert {item.check_id for item in first.checks} == {
        "external_release_readiness",
        "host_platform",
        "isaac_python",
        "nvidia_gpu",
        "official_act_artifacts",
        "request_contract",
        "univtac_checkout",
    }
    assert write_live_preflight_receipt(output, first) is True
    assert write_live_preflight_receipt(output, second) is False
    assert load_live_preflight_receipt(output) == first


def test_unrelated_legacy_act_pin_does_not_block_official_preflight(
    tmp_path: Path,
) -> None:
    receipt = run_live_preflight(
        _request_file(tmp_path),
        artifact_root=tmp_path / "artifacts",
        stats_sha256="c" * 64,
        encoder_sha256="d" * 64,
        isaac_python=tmp_path / "isaac-sim" / "python.sh",
        host_probe=_HostProbe(),
        checkout_verifier=_checkout,
        artifact_validator=_artifact,
    )

    assert tuple(item.check_id for item in receipt.checks if not item.passed) == ()
    assert receipt.passed is True


def test_noncanonical_or_symlinked_request_is_rejected(tmp_path: Path) -> None:
    request_path = _request_file(tmp_path)
    value = request_path.read_text(encoding="utf-8")
    request_path.write_text(value.replace(":", ": ", 1), encoding="utf-8")
    arguments = {
        "artifact_root": tmp_path / "artifacts",
        "stats_sha256": "c" * 64,
        "encoder_sha256": "d" * 64,
        "isaac_python": tmp_path / "isaac-sim" / "python.sh",
        "host_probe": _HostProbe(),
        "checkout_verifier": _checkout,
        "artifact_validator": _artifact,
        "release_gate": _release_ready,
    }

    with pytest.raises(LivePreflightError, match="not canonical JSON"):
        run_live_preflight(request_path, **arguments)

    request_path = _request_file(tmp_path)
    symlink = tmp_path / "request-link.json"
    symlink.symlink_to(request_path)
    with pytest.raises(LivePreflightError, match="regular file"):
        run_live_preflight(symlink, **arguments)


def test_receipt_tampering_and_different_existing_output_are_rejected(
    tmp_path: Path,
) -> None:
    receipt = LivePreflightReceipt(
        request_content_sha256="a" * 64,
        task_id="pull_out_key",
        condition="clean",
        policy_kind="act",
        checks=(LivePreflightCheck("only", True, {"value": "frozen"}),),
        passed=True,
    )
    output = tmp_path / "preflight.json"
    assert write_live_preflight_receipt(output, receipt)
    output.write_bytes(output.read_bytes().replace(b'"passed":true', b'"passed":false'))

    with pytest.raises(LivePreflightError):
        load_live_preflight_receipt(output)
    with pytest.raises(FileExistsError):
        write_live_preflight_receipt(output, receipt)


def test_v1_or_restored_preflight_receipt_fails_closed() -> None:
    kwargs = {
        "request_content_sha256": "a" * 64,
        "task_id": "pull_out_key",
        "condition": "clean",
        "policy_kind": "act",
        "checks": (LivePreflightCheck("only", True, {"value": "frozen"}),),
        "passed": True,
    }
    current = LivePreflightReceipt(**kwargs)
    assert current.semantic_version == LIVE_PREFLIGHT_SEMANTIC_VERSION

    with pytest.raises(LivePreflightError, match="semantic version"):
        LivePreflightReceipt(**kwargs, semantic_version="1.0")
    with pytest.raises(LivePreflightError, match="condition is invalid"):
        LivePreflightReceipt(**{**kwargs, "condition": "restored"})
    assert LivePreflightReceipt(**{**kwargs, "condition": "no_touch"}).condition == (
        "no_touch"
    )


def test_public_cli_exposes_live_preflight_not_cpu_qualification() -> None:
    help_text = _parser().format_help()

    assert "preflight-live" in help_text
    assert "univtac-cpu-qualify" not in help_text
    assert "policy-cpu-qualify" not in help_text
