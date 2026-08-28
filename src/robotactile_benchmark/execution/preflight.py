"""Strict GPU/Isaac readiness checks without allocating a simulator."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.official_act import (
    build_official_act_live_binding,
)
from robotactile_benchmark.execution.preflight_contracts import (
    LivePreflightCheck,
    LivePreflightError,
    LivePreflightReceipt,
)
from robotactile_benchmark.integrations.provenance import (
    ExternalCheckoutReceipt,
    ExternalIntegrationPin,
    IntegrationLock,
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
    validate_official_univtac_act_artifact,
)

_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 1024 * 1024
_LIVE_ISAAC_MODULES = (
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
)


class LiveHostProbe(Protocol):
    """Dependency-injected host checks used by the no-allocation preflight."""

    def platform(self) -> Mapping[str, str]: ...

    def nvidia(self) -> Mapping[str, str]: ...

    def isaac_python(self, executable: Path) -> Mapping[str, str]: ...


class CheckoutVerifier(Protocol):
    def __call__(
        self, pin: ExternalIntegrationPin, root: Path
    ) -> ExternalCheckoutReceipt: ...


class ArtifactValidator(Protocol):
    def __call__(
        self, manifest: OfficialUniVTACACTArtifactManifest
    ) -> Mapping[str, str]: ...


class ReleaseGate(Protocol):
    def __call__(self, lock: IntegrationLock) -> LivePreflightCheck: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_request_evidence(path: Path) -> Mapping[str, str]:
    target = Path(path).expanduser().absolute()
    if target.is_symlink() or not target.is_file():
        raise LivePreflightError("live request must be a regular file")
    raw = target.read_bytes()
    if not 1 <= len(raw) <= _MAX_REQUEST_BYTES:
        raise LivePreflightError("live request size is invalid")
    try:
        value = strict_json_bytes(raw, "live UniVTAC request")
    except ArtifactValidationError as error:
        raise LivePreflightError(str(error)) from error
    if canonical_json_bytes(value) != raw:
        raise LivePreflightError("live request is not canonical JSON")
    return {"request_file_sha256": hashlib.sha256(raw).hexdigest()}


class SystemLiveHostProbe:
    """Read-only checks for the canonical NVIDIA Linux deployment host."""

    def platform(self) -> Mapping[str, str]:
        system = platform.system()
        machine = platform.machine().lower()
        if system != "Linux" or machine not in {"x86_64", "amd64"}:
            raise LivePreflightError("live deployment requires Linux x86-64")
        return {"machine": machine, "system": system}

    def nvidia(self) -> Mapping[str, str]:
        completed = subprocess.run(
            ("nvidia-smi", "-L"),
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        inventory = completed.stdout.strip()
        if not inventory:
            raise LivePreflightError("nvidia-smi returned an empty GPU inventory")
        return {
            "gpu_inventory_sha256": canonical_hash(inventory),
            "nvidia_smi_exit": str(completed.returncode),
        }

    def isaac_python(self, executable: Path) -> Mapping[str, str]:
        selected = Path(executable)
        if (
            not selected.is_absolute()
            or selected.is_symlink()
            or not selected.is_file()
        ):
            raise LivePreflightError("Isaac Python must be an absolute regular file")
        resolved = selected.resolve(strict=True)
        if resolved != selected or not os.access(resolved, os.X_OK):
            raise LivePreflightError("Isaac Python must be executable without symlinks")
        module_names = repr(_LIVE_ISAAC_MODULES)
        probe = (
            "import importlib.util,json,platform,sys;"
            f"names={module_names};"
            "found={name:importlib.util.find_spec(name) is not None for name in names};"
            "print(json.dumps({'found':found,'machine':platform.machine(),"
            "'python':platform.python_version(),'system':platform.system()},"
            "sort_keys=True,separators=(',',':')))"
        )
        completed = subprocess.run(
            (os.fspath(resolved), "-c", probe),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise LivePreflightError(
                "Isaac Python probe returned invalid JSON"
            ) from error
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("found"), dict)
            or not all(value["found"].values())
        ):
            raise LivePreflightError("Isaac Python is missing required packages")
        return {
            "executable_sha256": _file_sha256(resolved),
            "probe_sha256": canonical_hash(value),
            "python_version": str(value.get("python", "unknown")),
        }


def _failure(error: Exception) -> Mapping[str, str]:
    return {
        "error_sha256": canonical_hash(
            {"message": str(error), "type": type(error).__name__}
        ),
        "error_type": type(error).__name__,
    }


def _capture(
    check_id: str, operation: Callable[[], Mapping[str, str]]
) -> LivePreflightCheck:
    try:
        return LivePreflightCheck(check_id, True, operation())
    except Exception as error:
        return LivePreflightCheck(check_id, False, _failure(error))


def _release_gate(lock: IntegrationLock) -> LivePreflightCheck:
    ids = ("act_runtime", "univtac")
    evidence = {
        integration_id: str(lock.by_id(integration_id).release_ready).lower()
        for integration_id in ids
    }
    return LivePreflightCheck(
        "external_release_readiness",
        all(lock.by_id(integration_id).release_ready for integration_id in ids),
        evidence,
    )


def _checkout_evidence(
    verifier: CheckoutVerifier, pin: ExternalIntegrationPin, root: Path
) -> Mapping[str, str]:
    receipt = verifier(pin, root)
    return {
        "commit_sha": receipt.commit_sha,
        "integration_id": receipt.integration_id,
        "lock_sha256": receipt.lock_sha256,
    }


def run_live_preflight(
    request_path: Path,
    *,
    act_checkout: Path,
    artifact_root: Path,
    stats_sha256: str,
    encoder_sha256: str,
    isaac_python: Path,
    host_probe: LiveHostProbe | None = None,
    checkout_verifier: CheckoutVerifier = verify_external_checkout,
    artifact_validator: ArtifactValidator = validate_official_univtac_act_artifact,
    release_gate: ReleaseGate = _release_gate,
) -> LivePreflightReceipt:
    """Check a complete ACT deployment without starting model or simulator."""

    request_evidence = _canonical_request_evidence(Path(request_path))
    request = load_live_univtac_request(Path(request_path))
    loaded = load_live_univtac_run(request)
    binding = build_official_act_live_binding(
        request,
        artifact_root=Path(artifact_root),
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
    )
    lock = load_integration_lock()
    univtac_pin = lock.by_id("univtac")
    act_pin = lock.by_id("act_runtime")
    probe = SystemLiveHostProbe() if host_probe is None else host_probe
    checks = (
        LivePreflightCheck(
            "request_contract",
            True,
            {
                **request_evidence,
                "run_content_sha256": loaded.content_sha256,
                "trial_manifest_sha256": loaded.trial.sha256,
            },
        ),
        release_gate(lock),
        _capture(
            "univtac_checkout",
            lambda: _checkout_evidence(
                checkout_verifier, univtac_pin, request.upstream_root
            ),
        ),
        _capture(
            "act_checkout",
            lambda: _checkout_evidence(checkout_verifier, act_pin, act_checkout),
        ),
        _capture(
            "official_act_artifacts",
            lambda: artifact_validator(binding.manifest),
        ),
        _capture("host_platform", probe.platform),
        _capture("nvidia_gpu", probe.nvidia),
        _capture("isaac_python", lambda: probe.isaac_python(isaac_python)),
    )
    return LivePreflightReceipt(
        request_content_sha256=loaded.content_sha256,
        task_id=request.task_id,
        condition=request.condition.value,
        policy_kind=request.policy_kind.value,
        checks=checks,
        passed=all(item.passed for item in checks),
    )


def live_preflight_receipt_bytes(receipt: LivePreflightReceipt) -> bytes:
    if type(receipt) is not LivePreflightReceipt:
        raise TypeError("receipt must be an exact LivePreflightReceipt")
    return canonical_json_bytes(receipt.to_dict())


def write_live_preflight_receipt(path: Path, receipt: LivePreflightReceipt) -> bool:
    """Publish once, accepting only an identical existing receipt."""

    target = Path(path).expanduser().absolute()
    payload = live_preflight_receipt_bytes(receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise LivePreflightError("preflight receipt target cannot be a symlink")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different preflight receipt")
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or target.read_bytes() != payload:
                raise FileExistsError(
                    "refusing to replace a concurrently written preflight receipt"
                ) from None
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_live_preflight_receipt(path: Path) -> LivePreflightReceipt:
    target = Path(path).expanduser().absolute()
    if target.is_symlink() or not target.is_file():
        raise LivePreflightError("preflight receipt must be a regular file")
    raw = target.read_bytes()
    if not 1 <= len(raw) <= _MAX_RECEIPT_BYTES:
        raise LivePreflightError("preflight receipt size is invalid")
    try:
        value = strict_json_bytes(raw, "live preflight receipt")
    except ArtifactValidationError as error:
        raise LivePreflightError(str(error)) from error
    if canonical_json_bytes(value) != raw:
        raise LivePreflightError("preflight receipt is not canonical JSON")
    return LivePreflightReceipt.from_dict(value)


__all__ = [
    "LiveHostProbe",
    "SystemLiveHostProbe",
    "live_preflight_receipt_bytes",
    "load_live_preflight_receipt",
    "run_live_preflight",
    "write_live_preflight_receipt",
]
