"""Lazy ACT runtime loading and the fail-closed matched no-touch gate."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, NoReturn, Optional, cast

import numpy as np

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.contracts import Array
from robotactile_benchmark.policies.act import (
    RuntimeInputTransform,
    StrictACTPolicy,
    StrictACTRuntime,
)

_STRICT_RUNTIME_MODULE = "deployment.ACTStrict.deploy_policy"
_STRICT_RUNTIME_CLASS = "StrictACTRuntime"
_STRICT_RUNTIME_RELATIVE_FILE = Path("deployment/ACTStrict/deploy_policy.py")


class ArtifactUnavailableError(RuntimeError):
    """Stable qualification failure for a missing matched artifact."""

    code = "artifact_unavailable"


def _torch_input_transform(value: Mapping[str, Array]) -> Mapping[str, object]:
    """Import torch only after a live ACT load was explicitly requested."""

    torch = importlib.import_module("torch")
    result: dict[str, object] = {"qpos": value["qpos"]}
    for key in value.keys() - {"qpos"}:
        result[key] = torch.from_numpy(np.ascontiguousarray(value[key]))
    return result


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required ACT environment is missing: {name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _RuntimeSourcePin:
    """Externally supplied origin and digest authority for ACT source."""

    source_root: Path
    source_file: Path
    source_sha256: str

    @classmethod
    def from_environment(cls) -> "_RuntimeSourcePin":
        root = Path(_required_environment("ACT_RUNTIME_SOURCE_ROOT")).resolve(
            strict=True
        )
        source = Path(_required_environment("ACT_EXPECTED_RUNTIME_FILE")).resolve(
            strict=True
        )
        if not root.is_dir() or not source.is_file():
            raise ValueError("ACT runtime root/file types do not match the source pin")
        try:
            relative_source = source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "ACT runtime source resolves outside the pinned root"
            ) from error
        if relative_source != _STRICT_RUNTIME_RELATIVE_FILE:
            raise ValueError("ACT runtime source is outside the frozen origin layout")
        expected_layout = (root / _STRICT_RUNTIME_RELATIVE_FILE).resolve(strict=True)
        if source != expected_layout:
            raise ValueError("ACT runtime source is outside the frozen origin layout")
        expected_sha256 = _required_environment("ACT_EXPECTED_RUNTIME_SHA256")
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(
                "ACT expected runtime source digest must be lowercase SHA256"
            )
        pin = cls(root, source, expected_sha256)
        pin._verify_file(source)
        return pin

    def _verify_file(self, value: Path) -> None:
        resolved = value.resolve(strict=True)
        if resolved != self.source_file:
            raise ValueError("ACT runtime module origin differs from the pinned file")
        if _sha256_file(resolved) != self.source_sha256:
            raise ValueError("ACT runtime source SHA256 mismatch")

    def discover(self) -> None:
        try:
            spec = importlib.util.find_spec(_STRICT_RUNTIME_MODULE)
        except (AttributeError, ImportError, ValueError) as error:
            raise ValueError("ACT runtime source discovery failed closed") from error
        if spec is None or not isinstance(spec.origin, str):
            raise ImportError("ACT runtime module has no filesystem origin")
        self._verify_file(Path(spec.origin))

    def verify_module(self, module: ModuleType) -> None:
        module_file = getattr(module, "__file__", None)
        module_spec = getattr(module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        if not isinstance(module_file, str) or not isinstance(spec_origin, str):
            raise ValueError("ACT runtime module lacks a pinned origin")
        self._verify_file(Path(module_file))
        self._verify_file(Path(spec_origin))


def _validate_artifact_identity(
    identity: PolicyIdentity,
    task: str,
    evidence: Mapping[str, object],
) -> None:
    if evidence.get("status") != "strict_policy_loaded":
        raise ValueError("ACT artifact evidence is not a strict policy load")
    if evidence.get("task") != task:
        raise ValueError("ACT artifact evidence task mismatch")
    hashes = evidence.get("hashes")
    if not isinstance(hashes, Mapping):
        raise TypeError("ACT artifact evidence hashes must be a mapping")
    if hashes.get("policy_best_sha256") != identity.checkpoint_sha256:
        raise ValueError("ACT artifact checkpoint identity mismatch")
    if hashes.get("run_manifest_sha256") != identity.config_sha256:
        raise ValueError("ACT artifact config identity mismatch")


def load_strict_act_policy(
    identity: PolicyIdentity,
    *,
    task: str,
    device_name: str,
) -> StrictACTPolicy:
    """Lazily construct only the qualified StrictACTRuntime boundary."""

    if not task or not device_name:
        raise ValueError("ACT task and device name must be non-empty")
    if (
        _required_environment("ACT_EXPECTED_CHECKPOINT_SHA256")
        != identity.checkpoint_sha256
    ):
        raise ValueError("ACT checkpoint identity differs from artifact preflight")
    if (
        _required_environment("ACT_EXPECTED_RUN_MANIFEST_SHA256")
        != identity.config_sha256
    ):
        raise ValueError("ACT config identity differs from artifact preflight")
    source_pin = _RuntimeSourcePin.from_environment()
    source_pin.discover()
    module = importlib.import_module(_STRICT_RUNTIME_MODULE)
    source_pin.verify_module(module)
    runtime_class = getattr(module, _STRICT_RUNTIME_CLASS, None)
    if runtime_class is None:
        raise ImportError(f"ACT runtime class is unavailable: {_STRICT_RUNTIME_CLASS}")
    if getattr(runtime_class, "__module__", None) != _STRICT_RUNTIME_MODULE:
        raise ValueError("ACT runtime class is not defined by the pinned module")
    class_file = inspect.getsourcefile(runtime_class)
    if not isinstance(class_file, str):
        raise ValueError("ACT runtime class has no pinned source file")
    source_pin._verify_file(Path(class_file))
    runtime = cast(StrictACTRuntime, runtime_class(task, device_name))
    evidence_path = (
        Path(_required_environment("ACT_RUNTIME_EVIDENCE_DIR"))
        / f"{task}_strict_policy_load.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping):
        raise TypeError("ACT runtime evidence must be a mapping")
    _validate_artifact_identity(identity, task, evidence)
    transform = cast(RuntimeInputTransform, _torch_input_transform)
    return StrictACTPolicy(
        identity,
        runtime,
        artifact_task=task,
        input_transform=transform,
    )


def load_matched_no_touch_policy(
    artifact_manifest: Optional[Path],
) -> NoReturn:
    """Fail closed until a separately qualified matched no-touch artifact exists."""

    location = "none" if artifact_manifest is None else str(artifact_manifest)
    raise ArtifactUnavailableError(
        f"artifact_unavailable: matched no-touch ACT artifact is unavailable ({location})"
    )
