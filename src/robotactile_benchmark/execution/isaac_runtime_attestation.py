"""Isaac-child runtime attestation for source-bound Clean episodes."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleIdentity
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
)
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.runtime_attestation import (
    load_n0_server_runtime_attestation,
    verify_live_n0_server_attestation,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

ISAAC_ATTESTATION_EVIDENCE_LEVEL = "isaac_child_runtime_attestation_v1"
ISAAC_ATTESTATION_SEMANTIC_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return value


def _relative(value: object, name: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{name} must be a POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} is unsafe")
    return value


def _member(root: Path, relative: str, name: str) -> Path:
    member = PurePosixPath(_relative(relative, name))
    target = root / Path(*member.parts)
    current = root
    for part in member.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    return target


def _relative_to_root(root: Path, path: Path, name: str) -> str:
    target = Path(path).absolute()
    try:
        return _relative(target.relative_to(root).as_posix(), name)
    except ValueError as error:
        raise ValueError(f"{name} must remain below deployment root") from error


def isaac_attestation_sha256(path: Path) -> str:
    """Hash one stable non-symlink Isaac attestation file."""

    target = Path(path).absolute()
    if target.is_symlink() or not target.is_file():
        raise ValueError("Isaac attestation must be a regular file")
    return hashlib.sha256(target.read_bytes()).hexdigest()


@dataclass(frozen=True)
class IsaacRuntimeAttestation:
    """Canonical runtime identity emitted by the actual Isaac child process."""

    campaign_id: str
    lifecycle_identity: LifecycleIdentity
    qualification_relpath: str
    qualification_sha256: str
    runtime_source_binding: RuntimeSourceBinding
    n0_server_attestation_relpath: str
    n0_server_attestation_sha256: str
    n0_server_content_sha256: str
    univtac_source_relpath: str
    univtac_source_commit: str
    n0_source_relpath: str
    n0_source_commit: str
    process_id: int
    process_group_id: int
    posix_session_id: int
    recorded_at_utc: str
    content_sha256: str
    evidence_level: str = ISAAC_ATTESTATION_EVIDENCE_LEVEL
    semantic_version: str = ISAAC_ATTESTATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        if type(self.lifecycle_identity) is not LifecycleIdentity:
            raise TypeError("lifecycle_identity must be LifecycleIdentity")
        if type(self.runtime_source_binding) is not RuntimeSourceBinding:
            raise TypeError("runtime_source_binding must be RuntimeSourceBinding")
        for name in (
            "qualification_relpath",
            "n0_server_attestation_relpath",
            "univtac_source_relpath",
            "n0_source_relpath",
        ):
            object.__setattr__(self, name, _relative(getattr(self, name), name))
        for name in (
            "qualification_sha256",
            "n0_server_attestation_sha256",
            "n0_server_content_sha256",
            "content_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("process_id", "process_group_id", "posix_session_id"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.recorded_at_utc, str)
            or not self.recorded_at_utc
            or self.evidence_level != ISAAC_ATTESTATION_EVIDENCE_LEVEL
            or self.semantic_version != ISAAC_ATTESTATION_SEMANTIC_VERSION
        ):
            raise ValueError("Isaac attestation version/timestamp mismatch")
        if self.content_sha256 != canonical_hash(self._content_dict()):
            raise ValueError("Isaac attestation content hash mismatch")

    def _content_dict(self) -> dict[str, object]:
        return {
            name: (
                self.lifecycle_identity.to_dict()
                if name == "lifecycle_identity"
                else self.runtime_source_binding.to_dict()
                if name == "runtime_source_binding"
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if name != "content_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def build(cls, **values: Any) -> "IsaacRuntimeAttestation":
        content = {
            name: (
                values[name].to_dict()
                if name in {"lifecycle_identity", "runtime_source_binding"}
                else values[name]
            )
            for name in cls.__dataclass_fields__
            if name != "content_sha256"
        }
        return cls(**values, content_sha256=canonical_hash(content))

    @classmethod
    def from_dict(cls, value: object) -> "IsaacRuntimeAttestation":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("Isaac attestation fields mismatch")
        parsed = dict(cast(Mapping[str, object], value))
        parsed["lifecycle_identity"] = LifecycleIdentity.from_dict(
            parsed["lifecycle_identity"]
        )
        parsed["runtime_source_binding"] = RuntimeSourceBinding.from_dict(
            parsed["runtime_source_binding"]
        )
        return cls(**cast(dict[str, Any], parsed))


@dataclass(frozen=True)
class IsaacAttestationRequest:
    """Parent-selected immutable paths consumed by the Isaac child."""

    campaign_id: str
    deployment_root: Path
    output_path: Path
    qualification_path: Path
    n0_server_attestation_path: Path
    n0_server_attestation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        for name in (
            "deployment_root",
            "output_path",
            "qualification_path",
            "n0_server_attestation_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
            object.__setattr__(self, name, value.absolute())
        object.__setattr__(
            self,
            "n0_server_attestation_sha256",
            _sha256(
                self.n0_server_attestation_sha256,
                "n0_server_attestation_sha256",
            ),
        )


def build_isaac_runtime_attestation(
    *,
    deployment_root: Path,
    campaign_id: str,
    lifecycle_identity: LifecycleIdentity,
    output_path: Path,
    qualification_path: Path,
    n0_server_attestation_path: Path,
    n0_server_attestation_sha256: str,
    request: LiveUniVTACRunRequest,
    manifest: N0TWAMArtifactManifest,
    n0_source_root: Path,
) -> IsaacRuntimeAttestation:
    """Verify the active Isaac/N0/UniVTAC stack before backend reset."""

    from robotactile_benchmark.clean_baseline.io import (
        write_canonical_no_clobber,
    )
    from robotactile_benchmark.clean_baseline.qualification import (
        QUALIFICATION_V3_SEMANTIC_VERSION,
        verify_all_task_qualification,
    )

    root = Path(deployment_root).resolve(strict=True)
    qualification = verify_all_task_qualification(root, qualification_path)
    if qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION:
        raise ValueError("Isaac source-bound execution requires qualification v3")
    if len(qualification.tasks) != len(qualification.task_source_bindings):
        raise ValueError("qualification task source inventory is incomplete")
    task_bindings = dict(zip(qualification.tasks, qualification.task_source_bindings))
    try:
        source = task_bindings[request.task_id]
    except KeyError as error:
        raise ValueError("qualification does not cover the requested task") from error
    source.verify_against_current_runtime()
    loaded = load_live_univtac_run(request)
    if (
        lifecycle_identity.task_id != request.task_id
        or lifecycle_identity.trial_manifest_sha256 != loaded.trial.sha256
        or manifest.task_id != request.task_id
        or any(
            getattr(source, name) != getattr(manifest, name)
            for name in (
                "checkpoint_sha256",
                "config_sha256",
                "normalizer_sha256",
                "serve_bundle_sha256",
                "prompt_manifest_sha256",
            )
        )
    ):
        raise ValueError("Isaac request/model/source binding mismatch")
    server = load_n0_server_runtime_attestation(
        root,
        n0_server_attestation_path,
        expected_sha256=n0_server_attestation_sha256,
        verify_members=True,
    )
    if (
        server.qualification_sha256 != qualification.sha256
        or server.task_source_binding != source
        or server.task_id != request.task_id
    ):
        raise ValueError("N0 server attestation disagrees with Isaac binding")
    verify_live_n0_server_attestation(
        server,
        expected_task_id=request.task_id,
        expected_session_id=server.session_id,
        expected_process_group_id=server.server_process_group_id,
    )
    lock = load_integration_lock()
    univtac = verify_external_checkout(lock.by_id("univtac"), request.upstream_root)
    n0 = verify_external_checkout(lock.by_id("n0_twam"), n0_source_root)
    if (
        univtac.commit_sha != source.univtac_source_commit
        or n0.commit_sha != source.n0_source_commit
    ):
        raise ValueError("active source checkout disagrees with qualification")
    attestation = IsaacRuntimeAttestation.build(
        campaign_id=campaign_id,
        lifecycle_identity=lifecycle_identity,
        qualification_relpath=_relative_to_root(
            root, qualification.path, "qualification"
        ),
        qualification_sha256=qualification.sha256,
        runtime_source_binding=source,
        n0_server_attestation_relpath=_relative_to_root(
            root, n0_server_attestation_path, "N0 server attestation"
        ),
        n0_server_attestation_sha256=n0_server_attestation_sha256,
        n0_server_content_sha256=server.content_sha256,
        univtac_source_relpath=_relative_to_root(
            root, request.upstream_root, "UniVTAC source"
        ),
        univtac_source_commit=univtac.commit_sha,
        n0_source_relpath=_relative_to_root(root, n0_source_root, "N0 source"),
        n0_source_commit=n0.commit_sha,
        process_id=os.getpid(),
        process_group_id=os.getpgrp(),
        posix_session_id=os.getsid(0),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_level=ISAAC_ATTESTATION_EVIDENCE_LEVEL,
        semantic_version=ISAAC_ATTESTATION_SEMANTIC_VERSION,
    )
    target = Path(output_path).absolute()
    relative = PurePosixPath(
        _relative_to_root(root, target, "Isaac attestation output")
    )
    if relative.parts[0] != "outputs":
        raise ValueError("Isaac attestation output must remain below outputs/")
    write_canonical_no_clobber(target, attestation.to_dict())
    return attestation


def load_isaac_runtime_attestation(
    deployment_root: Path,
    path: Path,
    *,
    expected_sha256: str | None = None,
    verify_members: bool = True,
) -> IsaacRuntimeAttestation:
    """Load canonical Isaac evidence and optionally re-verify bound files."""

    from robotactile_benchmark.clean_baseline.io import read_canonical_json_file
    from robotactile_benchmark.clean_baseline.qualification import (
        verify_all_task_qualification,
    )

    root = Path(deployment_root).resolve(strict=True)
    target = _member(
        root, _relative_to_root(root, path, "Isaac attestation"), "Isaac attestation"
    )
    actual = isaac_attestation_sha256(target)
    if expected_sha256 is not None and actual != _sha256(
        expected_sha256, "expected Isaac attestation SHA256"
    ):
        raise ValueError("Isaac attestation SHA256 mismatch")
    document, _ = read_canonical_json_file(target, "Isaac runtime attestation")
    attestation = IsaacRuntimeAttestation.from_dict(document)
    if verify_members:
        qualification = verify_all_task_qualification(
            root, _member(root, attestation.qualification_relpath, "qualification")
        )
        server = load_n0_server_runtime_attestation(
            root,
            _member(
                root, attestation.n0_server_attestation_relpath, "N0 server attestation"
            ),
            expected_sha256=attestation.n0_server_attestation_sha256,
            verify_members=True,
        )
        if len(qualification.tasks) != len(qualification.task_source_bindings):
            raise ValueError("qualification task source inventory is incomplete")
        bindings = dict(zip(qualification.tasks, qualification.task_source_bindings))
        if (
            qualification.sha256 != attestation.qualification_sha256
            or bindings.get(attestation.lifecycle_identity.task_id)
            != attestation.runtime_source_binding
            or server.content_sha256 != attestation.n0_server_content_sha256
            or server.task_source_binding != attestation.runtime_source_binding
        ):
            raise ValueError("Isaac attestation member binding mismatch")
        lock = load_integration_lock()
        for integration_id, relpath, commit in (
            (
                "univtac",
                attestation.univtac_source_relpath,
                attestation.univtac_source_commit,
            ),
            ("n0_twam", attestation.n0_source_relpath, attestation.n0_source_commit),
        ):
            receipt = verify_external_checkout(
                lock.by_id(integration_id), _member(root, relpath, "source checkout")
            )
            if receipt.commit_sha != commit:
                raise ValueError("Isaac attestation source checkout changed")
    return attestation


__all__ = [
    "ISAAC_ATTESTATION_EVIDENCE_LEVEL",
    "ISAAC_ATTESTATION_SEMANTIC_VERSION",
    "IsaacAttestationRequest",
    "IsaacRuntimeAttestation",
    "build_isaac_runtime_attestation",
    "isaac_attestation_sha256",
    "load_isaac_runtime_attestation",
]
