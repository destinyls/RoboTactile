"""N0 rank-zero runtime attestation for source-bound Clean evaluation."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Tuple, cast

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    load_n0_twam_artifact_manifest,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationEvidenceBinding,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock
from robotactile_benchmark.integrations.registry import (
    load_model_integration_config,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.runtime_source import (
    RuntimeSourceBinding,
    current_integrations_lock_sha256,
    current_source_manifest_sha256,
)

N0_SERVER_ATTESTATION_EVIDENCE_LEVEL = "n0_rank0_runtime_attestation_v1"
N0_SERVER_ATTESTATION_SEMANTIC_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INSTALLATION_KINDS = (
    "isaac_install_receipt",
    "n0_client_install_receipt",
    "n0_runtime_receipt",
    "tacex_install_receipt",
)
_ARTIFACT_HASH_FIELDS = (
    "checkpoint_sha256",
    "config_sha256",
    "normalizer_sha256",
    "serve_bundle_sha256",
    "prompt_manifest_sha256",
)


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
    if path.parts[0] not in {"artifacts", "outputs", "requests", "sources"}:
        raise ValueError(f"{name} is outside the deployment root")
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return cast(int, value)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return cast(bool, value)


def n0_server_attestation_sha256(path: Path) -> str:
    """Hash one stable, non-symlink attestation file."""

    target = Path(path).absolute()
    if target.is_symlink() or not target.is_file():
        raise ValueError("N0 server attestation must be a regular file")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class N0ServerRuntimeAttestation:
    """Canonical statement emitted by the actual N0 rank-zero process."""

    task_id: str
    session_id: str
    server_pid: int
    server_process_group_id: int
    server_posix_session_id: int
    rank: int
    world_size: int
    debug_offload: bool
    paper_gate_passed: bool
    n0_source_relpath: str
    n0_source_commit: str
    n0_source_clean: bool
    robotactile_source_manifest_sha256: str
    integrations_lock_sha256: str
    qualification_relpath: str
    qualification_sha256: str
    qualification_semantic_version: str
    task_source_binding: RuntimeSourceBinding
    integration_config_relpath: str
    integration_config_sha256: str
    artifact_manifest_relpath: str
    artifact_manifest_sha256: str
    n0_artifact_hashes: Mapping[str, str]
    installation_receipts: Tuple[ObservationEvidenceBinding, ...]
    recorded_at_utc: str
    content_sha256: str
    evidence_level: str = N0_SERVER_ATTESTATION_EVIDENCE_LEVEL
    semantic_version: str = N0_SERVER_ATTESTATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        from robotactile_benchmark.clean_baseline.qualification import (
            QUALIFICATION_V3_SEMANTIC_VERSION,
        )

        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(
            self, "session_id", _identifier(self.session_id, "session_id")
        )
        for name in (
            "server_pid",
            "server_process_group_id",
            "server_posix_session_id",
            "world_size",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        object.__setattr__(self, "rank", _integer(self.rank, "rank", minimum=0))
        if self.rank != 0 or self.rank >= self.world_size:
            raise ValueError("N0 server attestation must be emitted by rank zero")
        for name in ("debug_offload", "paper_gate_passed", "n0_source_clean"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if self.debug_offload or not self.paper_gate_passed or not self.n0_source_clean:
            raise ValueError("N0 server paper gate did not pass")
        for name in (
            "n0_source_relpath",
            "qualification_relpath",
            "integration_config_relpath",
            "artifact_manifest_relpath",
        ):
            object.__setattr__(self, name, _relative(getattr(self, name), name))
        if (
            not isinstance(self.n0_source_commit, str)
            or _COMMIT.fullmatch(self.n0_source_commit) is None
        ):
            raise ValueError("n0_source_commit must be a lowercase commit SHA")
        for name in (
            "robotactile_source_manifest_sha256",
            "integrations_lock_sha256",
            "qualification_sha256",
            "integration_config_sha256",
            "artifact_manifest_sha256",
            "content_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.qualification_semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION:
            raise ValueError("N0 server attestation requires qualification v3")
        if type(self.task_source_binding) is not RuntimeSourceBinding:
            raise TypeError("task_source_binding must be RuntimeSourceBinding")
        hashes = dict(self.n0_artifact_hashes)
        if set(hashes) != set(_ARTIFACT_HASH_FIELDS):
            raise ValueError("N0 artifact hash inventory mismatch")
        for name in _ARTIFACT_HASH_FIELDS:
            hashes[name] = _sha256(hashes[name], f"N0 artifact {name}")
        receipts = tuple(self.installation_receipts)
        if tuple(item.kind for item in receipts) != _INSTALLATION_KINDS:
            raise ValueError("N0 server installation receipt inventory mismatch")
        if any(item.content_sha256 is not None for item in receipts):
            raise ValueError("installation receipt content SHA256 must be null")
        if (
            not isinstance(self.recorded_at_utc, str)
            or not self.recorded_at_utc
            or self.evidence_level != N0_SERVER_ATTESTATION_EVIDENCE_LEVEL
            or self.semantic_version != N0_SERVER_ATTESTATION_SEMANTIC_VERSION
        ):
            raise ValueError("N0 server attestation version/timestamp mismatch")
        object.__setattr__(self, "n0_artifact_hashes", hashes)
        object.__setattr__(self, "installation_receipts", receipts)
        if self.content_sha256 != canonical_hash(self._content_dict()):
            raise ValueError("N0 server attestation content hash mismatch")

    def _content_dict(self) -> dict[str, object]:
        return {
            name: self._field_value(name)
            for name in self.__dataclass_fields__
            if name != "content_sha256"
        }

    def _field_value(self, name: str) -> object:
        value = getattr(self, name)
        if name == "task_source_binding":
            return self.task_source_binding.to_dict()
        if name == "installation_receipts":
            return [item.to_dict() for item in self.installation_receipts]
        if name == "n0_artifact_hashes":
            return dict(self.n0_artifact_hashes)
        return value

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def build(cls, **values: Any) -> "N0ServerRuntimeAttestation":
        content = {
            name: (
                values[name].to_dict()
                if name == "task_source_binding"
                else [item.to_dict() for item in values[name]]
                if name == "installation_receipts"
                else dict(values[name])
                if name == "n0_artifact_hashes"
                else values[name]
            )
            for name in cls.__dataclass_fields__
            if name != "content_sha256"
        }
        return cls(**values, content_sha256=canonical_hash(content))

    @classmethod
    def from_dict(cls, value: object) -> "N0ServerRuntimeAttestation":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("N0 server attestation fields mismatch")
        document = cast(Mapping[str, object], value)
        hashes = document["n0_artifact_hashes"]
        receipts = document["installation_receipts"]
        if not isinstance(hashes, Mapping):
            raise TypeError("n0_artifact_hashes must be an object")
        if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
            raise TypeError("installation_receipts must be a sequence")
        parsed = dict(document)
        parsed["task_source_binding"] = RuntimeSourceBinding.from_dict(
            document["task_source_binding"]
        )
        parsed["n0_artifact_hashes"] = dict(cast(Mapping[str, str], hashes))
        parsed["installation_receipts"] = tuple(
            ObservationEvidenceBinding.from_dict(item) for item in receipts
        )
        return cls(**cast(dict[str, Any], parsed))


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
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain below deployment root") from error
    return _relative(relative.as_posix(), name)


def _git_identity(source_root: Path) -> tuple[str, bool]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(source_root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError("N0 source checkout identity is unavailable")
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("N0 source checkout commit is invalid")
    clean = run("status", "--porcelain", "--untracked-files=all") == ""
    return commit, clean


def _file_sha256(path: Path, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    return n0_server_attestation_sha256(path)


def _qualification_receipts(path: Path) -> Tuple[ObservationEvidenceBinding, ...]:
    from robotactile_benchmark.clean_baseline.io import read_canonical_json_file

    document, _ = read_canonical_json_file(path, "all-task qualification")
    value = document.get("installation_receipts")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("qualification installation receipts are unavailable")
    receipts = tuple(ObservationEvidenceBinding.from_dict(item) for item in value)
    if tuple(item.kind for item in receipts) != _INSTALLATION_KINDS:
        raise ValueError("qualification installation receipt inventory mismatch")
    return receipts


def build_n0_server_runtime_attestation(
    *,
    deployment_root: Path,
    task_id: str,
    session_id: str,
    output_path: Path,
    qualification_path: Path,
    integration_config_path: Path,
    n0_source_root: Path,
    loaded_n0_module_path: Path,
    rank: int,
    world_size: int,
    debug_offload: bool,
) -> N0ServerRuntimeAttestation:
    """Verify the active rank-zero runtime and publish its canonical evidence."""

    from robotactile_benchmark.clean_baseline.io import (
        write_canonical_no_clobber,
    )
    from robotactile_benchmark.clean_baseline.qualification import (
        QUALIFICATION_V3_SEMANTIC_VERSION,
        verify_all_task_qualification,
    )

    root = Path(deployment_root).resolve(strict=True)
    task_id = _identifier(task_id, "task_id")
    session_id = _identifier(session_id, "session_id")
    if rank != 0 or debug_offload:
        raise ValueError("source-bound N0 attestation requires rank0 fast mode")
    qualification_target = _member(
        root,
        _relative_to_root(root, qualification_path, "qualification path"),
        "qualification path",
    )
    qualification = verify_all_task_qualification(root, qualification_target)
    if qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION:
        raise ValueError("source-bound N0 server requires qualification v3")
    try:
        task_index = qualification.tasks.index(task_id)
    except ValueError as error:
        raise ValueError("qualification does not cover the served task") from error
    task_source = qualification.task_source_bindings[task_index]
    task_source.verify_against_current_runtime()

    config_relative = _relative_to_root(
        root, integration_config_path, "integration config path"
    )
    config_target = _member(root, config_relative, "integration config path")
    runtime = resolve_n0_runtime_artifacts(config_target)
    manifest = runtime.manifest
    if manifest.task_id != task_id:
        raise ValueError("N0 artifact manifest task mismatch")
    expected_environment = {
        "TWAM_SERVE_ACTION_MODE": manifest.action_mode,
        "TWAM_SERVE_BUNDLE": str(manifest.serve_bundle_root),
        "TWAM_SERVE_POOL": str(manifest.serve_pool_root),
        "TWAM_SERVE_TASK": manifest.serve_task_id,
    }
    if any(
        os.environ.get(name) != value for name, value in expected_environment.items()
    ):
        raise ValueError(
            "active N0 serving environment disagrees with artifact manifest"
        )
    artifact_hashes = {
        name: cast(str, getattr(manifest, name)) for name in _ARTIFACT_HASH_FIELDS
    }
    if any(
        artifact_hashes[name] != getattr(task_source, name)
        for name in _ARTIFACT_HASH_FIELDS
    ):
        raise ValueError("N0 artifact manifest disagrees with task source binding")

    source_relative = _relative_to_root(root, n0_source_root, "N0 source path")
    source_target = _member(root, source_relative, "N0 source path")
    if source_target.is_symlink() or not source_target.is_dir():
        raise ValueError("N0 source must be a non-symlink directory")
    loaded_module = Path(loaded_n0_module_path).resolve(strict=True)
    try:
        loaded_module.relative_to(source_target.resolve(strict=True))
    except ValueError as error:
        raise ValueError("loaded N0 module is outside the declared checkout") from error
    commit, clean = _git_identity(source_target)
    if not clean or commit != task_source.n0_source_commit:
        raise ValueError("active N0 checkout does not match the qualified source")
    lock = load_integration_lock()
    if commit != lock.by_id("n0_twam").commit_sha or commit != manifest.external_commit:
        raise ValueError("active N0 checkout disagrees with pinned artifact source")

    artifact_relative = _relative_to_root(
        root, runtime.manifest_path, "N0 artifact manifest path"
    )
    receipts = _qualification_receipts(qualification_target)
    attestation = N0ServerRuntimeAttestation.build(
        task_id=task_id,
        session_id=session_id,
        server_pid=os.getpid(),
        server_process_group_id=os.getpgid(0),
        server_posix_session_id=os.getsid(0),
        rank=rank,
        world_size=world_size,
        debug_offload=False,
        paper_gate_passed=True,
        n0_source_relpath=source_relative,
        n0_source_commit=commit,
        n0_source_clean=True,
        robotactile_source_manifest_sha256=current_source_manifest_sha256(),
        integrations_lock_sha256=current_integrations_lock_sha256(),
        qualification_relpath=_relative_to_root(
            root, qualification_target, "qualification path"
        ),
        qualification_sha256=qualification.sha256,
        qualification_semantic_version=qualification.semantic_version,
        task_source_binding=task_source,
        integration_config_relpath=config_relative,
        integration_config_sha256=_file_sha256(config_target, "integration config"),
        artifact_manifest_relpath=artifact_relative,
        artifact_manifest_sha256=_file_sha256(
            runtime.manifest_path, "N0 artifact manifest"
        ),
        n0_artifact_hashes=artifact_hashes,
        installation_receipts=receipts,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_level=N0_SERVER_ATTESTATION_EVIDENCE_LEVEL,
        semantic_version=N0_SERVER_ATTESTATION_SEMANTIC_VERSION,
    )
    target = Path(output_path).absolute()
    output_relative = _relative_to_root(root, target, "N0 attestation output")
    if PurePosixPath(output_relative).parts[0] != "outputs":
        raise ValueError("N0 attestation output must be below outputs/")
    write_canonical_no_clobber(target, attestation.to_dict())
    return attestation


def load_n0_server_runtime_attestation(
    deployment_root: Path,
    path: Path,
    *,
    expected_sha256: str | None = None,
    verify_members: bool = True,
) -> N0ServerRuntimeAttestation:
    """Load canonical server evidence and optionally re-verify every member."""

    from robotactile_benchmark.clean_baseline.io import read_canonical_json_file

    root = Path(deployment_root).resolve(strict=True)
    target = _member(
        root,
        _relative_to_root(root, path, "N0 server attestation path"),
        "N0 server attestation path",
    )
    actual_sha256 = n0_server_attestation_sha256(target)
    if expected_sha256 is not None and actual_sha256 != _sha256(
        expected_sha256, "expected N0 server attestation SHA256"
    ):
        raise ValueError("N0 server attestation SHA256 mismatch")
    document, _ = read_canonical_json_file(target, "N0 server attestation")
    attestation = N0ServerRuntimeAttestation.from_dict(document)
    if verify_members:
        _verify_attestation_members(root, attestation)
    return attestation


def _verify_attestation_members(
    root: Path, attestation: N0ServerRuntimeAttestation
) -> None:
    from robotactile_benchmark.clean_baseline.qualification import (
        QUALIFICATION_V3_SEMANTIC_VERSION,
        verify_all_task_qualification,
    )

    qualification_path = _member(
        root, attestation.qualification_relpath, "qualification path"
    )
    qualification = verify_all_task_qualification(root, qualification_path)
    if (
        qualification.sha256 != attestation.qualification_sha256
        or qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION
    ):
        raise ValueError("attested qualification identity mismatch")
    task_index = qualification.tasks.index(attestation.task_id)
    if (
        qualification.task_source_bindings[task_index]
        != attestation.task_source_binding
    ):
        raise ValueError("attested qualification task binding mismatch")
    attestation.task_source_binding.verify_against_current_runtime()
    if (
        attestation.robotactile_source_manifest_sha256
        != current_source_manifest_sha256()
        or attestation.integrations_lock_sha256 != current_integrations_lock_sha256()
    ):
        raise ValueError("attested RoboTactile runtime identity mismatch")
    if attestation.installation_receipts != _qualification_receipts(qualification_path):
        raise ValueError("attested qualification installation bindings mismatch")

    config_path = _member(
        root, attestation.integration_config_relpath, "integration config path"
    )
    if _file_sha256(config_path, "integration config") != (
        attestation.integration_config_sha256
    ):
        raise ValueError("attested integration config hash mismatch")
    config = load_model_integration_config("n0_twam", config_path)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = config_path.parent / manifest_path
    expected_manifest = _member(
        root, attestation.artifact_manifest_relpath, "artifact manifest path"
    )
    if manifest_path.absolute() != expected_manifest.absolute():
        raise ValueError("attested integration config artifact path mismatch")
    if _file_sha256(expected_manifest, "N0 artifact manifest") != (
        attestation.artifact_manifest_sha256
    ):
        raise ValueError("attested artifact manifest hash mismatch")
    manifest = load_n0_twam_artifact_manifest(expected_manifest)
    validate_n0_twam_artifact(manifest)
    if manifest.task_id != attestation.task_id or any(
        getattr(manifest, name) != attestation.n0_artifact_hashes[name]
        for name in _ARTIFACT_HASH_FIELDS
    ):
        raise ValueError("attested N0 artifact identity mismatch")
    source = _member(root, attestation.n0_source_relpath, "N0 source path")
    commit, clean = _git_identity(source)
    if not clean or commit != attestation.n0_source_commit:
        raise ValueError("attested N0 checkout changed after server startup")
    for receipt in attestation.installation_receipts:
        receipt_path = _member(root, receipt.relpath, f"{receipt.kind} path")
        if _file_sha256(receipt_path, receipt.kind) != receipt.sha256:
            raise ValueError(f"attested installation receipt changed: {receipt.kind}")


def verify_live_n0_server_attestation(
    attestation: N0ServerRuntimeAttestation,
    *,
    expected_task_id: str,
    expected_session_id: str,
    expected_process_group_id: int,
) -> None:
    """Bind an attestation to the server process group owned by the shard."""

    if (
        attestation.task_id != expected_task_id
        or attestation.session_id != expected_session_id
        or attestation.server_process_group_id != expected_process_group_id
        or attestation.server_posix_session_id != expected_process_group_id
    ):
        raise ValueError("N0 server attestation process/session identity mismatch")
    try:
        actual_group = os.getpgid(attestation.server_pid)
        actual_session = os.getsid(attestation.server_pid)
    except ProcessLookupError as error:
        raise ValueError("attested N0 rank-zero process is not running") from error
    if actual_group != expected_process_group_id or actual_session != (
        expected_process_group_id
    ):
        raise ValueError("attested N0 rank-zero process escaped the owned group")


__all__ = [
    "N0_SERVER_ATTESTATION_EVIDENCE_LEVEL",
    "N0_SERVER_ATTESTATION_SEMANTIC_VERSION",
    "N0ServerRuntimeAttestation",
    "build_n0_server_runtime_attestation",
    "load_n0_server_runtime_attestation",
    "n0_server_attestation_sha256",
    "verify_live_n0_server_attestation",
]
