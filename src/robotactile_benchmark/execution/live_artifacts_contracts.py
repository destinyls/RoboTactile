"""Typed contracts and independent limits for live UniVTAC trace artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Tuple

from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.execution.live_univtac import (
    LIVE_ARTIFACT_EVIDENCE_LEVEL,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.trials import TrialManifest

LIVE_ARTIFACT_SEMANTIC_VERSION = "1.0"
LIVE_ROOT_RECEIPT_PATH = "root_receipt.json"
LIVE_MAX_MEMBER_BYTES = 1024 * 1024 * 1024
LIVE_MAX_JSON_BYTES = 128 * 1024 * 1024
LIVE_MAX_BUNDLE_BYTES = 32 * 1024 * 1024 * 1024
LIVE_MAX_MEMBER_COUNT = 100_000
LIVE_MAX_TRACE_RECORDS = 4096
LIVE_JSON_RESERVE_BYTES = 256 * 1024 * 1024
LIVE_REQUIRED_JSON_MEMBERS = frozenset(
    {
        "request_identity.json",
        "trial_manifest.json",
        "run_spec.json",
        "fault_manifest.json",
        "rest_references.json",
        "terminal_result.json",
        "action_trace.json",
        "delivery_trace.json",
        "validation_report.json",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class LiveArtifactValidationError(ValueError):
    """Raised when a live trace byte or semantic link fails closed."""


def require_live_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LiveArtifactValidationError(f"{name} must be a lowercase SHA256")
    return value


def require_optional_live_sha256(value: object, name: str) -> str | None:
    return None if value is None else require_live_sha256(value, name)


def validate_live_member_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise LiveArtifactValidationError("live artifact path is not normalized POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise LiveArtifactValidationError("live artifact path escapes the bundle")
    if any(part in {"", "."} for part in path.parts):
        raise LiveArtifactValidationError("live artifact path is not normalized")
    return value


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LiveArtifactValidationError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class LiveArtifactMember:
    """One non-root live member pinned by normalized path, bytes, and size."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", validate_live_member_path(self.path))
        object.__setattr__(self, "sha256", require_live_sha256(self.sha256, "member"))
        size = _strict_int(self.size_bytes, "member size", 1)
        if size > LIVE_MAX_MEMBER_BYTES:
            raise LiveArtifactValidationError("live artifact member exceeds its cap")
        object.__setattr__(self, "size_bytes", size)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> LiveArtifactMember:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise LiveArtifactValidationError("live artifact member fields mismatch")
        return cls(value["path"], value["sha256"], value["size_bytes"])


@dataclass(frozen=True)
class LiveArtifactRootReceipt:
    """Non-recursive live receipt whose file SHA is the external root hash."""

    request_sha256: str
    run_content_sha256: str
    source_binding_sha256: str
    trial_manifest_sha256: str
    pair_key: str
    run_spec_sha256: str
    fault_manifest_sha256: str | None
    rest_references_sha256: str | None
    result_sha256: str
    terminal_trace_sha256: str
    action_trace_sha256: str
    clean_trace_sha256: str
    delivered_trace_sha256: str
    members: Tuple[LiveArtifactMember, ...]
    evidence_level: str = LIVE_ARTIFACT_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = LIVE_ARTIFACT_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if self.evidence_level != LIVE_ARTIFACT_EVIDENCE_LEVEL:
            raise LiveArtifactValidationError("live evidence level mismatch")
        if self.simulator_qualification_claimed is not False:
            raise LiveArtifactValidationError("live trace cannot claim qualification")
        if self.semantic_version != LIVE_ARTIFACT_SEMANTIC_VERSION:
            raise LiveArtifactValidationError("live artifact version mismatch")
        for name in (
            "request_sha256",
            "run_content_sha256",
            "source_binding_sha256",
            "trial_manifest_sha256",
            "pair_key",
            "run_spec_sha256",
            "result_sha256",
            "terminal_trace_sha256",
            "action_trace_sha256",
            "clean_trace_sha256",
            "delivered_trace_sha256",
        ):
            object.__setattr__(
                self, name, require_live_sha256(getattr(self, name), name)
            )
        for name in ("fault_manifest_sha256", "rest_references_sha256"):
            object.__setattr__(
                self, name, require_optional_live_sha256(getattr(self, name), name)
            )
        members = tuple(self.members)
        paths = tuple(item.path for item in members)
        if not members or len(members) > LIVE_MAX_MEMBER_COUNT:
            raise LiveArtifactValidationError("live member count is outside bounds")
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise LiveArtifactValidationError(
                "live member paths must be unique and sorted"
            )
        if LIVE_ROOT_RECEIPT_PATH in paths:
            raise LiveArtifactValidationError("root receipt cannot enumerate itself")
        if {path for path in paths if not path.startswith("arrays/")} != (
            LIVE_REQUIRED_JSON_MEMBERS
        ):
            raise LiveArtifactValidationError("live JSON member inventory mismatch")
        if sum(item.size_bytes for item in members) > LIVE_MAX_BUNDLE_BYTES:
            raise LiveArtifactValidationError("live artifact exceeds total byte cap")
        object.__setattr__(self, "members", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "request_sha256": self.request_sha256,
            "run_content_sha256": self.run_content_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "pair_key": self.pair_key,
            "run_spec_sha256": self.run_spec_sha256,
            "fault_manifest_sha256": self.fault_manifest_sha256,
            "rest_references_sha256": self.rest_references_sha256,
            "result_sha256": self.result_sha256,
            "terminal_trace_sha256": self.terminal_trace_sha256,
            "action_trace_sha256": self.action_trace_sha256,
            "clean_trace_sha256": self.clean_trace_sha256,
            "delivered_trace_sha256": self.delivered_trace_sha256,
            "members": [item.to_dict() for item in self.members],
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> LiveArtifactRootReceipt:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != fields:
            raise LiveArtifactValidationError("live root receipt fields mismatch")
        members = value["members"]
        if not isinstance(members, list):
            raise LiveArtifactValidationError("live root members must be a list")
        kwargs = dict(value)
        kwargs["members"] = tuple(
            LiveArtifactMember.from_dict(item) for item in members
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class LoadedLiveUniVTACArtifact:
    """Fully reconstructed and independently revalidated live trace bundle."""

    request_identity: Mapping[str, object]
    run_content_sha256: str
    trial: TrialManifest
    run_spec: ClosedLoopRunSpec
    fault_manifest: FaultManifest | None
    rest_references: RestReferenceBundle | None
    evidence: ClosedLoopExecutionEvidence
    root_receipt: LiveArtifactRootReceipt
    root_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_identity", freeze_value(self.request_identity)
        )
        object.__setattr__(
            self,
            "run_content_sha256",
            require_live_sha256(self.run_content_sha256, "run content"),
        )
        object.__setattr__(
            self,
            "root_receipt_sha256",
            require_live_sha256(self.root_receipt_sha256, "external root"),
        )

    @property
    def external_root_sha256(self) -> str:
        """Return the pin callers should record outside the artifact directory."""

        return self.root_receipt_sha256
