"""Typed receipts for one bounded, content-addressed closed-loop bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from robotactile_benchmark.artifacts.primitives import (
    require_lowercase_sha256,
    validate_posix_member_path,
)
from robotactile_benchmark.closed_loop.capture import ActionTraceEntry
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.trials import TrialManifest

BUNDLE_SEMANTIC_VERSION = "1.0"
EVIDENCE_LEVEL = "deterministic_cpu_fake_closed_loop"
ROOT_RECEIPT_PATH = "root_receipt.json"
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_COUNT = 512
REQUIRED_JSON_MEMBERS = frozenset(
    {
        "trial_manifest.json",
        "run_spec.json",
        "fault_manifest.json",
        "terminal_result.json",
        "action_trace.json",
        "delivery_trace.json",
        "validation_report.json",
    }
)


class ArtifactValidationError(ValueError):
    """Raised when any bundle byte or semantic cross-link fails closed."""


def require_sha256(value: object, name: str) -> str:
    return require_lowercase_sha256(value, name, ArtifactValidationError)


def validate_member_path(value: object) -> str:
    return validate_posix_member_path(
        value,
        label="artifact member path",
        error_type=ArtifactValidationError,
    )


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError(f"{name} must be an integer")
    if value < minimum:
        raise ArtifactValidationError(f"{name} is below its lower bound")
    return value


@dataclass(frozen=True)
class ArtifactMember:
    """One non-root bundle member pinned by path, bytes, and bounded size."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", validate_member_path(self.path))
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "member sha256"))
        size = _strict_int(self.size_bytes, "member size", minimum=1)
        if size > MAX_MEMBER_BYTES:
            raise ArtifactValidationError("artifact member exceeds the size cap")
        object.__setattr__(self, "size_bytes", size)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactMember:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ArtifactValidationError("artifact member fields mismatch")
        return cls(
            path=value["path"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
        )


@dataclass(frozen=True)
class RootReceipt:
    """Non-recursive root receipt; its file SHA is the externally pinned root."""

    evidence_level: str
    trial_manifest_sha256: str
    pair_key: str
    run_spec_sha256: str
    fault_manifest_sha256: str
    result_sha256: str
    terminal_trace_sha256: str
    action_trace_sha256: str
    clean_trace_sha256: str
    delivered_trace_sha256: str
    members: Tuple[ArtifactMember, ...]
    semantic_version: str = BUNDLE_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if self.evidence_level != EVIDENCE_LEVEL:
            raise ArtifactValidationError("unsupported closed-loop evidence level")
        if self.semantic_version != BUNDLE_SEMANTIC_VERSION:
            raise ArtifactValidationError("unsupported bundle semantic version")
        for name in (
            "trial_manifest_sha256",
            "pair_key",
            "run_spec_sha256",
            "fault_manifest_sha256",
            "result_sha256",
            "terminal_trace_sha256",
            "action_trace_sha256",
            "clean_trace_sha256",
            "delivered_trace_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        members = tuple(self.members)
        if not members or len(members) > MAX_MEMBER_COUNT:
            raise ArtifactValidationError("artifact member count is outside bounds")
        paths = tuple(member.path for member in members)
        if len(set(paths)) != len(paths) or tuple(sorted(paths)) != paths:
            raise ArtifactValidationError(
                "artifact member paths must be unique and sorted"
            )
        if ROOT_RECEIPT_PATH in paths:
            raise ArtifactValidationError(
                "root receipt cannot recursively enumerate itself"
            )
        if {
            path for path in paths if not path.startswith("arrays/")
        } != REQUIRED_JSON_MEMBERS:
            raise ArtifactValidationError("required JSON artifact inventory mismatch")
        if sum(member.size_bytes for member in members) > MAX_BUNDLE_BYTES:
            raise ArtifactValidationError("artifact bundle exceeds the size cap")
        object.__setattr__(self, "members", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_level": self.evidence_level,
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "pair_key": self.pair_key,
            "run_spec_sha256": self.run_spec_sha256,
            "fault_manifest_sha256": self.fault_manifest_sha256,
            "result_sha256": self.result_sha256,
            "terminal_trace_sha256": self.terminal_trace_sha256,
            "action_trace_sha256": self.action_trace_sha256,
            "clean_trace_sha256": self.clean_trace_sha256,
            "delivered_trace_sha256": self.delivered_trace_sha256,
            "members": [member.to_dict() for member in self.members],
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RootReceipt:
        fields = {
            "evidence_level",
            "trial_manifest_sha256",
            "pair_key",
            "run_spec_sha256",
            "fault_manifest_sha256",
            "result_sha256",
            "terminal_trace_sha256",
            "action_trace_sha256",
            "clean_trace_sha256",
            "delivered_trace_sha256",
            "members",
            "semantic_version",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ArtifactValidationError("root receipt fields mismatch")
        members = value["members"]
        if not isinstance(members, list):
            raise ArtifactValidationError("root members must be a list")
        return cls(
            evidence_level=value["evidence_level"],
            trial_manifest_sha256=value["trial_manifest_sha256"],
            pair_key=value["pair_key"],
            run_spec_sha256=value["run_spec_sha256"],
            fault_manifest_sha256=value["fault_manifest_sha256"],
            result_sha256=value["result_sha256"],
            terminal_trace_sha256=value["terminal_trace_sha256"],
            action_trace_sha256=value["action_trace_sha256"],
            clean_trace_sha256=value["clean_trace_sha256"],
            delivered_trace_sha256=value["delivered_trace_sha256"],
            members=tuple(ArtifactMember.from_dict(item) for item in members),
            semantic_version=value["semantic_version"],
        )


@dataclass(frozen=True)
class LoadedClosedLoopBundle:
    """Typed, fully revalidated view of one single-trial artifact directory."""

    trial: TrialManifest
    run_spec: ClosedLoopRunSpec
    fault_manifest: FaultManifest
    result: ClosedLoopTrialResult
    finalization: DeliveryFinalization
    action_entries: Tuple[ActionTraceEntry, ...]
    root_receipt: RootReceipt
    root_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_receipt_sha256",
            require_sha256(self.root_receipt_sha256, "root receipt file sha256"),
        )
        object.__setattr__(self, "action_entries", tuple(self.action_entries))
