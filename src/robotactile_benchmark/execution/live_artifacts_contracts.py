"""Typed contracts and independent limits for live UniVTAC trace artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from robotactile_benchmark.artifacts.primitives import (
    require_lowercase_sha256,
    validate_posix_member_path,
)
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_univtac import (
    LIVE_ARTIFACT_EVIDENCE_LEVEL,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.trials import TrialManifest

if TYPE_CHECKING:
    from robotactile_benchmark.execution.live_artifacts_preview import (
        LivePreviewTrace,
    )

LIVE_ARTIFACT_SEMANTIC_VERSION = "1.0"
LIVE_ROOT_RECEIPT_SEMANTIC_VERSION = "1.1"
LIVE_CAPTURE_ROOT_RECEIPT_SEMANTIC_VERSION = "1.2"
LIVE_SUPPORTED_ROOT_RECEIPT_VERSIONS = frozenset({"1.0", "1.1", "1.2"})
LIVE_COMPACT_ARTIFACT_EVIDENCE_LEVEL = "live_univtac_diagnostic_capture_v1"
LIVE_ROOT_RECEIPT_PATH = "root_receipt.json"
LIVE_MAX_MEMBER_BYTES = 1024 * 1024 * 1024
LIVE_MAX_JSON_BYTES = 128 * 1024 * 1024
LIVE_MAX_BUNDLE_BYTES = 32 * 1024 * 1024 * 1024
LIVE_MAX_MEMBER_COUNT = 100_000
LIVE_MAX_TRACE_RECORDS = 4096
LIVE_JSON_RESERVE_BYTES = 256 * 1024 * 1024
LIVE_REQUIRED_JSON_MEMBERS_V1_0 = frozenset(
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
LIVE_REQUIRED_JSON_MEMBERS = LIVE_REQUIRED_JSON_MEMBERS_V1_0 | frozenset(
    {"transition_trace.json"}
)
LIVE_CAPTURE_REQUIRED_JSON_MEMBERS = LIVE_REQUIRED_JSON_MEMBERS | frozenset(
    {"capture_summary.json"}
)


class LiveArtifactValidationError(ValueError):
    """Raised when a live trace byte or semantic link fails closed."""


def require_live_sha256(value: object, name: str) -> str:
    return require_lowercase_sha256(value, name, LiveArtifactValidationError)


def require_optional_live_sha256(value: object, name: str) -> str | None:
    return None if value is None else require_live_sha256(value, name)


def validate_live_member_path(value: object) -> str:
    return validate_posix_member_path(
        value,
        label="live artifact path",
        error_type=LiveArtifactValidationError,
    )


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
    semantic_version: str = LIVE_ROOT_RECEIPT_SEMANTIC_VERSION
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL

    def __post_init__(self) -> None:
        try:
            capture_profile = LiveCaptureProfile(self.capture_profile)
        except (TypeError, ValueError) as error:
            raise LiveArtifactValidationError(
                "live capture profile mismatch"
            ) from error
        object.__setattr__(self, "capture_profile", capture_profile)
        expected_evidence_level = (
            LIVE_ARTIFACT_EVIDENCE_LEVEL
            if capture_profile.is_full_trace
            else LIVE_COMPACT_ARTIFACT_EVIDENCE_LEVEL
        )
        if self.evidence_level != expected_evidence_level:
            raise LiveArtifactValidationError("live evidence level mismatch")
        if self.simulator_qualification_claimed is not False:
            raise LiveArtifactValidationError("live trace cannot claim qualification")
        if self.semantic_version not in LIVE_SUPPORTED_ROOT_RECEIPT_VERSIONS:
            raise LiveArtifactValidationError("live artifact version mismatch")
        if (
            self.semantic_version in {"1.0", "1.1"}
            and not capture_profile.is_full_trace
        ):
            raise LiveArtifactValidationError(
                "legacy live artifacts must be full trace"
            )
        if self.semantic_version == "1.2" and capture_profile.is_full_trace:
            raise LiveArtifactValidationError("v1.2 is reserved for compact captures")
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
        if self.semantic_version == "1.0":
            required_members = LIVE_REQUIRED_JSON_MEMBERS_V1_0
        elif self.semantic_version == "1.1":
            required_members = LIVE_REQUIRED_JSON_MEMBERS
        elif capture_profile is LiveCaptureProfile.METRICS_ONLY:
            required_members = LIVE_CAPTURE_REQUIRED_JSON_MEMBERS
        else:
            required_members = LIVE_CAPTURE_REQUIRED_JSON_MEMBERS | frozenset(
                {"preview_trace.json"}
            )
        if {
            path for path in paths if not path.startswith("arrays/")
        } != required_members:
            raise LiveArtifactValidationError("live JSON member inventory mismatch")
        if sum(item.size_bytes for item in members) > LIVE_MAX_BUNDLE_BYTES:
            raise LiveArtifactValidationError("live artifact exceeds total byte cap")
        object.__setattr__(self, "members", members)

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
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
        if self.semantic_version == LIVE_CAPTURE_ROOT_RECEIPT_SEMANTIC_VERSION:
            document["capture_profile"] = self.capture_profile.value
        return document

    @classmethod
    def from_dict(cls, value: object) -> LiveArtifactRootReceipt:
        if not isinstance(value, dict):
            raise LiveArtifactValidationError("live root receipt fields mismatch")
        legacy_fields = set(cls.__dataclass_fields__) - {"capture_profile"}
        semantic_version = value.get("semantic_version")
        expected_fields = (
            set(cls.__dataclass_fields__)
            if semantic_version == LIVE_CAPTURE_ROOT_RECEIPT_SEMANTIC_VERSION
            else legacy_fields
        )
        if set(value) != expected_fields:
            raise LiveArtifactValidationError("live root receipt fields mismatch")
        members = value["members"]
        if not isinstance(members, list):
            raise LiveArtifactValidationError("live root members must be a list")
        kwargs = dict(value)
        if "capture_profile" not in kwargs:
            kwargs["capture_profile"] = LiveCaptureProfile.PAPER_FULL
        kwargs["members"] = tuple(
            LiveArtifactMember.from_dict(item) for item in members
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class LoadedLiveUniVTACArtifact:
    """Independently revalidated full or diagnostic live capture bundle."""

    request_identity: Mapping[str, object]
    run_content_sha256: str
    trial: TrialManifest
    run_spec: ClosedLoopRunSpec
    fault_manifest: FaultManifest | None
    rest_references: RestReferenceBundle | None
    evidence: ClosedLoopExecutionEvidence
    root_receipt: LiveArtifactRootReceipt
    root_receipt_sha256: str
    preview_trace: LivePreviewTrace | None = None

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
        if (
            not self.root_receipt.capture_profile.is_full_trace
            and self.evidence.finalization is not None
        ):
            raise LiveArtifactValidationError(
                "compact capture cannot expose a full finalization"
            )
        has_preview = self.preview_trace is not None
        if has_preview != (
            self.root_receipt.capture_profile is LiveCaptureProfile.PREVIEW
        ):
            raise LiveArtifactValidationError("loaded preview/profile mismatch")

    @property
    def external_root_sha256(self) -> str:
        """Return the pin callers should record outside the artifact directory."""

        return self.root_receipt_sha256

    @property
    def capture_profile(self) -> LiveCaptureProfile:
        """Return the explicit persisted-evidence profile."""

        return self.root_receipt.capture_profile

    @property
    def has_full_trace(self) -> bool:
        """Return whether all observation records are present."""

        return self.capture_profile.is_full_trace
