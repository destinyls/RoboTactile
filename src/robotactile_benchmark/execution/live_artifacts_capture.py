"""Compact-capture support shared by the live artifact writer and loader."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from robotactile_benchmark.contracts import Array
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_CAPTURE_ROOT_RECEIPT_SEMANTIC_VERSION,
    LIVE_COMPACT_ARTIFACT_EVIDENCE_LEVEL,
    LiveArtifactMember,
    LiveArtifactRootReceipt,
    LiveArtifactValidationError,
)
from robotactile_benchmark.execution.live_artifacts_fs import LiveBundleSnapshot
from robotactile_benchmark.execution.live_artifacts_preview import (
    LivePreviewTrace,
    iter_preview_arrays,
    preview_trace_from_live_dict,
)


def iter_capture_arrays(
    base_arrays: Iterable[Array], preview: LivePreviewTrace | None
) -> Iterable[Array]:
    """Yield base arrays and the explicitly selected preview subset."""

    yield from base_arrays
    if preview is not None:
        yield from iter_preview_arrays(preview)


def live_capture_summary(
    profile: LiveCaptureProfile,
    preview: LivePreviewTrace | None,
    *,
    clean_record_count: int,
    delivered_record_count: int,
) -> dict[str, object]:
    """Describe exactly what a compact artifact omitted and retained."""

    return {
        "capture_profile": profile.value,
        "full_observation_trace_included": False,
        "preview_record_count": (
            0 if preview is None else len(preview.selected_indices)
        ),
        "total_clean_record_count": clean_record_count,
        "total_delivered_record_count": delivered_record_count,
        "semantic_version": "1.0",
    }


def load_live_capture_members(
    receipt: LiveArtifactRootReceipt,
    documents: Mapping[str, object],
    snapshot: LiveBundleSnapshot,
    referenced_arrays: set[str],
) -> LivePreviewTrace | None:
    """Validate compact metadata and optionally reconstruct its preview."""

    summary = documents["capture_summary.json"]
    fields = {
        "capture_profile",
        "full_observation_trace_included",
        "preview_record_count",
        "total_clean_record_count",
        "total_delivered_record_count",
        "semantic_version",
    }
    if not isinstance(summary, dict) or set(summary) != fields:
        raise LiveArtifactValidationError("capture summary fields mismatch")
    if (
        summary["capture_profile"] != receipt.capture_profile.value
        or summary["full_observation_trace_included"] is not False
        or summary["semantic_version"] != "1.0"
    ):
        raise LiveArtifactValidationError("capture summary contract mismatch")
    for name in (
        "preview_record_count",
        "total_clean_record_count",
        "total_delivered_record_count",
    ):
        value = summary[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LiveArtifactValidationError("capture summary count is invalid")
    if receipt.capture_profile is LiveCaptureProfile.METRICS_ONLY:
        if summary["preview_record_count"] != 0:
            raise LiveArtifactValidationError(
                "metrics-only capture cannot carry preview"
            )
        return None
    if receipt.capture_profile is not LiveCaptureProfile.PREVIEW:
        raise LiveArtifactValidationError("unsupported compact capture profile")
    preview = preview_trace_from_live_dict(
        documents["preview_trace.json"], snapshot, referenced_arrays
    )
    if len(preview.selected_indices) != summary["preview_record_count"]:
        raise LiveArtifactValidationError(
            "preview count disagrees with capture summary"
        )
    return preview


def build_live_capture_root_receipt(
    *,
    request_sha256: str,
    run_content_sha256: str,
    source_binding_sha256: str,
    trial_manifest_sha256: str,
    pair_key: str,
    run_spec_sha256: str,
    fault_manifest_sha256: str | None,
    rest_references_sha256: str | None,
    result_sha256: str,
    terminal_trace_sha256: str,
    action_trace_sha256: str,
    clean_trace_sha256: str,
    delivered_trace_sha256: str,
    members: tuple[LiveArtifactMember, ...],
    profile: LiveCaptureProfile,
) -> LiveArtifactRootReceipt:
    """Build a legacy-compatible full root or an explicit compact root."""

    extra: dict[str, object] = {}
    if not profile.is_full_trace:
        extra = {
            "capture_profile": profile,
            "evidence_level": LIVE_COMPACT_ARTIFACT_EVIDENCE_LEVEL,
            "semantic_version": LIVE_CAPTURE_ROOT_RECEIPT_SEMANTIC_VERSION,
        }
    if extra:
        return LiveArtifactRootReceipt(
            request_sha256=request_sha256,
            run_content_sha256=run_content_sha256,
            source_binding_sha256=source_binding_sha256,
            trial_manifest_sha256=trial_manifest_sha256,
            pair_key=pair_key,
            run_spec_sha256=run_spec_sha256,
            fault_manifest_sha256=fault_manifest_sha256,
            rest_references_sha256=rest_references_sha256,
            result_sha256=result_sha256,
            terminal_trace_sha256=terminal_trace_sha256,
            action_trace_sha256=action_trace_sha256,
            clean_trace_sha256=clean_trace_sha256,
            delivered_trace_sha256=delivered_trace_sha256,
            members=members,
            capture_profile=profile,
            evidence_level=LIVE_COMPACT_ARTIFACT_EVIDENCE_LEVEL,
            semantic_version=LIVE_CAPTURE_ROOT_RECEIPT_SEMANTIC_VERSION,
        )
    return LiveArtifactRootReceipt(
        request_sha256=request_sha256,
        run_content_sha256=run_content_sha256,
        source_binding_sha256=source_binding_sha256,
        trial_manifest_sha256=trial_manifest_sha256,
        pair_key=pair_key,
        run_spec_sha256=run_spec_sha256,
        fault_manifest_sha256=fault_manifest_sha256,
        rest_references_sha256=rest_references_sha256,
        result_sha256=result_sha256,
        terminal_trace_sha256=terminal_trace_sha256,
        action_trace_sha256=action_trace_sha256,
        clean_trace_sha256=clean_trace_sha256,
        delivered_trace_sha256=delivered_trace_sha256,
        members=members,
    )


__all__ = [
    "build_live_capture_root_receipt",
    "iter_capture_arrays",
    "live_capture_summary",
    "load_live_capture_members",
]
