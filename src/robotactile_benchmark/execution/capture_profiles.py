"""Storage profiles for live UniVTAC execution evidence."""

from __future__ import annotations

from enum import Enum

from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence


class LiveCaptureProfile(str, Enum):
    """Select how much post-run evidence is persisted."""

    METRICS_ONLY = "metrics_only_v1"
    PREVIEW = "preview_v1"
    PAPER_FULL = "paper_full_v1"

    @property
    def is_full_trace(self) -> bool:
        """Return whether the profile preserves every observation record."""

        return self is LiveCaptureProfile.PAPER_FULL


DEFAULT_PREVIEW_RECORD_LIMIT = 64


def project_evidence_for_capture(
    evidence: ClosedLoopExecutionEvidence,
    profile: LiveCaptureProfile,
) -> ClosedLoopExecutionEvidence:
    """Drop full observations while retaining the metric and action hash domain."""

    if not isinstance(evidence, ClosedLoopExecutionEvidence):
        raise TypeError("evidence must be ClosedLoopExecutionEvidence")
    selected = LiveCaptureProfile(profile)
    if selected.is_full_trace:
        return evidence
    return ClosedLoopExecutionEvidence(
        result=evidence.result,
        finalization=None,
        action_entries=evidence.action_entries,
        transition_entries=evidence.transition_entries,
        initial_diagnostics=evidence.initial_diagnostics,
    )


__all__ = [
    "DEFAULT_PREVIEW_RECORD_LIMIT",
    "LiveCaptureProfile",
    "project_evidence_for_capture",
]
