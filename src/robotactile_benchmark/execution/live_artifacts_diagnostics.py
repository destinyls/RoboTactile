"""Versioned backend-transition diagnostics for live trace artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from robotactile_benchmark.closed_loop.capture import TransitionTraceEntry
from robotactile_benchmark.closed_loop.contracts import BackendSignal
from robotactile_benchmark.contracts import freeze_value, thaw_value
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_ARTIFACT_SEMANTIC_VERSION,
    LIVE_MAX_TRACE_RECORDS,
    LiveArtifactValidationError,
)


def transition_trace_to_live_dict(
    entries: tuple[TransitionTraceEntry, ...],
    initial_diagnostics: object,
) -> dict[str, object]:
    """Serialize evaluator-owned transition witnesses without array payloads."""

    if len(entries) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live transition trace exceeds record cap")
    return {
        "entries": [
            {
                "benchmark_step_index": entry.benchmark_step_index,
                "native_step_id": entry.native_step_id,
                "signal": entry.signal.value,
                "diagnostics": thaw_value(entry.diagnostics),
            }
            for entry in entries
        ],
        "initial_diagnostics": thaw_value(initial_diagnostics),
        "semantic_version": LIVE_ARTIFACT_SEMANTIC_VERSION,
    }


def transition_trace_from_live_dict(
    value: object,
) -> tuple[
    tuple[TransitionTraceEntry, ...],
    Mapping[str, object],
]:
    """Strictly reconstruct transition witnesses from one v1.1 member."""

    if not isinstance(value, dict) or set(value) != {
        "entries",
        "initial_diagnostics",
        "semantic_version",
    }:
        raise LiveArtifactValidationError("live transition trace fields mismatch")
    if value["semantic_version"] != LIVE_ARTIFACT_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live transition trace version mismatch")
    entries = value["entries"]
    initial_diagnostics = value["initial_diagnostics"]
    if not isinstance(initial_diagnostics, dict):
        raise LiveArtifactValidationError("initial diagnostics must be an object")
    if not isinstance(entries, list) or len(entries) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live transition entries are outside bounds")
    loaded = []
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "benchmark_step_index",
            "native_step_id",
            "signal",
            "diagnostics",
        }:
            raise LiveArtifactValidationError("live transition entry fields mismatch")
        diagnostics = item["diagnostics"]
        if not isinstance(diagnostics, dict):
            raise LiveArtifactValidationError(
                "live transition diagnostics must be object"
            )
        try:
            loaded.append(
                TransitionTraceEntry(
                    benchmark_step_index=item["benchmark_step_index"],
                    native_step_id=item["native_step_id"],
                    signal=BackendSignal(item["signal"]),
                    diagnostics=diagnostics,
                )
            )
        except (TypeError, ValueError) as error:
            raise LiveArtifactValidationError(
                "live transition entry failed typed validation"
            ) from error
    return (
        tuple(loaded),
        cast(Mapping[str, object], freeze_value(initial_diagnostics)),
    )


__all__ = ["transition_trace_from_live_dict", "transition_trace_to_live_dict"]
