"""Bounded preview selection and live record codec tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.delivery import (
    DeliveryFinalization,
    IdentityDeliverySession,
    OnlineFaultSession,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LiveArtifactValidationError,
)
from robotactile_benchmark.execution.live_artifacts_fs import scan_live_bundle
from robotactile_benchmark.execution.live_artifacts_io import LiveArrayWriter
from robotactile_benchmark.execution.live_artifacts_preview import (
    DEFAULT_PREVIEW_RECORD_LIMIT,
    LivePreviewTrace,
    build_live_preview_trace,
    iter_preview_arrays,
    preview_trace_from_live_dict,
    preview_trace_to_live_dict,
    select_preview_indices,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability


def _identity_finalization(length: int = 100) -> DeliveryFinalization:
    session = IdentityDeliverySession()
    for record in make_synthetic_episode(length=length):
        session.deliver(record)
    return session.finalize()


def _fault_finalization(length: int = 100) -> DeliveryFinalization:
    fault = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=23,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )
    session = OnlineFaultSession(fault)
    for record in make_synthetic_episode(length=length):
        session.deliver(record)
    return session.finalize()


def test_preview_selection_preserves_endpoints_and_phase_transitions() -> None:
    selected = select_preview_indices(_identity_finalization(), limit=8)

    assert selected == (0, 2, 3, 4, 7, 8, 9, 99)


def test_preview_selection_includes_active_fault_transitions_and_fills_limit() -> None:
    selected = select_preview_indices(_fault_finalization(), limit=12)

    assert selected[0] == 0
    assert selected[-1] == 99
    assert {1, 2, 3, 4, 7, 8, 9} <= set(selected)
    assert len(selected) == 12
    assert selected == select_preview_indices(_fault_finalization(), limit=12)


def test_preview_codec_round_trip_tracks_every_referenced_array(
    tmp_path: Path,
) -> None:
    trace = build_live_preview_trace(_identity_finalization(length=12), limit=8)
    bundle = tmp_path / "preview"
    bundle.mkdir()
    arrays = LiveArrayWriter(bundle)

    document = preview_trace_to_live_dict(trace, arrays)
    snapshot = scan_live_bundle(bundle)
    referenced_paths: set[str] = set()
    loaded = preview_trace_from_live_dict(document, snapshot, referenced_paths)

    assert loaded.selected_indices == trace.selected_indices
    assert canonical_hash(loaded) == canonical_hash(trace)
    assert referenced_paths == {
        str(descriptor["path"]) for descriptor in arrays.descriptors.values()
    }
    assert len(tuple(iter_preview_arrays(trace))) == len(trace.selected_indices) * 10


def test_empty_preview_is_valid_for_artifactless_delivery(tmp_path: Path) -> None:
    trace = build_live_preview_trace(None)
    assert trace == LivePreviewTrace((), (), ())
    assert select_preview_indices(None) == ()

    bundle = tmp_path / "empty-preview"
    bundle.mkdir()
    document = preview_trace_to_live_dict(trace, LiveArrayWriter(bundle))
    loaded = preview_trace_from_live_dict(document, scan_live_bundle(bundle), set())

    assert loaded == trace
    assert tuple(iter_preview_arrays(loaded)) == ()


def test_preview_limits_and_tampering_fail_closed(tmp_path: Path) -> None:
    finalization = _identity_finalization()
    with pytest.raises(ValueError, match="preview limit"):
        select_preview_indices(finalization, limit=1)
    with pytest.raises(ValueError, match="preview limit"):
        select_preview_indices(finalization, limit=DEFAULT_PREVIEW_RECORD_LIMIT + 1)

    trace = build_live_preview_trace(finalization, limit=8)
    bundle = tmp_path / "tampered-preview"
    bundle.mkdir()
    arrays = LiveArrayWriter(bundle)
    document = preview_trace_to_live_dict(trace, arrays)
    snapshot = scan_live_bundle(bundle)

    bad_fields = dict(document)
    bad_fields["unknown"] = None
    with pytest.raises(LiveArtifactValidationError, match="fields"):
        preview_trace_from_live_dict(bad_fields, snapshot, set())

    bad_index = dict(document)
    bad_index["selected_indices"] = [True, *trace.selected_indices[1:]]
    with pytest.raises(LiveArtifactValidationError, match="index"):
        preview_trace_from_live_dict(bad_index, snapshot, set())

    wrong_pairing = dict(document)
    wrong_pairing["selected_indices"] = [1, *trace.selected_indices[1:]]
    with pytest.raises(ValueError, match="pairing"):
        preview_trace_from_live_dict(wrong_pairing, snapshot, set())
