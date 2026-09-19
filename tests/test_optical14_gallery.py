"""Integrity and presentation boundaries of the recorded-source gallery."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from robotactile_benchmark.fixtures import make_synthetic_rest_references
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.visualization.optical14 import (
    delta,
    display_index,
    file_hash,
    load_source,
    make_manifest,
    render_gallery,
    sequence_caption,
    source_array_hash,
    tile,
)


def test_missing_payload_has_no_fabricated_delta() -> None:
    clean = np.full((240, 320, 3), 80, dtype=np.uint8)
    assert delta(clean, None) is None
    assert tile(None, "Missing payload").size == (320, 294)
    changed = clean.copy()
    changed[0, 0] = [79, 82, 0]
    assert delta(clean, changed)[0, 0].tolist() == [1, 2, 80]


def test_existing_output_fails_before_reading_source(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="no-clobber"):
        render_gallery(tmp_path / "missing-input", tmp_path)


def test_loader_rejects_array_tampering_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    value = np.zeros((2, 2, 240, 320, 3), dtype=np.uint8)
    expected = source_array_hash(value)
    value[0, 0, 0, 0, 0] = 1
    path = tmp_path / "episode.npz"
    np.savez(path, tactile_rgb=value)
    spec = {
        "artifact": path.name,
        "artifact_sha256": file_hash(path),
        "arrays": {
            "tactile_rgb": {
                "dtype": "uint8",
                "shape": list(value.shape),
                "sha256": expected,
            }
        },
    }
    with pytest.raises(ValueError, match="array hash/shape/dtype mismatch"):
        load_source(tmp_path, spec)


def test_loader_rejects_artifact_path_escape(tmp_path: Path) -> None:
    inside = tmp_path / "source"
    inside.mkdir()
    path = tmp_path / "outside.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="escapes"):
        load_source(inside, {"artifact": "../outside.npz"})


def test_display_selection_requires_no_corrupted_payload() -> None:
    manifest = SimpleNamespace(
        operator_id="F4_local_nonresponsive_patch",
        start_index=105,
        stop_index=131,
        parameters={},
    )
    assert display_index(cast(FaultManifest, manifest), {"contact_peak": 121}) == 121
    manifest.operator_id = "F6_history_residual_imprint"
    assert (
        display_index(
            cast(FaultManifest, manifest),
            {"f6_local_unloading_witness": {"indices": [105, 106, 107]}},
        )
        == 107
    )


def test_temporal_common_delay_and_inter_sensor_skew_have_distinct_maps() -> None:
    arrays = {"source_time_s": np.arange(100, dtype=np.float64) / 60}
    spec = {"contact_peak": 40}
    rest = make_synthetic_rest_references()
    common = make_manifest("T1_fixed_source_delay", 3, arrays, spec, rest)
    skew = make_manifest("T3_inter_sensor_skew", 3, arrays, spec, rest)
    assert common.sensor_slots == skew.sensor_slots == ("left", "right")
    assert common.parameters["source_index_map"][0] < common.start_index
    assert skew.parameters["source_index_map"]["left"][0] == skew.start_index
    assert skew.parameters["source_index_map"]["right"][0] < skew.start_index


def test_unloading_caption_never_leaks_into_temporal_fault_figures() -> None:
    for prefix in ("T1", "T2", "T3"):
        title = sequence_caption(prefix, "pull_out_key", 55)
        assert "F6" not in title
        assert "release" not in title
    assert "not global release" in sequence_caption("F6", "pull_out_key", 55)
