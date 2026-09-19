"""Display selection and source-frame assertions, not a claim of paper acceptance."""

import importlib.util
from pathlib import Path

import pytest

from robotactile_benchmark.constants import (
    LOGICAL_STEP_SECONDS,
    OPTICAL_MARKER_REGISTRY_ID,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.runtime import apply_fault


@pytest.fixture
def panels():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/render_optical14_paper_panels.py"
    )
    spec = importlib.util.spec_from_file_location("paper_panels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_panels_no_clobber_before_reading_sources(panels, tmp_path):
    with pytest.raises(FileExistsError):
        panels.render(
            tmp_path / "missing", tmp_path / "missing", tmp_path, tmp_path / "font"
        )


def test_freeze_witness_requires_old_frame_and_actual_resumption(panels):
    clean = make_synthetic_episode(30)
    manifest = FaultManifest(
        "T2_held_last_freeze",
        1,
        23,
        8,
        22,
        ("right",),
        Observability.BLIND,
        {"sample_period_s": LOGICAL_STEP_SECONDS},
        severity_registry=OPTICAL_MARKER_REGISTRY_ID,
    )
    result = apply_fault(clean, manifest)
    duration = manifest.parameters["hold_duration_frames"]
    indices = [7, 8, 8 + duration - 1, 8 + duration]
    witness = panels.freeze_witness(clean, result.records, indices)
    assert all(witness["checks"].values())
    with pytest.raises(ValueError, match="freeze has no intelligible witness"):
        panels.freeze_witness(clean, clean, indices)
