"""Fixed visual witnesses and numerical dose checks do not invent significance."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gallery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("scripts.render_optical14_stress")


def test_stress_source_layout_rejects_a_different_clock(gallery):
    spec = {"task": "pull_out_key", "raw_episode_id": 55, "contact_peak": 121}
    gallery.check_source_layout(spec, {"source_time_s": np.arange(178) / 60})
    with pytest.raises(ValueError, match="source layout"):
        gallery.check_source_layout(spec, {"source_time_s": np.arange(178) / 120})


def test_stress_equal_effect_is_not_reported_as_stronger(gallery):
    for standard, stress, expected in (
        (2.0, 4.0, True),
        (2.0, 2.0, False),
        (4.0, 2.0, False),
    ):
        checks = gallery.stronger_checks(
            {
                "F2": {
                    "standard": {"display_mean_absolute_delta_u8": standard},
                    "stress": {"display_mean_absolute_delta_u8": stress},
                }
            }
        )
        assert checks["F2"] is expected


def test_stress_gallery_no_clobber(gallery, tmp_path):
    with pytest.raises(FileExistsError):
        gallery.render(tmp_path / "missing", tmp_path, tmp_path / "font")


def test_extreme_cap_is_not_reported_as_an_increase(gallery):
    cases = {
        prefix: {"standard": {"affected_count": 87}, "stress": {"affected_count": 87}}
        for prefix in ("A1", "C1")
    }
    assert not any(gallery.stronger_checks(cases).values())
    assert set(gallery.accepted_checks(cases, extreme=True).values()) == {
        "unchanged_at_100_percent_cap"
    }
    with pytest.raises(ValueError, match="no demonstrated increase"):
        gallery.accepted_checks(cases, extreme=False)
    cases["A1"]["stress"]["affected_count"] = 86
    with pytest.raises(ValueError, match="no demonstrated increase"):
        gallery.accepted_checks(cases, extreme=True)


def test_extreme_cannot_excuse_equal_or_reduced_optical_effect(gallery):
    cases = {
        "F2": {
            "standard": {"display_mean_absolute_delta_u8": 2.0},
            "stress": {"display_mean_absolute_delta_u8": 2.0},
        }
    }
    with pytest.raises(ValueError, match="F2"):
        gallery.accepted_checks(cases, extreme=True)


def test_freeze_witness_uses_actual_durations_and_requires_resumption(gallery):
    assert gallery.freeze_indices(80, 72, 90, 178) == [79, 80, 152, 169, 170]
    assert gallery.freeze_indices(90, 48, 72, 177) == [89, 90, 138, 161, 162]
    with pytest.raises(ValueError, match="resumption"):
        gallery.freeze_indices(90, 72, 90, 178)


def test_optical_dose_caption_reads_materialized_parameters(gallery):
    assert (
        gallery.dose_label("F5", {"displacement_fraction": 0.2}, 240) == "最大位移48 px"
    )
    assert gallery.dose_label("C2", {"translation_xy_px": [96, 0]}, 240) == "向右96 px"
    assert (
        gallery.dose_label(
            "F2", {"retained_gain": 0.005, "core_radius_fraction": 0.36}, 240
        )
        == "核心保留0.5% / 半径0.36"
    )


def test_misregistration_uses_displacement_not_periodic_marker_mae(gallery):
    case = {
        "C2": {
            key: {
                "manifest": {"parameters": {"translation_xy_px": [shift, 0]}},
                "display_mean_absolute_delta_u8": mae,
            }
            for key, shift, mae in (("standard", 60, 27.0), ("stress", 96, 24.0))
        }
    }
    assert gallery.stronger_checks(case)["C2"] is True
    case["C2"]["stress"]["manifest"]["parameters"]["translation_xy_px"] = [60, 0]
    assert gallery.stronger_checks(case)["C2"] is False
