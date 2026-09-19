import json

import numpy as np
import pytest

from robotactile_benchmark.n0_fault_campaign.action_effects import (
    action_distance,
    delivery_exposure,
)
from scripts.n0_twam.run_decision_stress import prepare_pilot


def test_action_distance_reports_geometry_not_quaternion_sign():
    a = np.array([[0, 0, 0, 1, 0, 0, 0, 0.02]], dtype=np.float32)
    b = np.array([[0.003, 0.004, 0, -1, 0, 0, 0, 0.023]], dtype=np.float32)
    result = action_distance(a, b)
    assert result["translation_mean_mm"] == pytest.approx(5)
    assert result["rotation_max_deg"] == 0
    assert result["gripper_max_mm"] == pytest.approx(3)


def test_action_distance_rejects_empty_or_invalid_geometry():
    a = np.zeros((1, 8))
    with pytest.raises(ValueError, match="quaternion"):
        action_distance(a, a)
    with pytest.raises(ValueError, match="empty"):
        action_distance(a[:0], a[:0])


def test_pilot_rejects_duplicate_seeds_before_writing(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        prepare_pilot(
            tmp_path / "prior", tmp_path / "new", (5, 5), ("lift_can",), 29695
        )
    assert not (tmp_path / "new").exists()


def test_exposure_falls_back_from_null_finalization_to_preview(tmp_path):
    (tmp_path / "delivery_trace.json").write_text(json.dumps({"finalization": None}))
    clean = {
        "observation": {
            "step_index": 6,
            "vision": [],
            "proprio": {},
            "tactile": [{"slot_id": "left", "payload": {"array_sha256": "a"}}],
        },
        "provenance": [],
    }
    delivered = json.loads(json.dumps(clean))
    delivered["observation"]["tactile"][0]["payload"]["array_sha256"] = "b"
    (tmp_path / "preview_trace.json").write_text(
        json.dumps({"clean_records": [clean], "delivered_records": [delivered]})
    )
    result = delivery_exposure(tmp_path, [{"source_step_index": 12}], 5)
    assert result["expected_committed_keyframes"] == 4
    assert result["captured_committed_keyframes"] == 1
    assert result["frames"][0]["post_onset"] is True
    assert result["frames"][0]["raw_tactile_hashes"][0]["delivered"] == "b"


def test_exposure_without_capture_is_not_evidence_of_no_effect(tmp_path):
    assert delivery_exposure(tmp_path, [], 5) == {"status": "not_captured"}
