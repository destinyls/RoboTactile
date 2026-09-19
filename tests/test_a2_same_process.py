"""A2 retry uses a fresh paired control, not a historical reset reference."""

import json
from pathlib import Path

import pytest
from test_tactile_availability_groups import binding

from scripts.retrained_evaluation.a2_same_process import A2, TASK, prepare
from scripts.retrained_evaluation.group import write_json


def test_prepare_clean_a2_preserves_source_and_erasure_contract(tmp_path: Path) -> None:
    value = binding(tmp_path)
    value["tasks"] = {
        TASK: {"prompt": "Reorient a bottle upright and place it on a shelf"}
    }
    source = tmp_path / "source.json"
    write_json(source, value)
    before = source.read_bytes()
    output = tmp_path / "new-pair"
    prepare(source, output, 0)
    plan = json.loads((output / "prepared/group.json").read_text())
    assert [Path(name).stem for name in plan["ordered_requests"]] == ["clean", A2]
    assert "recovery_protocol" not in plan
    assert not (output / "prepared/reset_reference.json").exists()
    fault = json.loads(
        (
            output / "prepared/faults/optical_marker_extreme_v1/A2_frame_erasure.json"
        ).read_text()
    )
    assert fault["start_index"] == 0 and fault["stop_index"] == 301
    assert fault["sensor_slots"] == ["left", "right"]
    assert fault["parameters"]["a2_end_policy"] == "episode_censored_v1"
    assert fault["parameters"]["erased_offsets"] == [
        i for i in range(301) if i not in {25, 75, 125, 175, 225, 275}
    ]
    assert source.read_bytes() == before
    assert not (output / "campaign").exists()
    with pytest.raises(FileExistsError):
        prepare(source, output, 0)


def test_wrong_model_rejected_without_output(tmp_path: Path) -> None:
    value = binding(tmp_path)
    value["model"] = "n0_twam"
    source = tmp_path / "source.json"
    write_json(source, value)
    with pytest.raises(ValueError, match="N0-VTLA shelf"):
        prepare(source, tmp_path / "output", 0)
    assert not (tmp_path / "output").exists()
