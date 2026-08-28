"""Static contracts for the bounded all-task UniVTAC qualifier."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/live_univtac/qualify_univtac_all_tasks.sh"


def test_all_task_qualifier_freezes_scope_and_evidence_boundary() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for task_id in (
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    ):
        assert f"  {task_id}\n" in text
    assert 'ACTION_SPEC="ee8_absolute"' in text
    assert 'REPETITIONS="1"' in text
    assert "same-seed reset is not repeatable" in text
    assert "same-seed action pairing is not repeatable" in text
    assert '"closed_loop_episode_executed": False' in text
    assert '"policy_loaded": False' in text
    assert '"success_rate_claimed": False' in text
