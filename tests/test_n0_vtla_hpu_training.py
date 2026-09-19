from pathlib import Path

import numpy as np
import pytest

from scripts.n0_vtla.hpu_training.contracts import SourceRecord, select_train_records
from scripts.n0_vtla.hpu_training.qpos8 import pack_next_step_qpos8, vector_stats
from scripts.n0_vtla.hpu_training.runtime_patches import patch_policy_source


def _record(episode_id: int, split: str = "train") -> SourceRecord:
    return SourceRecord(
        raw_root=Path("/raw"),
        relative_path=f"insert_hole/clean/{episode_id}.hdf5",
        task="insert_hole",
        episode_id=episode_id,
        split=split,
        size_bytes=1,
        mtime_ns=1,
        sha256="0" * 64,
        usable_source_range=None,
    )


def test_qpos8_uses_current_state_and_strict_next_action() -> None:
    joint = np.arange(36, dtype=np.float32).reshape(4, 9)

    state, action = pack_next_step_qpos8(joint)

    np.testing.assert_array_equal(state, joint[:-1, :8])
    np.testing.assert_array_equal(action, joint[1:, :8])
    assert state.dtype == np.float32
    assert action.dtype == np.float32


def test_qpos8_respects_pinned_usable_prefix() -> None:
    joint = np.arange(45, dtype=np.float32).reshape(5, 9)

    state, action = pack_next_step_qpos8(joint, usable_start=1, usable_stop=4)

    np.testing.assert_array_equal(state, joint[1:3, :8])
    np.testing.assert_array_equal(action, joint[2:4, :8])


def test_qpos8_rejects_nonfinite_joint_rows() -> None:
    joint = np.zeros((3, 9), dtype=np.float32)
    joint[1, 0] = np.nan

    try:
        pack_next_step_qpos8(joint)
    except ValueError as exc:
        assert "finite float32" in str(exc)
    else:
        raise AssertionError("nonfinite qpos rows must be rejected")


def test_vector_stats_preserve_eight_dimensions() -> None:
    rows = np.stack(
        (np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 2)
    )

    stats = vector_stats(rows)

    assert stats["count"] == [2]
    assert stats["mean"] == (np.arange(8, dtype=float) + 1).tolist()
    assert len(stats["std"]) == 8


def test_task_selection_excludes_non_train_records() -> None:
    train = [_record(index) for index in range(95)]
    frozen = _record(99, split="frozen")

    selected = select_train_records([frozen, *reversed(train)], task="insert_hole")

    assert [record.episode_id for record in selected] == list(range(95))
    assert all(record.split == "train" for record in selected)


def test_formal_launcher_is_persistent_and_uses_official_workers() -> None:
    tooling = Path("scripts/n0_vtla/hpu_training")
    rank_entrypoint = (tooling / "rank_entrypoint.sh").read_text()
    launcher = (tooling / "launch_formal_task.sh").read_text()

    assert '[[ "$mode" == "formal" ]] && printf 8 || printf 2' in rank_entrypoint
    assert '--num-workers="$num_workers"' in rank_entrypoint
    assert 'PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"' in rank_entrypoint
    hcu_entry = (tooling / "hcu_train_entry.py").read_text()
    assert "ROBOTACTILE_N0_VTLA_DETACH_VL_CTX" in hcu_entry
    assert "ROBOTACTILE_N0_VTLA_GRAD_ACCUM" in hcu_entry
    assert "ROBOTACTILE_N0_VTLA_TASK_BALANCED" in hcu_entry
    assert 'nohup setsid "$0" __worker' in launcher
    assert 'formal "$scope" "$project_root" 8 "$num_steps" "$run_id"' in launcher
    assert '[[ "$scope" == "mixed8" ]] && printf 160000 || printf 20000' in launcher
    assert 'mkdir "$supervisor_dir" || die' in launcher


def test_runtime_patch_is_source_bound_and_semantics_preserving() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "N0-VTLA/n0vtla/models_pytorch/n0vtla_policy.py"
    )
    if not source_path.is_file():
        pytest.skip("requires the reviewed external N0-VTLA source checkout")

    patched = patch_policy_source(source_path)

    assert "vl_ctx = vl_ctx.detach()" in patched
    assert "self._compute_z(vl_ctx, prefix_pad_masks)" in patched
    assert "self._compute_z(vl_ctx.detach(), prefix_pad_masks)" not in patched
    compile(patched, str(source_path), "exec")
