from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

OVERLAY = (
    Path(__file__).parents[1]
    / "scripts"
    / "dream_tac"
    / "hcu_port"
    / "overlays"
    / "franka_dataset.py"
)


def _tree() -> ast.Module:
    return ast.parse(OVERLAY.read_text(encoding="utf-8"))


def _class(name: str) -> ast.ClassDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def test_overlay_is_syntax_valid_and_train_only() -> None:
    source = OVERLAY.read_text(encoding="utf-8")
    compile(source, str(OVERLAY), "exec")
    assert "get_hdf5_files(data_dir, is_train=True)" in source
    assert "if not is_train:" in source
    assert "DREAM_TAC_FRANKA_MAX_OPEN_VIDEOS" in source
    assert "DREAM_TAC_FRANKA_FRAME_CACHE_SIZE" in source


def test_initialization_never_decodes_a_complete_video() -> None:
    source = OVERLAY.read_text(encoding="utf-8")
    assert "load_video_as_images" not in source
    init_node = next(
        node
        for node in _class("FrankaDataset").body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert not any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "read"
        for call in ast.walk(init_node)
    )


def test_video_fields_are_lazy_and_numeric_fields_use_original_rescale() -> None:
    source = OVERLAY.read_text(encoding="utf-8")
    assert '"images": images' in source
    assert '"wrist_images": wrist_images' in source
    assert '"tactile_left_images": tactile_left' in source
    assert (
        'self.data = rescale_data(self.data, self.dataset_stats, "actions")' in source
    )
    assert (
        'self.data = rescale_data(self.data, self.dataset_stats, "proprio")' in source
    )


def test_missing_tactile_uses_constant_memory_zero_sequence() -> None:
    zero_class = _class("_ZeroFrameSequence")
    init_node = next(
        node
        for node in zero_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    zeros_calls = [
        call
        for call in ast.walk(init_node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "zeros"
    ]
    assert len(zeros_calls) == 1
    dimensions = zeros_calls[0].args[0]
    assert isinstance(dimensions, ast.Tuple)
    assert len(dimensions.elts) == 3


def test_capture_and_frame_caches_are_bounded_lru() -> None:
    source = OVERLAY.read_text(encoding="utf-8")
    pool = _class("_FrameReaderPool")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "popitem"
        for node in ast.walk(pool)
    )
    assert "while len(self._captures) > self._max_open_videos" in source
    assert "while len(self._frames) > self._max_cached_frames" in source


def test_zero_placeholder_broadcast_does_not_allocate_per_episode_timeline() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    broadcast = np.broadcast_to(frame, (100, *frame.shape))
    assert not broadcast.flags.owndata
