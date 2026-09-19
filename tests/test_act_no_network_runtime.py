"""No-network construction contract for the pinned official UniVTAC ACT."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from robotactile_benchmark.policies.univtac_official_act import (
    OfficialACTProfile,
    _construct_pinned_upstream_runtime,
    _OwnedRuntime,
)


class _FakeCuda:
    @staticmethod
    def device(_device_name: str) -> nullcontext[None]:
        return nullcontext()


class _FakeTorch:
    cuda = _FakeCuda()


def _write_upstream(
    root: Path,
    *,
    fail: bool = False,
    capture_module: str | None = None,
) -> Path:
    source = root / "act_policy.py"
    capture_import = (
        "" if capture_module is None else f"import {capture_module} as capture\n"
    )
    capture_state = (
        ""
        if capture_module is None
        else (
            "        capture.backbone = backbone\n"
            "        capture.original = misc.is_main_process\n"
        )
    )
    files = {
        root / "detr/__init__.py": "",
        root / "detr/models/__init__.py": "",
        root / "detr/util/__init__.py": "",
        root / "detr/util/misc.py": ("def is_main_process():\n    return True\n"),
        root / "detr/models/backbone.py": ("from util.misc import is_main_process\n"),
        source: (
            "from detr.models import backbone\n"
            "from util import misc\n" + capture_import + "class ACT:\n"
            "    def __init__(self, _args):\n"
            "        self.backbone = backbone\n"
            "        self.original = misc.is_main_process\n"
            "        self.pretraining_enabled = backbone.is_main_process()\n"
            + capture_state
            + ("        raise RuntimeError('constructor failed')\n" if fail else "")
        ),
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return source


def _construct(source: Path) -> object:
    return _construct_pinned_upstream_runtime(
        source_path=source,
        source_sha256="a" * 64,
        profile=OfficialACTProfile.UNIVTAC,
        task_id="pull_out_key",
        encoder_path=source.parent / "encoder.pth",
        device_name="cuda:0",
        torch_module=_FakeTorch(),
    )


def test_runtime_construction_disables_download_then_restores(tmp_path: Path) -> None:
    runtime = _construct(_write_upstream(tmp_path / "ACT"))

    assert runtime.pretraining_enabled is False  # type: ignore[attr-defined]
    assert runtime.backbone.is_main_process is runtime.original  # type: ignore[attr-defined]
    assert runtime.backbone.is_main_process() is True  # type: ignore[attr-defined]
    assert "detr.models.backbone" not in sys.modules
    assert "util.misc" not in sys.modules


def test_runtime_constructor_failure_restores_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_name = "_robotactile_act_guard_capture"
    capture = ModuleType(capture_name)
    monkeypatch.setitem(sys.modules, capture_name, capture)
    source = _write_upstream(
        tmp_path / "ACT",
        fail=True,
        capture_module=capture_name,
    )
    with pytest.raises(RuntimeError, match="constructor failed"):
        _construct(source)

    assert capture.backbone.is_main_process is capture.original  # type: ignore[attr-defined]
    assert capture.backbone.is_main_process() is True  # type: ignore[attr-defined]
    assert "detr.models.backbone" not in sys.modules
    assert "util.misc" not in sys.modules


def test_runtime_construction_rejects_rebound_upstream_guard(tmp_path: Path) -> None:
    source = _write_upstream(tmp_path / "ACT")
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "from util import misc\n",
            "from util import misc\nbackbone.is_main_process = lambda: True\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        _construct(source)


class _TemporalAggregationRuntime:
    def __init__(self, action: object) -> None:
        self.action = action

    def get_action(self, _observation: object) -> object:
        return self.action


def test_owned_runtime_canonicalizes_exact_temporal_float64_action() -> None:
    source = np.arange(8, dtype=np.float64)[None, :]
    runtime = _OwnedRuntime(_TemporalAggregationRuntime(source), _FakeTorch())

    action = runtime.get_action({})

    assert action.dtype == np.float32
    assert action.shape == (1, 8)
    assert action.flags.c_contiguous
    np.testing.assert_array_equal(action, source.astype(np.float32))


@pytest.mark.parametrize(
    "source",
    (
        np.zeros((8,), dtype=np.float64),
        np.full((1, 8), np.nan, dtype=np.float64),
    ),
)
def test_owned_runtime_leaves_other_invalid_outputs_for_policy_rejection(
    source: np.ndarray,
) -> None:
    runtime = _OwnedRuntime(_TemporalAggregationRuntime(source), _FakeTorch())

    assert runtime.get_action({}) is source
