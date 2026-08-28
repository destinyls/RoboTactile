from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam.dynamic_probe_io import (
    write_dynamic_probe_bundle,
)
from scripts.live_univtac.probe_n0_dynamic_contract_isaac import (
    _raise_probe_failure,
    _validate_args,
)


def _args(**updates: object) -> argparse.Namespace:
    values = {
        "exogenous_seed": 20260825,
        "initial_seed": 90,
        "static_render_count": 4,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_dynamic_probe_arguments_are_bounded() -> None:
    _validate_args(_args())
    with pytest.raises(ValueError, match="static_render_count"):
        _validate_args(_args(static_render_count=1))
    with pytest.raises(ValueError, match="initial_seed"):
        _validate_args(_args(initial_seed=-1))


def test_dynamic_probe_rejects_clean_looking_early_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError, match="SystemExit during reset"):
        _raise_probe_failure("reset", SystemExit(0))

    captured = capsys.readouterr()
    assert '"error_type":"SystemExit"' in captured.err
    assert '"failed_stage":"reset"' in captured.err


def test_dynamic_probe_bundle_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    streams = {
        name: np.zeros((4, 5, 3), dtype=np.uint8)
        for name in ("top", "wrist_l", "tactile_a", "tactile_b")
    }
    output = tmp_path / "probe"
    document = write_dynamic_probe_bundle(
        output,
        domains={"hdf5_checkpoint": streams},
        states={"live_pre_ee8": np.zeros(8, dtype=np.float32)},
        panel=b"real-panel-bytes",
        metadata={
            "task_id": "lift_bottle",
            "transition_diagnostics": MappingProxyType(
                {
                    "task": MappingProxyType({"predicate_success": False}),
                    "steps": (1, 2),
                }
            ),
        },
    )

    assert json.loads((output / "probe.json").read_text()) == document
    assert document["transition_diagnostics"] == {
        "steps": [1, 2],
        "task": {"predicate_success": False},
    }
    assert (output / "comparison_panel.png").read_bytes() == b"real-panel-bytes"
    with pytest.raises(FileExistsError):
        write_dynamic_probe_bundle(
            output,
            domains={"hdf5_checkpoint": streams},
            states={"live_pre_ee8": np.zeros(8, dtype=np.float32)},
            panel=b"real-panel-bytes",
            metadata={"task_id": "lift_bottle"},
        )
