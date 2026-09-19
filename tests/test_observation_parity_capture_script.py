from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.live_univtac.capture_observation_parity_isaac import (
    _parser,
    _sha256_file,
    _validate_args,
    _write_bundle,
)


def _args(**updates: object) -> argparse.Namespace:
    values = {
        "exogenous_seed": 29,
        "initial_seed": 90,
        "static_render_count": 8,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_capture_arguments_reject_invalid_render_count() -> None:
    _validate_args(_args())
    with pytest.raises(ValueError, match="static_render_count"):
        _validate_args(_args(static_render_count=0))


def test_capture_defaults_to_shared_n0_twam_renderer() -> None:
    defaults = _parser().parse_args(
        [
            "--upstream-root",
            "/upstream",
            "--runtime-dir",
            "/runtime",
            "--output-dir",
            "/output",
            "--initial-seed",
            "90",
            "--exogenous-seed",
            "29",
        ]
    )

    assert defaults.rendering_mode == "balanced"
    assert defaults.antialiasing_mode == "TAA"


def test_source_manifest_is_content_hashed(tmp_path: Path) -> None:
    path = tmp_path / "source_manifest.sha256"
    payload = b"a" * 64 + b"  tracked.py\n"
    path.write_bytes(payload)
    assert _sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_capture_bundle_is_atomic_content_bound_and_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture"
    streams = {
        name: np.full((2, 4, 5, 3), index, dtype=np.uint8)
        for index, name in enumerate(("top", "wrist_l", "tactile_a", "tactile_b"))
    }
    document = _write_bundle(
        output,
        raw_frames=streams,
        model_frames=streams,
        metadata={"task_id": "lift_bottle"},
    )

    stored = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    assert stored == document
    assert len(stored["content_sha256"]) == 64
    assert np.array_equal(
        np.load(output / stored["arrays"]["raw"]["top"]["path"]),
        streams["top"],
    )
    with pytest.raises(FileExistsError, match="already exist"):
        _write_bundle(
            output,
            raw_frames=streams,
            model_frames=streams,
            metadata={"task_id": "lift_bottle"},
        )
