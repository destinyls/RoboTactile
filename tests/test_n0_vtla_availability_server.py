"""Missing-input overlay tests without loading the model or GPU runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robotactile_benchmark.contracts import Array, array_sha256
from robotactile_benchmark.integrations.n0_vtla.availability_server import (
    BASELINE_RULE,
    PROTOCOL_KEY,
    install_availability_protocol,
)


def decode(value: object) -> Array:
    return np.asarray(value, dtype=np.uint8)


def frame(value: int) -> Array:
    return np.full((4, 5, 3), value, dtype=np.uint8)


class FakeWorker:
    """Only the upstream collector/reset and baseline-pairing control flow."""

    def __init__(self) -> None:
        self._tactile_views = ["view_left", "view_right"]
        self._tactile_enabled = True
        self._baseline: dict[str, Array] | None = None
        self.reset_calls = 0
        self._reset: Callable[[], dict[str, Any]] = self.reset_impl
        self._collect_current_tactile: Callable[
            [Mapping[str, object]], dict[str, Array]
        ] = self.collect_impl

    def reset_impl(self) -> dict[str, Any]:
        self.reset_calls += 1
        self._baseline = None
        return {"status": "ok"}

    def collect_impl(self, message: Mapping[str, object]) -> dict[str, Array]:
        keys = [
            key
            for key in ("observation/left_tactile", "observation/right_tactile")
            if key in message
        ]
        return {
            view: decode(message[key]) for view, key in zip(self._tactile_views, keys)
        }

    def predict(self, message: Mapping[str, object]) -> dict[str, Array]:
        current = self._collect_current_tactile(message)
        if not current:
            return {}
        if self._baseline is None:
            self._baseline = {view: image.copy() for view, image in current.items()}
        return {
            view: np.stack([self._baseline.get(view, image), image])
            for view, image in current.items()
        }


def message(mode: str = "native_missing_v1", **slots: Array) -> dict[str, object]:
    return {
        PROTOCOL_KEY: mode,
        **{f"observation/{slot}_tactile": image for slot, image in slots.items()},
    }


def test_missing_left_keeps_right_identity_and_first_available_baselines() -> None:
    worker = FakeWorker()
    install_availability_protocol(worker, "native_missing_v1", decoder=decode)
    right_only = worker.predict(message(right=frame(10)))
    assert set(right_only) == {"view_right"}
    np.testing.assert_array_equal(right_only["view_right"], np.stack([frame(10)] * 2))
    both = worker.predict(message(left=frame(20), right=frame(30)))
    np.testing.assert_array_equal(both["view_left"][0], frame(20))
    np.testing.assert_array_equal(both["view_right"][0], frame(10))
    worker.predict(message(right=frame(40)))
    recovered = worker.predict(message(left=frame(50), right=frame(60)))
    np.testing.assert_array_equal(recovered["view_left"][0], frame(20))
    np.testing.assert_array_equal(recovered["view_right"][0], frame(10))


def test_both_missing_does_not_create_baseline() -> None:
    worker = FakeWorker()
    install_availability_protocol(worker, "native_missing_v1", decoder=decode)
    assert worker.predict(message()) == {}
    assert worker._baseline is None
    result = worker.predict(message(left=frame(25)))
    np.testing.assert_array_equal(result["view_left"][0], frame(25))


def test_reset_preserves_upstream_reset_and_clears_overlay_state(
    tmp_path: Path,
) -> None:
    worker = FakeWorker()
    witness = tmp_path / "availability.jsonl"
    install_availability_protocol(
        worker, "native_missing_v1", decoder=decode, witness_path=witness
    )
    worker.predict(message())
    worker.predict(message(right=frame(10)))
    assert worker._reset() == {"status": "ok"}
    assert worker.reset_calls == 1
    assert worker._baseline is None
    result = worker.predict(message(right=frame(35)))
    np.testing.assert_array_equal(result["view_right"][0], frame(35))
    records = [json.loads(line) for line in witness.read_text().splitlines()]
    assert records[1]["baseline_first_query"] == {"right": 1}
    assert records[-1]["baseline_first_query"] == {"right": 0}
    assert records[-1]["episode_index"] == 1
    assert records[-1]["baseline_rule"] == BASELINE_RULE
    assert "data_hex" not in witness.read_text()


def test_zero_fill_black_first_frame_remains_baseline(tmp_path: Path) -> None:
    worker = FakeWorker()
    path = tmp_path / "zero.availability.jsonl"
    install_availability_protocol(
        worker, "zero_fill_v1", decoder=decode, witness_path=path
    )
    worker.predict(message("zero_fill_v1", left=frame(0), right=frame(15)))
    result = worker.predict(message("zero_fill_v1", left=frame(55), right=frame(45)))
    np.testing.assert_array_equal(result["view_left"][0], frame(0))
    with pytest.raises(ValueError, match="two supplied"):
        worker.predict(message("zero_fill_v1", right=frame(5)))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["baseline_rule"] == "first_delivered_including_fill_v1"
    received = records[0]["received_slots"]["left"]
    assert received == {
        "array_sha256": array_sha256(frame(0)),
        "shape": [4, 5, 3],
        "dtype": "uint8",
        "max_abs": 0,
    }
    assert records[1]["received_slots"]["left"]["max_abs"] == 55
    assert (
        records[0]["baseline_sha256"]["left"] == records[1]["baseline_sha256"]["left"]
    )


@pytest.mark.parametrize("mode", ["native_missing_v1", "zero_fill_v1"])
def test_full_input_matches_legacy_pairing(mode: str) -> None:
    legacy, overlay = FakeWorker(), FakeWorker()
    install_availability_protocol(overlay, mode, decoder=decode)
    for value in (10, 20, 30):
        request = message(mode, left=frame(value), right=frame(value + 1))
        expected, actual = legacy.predict(request), overlay.predict(request)
        assert set(actual) == set(expected)
        for view in expected:
            np.testing.assert_array_equal(actual[view], expected[view])


def test_required_mode_does_not_install_or_create_witness(tmp_path: Path) -> None:
    worker = FakeWorker()
    original_reset, original_collect = worker._reset, worker._collect_current_tactile
    path = tmp_path / "unused.jsonl"
    install_availability_protocol(worker, "required", decoder=decode, witness_path=path)
    assert worker._reset is original_reset
    assert worker._collect_current_tactile is original_collect
    assert not path.exists()


@pytest.mark.parametrize(
    "wire_message",
    [
        {},
        {PROTOCOL_KEY: "zero_fill_v1"},
        {PROTOCOL_KEY: "native_missing_v1", "observation/extra_tactile": frame(1)},
        {PROTOCOL_KEY: "native_missing_v1", "observation/left_tactile": None},
    ],
)
def test_protocol_mismatch_unknown_slot_and_null_fail_before_state_change(
    wire_message: dict[str, object],
) -> None:
    worker = FakeWorker()
    install_availability_protocol(worker, "native_missing_v1", decoder=decode)
    with pytest.raises(ValueError):
        worker.predict(wire_message)
    assert worker._baseline is None


def test_witness_creation_never_clobbers(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text("preserve")
    with pytest.raises(FileExistsError):
        install_availability_protocol(
            FakeWorker(),
            "native_missing_v1",
            decoder=decode,
            witness_path=path,
        )
    assert path.read_text() == "preserve"
