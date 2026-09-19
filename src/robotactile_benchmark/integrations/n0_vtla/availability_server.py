"""Explicit missing-input overlay for the pinned N0-VTLA server.

The upstream predictor remains untouched. Each slot captures its first available
image as its baseline; zero-filled images count as available in the control arm.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robotactile_benchmark.contracts import Array, array_sha256

AVAILABILITY_PROTOCOLS = frozenset({"native_missing_v1", "zero_fill_v1"})
BASELINE_RULE = "first_available_per_slot_v1"
PROTOCOL_KEY = "robotactile_tactile_protocol"
_WIRE_SLOTS = ("observation/left_tactile", "observation/right_tactile")


def baseline_rule(protocol: str) -> str:
    """Describe whether an imputed first input can establish the baseline."""
    if protocol == "native_missing_v1":
        return BASELINE_RULE
    if protocol == "zero_fill_v1":
        return "first_delivered_including_fill_v1"
    raise ValueError(f"unsupported availability baseline protocol: {protocol}")


class AvailabilityWorker(Protocol):
    """The small mutable surface of the upstream server used by this overlay."""

    _tactile_views: list[str]
    _tactile_enabled: bool
    _baseline: dict[str, Array] | None
    _collect_current_tactile: Callable[[Mapping[str, object]], dict[str, Array]]
    _reset: Callable[[], dict[str, Any]]


def install_availability_protocol(
    worker: AvailabilityWorker,
    protocol: str,
    *,
    decoder: Callable[[object], Array],
    witness_path: Path | None = None,
) -> None:
    """Install an opt-in collector, preserving the existing reset/RNG behavior.

    ``required`` is a strict no-op. Missing native views must be omitted, not
    serialized as null. The witness file is exclusively created and only appended.
    """
    if protocol == "required":
        return
    if protocol not in AVAILABILITY_PROTOCOLS:
        raise ValueError(f"unsupported N0-VTLA tactile protocol: {protocol}")
    if not worker._tactile_enabled or len(worker._tactile_views) != 2:
        raise ValueError("availability overlay requires two enabled tactile views")
    if len(set(worker._tactile_views)) != 2:
        raise ValueError("availability overlay requires distinct tactile views")
    if worker._baseline is not None:
        raise ValueError(
            "availability overlay must be installed before first prediction"
        )
    if witness_path is not None:
        witness_path.parent.mkdir(parents=True, exist_ok=True)
        with witness_path.open("x", encoding="utf-8"):
            pass
    views = tuple(worker._tactile_views)
    upstream_reset = worker._reset
    first_query: dict[str, int] = {}
    query_index = 0
    episode_index = 0

    def witness(event: dict[str, object]) -> None:
        if witness_path is not None:
            with witness_path.open("a", encoding="utf-8") as stream:
                json.dump(
                    {
                        "protocol": protocol,
                        "baseline_rule": baseline_rule(protocol),
                        "episode_index": episode_index,
                        **event,
                    },
                    stream,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")

    def collect(message: Mapping[str, object]) -> dict[str, Array]:
        nonlocal query_index
        if message.get(PROTOCOL_KEY) != protocol:
            raise ValueError("N0-VTLA wire tactile protocol does not match server")
        tactile_keys = {
            key
            for key in message
            if key.startswith("observation/") and key.endswith("_tactile")
        }
        if tactile_keys - set(_WIRE_SLOTS):
            raise ValueError("unknown N0-VTLA tactile wire slot")
        if protocol == "zero_fill_v1" and tactile_keys != set(_WIRE_SLOTS):
            raise ValueError("zero_fill_v1 requires two supplied tactile arrays")
        current: dict[str, Array] = {}
        available: list[str] = []
        for key, view, slot in zip(_WIRE_SLOTS, views, ("left", "right")):
            if key not in message:
                continue
            value = message[key]
            if value is None:
                raise ValueError("missing tactile views must omit their wire key")
            if protocol == "zero_fill_v1" and not isinstance(value, (np.ndarray, list)):
                raise ValueError("zero_fill_v1 requires numeric tactile arrays")
            image = decoder(value)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
                or min(image.shape) < 1
            ):
                raise ValueError("decoded tactile view must be nonempty uint8 HWC RGB")
            current[view] = image
            available.append(slot)
        # Decode the whole request before changing causal history, so errors cannot
        # establish a partial baseline. Upstream _predict will reuse this mapping.
        established = []
        if current:
            if worker._baseline is None:
                worker._baseline = {}
            for slot, view in zip(("left", "right"), views):
                if view in current and view not in worker._baseline:
                    worker._baseline[view] = current[view].copy()
                    first_query[slot] = query_index
                    established.append(slot)
        witness(
            {
                "event": "query",
                "query_index": query_index,
                "stage": "pre_predict_tactile_collection",
                "available_slots": available,
                "missing_slots": [
                    slot for slot in ("left", "right") if slot not in available
                ],
                "baseline_established_slots": established,
                "baseline_first_query": dict(first_query),
                "received_slots": {
                    slot: {
                        "array_sha256": array_sha256(current[view]),
                        "shape": list(current[view].shape),
                        "dtype": str(current[view].dtype),
                        "max_abs": int(np.max(current[view])),
                    }
                    for slot, view in zip(("left", "right"), views)
                    if view in current
                },
                "baseline_sha256": {
                    slot: array_sha256(worker._baseline[view])
                    for slot, view in zip(("left", "right"), views)
                    if worker._baseline is not None and view in worker._baseline
                },
            }
        )
        query_index += 1
        return current

    def reset() -> dict[str, Any]:
        nonlocal query_index, episode_index
        result = upstream_reset()
        worker._baseline = None
        first_query.clear()
        query_index = 0
        episode_index += 1
        witness({"event": "reset"})
        return result

    worker._collect_current_tactile = collect
    worker._reset = reset
