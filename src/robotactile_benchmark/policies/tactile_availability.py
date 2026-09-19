"""Explicit model-side handling of structurally missing tactile observations.

Fault delivery remains absent. Zero filling is a policy input adaptation, never
a replacement of evaluator records or evidence that the sensor delivered pixels.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.contracts import ObservationRecord


class TactileAvailabilityMode(str, Enum):
    REQUIRED = "required"
    NATIVE_MISSING = "native_missing_v1"
    ZERO_FILL = "zero_fill_v1"


RGBShape = Tuple[int, int, int]


def normalize_zero_shape(value: object) -> Optional[RGBShape]:
    """Require a declared native shape, including for absence at the first query."""
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(item) is not int or item < 1 for item in value)
        or value[2] != 3
    ):
        raise ValueError("tactile_zero_shape must be positive integer [H, W, 3]")
    return (value[0], value[1], value[2])


def validate_availability_config(
    mode: TactileAvailabilityMode, shape: Optional[RGBShape], model: str
) -> None:
    if (mode is TactileAvailabilityMode.ZERO_FILL) != (shape is not None):
        raise ValueError("tactile_zero_shape is required only for zero_fill_v1")
    if mode is TactileAvailabilityMode.NATIVE_MISSING and model != "n0_vtla":
        raise ValueError("native_missing_v1 is currently implemented for N0-VTLA only")
    if mode is TactileAvailabilityMode.ZERO_FILL and model not in {
        "act",
        "n0",
        "n0_twam",
        "n0_vtla",
        "ftp1_policy",
        "dream_tac",
    }:
        raise ValueError("zero_fill_v1 requires a supported tactile policy")


def zero_fill_observation(
    observation: ObservationRecord, shape: RGBShape
) -> ObservationRecord:
    """Fill only missing payloads in raw uint8 RGB space, without a clean oracle."""
    output = observation
    for sensor in observation.tactile:
        if sensor.payload is not None:
            if sensor.payload.shape != shape:
                raise ValueError(
                    "tactile_zero_shape differs from delivered image shape"
                )
            continue
        output = output.replace_sensor(
            replace(
                sensor, payload=np.zeros(shape, dtype=np.uint8), payload_present=True
            )
        )
    return output


class ZeroFillTactilePolicy:
    """Compatibility wrapper for a fixed-payload policy; no new model inputs."""

    def __init__(
        self, identity: PolicyIdentity, policy: ClosedLoopPolicy, shape: RGBShape
    ) -> None:
        if not identity.consumes_tactile or not identity.supports_structural_absence:
            raise ValueError("zero-fill wrapper must explicitly accept missing streams")
        if policy.identity != replace(identity, supports_structural_absence=False):
            raise ValueError("zero-fill inner policy identity mismatch")
        if normalize_zero_shape(shape) is None:
            raise ValueError("zero-fill requires a shape")
        self.identity = identity
        self._policy = policy
        self._shape = shape

    def reset(self, context: PolicyEpisodeContext) -> None:
        self._policy.reset(context)

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        return self._policy.infer(zero_fill_observation(observation, self._shape))

    def commit(self, execution: PolicyExecution) -> None:
        # The inner adapter also sees its own adapted inputs at commit boundaries.
        self._policy.commit(
            replace(
                execution,
                delivered_observations=tuple(
                    zero_fill_observation(obs, self._shape)
                    for obs in execution.delivered_observations
                ),
            )
        )

    def abort(self, reason_code: str) -> None:
        self._policy.abort(reason_code)

    def close(self) -> None:
        self._policy.close()
