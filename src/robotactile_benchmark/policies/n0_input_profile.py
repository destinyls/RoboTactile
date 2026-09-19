"""Explicit image-domain contracts for the released UniVTAC N0 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash, freeze_array


class N0SourceColorDomain(str, Enum):
    """Numerical channel domain at the RoboTactile-to-N0 boundary."""

    CHECKPOINT_PIL_RGB = "checkpoint_pil_decoded_rgb"
    UNIVTAC_SIMULATOR_RGB = "univtac_simulator_numeric_rgb"
    RETRAINED_NUMERIC_RGB = "retrained_train759_numeric_rgb"


@dataclass(frozen=True)
class N0InputProfile:
    """Content-addressable preprocessing selected by an evaluation pathway."""

    profile_id: str
    source_color_domain: N0SourceColorDomain

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or not self.profile_id
            or self.profile_id.strip() != self.profile_id
        ):
            raise ValueError("N0 input profile_id must be a non-empty string")
        if type(self.source_color_domain) is not N0SourceColorDomain:
            raise TypeError("N0 source_color_domain must be N0SourceColorDomain")

    @property
    def channel_transform(self) -> str:
        """Return the deterministic transform applied before websocket transport."""

        if self.source_color_domain is N0SourceColorDomain.UNIVTAC_SIMULATOR_RGB:
            return "reverse_rgb"
        return "identity"

    @property
    def sha256(self) -> str:
        """Return a stable hash suitable for diagnostic provenance."""

        return canonical_hash(self)

    def to_dict(self) -> dict[str, str]:
        """Return the exact JSON-domain preprocessing contract."""

        return {
            "profile_id": self.profile_id,
            "source_color_domain": self.source_color_domain.value,
            "channel_transform": self.channel_transform,
            "sha256": self.sha256,
        }


N0_LIVE_UNIVTAC_INPUT_PROFILE = N0InputProfile(
    profile_id="n0-live-univtac-isaaclab-rgb-to-checkpoint-pil-v2",
    source_color_domain=N0SourceColorDomain.UNIVTAC_SIMULATOR_RGB,
)
N0_RECORDED_CHECKPOINT_INPUT_PROFILE = N0InputProfile(
    profile_id="n0-recorded-checkpoint-pil-color-v1",
    source_color_domain=N0SourceColorDomain.CHECKPOINT_PIL_RGB,
)
N0_RETRAINED_INPUT_PROFILE = N0InputProfile(
    profile_id="n0-train759-vt-10hz-numeric-rgb-v1",
    source_color_domain=N0SourceColorDomain.RETRAINED_NUMERIC_RGB,
)


def prepare_n0_image(
    value: object,
    *,
    profile: N0InputProfile,
    name: str,
) -> Array:
    """Map one uint8 HWC image into the checkpoint's documented PIL domain."""

    if type(profile) is not N0InputProfile:
        raise TypeError("profile must be an exact N0InputProfile")
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be uint8 HWC with three channels")
    if profile.source_color_domain is N0SourceColorDomain.UNIVTAC_SIMULATOR_RGB:
        array = array[..., ::-1]
    return freeze_array(array, np.uint8)


__all__ = [
    "N0InputProfile",
    "N0SourceColorDomain",
    "N0_LIVE_UNIVTAC_INPUT_PROFILE",
    "N0_RECORDED_CHECKPOINT_INPUT_PROFILE",
    "N0_RETRAINED_INPUT_PROFILE",
    "prepare_n0_image",
]
