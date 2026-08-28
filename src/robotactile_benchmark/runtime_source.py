"""Shared immutable source identity for N0-TWAM x UniVTAC evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.provenance import load_integration_lock
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.resources import load_source_manifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _commit_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character commit SHA")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def current_integrations_lock_sha256() -> str:
    """Return the canonical full-lock identity shipped by this runtime."""

    lock = load_integration_lock()
    return sha256_bytes(canonical_json_bytes(lock.to_dict()))


def current_source_manifest_sha256() -> str:
    """Return the source manifest identity visible to this installed runtime."""

    return hashlib.sha256(load_source_manifest().encode("utf-8")).hexdigest()


def current_n0_input_profile_sha256() -> str:
    """Return the exact live N0 input-profile identity."""

    return canonical_hash(N0_LIVE_UNIVTAC_INPUT_PROFILE.to_dict())


@dataclass(frozen=True)
class RuntimeSourceBinding:
    """Exact code, wheel, external source, model, and action identities."""

    robotactile_source_manifest_sha256: str
    robotactile_wheel_sha256: str
    integrations_lock_sha256: str
    univtac_source_commit: str
    n0_source_commit: str
    checkpoint_sha256: str
    config_sha256: str
    normalizer_sha256: str
    serve_bundle_sha256: str
    prompt_manifest_sha256: str
    input_profile_sha256: str
    action_execution_contract: str
    native_step_contract: str

    def __post_init__(self) -> None:
        for name in (
            "robotactile_source_manifest_sha256",
            "robotactile_wheel_sha256",
            "integrations_lock_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "normalizer_sha256",
            "serve_bundle_sha256",
            "prompt_manifest_sha256",
            "input_profile_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("univtac_source_commit", "n0_source_commit"):
            object.__setattr__(
                self,
                name,
                _commit_sha(getattr(self, name), name),
            )
        for name in ("action_execution_contract", "native_step_contract"):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name),
            )

    def verify_against_current_runtime(self) -> None:
        """Fail unless this historical binding matches the active runtime.

        Parsing is deliberately checkout-independent so archived evidence remains
        readable.  Builders and live execution entry points call this explicit
        verification method when they need to assert the active installation.
        """

        if self.robotactile_source_manifest_sha256 != current_source_manifest_sha256():
            raise ValueError("runtime source binding source manifest mismatch")
        lock = load_integration_lock()
        if self.integrations_lock_sha256 != current_integrations_lock_sha256():
            raise ValueError("runtime source binding integrations lock mismatch")
        if self.univtac_source_commit != lock.by_id("univtac").commit_sha:
            raise ValueError("runtime source binding UniVTAC commit mismatch")
        if self.n0_source_commit != lock.by_id("n0_twam").commit_sha:
            raise ValueError("runtime source binding N0-TWAM commit mismatch")
        if (
            self.input_profile_sha256 != current_n0_input_profile_sha256()
            or self.action_execution_contract
            != N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
            or self.native_step_contract != FIXED_NATIVE_STEP_CONTRACT
        ):
            raise ValueError("runtime source binding execution contract mismatch")

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "RuntimeSourceBinding":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("runtime source binding fields mismatch")
        if any(not isinstance(value[name], str) for name in fields):
            raise TypeError("runtime source binding values must be strings")
        document = cast(Mapping[str, str], value)
        return cls(**{name: document[name] for name in fields})


__all__ = [
    "RuntimeSourceBinding",
    "current_integrations_lock_sha256",
    "current_n0_input_profile_sha256",
    "current_source_manifest_sha256",
]
