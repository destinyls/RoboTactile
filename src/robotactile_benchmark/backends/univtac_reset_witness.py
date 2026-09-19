"""Source-bound reset references and observations for live UniVTAC runs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Tuple, cast

import numpy as np

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_conversion import (
    ConvertedUniVTACObservation,
)
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import Array, canonical_hash

UNIVTAC_RESET_REFERENCE_SCHEMA = "univtac-reset-reference-v1"
UNIVTAC_RESET_WITNESS_SCHEMA = "univtac-reset-witness-v1"
UNIVTAC_RESET_SEMANTIC_VERSION = "1.0"
STRICT_RESET_QUALIFICATION_PROFILE = "strict_simulator_state_native_step_qpos_v1"
TRAJECTORY_RESET_QUALIFICATION_PROFILE = "trajectory_native_step_qpos_clean_gate_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class UniVTACResetReferenceError(ValueError):
    """A reset reference or its runtime witness failed strict validation."""


def _nonempty(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise UniVTACResetReferenceError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise UniVTACResetReferenceError(f"{name} must be a lowercase SHA256")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise UniVTACResetReferenceError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise UniVTACResetReferenceError(f"{name} must be non-negative")
    return result


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise UniVTACResetReferenceError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise UniVTACResetReferenceError(f"{name} must be finite and positive")
    return result


def _finite_vector(
    value: object,
    name: str,
    length: int,
) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UniVTACResetReferenceError(f"{name} must be a sequence")
    if len(value) != length:
        raise UniVTACResetReferenceError(f"{name} must have length {length}")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise UniVTACResetReferenceError(f"{name} must contain real numbers")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise UniVTACResetReferenceError(f"{name} must be finite")
        result.append(normalized)
    return tuple(result)


@dataclass(frozen=True)
class UniVTACResetReference:
    """A source-bound expected initial state for one UniVTAC seed pair."""

    task_id: str
    initial_seed: int
    exogenous_seed: int
    pair_key: str
    dataset_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    source_artifact_root_sha256: str
    source_result_sha256: str
    source_run_content_sha256: str
    expected_simulator_state_sha256: str
    expected_native_step: int
    expected_qpos8: Tuple[float, ...]
    qpos_atol: float
    schema: str = UNIVTAC_RESET_REFERENCE_SCHEMA
    semantic_version: str = UNIVTAC_RESET_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        for name in ("initial_seed", "exogenous_seed", "expected_native_step"):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        for name in (
            "pair_key",
            "dataset_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "source_artifact_root_sha256",
            "source_result_sha256",
            "source_run_content_sha256",
            "expected_simulator_state_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "expected_qpos8",
            _finite_vector(self.expected_qpos8, "expected_qpos8", 8),
        )
        object.__setattr__(
            self, "qpos_atol", _positive_finite(self.qpos_atol, "qpos_atol")
        )
        if self.schema != UNIVTAC_RESET_REFERENCE_SCHEMA:
            raise UniVTACResetReferenceError("reset reference schema mismatch")
        if self.semantic_version != UNIVTAC_RESET_SEMANTIC_VERSION:
            raise UniVTACResetReferenceError(
                "reset reference semantic version mismatch"
            )

    @property
    def sha256(self) -> str:
        """Return the canonical content hash of the complete reference."""

        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a new JSON-safe representation."""

        return {
            "task_id": self.task_id,
            "initial_seed": self.initial_seed,
            "exogenous_seed": self.exogenous_seed,
            "pair_key": self.pair_key,
            "dataset_sha256": self.dataset_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "source_artifact_root_sha256": self.source_artifact_root_sha256,
            "source_result_sha256": self.source_result_sha256,
            "source_run_content_sha256": self.source_run_content_sha256,
            "expected_simulator_state_sha256": (self.expected_simulator_state_sha256),
            "expected_native_step": self.expected_native_step,
            "expected_qpos8": list(self.expected_qpos8),
            "qpos_atol": self.qpos_atol,
            "schema": self.schema,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> UniVTACResetReference:
        """Load only the exact JSON object emitted by :meth:`to_dict`."""

        if not isinstance(value, Mapping):
            raise UniVTACResetReferenceError("reset reference must be a mapping")
        expected_fields = set(cls.__dataclass_fields__)
        if set(value) != expected_fields:
            raise UniVTACResetReferenceError("reset reference fields mismatch")
        document = dict(value)
        qpos8 = document["expected_qpos8"]
        if not isinstance(qpos8, list):
            raise UniVTACResetReferenceError("expected_qpos8 must be a JSON array")
        document["expected_qpos8"] = tuple(qpos8)
        return cls(**cast(Any, document))


def _actual_vector(value: Array, name: str, length: int) -> Array:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (length,)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
    ):
        raise UniVTACResetReferenceError(
            f"converted {name} must be a finite floating [{length}] array"
        )
    return cast(Array, np.asarray(value, dtype=np.float32))


def _validate_context_observation(
    context: PolicyEpisodeContext,
    converted: ConvertedUniVTACObservation,
) -> None:
    observation = converted.record.observation
    if observation.episode_id != context.episode_id:
        raise UniVTACResetReferenceError("converted episode_id does not match context")
    if observation.task != context.task:
        raise UniVTACResetReferenceError("converted task does not match context")
    if observation.seed != context.initial_seed:
        raise UniVTACResetReferenceError("converted seed does not match context")


def _validate_reference_identity(
    context: PolicyEpisodeContext,
    reference: UniVTACResetReference,
) -> None:
    if reference.task_id != context.task:
        raise UniVTACResetReferenceError("reference task does not match context")
    if reference.initial_seed != context.initial_seed:
        raise UniVTACResetReferenceError(
            "reference initial_seed does not match context"
        )
    if reference.exogenous_seed != context.exogenous_seed:
        raise UniVTACResetReferenceError(
            "reference exogenous_seed does not match context"
        )


def build_univtac_reset_witness(
    context: PolicyEpisodeContext,
    converted: ConvertedUniVTACObservation,
    reference: UniVTACResetReference | None = None,
    *,
    require_simulator_state_match: bool = True,
) -> dict[str, object]:
    """Record one initial state and optionally assess it against a reference.

    An observation without a reference deliberately omits ``reset_viable``.  This
    prevents a diagnostic capture from being upgraded into a reset qualification.
    """

    if not isinstance(context, PolicyEpisodeContext):
        raise TypeError("context must be a PolicyEpisodeContext")
    if not isinstance(converted, ConvertedUniVTACObservation):
        raise TypeError("converted must be a ConvertedUniVTACObservation")
    if reference is not None and not isinstance(reference, UniVTACResetReference):
        raise TypeError("reference must be a UniVTACResetReference or None")
    if type(require_simulator_state_match) is not bool:
        raise TypeError("require_simulator_state_match must be bool")
    if reference is None and not require_simulator_state_match:
        raise ValueError(
            "simulator-state matching can only be relaxed for a reset reference"
        )
    _validate_context_observation(context, converted)
    native_step = _nonnegative_integer(converted.native_step_id, "native_step_id")
    canonical_joint9 = _actual_vector(converted.canonical_joint9, "canonical_joint9", 9)
    qpos8 = _actual_vector(converted.model_visible_qpos8, "qpos8", 8)
    proprio8 = _actual_vector(
        converted.record.observation.proprio,
        "model_visible_proprio8",
        8,
    )
    qpos8_is_model_visible = bool(np.array_equal(proprio8, qpos8))
    if context.action_spec == QPOS8_ACTION_SPEC and not qpos8_is_model_visible:
        raise UniVTACResetReferenceError(
            "converted qpos8 does not match model-visible proprioception"
        )
    if reference is not None and context.action_spec != QPOS8_ACTION_SPEC:
        raise UniVTACResetReferenceError(
            "qpos8 reset references require the qpos8 action contract"
        )
    result: dict[str, object] = {
        "schema": UNIVTAC_RESET_WITNESS_SCHEMA,
        "semantic_version": UNIVTAC_RESET_SEMANTIC_VERSION,
        "episode_id": context.episode_id,
        "task_id": context.task,
        "initial_seed": context.initial_seed,
        "exogenous_seed": context.exogenous_seed,
        "native_step": native_step,
        "canonical_joint9": [float(item) for item in canonical_joint9],
        "canonical_joint9_sha256": canonical_hash(canonical_joint9),
        "qpos8": [float(item) for item in qpos8],
        "qpos8_sha256": canonical_hash(qpos8),
        "model_visible_proprio8": [float(item) for item in proprio8],
        "model_visible_proprio8_sha256": canonical_hash(proprio8),
        "qpos8_is_model_visible_proprio": qpos8_is_model_visible,
        "simulator_state_sha256": _sha256(
            converted.simulator_state_sha256, "simulator_state_sha256"
        ),
        "joint_reorder_witness_sha256": _sha256(
            converted.joint_reorder_witness_sha256,
            "joint_reorder_witness_sha256",
        ),
        "clean_record_sha256": _sha256(
            converted.record.clean_record_sha256, "clean_record_sha256"
        ),
        "delivered_record_sha256": _sha256(
            converted.record.delivered_record_sha256, "delivered_record_sha256"
        ),
        "reference_checks": {"reference_provided": False},
    }
    if reference is None:
        return result

    _validate_reference_identity(context, reference)
    expected_qpos8 = np.asarray(reference.expected_qpos8, dtype=np.float32)
    absolute_error = np.abs(qpos8 - expected_qpos8)
    max_error = float(np.max(absolute_error))
    simulator_state_match = (
        converted.simulator_state_sha256 == reference.expected_simulator_state_sha256
    )
    native_step_match = native_step == reference.expected_native_step
    qpos8_match = bool(np.all(absolute_error <= reference.qpos_atol))
    result["reference_checks"] = {
        "reference_provided": True,
        "reference_sha256": reference.sha256,
        "pair_key": reference.pair_key,
        "dataset_sha256": reference.dataset_sha256,
        "checkpoint_sha256": reference.checkpoint_sha256,
        "config_sha256": reference.config_sha256,
        "source_artifact_root_sha256": reference.source_artifact_root_sha256,
        "source_result_sha256": reference.source_result_sha256,
        "source_run_content_sha256": reference.source_run_content_sha256,
        "expected_simulator_state_sha256": (reference.expected_simulator_state_sha256),
        "task_id_match": True,
        "initial_seed_match": True,
        "exogenous_seed_match": True,
        "simulator_state_match": simulator_state_match,
        "simulator_state_match_required": require_simulator_state_match,
        "qualification_profile": (
            STRICT_RESET_QUALIFICATION_PROFILE
            if require_simulator_state_match
            else TRAJECTORY_RESET_QUALIFICATION_PROFILE
        ),
        "native_step_match": native_step_match,
        "qpos8_atol_match": qpos8_match,
        "qpos8_atol": reference.qpos_atol,
        "qpos8_max_abs_error": max_error,
    }
    state_qualified = simulator_state_match or not require_simulator_state_match
    result["reset_viable"] = state_qualified and native_step_match and qpos8_match
    return result
