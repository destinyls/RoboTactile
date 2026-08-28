"""Immutable contracts for frozen clean-only evaluation campaigns."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Tuple, cast

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACTaskRegistry,
    UniVTACTaskSpec,
)
from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.clean_baseline.seeds import (
    expected_clean_seed_pairs,
    univtac_task_seed_start,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LivePolicyKind

CLEAN_CAMPAIGN_SEMANTIC_VERSION = "1.0"
CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION = "2.0"
CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION = "1.0"
CLEAN_SEED_DERIVATION = "sha256_signed31_v1"
CLEAN_UNIVTAC_SEED_DERIVATION = "univtac_consecutive_task_seed_v1"
CLEAN_SIMULATOR_SEED_ROLE = "initial_seed"
CLEAN_POLICY_SEED_ROLE = "exogenous_seed"
CLEAN_BASELINE_EVIDENCE_LEVEL = (
    "derived_from_verified_unqualified_clean_live_artifacts_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUPPORTED_CAMPAIGN_VERSIONS = frozenset(
    {CLEAN_CAMPAIGN_SEMANTIC_VERSION, CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION}
)
_CRASH_AS_FAILURE = "crash_counted_as_failure_v1"
_REPLACE_EXCEPTION = "replace_exception_until_target_valid_v1"
_SEPARATE_EXOGENOUS = "separate_exogenous_v1"
_SAME_AS_TASK_SEED = "same_as_task_seed_v1"


class CleanCampaignProtocol(str, Enum):
    """Frozen sampling tiers with explicit paper-claim eligibility."""

    DIAGNOSTIC = "diagnostic_v1"
    PILOT = "pilot_v1"
    PAPER = "paper_v1"

    @property
    def minimum_trials_per_task(self) -> int:
        return {
            CleanCampaignProtocol.DIAGNOSTIC: 1,
            CleanCampaignProtocol.PILOT: 10,
            CleanCampaignProtocol.PAPER: 100,
        }[self]

    @property
    def allows_paper_claim(self) -> bool:
        return self is CleanCampaignProtocol.PAPER


class CleanCampaignError(ValueError):
    """A campaign manifest, artifact link, or summary failed closed."""


@dataclass(frozen=True)
class CleanCampaignSamplingSpec:
    """Frozen candidate sampling and exception-replacement contract."""

    seed_protocol: str
    exception_handling: str
    target_valid_trials_per_task: int
    candidate_trials_per_task: int
    official_eval_seed: int
    task_seed_start: int
    policy_seed_mode: str

    def __post_init__(self) -> None:
        if self.seed_protocol not in {
            CLEAN_SEED_DERIVATION,
            CLEAN_UNIVTAC_SEED_DERIVATION,
        }:
            raise CleanCampaignError("unsupported clean sampling seed protocol")
        if self.exception_handling not in {_CRASH_AS_FAILURE, _REPLACE_EXCEPTION}:
            raise CleanCampaignError("unsupported clean exception handling")
        target = require_integer(
            self.target_valid_trials_per_task,
            "target_valid_trials_per_task",
            minimum=1,
        )
        candidate = require_integer(
            self.candidate_trials_per_task,
            "candidate_trials_per_task",
            minimum=1,
        )
        if candidate < target:
            raise CleanCampaignError(
                "candidate_trials_per_task cannot be below the valid target"
            )
        eval_seed = require_integer(self.official_eval_seed, "official_eval_seed")
        seed_start = require_integer(self.task_seed_start, "task_seed_start")
        if self.policy_seed_mode not in {
            _SEPARATE_EXOGENOUS,
            _SAME_AS_TASK_SEED,
        }:
            raise CleanCampaignError("unsupported clean policy seed mode")
        if self.seed_protocol == CLEAN_UNIVTAC_SEED_DERIVATION:
            if (
                self.exception_handling != _REPLACE_EXCEPTION
                or self.policy_seed_mode != _SAME_AS_TASK_SEED
                or seed_start != univtac_task_seed_start(eval_seed)
            ):
                raise CleanCampaignError(
                    "UniVTAC sampling requires replacement, shared task seeds, "
                    "and the official seed start"
                )
        elif (
            self.exception_handling != _CRASH_AS_FAILURE
            or self.policy_seed_mode != _SEPARATE_EXOGENOUS
        ):
            raise CleanCampaignError(
                "legacy SHA sampling requires failure counting and separate seeds"
            )
        object.__setattr__(self, "target_valid_trials_per_task", target)
        object.__setattr__(self, "candidate_trials_per_task", candidate)
        object.__setattr__(self, "official_eval_seed", eval_seed)
        object.__setattr__(self, "task_seed_start", seed_start)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> CleanCampaignSamplingSpec:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CleanCampaignError("clean campaign sampling fields mismatch")
        return cls(**cast(Any, dict(value)))


def require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CleanCampaignError(f"{name} must be a non-empty string")
    return value


def require_identifier(value: object, name: str) -> str:
    normalized = require_nonempty(value, name)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise CleanCampaignError(f"{name} must be a safe identifier")
    return normalized


def require_sha256(value: object, name: str) -> str:
    normalized = require_nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise CleanCampaignError(f"{name} must be a lowercase SHA256")
    return normalized


def require_integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CleanCampaignError(f"{name} must be an integer")
    if value < minimum:
        raise CleanCampaignError(f"{name} must be at least {minimum}")
    return value


def require_relative_path(value: object, name: str, prefix: str) -> str:
    normalized = require_nonempty(value, name)
    if "\\" in normalized:
        raise CleanCampaignError(f"{name} must use POSIX separators")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or path.as_posix() != normalized
        or not path.parts
        or path.parts[0] != prefix
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CleanCampaignError(
            f"{name} must be a canonical relative path below {prefix}/"
        )
    return normalized


@dataclass(frozen=True)
class CleanCampaignTrialSpec:
    """One expected clean request and its future strict artifact location."""

    ordinal: int
    task: str
    initial_seed: int
    exogenous_seed: int
    base_system_id: str
    dataset_sha256: str
    base_system_manifest_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    trial_manifest_sha256: str
    pair_key: str
    run_spec_sha256: str
    run_content_sha256: str
    request_file_sha256: str
    request_relpath: str
    artifact_relpath: str
    max_control_cycles: int
    max_observation_steps: int
    execute_action_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordinal", require_integer(self.ordinal, "ordinal"))
        for name in ("task", "base_system_id"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name))
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(self, name, require_integer(getattr(self, name), name))
        for name in (
            "dataset_sha256",
            "base_system_manifest_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "trial_manifest_sha256",
            "pair_key",
            "run_spec_sha256",
            "run_content_sha256",
            "request_file_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "request_relpath",
            require_relative_path(self.request_relpath, "request_relpath", "requests"),
        )
        object.__setattr__(
            self,
            "artifact_relpath",
            require_relative_path(
                self.artifact_relpath, "artifact_relpath", "artifacts"
            ),
        )
        for name in (
            "max_control_cycles",
            "max_observation_steps",
            "execute_action_steps",
        ):
            object.__setattr__(
                self, name, require_integer(getattr(self, name), name, minimum=1)
            )
        if self.max_control_cycles > self.max_observation_steps:
            raise CleanCampaignError(
                "max_control_cycles cannot exceed max_observation_steps"
            )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> CleanCampaignTrialSpec:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CleanCampaignError("clean campaign trial fields mismatch")
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True)
class CleanCampaignManifest:
    """One precommitted clean campaign over exact requests and seed identities."""

    campaign_id: str
    protocol_id: CleanCampaignProtocol
    policy_kind: LivePolicyKind
    task_registry_sha256: str
    master_seed: int
    seed_derivation: str
    simulator_seed_role: str
    policy_seed_role: str
    planned_trial_count: int
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    trials: Tuple[CleanCampaignTrialSpec, ...]
    semantic_version: str = CLEAN_CAMPAIGN_SEMANTIC_VERSION
    sampling: CleanCampaignSamplingSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", require_identifier(self.campaign_id, "campaign_id")
        )
        protocol = (
            self.protocol_id
            if isinstance(self.protocol_id, CleanCampaignProtocol)
            else CleanCampaignProtocol(self.protocol_id)
        )
        policy = (
            self.policy_kind
            if isinstance(self.policy_kind, LivePolicyKind)
            else LivePolicyKind(self.policy_kind)
        )
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "policy_kind", policy)
        semantic_version = require_nonempty(self.semantic_version, "semantic_version")
        if semantic_version not in _SUPPORTED_CAMPAIGN_VERSIONS:
            raise CleanCampaignError("unsupported clean campaign semantic version")
        sampling = self.sampling
        if semantic_version == CLEAN_CAMPAIGN_SEMANTIC_VERSION:
            if sampling is not None:
                raise CleanCampaignError("clean campaign v1 cannot carry sampling")
        elif type(sampling) is not CleanCampaignSamplingSpec:
            raise CleanCampaignError(
                "clean campaign v2 requires a typed sampling contract"
            )
        registry = load_registry()
        registry_hash = require_sha256(
            self.task_registry_sha256, "task_registry_sha256"
        )
        if registry_hash != registry.resource_sha256:
            raise CleanCampaignError("task registry hash does not match the package")
        master_seed = require_integer(self.master_seed, "master_seed")
        expected_seed_derivation = (
            CLEAN_SEED_DERIVATION if sampling is None else sampling.seed_protocol
        )
        if self.seed_derivation != expected_seed_derivation:
            raise CleanCampaignError(
                "clean campaign seed derivation does not match its sampling contract"
            )
        if self.simulator_seed_role != CLEAN_SIMULATOR_SEED_ROLE:
            raise CleanCampaignError("simulator seed role must be initial_seed")
        if self.policy_seed_role != CLEAN_POLICY_SEED_ROLE:
            raise CleanCampaignError("policy seed role must be exogenous_seed")
        planned = require_integer(
            self.planned_trial_count, "planned_trial_count", minimum=1
        )
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise CleanCampaignError("confidence_level must lie in (0, 1)")
        resamples = require_integer(
            self.bootstrap_resamples, "bootstrap_resamples", minimum=1
        )
        bootstrap_seed = require_integer(self.bootstrap_seed, "bootstrap_seed")
        trials = tuple(self.trials)
        if planned != len(trials) or any(
            type(item) is not CleanCampaignTrialSpec for item in trials
        ):
            raise CleanCampaignError(
                "planned_trial_count must cover typed clean campaign trials"
            )
        if tuple(item.ordinal for item in trials) != tuple(range(len(trials))):
            raise CleanCampaignError("clean campaign ordinals must be dense and sorted")
        self._validate_unique_trials(trials)
        self._validate_protocol(trials, registry, master_seed, sampling)
        object.__setattr__(self, "task_registry_sha256", registry_hash)
        object.__setattr__(self, "master_seed", master_seed)
        object.__setattr__(self, "planned_trial_count", planned)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "bootstrap_resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "semantic_version", semantic_version)
        object.__setattr__(self, "sampling", sampling)

    @staticmethod
    def _validate_unique_trials(
        trials: Tuple[CleanCampaignTrialSpec, ...],
    ) -> None:
        identities = {
            "trial manifest": tuple(item.trial_manifest_sha256 for item in trials),
            "pair key": tuple(item.pair_key for item in trials),
            "request path": tuple(item.request_relpath for item in trials),
            "artifact path": tuple(item.artifact_relpath for item in trials),
            "task/initial seed": tuple(
                (item.task, item.initial_seed) for item in trials
            ),
        }
        for name, values in identities.items():
            if len(values) != len(set(values)):
                raise CleanCampaignError(f"duplicate clean campaign {name}")

    def _validate_protocol(
        self,
        trials: Tuple[CleanCampaignTrialSpec, ...],
        registry: UniVTACTaskRegistry,
        master_seed: int,
        sampling: CleanCampaignSamplingSpec | None,
    ) -> None:
        known = {task.task_id: task for task in registry.tasks}
        counts = Counter(item.task for item in trials)
        if set(counts) - set(known):
            raise CleanCampaignError("clean campaign contains an unknown UniVTAC task")
        if sampling is not None:
            self._validate_sampling_protocol(
                trials, known, counts, master_seed, sampling
            )
            return
        self._validate_legacy_protocol(trials, known, counts, master_seed)

    def _validate_legacy_protocol(
        self,
        trials: Tuple[CleanCampaignTrialSpec, ...],
        known: Mapping[str, UniVTACTaskSpec],
        counts: Counter[str],
        master_seed: int,
    ) -> None:
        minimum = self.protocol_id.minimum_trials_per_task
        if any(count < minimum for count in counts.values()):
            raise CleanCampaignError(
                f"{self.protocol_id.value} requires at least {minimum} trials per task"
            )
        realized_seed_values: list[int] = []
        for task, count in counts.items():
            actual = {
                (item.initial_seed, item.exogenous_seed)
                for item in trials
                if item.task == task
            }
            expected_pairs = expected_clean_seed_pairs(
                protocol_id=self.protocol_id.value,
                master_seed=master_seed,
                task_id=task,
                trial_count=count,
            )
            if actual != expected_pairs:
                raise CleanCampaignError(
                    "realized clean seeds do not match master seed derivation"
                )
            realized_seed_values.extend(seed for pair in actual for seed in pair)
        if len(realized_seed_values) != len(set(realized_seed_values)):
            raise CleanCampaignError("clean campaign seed derivation collided")
        if self.protocol_id is CleanCampaignProtocol.PILOT and (
            set(counts) != set(known) or any(count != 10 for count in counts.values())
        ):
            raise CleanCampaignError(
                "pilot_v1 requires all eight tasks and exactly 10 trials per task"
            )
        if self.protocol_id is CleanCampaignProtocol.PAPER:
            if set(counts) != set(known):
                raise CleanCampaignError(
                    "paper_v1 requires the frozen complete eight-task registry"
                )
            for item in trials:
                expected_steps = known[item.task].action_horizon + 1
                if item.max_observation_steps != expected_steps:
                    raise CleanCampaignError(
                        "paper_v1 requires each task's complete observation horizon"
                    )

    def _validate_sampling_protocol(
        self,
        trials: Tuple[CleanCampaignTrialSpec, ...],
        known: Mapping[str, UniVTACTaskSpec],
        counts: Counter[str],
        master_seed: int,
        sampling: CleanCampaignSamplingSpec,
    ) -> None:
        target = sampling.target_valid_trials_per_task
        candidate = sampling.candidate_trials_per_task
        if any(count != candidate for count in counts.values()):
            raise CleanCampaignError(
                "clean campaign candidate inventory must be equal per task"
            )
        if target < self.protocol_id.minimum_trials_per_task:
            raise CleanCampaignError(
                f"{self.protocol_id.value} requires at least "
                f"{self.protocol_id.minimum_trials_per_task} valid trials per task"
            )
        if self.protocol_id in {
            CleanCampaignProtocol.PILOT,
            CleanCampaignProtocol.PAPER,
        } and set(counts) != set(known):
            raise CleanCampaignError(
                f"{self.protocol_id.value} requires the frozen complete eight-task registry"
            )
        if self.protocol_id is CleanCampaignProtocol.PILOT and target != 10:
            raise CleanCampaignError(
                "pilot_v1 requires exactly 10 valid trials per task"
            )
        if self.protocol_id is CleanCampaignProtocol.PAPER and (
            target != 100
            or sampling.seed_protocol != CLEAN_UNIVTAC_SEED_DERIVATION
            or sampling.exception_handling != _REPLACE_EXCEPTION
            or sampling.policy_seed_mode != _SAME_AS_TASK_SEED
        ):
            raise CleanCampaignError(
                "paper_v1 requires the official 100-valid-trial UniVTAC sampling contract"
            )
        if sampling.seed_protocol == CLEAN_UNIVTAC_SEED_DERIVATION:
            if master_seed != sampling.official_eval_seed:
                raise CleanCampaignError(
                    "master_seed must equal official_eval_seed for UniVTAC sampling"
                )
            expected_pairs = tuple(
                (
                    sampling.task_seed_start + index,
                    sampling.task_seed_start + index,
                )
                for index in range(candidate)
            )
            for task in counts:
                actual_pairs = tuple(
                    (item.initial_seed, item.exogenous_seed)
                    for item in trials
                    if item.task == task
                )
                if actual_pairs != expected_pairs:
                    raise CleanCampaignError(
                        "realized clean seeds do not match the ordered UniVTAC "
                        "task-local sampling candidate sequence"
                    )
        else:
            realized_seed_values: list[int] = []
            for task in counts:
                actual_seed_pairs = {
                    (item.initial_seed, item.exogenous_seed)
                    for item in trials
                    if item.task == task
                }
                expected = expected_clean_seed_pairs(
                    protocol_id=self.protocol_id.value,
                    master_seed=master_seed,
                    task_id=task,
                    trial_count=candidate,
                )
                if actual_seed_pairs != expected:
                    raise CleanCampaignError(
                        "realized clean seeds do not match master seed derivation"
                    )
                realized_seed_values.extend(
                    seed for pair in actual_seed_pairs for seed in pair
                )
            if len(realized_seed_values) != len(set(realized_seed_values)):
                raise CleanCampaignError("clean campaign seed derivation collided")
        if self.protocol_id is CleanCampaignProtocol.PAPER:
            for item in trials:
                if item.max_observation_steps != known[item.task].action_horizon + 1:
                    raise CleanCampaignError(
                        "paper_v1 requires each task's complete observation horizon"
                    )

    @property
    def protocol_allows_paper_claim(self) -> bool:
        return (
            self.protocol_id.allows_paper_claim
            and self.sampling is not None
            and self.sampling.seed_protocol == CLEAN_UNIVTAC_SEED_DERIVATION
            and self.sampling.exception_handling == _REPLACE_EXCEPTION
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "campaign_id": self.campaign_id,
            "protocol_id": self.protocol_id.value,
            "policy_kind": self.policy_kind.value,
            "task_registry_sha256": self.task_registry_sha256,
            "master_seed": self.master_seed,
            "seed_derivation": self.seed_derivation,
            "simulator_seed_role": self.simulator_seed_role,
            "policy_seed_role": self.policy_seed_role,
            "planned_trial_count": self.planned_trial_count,
            "confidence_level": self.confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "trials": [item.to_dict() for item in self.trials],
            "semantic_version": self.semantic_version,
        }
        if self.sampling is not None:
            document["sampling"] = self.sampling.to_dict()
        return document

    @classmethod
    def from_dict(cls, value: object) -> CleanCampaignManifest:
        if not isinstance(value, Mapping):
            raise CleanCampaignError("clean campaign manifest fields mismatch")
        version = value.get("semantic_version")
        v1_fields = set(cls.__dataclass_fields__) - {"sampling"}
        expected_fields = (
            v1_fields
            if version == CLEAN_CAMPAIGN_SEMANTIC_VERSION
            else v1_fields | {"sampling"}
        )
        if version not in _SUPPORTED_CAMPAIGN_VERSIONS or set(value) != expected_fields:
            raise CleanCampaignError("clean campaign manifest fields mismatch")
        trials = value["trials"]
        if not isinstance(trials, list):
            raise CleanCampaignError("clean campaign trials must be a list")
        sampling_value = value.get("sampling")
        return cls(
            campaign_id=value["campaign_id"],
            protocol_id=CleanCampaignProtocol(value["protocol_id"]),
            policy_kind=LivePolicyKind(value["policy_kind"]),
            task_registry_sha256=value["task_registry_sha256"],
            master_seed=value["master_seed"],
            seed_derivation=value["seed_derivation"],
            simulator_seed_role=value["simulator_seed_role"],
            policy_seed_role=value["policy_seed_role"],
            planned_trial_count=value["planned_trial_count"],
            confidence_level=value["confidence_level"],
            bootstrap_resamples=value["bootstrap_resamples"],
            bootstrap_seed=value["bootstrap_seed"],
            trials=tuple(CleanCampaignTrialSpec.from_dict(item) for item in trials),
            semantic_version=value["semantic_version"],
            sampling=(
                None
                if sampling_value is None
                else CleanCampaignSamplingSpec.from_dict(sampling_value)
            ),
        )


__all__ = [
    "CLEAN_BASELINE_EVIDENCE_LEVEL",
    "CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION",
    "CLEAN_CAMPAIGN_SEMANTIC_VERSION",
    "CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION",
    "CLEAN_POLICY_SEED_ROLE",
    "CLEAN_SEED_DERIVATION",
    "CLEAN_SIMULATOR_SEED_ROLE",
    "CLEAN_UNIVTAC_SEED_DERIVATION",
    "CleanCampaignError",
    "CleanCampaignManifest",
    "CleanCampaignProtocol",
    "CleanCampaignSamplingSpec",
    "CleanCampaignTrialSpec",
]
