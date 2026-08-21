"""Immutable contracts for one UniVTAC rest-reference calibration artifact."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.rest_references import ReferenceSplit, RestReferenceBundle

CALIBRATION_SEMANTIC_VERSION = "1.0"
NO_CONTACT_PREDICATE_ID = "univtac-depth-hysteresis-free-consecutive-v1"
CALIBRATION_EVIDENCE_LEVEL = "unqualified_univtac_no_contact_calibration_v1"
REST_REFERENCE_PATH = "rest_reference.json"
VALIDATION_PATH = "no_contact_validation.json"
ROOT_RECEIPT_PATH = "root_receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RestReferenceArtifactError(ValueError):
    """Raised when calibration evidence is incomplete or inconsistent."""


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RestReferenceArtifactError(f"{name} must be a lowercase SHA256")
    return value


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RestReferenceArtifactError(f"{name} must be an integer >= {minimum}")
    return value


def _slot_mapping(value: object, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(SENSOR_SLOTS):
        raise RestReferenceArtifactError(f"{name} must cover left and right")
    result = {
        slot: require_sha256(value[slot], f"{name}.{slot}") for slot in SENSOR_SLOTS
    }
    return MappingProxyType(result)


@dataclass(frozen=True)
class NoContactValidationReceipt:
    """Evidence that one selected record lies in a frozen no-contact run."""

    source_live_artifact_root_sha256: str
    source_trial_manifest_sha256: str
    dataset_split: ReferenceSplit
    split_manifest_sha256: str
    task: str
    episode_id: str
    initial_seed: int
    no_contact_predicate_id: str
    phase_tracker_sha256: str
    minimum_consecutive_free_records: int
    qualified_start_index: int
    qualified_stop_index: int
    selected_step_index: int
    qualified_clean_record_sha256s: Tuple[str, ...]
    qualified_record_ids: Mapping[str, str]
    calibration_sha256: Mapping[str, str]
    selected_payload_sha256: Mapping[str, str]
    semantic_version: str = CALIBRATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_live_artifact_root_sha256",
            "source_trial_manifest_sha256",
            "split_manifest_sha256",
            "phase_tracker_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        if not isinstance(self.dataset_split, ReferenceSplit):
            object.__setattr__(
                self, "dataset_split", ReferenceSplit(self.dataset_split)
            )
        if self.dataset_split is ReferenceSplit.SYNTHETIC_DEVELOPMENT:
            raise RestReferenceArtifactError(
                "production calibration cannot use synthetic-development"
            )
        if not self.task or not self.episode_id:
            raise RestReferenceArtifactError("task and episode_id must be non-empty")
        if self.no_contact_predicate_id != NO_CONTACT_PREDICATE_ID:
            raise RestReferenceArtifactError("no-contact predicate identity mismatch")
        initial_seed = _strict_int(self.initial_seed, "initial_seed")
        minimum = _strict_int(
            self.minimum_consecutive_free_records,
            "minimum_consecutive_free_records",
            1,
        )
        start = _strict_int(self.qualified_start_index, "qualified_start_index")
        stop = _strict_int(self.qualified_stop_index, "qualified_stop_index", 1)
        selected = _strict_int(self.selected_step_index, "selected_step_index")
        record_hashes = tuple(
            require_sha256(value, "qualified clean record")
            for value in self.qualified_clean_record_sha256s
        )
        if stop <= start or len(record_hashes) != stop - start:
            raise RestReferenceArtifactError("qualified FREE interval is inconsistent")
        if len(record_hashes) < minimum or not start <= selected < stop:
            raise RestReferenceArtifactError(
                "selected step is outside qualified FREE run"
            )
        object.__setattr__(self, "initial_seed", initial_seed)
        object.__setattr__(self, "minimum_consecutive_free_records", minimum)
        object.__setattr__(self, "qualified_start_index", start)
        object.__setattr__(self, "qualified_stop_index", stop)
        object.__setattr__(self, "selected_step_index", selected)
        object.__setattr__(self, "qualified_clean_record_sha256s", record_hashes)
        for name in (
            "qualified_record_ids",
            "calibration_sha256",
            "selected_payload_sha256",
        ):
            object.__setattr__(self, name, _slot_mapping(getattr(self, name), name))
        if self.semantic_version != CALIBRATION_SEMANTIC_VERSION:
            raise RestReferenceArtifactError("calibration semantic version mismatch")

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_live_artifact_root_sha256": self.source_live_artifact_root_sha256,
            "source_trial_manifest_sha256": self.source_trial_manifest_sha256,
            "dataset_split": self.dataset_split.value,
            "split_manifest_sha256": self.split_manifest_sha256,
            "task": self.task,
            "episode_id": self.episode_id,
            "initial_seed": self.initial_seed,
            "no_contact_predicate_id": self.no_contact_predicate_id,
            "phase_tracker_sha256": self.phase_tracker_sha256,
            "minimum_consecutive_free_records": self.minimum_consecutive_free_records,
            "qualified_start_index": self.qualified_start_index,
            "qualified_stop_index": self.qualified_stop_index,
            "selected_step_index": self.selected_step_index,
            "qualified_clean_record_sha256s": list(self.qualified_clean_record_sha256s),
            "qualified_record_ids": dict(self.qualified_record_ids),
            "calibration_sha256": dict(self.calibration_sha256),
            "selected_payload_sha256": dict(self.selected_payload_sha256),
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> NoContactValidationReceipt:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise RestReferenceArtifactError("validation receipt fields mismatch")
        values = dict(value)
        hashes = values["qualified_clean_record_sha256s"]
        if not isinstance(hashes, list):
            raise RestReferenceArtifactError("qualified record hashes must be a list")
        values["qualified_clean_record_sha256s"] = tuple(hashes)
        return cls(**values)


@dataclass(frozen=True)
class RestReferenceArtifactMember:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.path not in {REST_REFERENCE_PATH, VALIDATION_PATH}:
            raise RestReferenceArtifactError("unknown calibration artifact member")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "member"))
        object.__setattr__(self, "size_bytes", _strict_int(self.size_bytes, "size", 1))

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: object) -> RestReferenceArtifactMember:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise RestReferenceArtifactError("member fields mismatch")
        return cls(**value)


@dataclass(frozen=True)
class RestReferenceArtifactRootReceipt:
    rest_reference_sha256: str
    no_contact_validation_sha256: str
    source_live_artifact_root_sha256: str
    members: Tuple[RestReferenceArtifactMember, ...]
    evidence_level: str = CALIBRATION_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = CALIBRATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "rest_reference_sha256",
            "no_contact_validation_sha256",
            "source_live_artifact_root_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        members = tuple(self.members)
        if tuple(item.path for item in members) != (
            VALIDATION_PATH,
            REST_REFERENCE_PATH,
        ):
            raise RestReferenceArtifactError("calibration member inventory mismatch")
        if self.evidence_level != CALIBRATION_EVIDENCE_LEVEL:
            raise RestReferenceArtifactError("calibration evidence level mismatch")
        if self.simulator_qualification_claimed is not False:
            raise RestReferenceArtifactError("calibration cannot claim qualification")
        if self.semantic_version != CALIBRATION_SEMANTIC_VERSION:
            raise RestReferenceArtifactError("calibration semantic version mismatch")
        object.__setattr__(self, "members", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "rest_reference_sha256": self.rest_reference_sha256,
            "no_contact_validation_sha256": self.no_contact_validation_sha256,
            "source_live_artifact_root_sha256": self.source_live_artifact_root_sha256,
            "members": [item.to_dict() for item in self.members],
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RestReferenceArtifactRootReceipt:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise RestReferenceArtifactError("root receipt fields mismatch")
        members = value["members"]
        if not isinstance(members, list):
            raise RestReferenceArtifactError("root members must be a list")
        values = dict(value)
        values["members"] = tuple(
            RestReferenceArtifactMember.from_dict(item) for item in members
        )
        return cls(**values)


@dataclass(frozen=True)
class LoadedRestReferenceArtifact:
    references: RestReferenceBundle
    validation: NoContactValidationReceipt
    root_receipt: RestReferenceArtifactRootReceipt
    root_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_receipt_sha256",
            require_sha256(self.root_receipt_sha256, "external calibration root"),
        )
