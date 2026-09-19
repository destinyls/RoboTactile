"""Immutable contracts for official ACT Clean/Faulted fault campaigns."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Optional, Tuple, cast

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    SEVERITY_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition, TerminalStatus

ACT_FAULT_CAMPAIGN_SEMANTIC_VERSION = "1.0"
ACT_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION = "1.0"
ACT_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION = "1.0"
ACT_FAULT_TEMPLATE_SEED_DERIVATION = "sha256_signed31_pair_operator_v1"
ACT_FAULT_GENERATION_EVIDENCE_LEVEL = "request_generation_only_no_execution_v1"
ACT_UNSUPPORTED_REASON_CODE = "act_univtac_requires_both_tactile_streams_v1"
ACT_UNSUPPORTED_OPERATOR_IDS = frozenset({"A1_stream_absence", "A2_frame_erasure"})
ACT_SUPPORTED_OPERATOR_IDS = CORE_OPERATOR_IDS - ACT_UNSUPPORTED_OPERATOR_IDS

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ACTFaultCampaignError(ValueError):
    """An ACT fault campaign failed its frozen contract."""


class ACTFaultCellDisposition(str, Enum):
    """How one contract cell is represented before execution."""

    LIVE_REQUEST = "live_request"
    UNSUPPORTED_CONTRACT = "unsupported_contract"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ACTFaultCampaignError(f"{name} must be a non-empty string")
    return value


def _identifier(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ACTFaultCampaignError(f"{name} must be a safe identifier")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ACTFaultCampaignError(f"{name} must be a lowercase SHA256")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ACTFaultCampaignError(f"{name} must be an integer >= {minimum}")
    return value


def _relative(value: object, name: str, prefix: str) -> str:
    text = _nonempty(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or not path.parts
        or path.parts[0] != prefix
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise ACTFaultCampaignError(f"{name} must be below {prefix}/")
    return text


require_nonempty = _nonempty
require_sha256 = _sha256


@dataclass(frozen=True)
class ACTFaultCampaignCellSpec:
    """One Clean, executable Faulted, or unsupported ACT contract cell."""

    ordinal: int
    task: str
    initial_seed: int
    exogenous_seed: int
    pair_key: str
    condition: Condition
    disposition: ACTFaultCellDisposition
    operator_id: Optional[str]
    severity_level: Optional[int]
    operator_template_seed: Optional[int]
    request_relpath: Optional[str]
    artifact_relpath: Optional[str]
    request_file_sha256: Optional[str]
    trial_manifest_sha256: str
    fault_manifest_sha256: Optional[str]
    fault_manifest_relpath: Optional[str]
    unsupported_receipt_relpath: Optional[str]
    unsupported_receipt_sha256: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordinal", _integer(self.ordinal, "ordinal"))
        object.__setattr__(self, "task", _identifier(self.task, "task"))
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "pair_key", _sha256(self.pair_key, "pair_key"))
        object.__setattr__(self, "condition", Condition(self.condition))
        object.__setattr__(
            self, "disposition", ACTFaultCellDisposition(self.disposition)
        )
        object.__setattr__(
            self,
            "trial_manifest_sha256",
            _sha256(self.trial_manifest_sha256, "trial_manifest_sha256"),
        )
        for name in (
            "request_file_sha256",
            "fault_manifest_sha256",
            "unsupported_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(value, name))
        self._validate_shape()

    def _validate_shape(self) -> None:
        if self.condition is Condition.CLEAN:
            if self.disposition is not ACTFaultCellDisposition.LIVE_REQUEST or any(
                value is not None
                for value in (
                    self.operator_id,
                    self.severity_level,
                    self.operator_template_seed,
                    self.fault_manifest_sha256,
                    self.fault_manifest_relpath,
                    self.unsupported_receipt_relpath,
                    self.unsupported_receipt_sha256,
                )
            ):
                raise ACTFaultCampaignError("clean cell fields are inconsistent")
            label = "clean"
        elif self.condition is Condition.FAULTED:
            if self.operator_id not in CORE_OPERATOR_IDS:
                raise ACTFaultCampaignError("fault cell has an unknown operator")
            if self.severity_level not in range(1, 6):
                raise ACTFaultCampaignError("fault severity must be in [1, 5]")
            _integer(self.operator_template_seed, "operator_template_seed")
            if self.fault_manifest_sha256 is None:
                raise ACTFaultCampaignError("fault cell requires a fault manifest")
            label = f"{self.operator_id}/s{self.severity_level}"
            expected_fault = f"fault_manifests/{self.task}/{self.pair_key}/{label}.json"
            if self.fault_manifest_relpath != expected_fault:
                raise ACTFaultCampaignError("fault manifest path is not canonical")
            object.__setattr__(
                self,
                "fault_manifest_relpath",
                _relative(expected_fault, "fault_manifest_relpath", "fault_manifests"),
            )
        else:
            raise ACTFaultCampaignError("ACT campaigns allow only clean/faulted")
        self._validate_representation(label)

    def _validate_representation(self, label: str) -> None:
        request = f"requests/{self.task}/{self.pair_key}/{label}.json"
        artifact = f"artifacts/{self.task}/{self.pair_key}/{label}"
        if self.disposition is ACTFaultCellDisposition.LIVE_REQUEST:
            if self.operator_id in ACT_UNSUPPORTED_OPERATOR_IDS:
                raise ACTFaultCampaignError("A1/A2 must use unsupported_contract")
            if (
                self.request_relpath != request
                or self.artifact_relpath != artifact
                or self.request_file_sha256 is None
                or self.unsupported_receipt_relpath is not None
                or self.unsupported_receipt_sha256 is not None
            ):
                raise ACTFaultCampaignError(
                    "live cell paths or hashes are inconsistent"
                )
            object.__setattr__(
                self,
                "request_relpath",
                _relative(request, "request_relpath", "requests"),
            )
            object.__setattr__(
                self,
                "artifact_relpath",
                _relative(artifact, "artifact_relpath", "artifacts"),
            )
            return
        if self.operator_id not in ACT_UNSUPPORTED_OPERATOR_IDS:
            raise ACTFaultCampaignError("only A1/A2 may be unsupported for ACT")
        receipt = f"unsupported_contracts/{self.task}/{self.pair_key}/{label}.json"
        if (
            self.request_relpath is not None
            or self.artifact_relpath is not None
            or self.request_file_sha256 is not None
            or self.unsupported_receipt_relpath != receipt
            or self.unsupported_receipt_sha256 is None
        ):
            raise ACTFaultCampaignError("unsupported cell cannot be a live request")
        object.__setattr__(
            self,
            "unsupported_receipt_relpath",
            _relative(receipt, "unsupported_receipt_relpath", "unsupported_contracts"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            name: item.value if isinstance(item := getattr(self, name), Enum) else item
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> ACTFaultCampaignCellSpec:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ACTFaultCampaignError("campaign cell fields mismatch")
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True)
class ACTUnsupportedContractSpec:
    """Evidence that A1/A2 were not substituted by synthetic black frames."""

    campaign_id: str
    task: str
    initial_seed: int
    exogenous_seed: int
    pair_key: str
    operator_id: str
    severity_level: int
    operator_template_seed: int
    fault_manifest_sha256: str
    trial_manifest_sha256: str
    terminal_status: TerminalStatus = TerminalStatus.UNSUPPORTED_CONTRACT
    reason_code: str = ACT_UNSUPPORTED_REASON_CODE
    live_request_generated: bool = False
    simulator_execution_claimed: bool = False
    black_frame_substitution_used: bool = False
    semantic_version: str = ACT_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        object.__setattr__(self, "task", _identifier(self.task, "task"))
        for name in ("initial_seed", "exogenous_seed", "operator_template_seed"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "pair_key", _sha256(self.pair_key, "pair_key"))
        if self.operator_id not in ACT_UNSUPPORTED_OPERATOR_IDS:
            raise ACTFaultCampaignError("unsupported receipt must identify A1 or A2")
        if self.severity_level not in range(1, 6):
            raise ACTFaultCampaignError("unsupported severity must be in [1, 5]")
        for name in ("fault_manifest_sha256", "trial_manifest_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self, "terminal_status", TerminalStatus(self.terminal_status)
        )
        if (
            self.terminal_status is not TerminalStatus.UNSUPPORTED_CONTRACT
            or self.reason_code != ACT_UNSUPPORTED_REASON_CODE
            or self.live_request_generated
            or self.simulator_execution_claimed
            or self.black_frame_substitution_used
            or self.semantic_version != ACT_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION
        ):
            raise ACTFaultCampaignError("unsupported receipt evidence mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            name: item.value if isinstance(item := getattr(self, name), Enum) else item
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> ACTUnsupportedContractSpec:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ACTFaultCampaignError("unsupported receipt fields mismatch")
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True)
class ACTFaultCampaignManifest:
    """Frozen full-operator grid for official tactile ACT."""

    campaign_id: str
    policy_kind: LivePolicyKind
    profile: OfficialACTProfile
    operator_ids: Tuple[str, ...]
    severity_levels: Tuple[int, ...]
    operator_template_seed_derivation: str
    pair_count: int
    cell_count: int
    live_request_count: int
    unsupported_contract_count: int
    rest_reference_bindings: Mapping[str, Mapping[str, str]]
    cells: Tuple[ACTFaultCampaignCellSpec, ...]
    severity_registry: str = SEVERITY_REGISTRY_ID
    semantic_version: str = ACT_FAULT_CAMPAIGN_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        object.__setattr__(self, "policy_kind", LivePolicyKind(self.policy_kind))
        object.__setattr__(self, "profile", OfficialACTProfile(self.profile))
        operators, levels, cells = (
            tuple(self.operator_ids),
            tuple(self.severity_levels),
            tuple(self.cells),
        )
        if (
            self.policy_kind is not LivePolicyKind.ACT
            or self.profile is not OfficialACTProfile.UNIVTAC
        ):
            raise ACTFaultCampaignError("campaign must select official ACT univtac")
        if (
            operators != tuple(sorted(CORE_OPERATOR_IDS))
            or len(levels) != 1
            or levels[0] not in range(1, 6)
        ):
            raise ACTFaultCampaignError(
                "campaign requires all 14 operators at one severity"
            )
        if self.operator_template_seed_derivation != ACT_FAULT_TEMPLATE_SEED_DERIVATION:
            raise ACTFaultCampaignError("template seed derivation mismatch")
        self._validate_cells(cells, operators, levels)
        bindings = self._freeze_bindings(self.rest_reference_bindings)
        required = {
            cell.task
            for cell in cells
            if cell.operator_id is not None
            and operator_requires_rest_reference(cell.operator_id)
        }
        if set(bindings) != required:
            raise ACTFaultCampaignError(
                "rest-reference bindings do not exactly cover required tasks"
            )
        if (
            self.severity_registry != SEVERITY_REGISTRY_ID
            or self.semantic_version != ACT_FAULT_CAMPAIGN_SEMANTIC_VERSION
        ):
            raise ACTFaultCampaignError("campaign registry or version mismatch")
        object.__setattr__(self, "operator_ids", operators)
        object.__setattr__(self, "severity_levels", levels)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "rest_reference_bindings", bindings)

    def _validate_cells(
        self,
        cells: Tuple[ACTFaultCampaignCellSpec, ...],
        operators: Tuple[str, ...],
        levels: Tuple[int, ...],
    ) -> None:
        if tuple(cell.ordinal for cell in cells) != tuple(range(len(cells))):
            raise ACTFaultCampaignError("campaign cell ordinals must be dense")
        live = sum(
            cell.disposition is ACTFaultCellDisposition.LIVE_REQUEST for cell in cells
        )
        unsupported = len(cells) - live
        pair_keys = {cell.pair_key for cell in cells}
        if (
            any(
                type(getattr(self, name)) is not int
                for name in (
                    "pair_count",
                    "cell_count",
                    "live_request_count",
                    "unsupported_contract_count",
                )
            )
            or self.pair_count != len(pair_keys)
            or self.cell_count != len(cells)
            or self.live_request_count != live
            or self.unsupported_contract_count != unsupported
            or Counter(cell.pair_key for cell in cells)
            != Counter({key: 15 for key in pair_keys})
        ):
            raise ACTFaultCampaignError("campaign counts disagree with full cells")
        for pair in pair_keys:
            selected = [cell for cell in cells if cell.pair_key == pair]
            clean = [cell for cell in selected if cell.condition is Condition.CLEAN]
            actual = {
                (cast(str, cell.operator_id), cast(int, cell.severity_level))
                for cell in selected
                if cell.condition is Condition.FAULTED
            }
            expected = {(operator, levels[0]) for operator in operators}
            identities = {
                (cell.task, cell.initial_seed, cell.exogenous_seed) for cell in selected
            }
            if len(clean) != 1 or actual != expected or len(identities) != 1:
                raise ACTFaultCampaignError(
                    "pair grid or source identity is inconsistent"
                )
            for cell in selected:
                should_be_unsupported = cell.operator_id in ACT_UNSUPPORTED_OPERATOR_IDS
                if should_be_unsupported != (
                    cell.disposition is ACTFaultCellDisposition.UNSUPPORTED_CONTRACT
                ):
                    raise ACTFaultCampaignError("operator disposition is inconsistent")

    @staticmethod
    def _freeze_bindings(
        value: Mapping[str, Mapping[str, str]],
    ) -> Mapping[str, Mapping[str, str]]:
        result: dict[str, Mapping[str, str]] = {}
        for task, binding in sorted(value.items()):
            if set(binding) != {
                "artifact_relpath",
                "artifact_root_sha256",
                "rest_reference_sha256",
            }:
                raise ACTFaultCampaignError("rest-reference binding fields mismatch")
            result[_identifier(task, "rest-reference task")] = MappingProxyType(
                {
                    "artifact_relpath": _relative(
                        binding["artifact_relpath"],
                        "artifact_relpath",
                        "rest_references",
                    ),
                    "artifact_root_sha256": _sha256(
                        binding["artifact_root_sha256"], "artifact_root_sha256"
                    ),
                    "rest_reference_sha256": _sha256(
                        binding["rest_reference_sha256"], "rest_reference_sha256"
                    ),
                }
            )
        return MappingProxyType(result)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "policy_kind": self.policy_kind.value,
            "profile": self.profile.value,
            "operator_ids": list(self.operator_ids),
            "severity_levels": list(self.severity_levels),
            "operator_template_seed_derivation": self.operator_template_seed_derivation,
            "pair_count": self.pair_count,
            "cell_count": self.cell_count,
            "live_request_count": self.live_request_count,
            "unsupported_contract_count": self.unsupported_contract_count,
            "rest_reference_bindings": {
                task: dict(binding)
                for task, binding in self.rest_reference_bindings.items()
            },
            "cells": [cell.to_dict() for cell in self.cells],
            "severity_registry": self.severity_registry,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ACTFaultCampaignManifest:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ACTFaultCampaignError("campaign manifest fields mismatch")
        document = dict(value)
        if not isinstance(document["cells"], list):
            raise ACTFaultCampaignError("campaign cells must be a list")
        document["cells"] = tuple(
            ACTFaultCampaignCellSpec.from_dict(cell) for cell in document["cells"]
        )
        return cls(**cast(Any, document))
