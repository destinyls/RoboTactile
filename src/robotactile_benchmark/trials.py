"""Immutable paired-trial contracts for the four benchmark conditions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple, cast

from robotactile_benchmark.action_specs import validate_action_spec
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operator_parameters import static_parameter_view


class Condition(str, Enum):
    """Matched closed-loop conditions defined by the paper."""

    CLEAN = "clean"
    FAULTED = "faulted"
    NO_TOUCH = "no_touch"
    RESTORED = "restored"


class RestorationMode(str, Enum):
    """How a contract-valid tactile stream is restored."""

    VALID_STREAM_RESUME = "valid_stream_resume"
    MAINTENANCE_OR_REPLACEMENT = "maintenance_or_replacement"


class TerminalStatus(str, Enum):
    """Every requested trial must terminate with one explicit status."""

    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    EARLY_STOP = "early_stop"
    TIMEOUT = "timeout"
    CRASH = "crash"
    VALIDATOR_REJECTED = "validator_rejected"
    UNSUPPORTED_CONTRACT = "unsupported_contract"


def _validate_sha256(name: str, value: Optional[str]) -> None:
    if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def system_manifest_hash(
    system_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
    action_spec: str,
) -> str:
    """Bind a base system to its executable artifacts and action contract."""

    return canonical_hash(
        {
            "system_id": system_id,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "action_spec": action_spec,
        }
    )


_TRIAL_FIELDS = frozenset(
    {
        "task",
        "initial_seed",
        "exogenous_seed",
        "condition",
        "base_system_id",
        "executed_system_id",
        "dataset_sha256",
        "base_system_manifest_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "action_spec",
        "fault_manifest_sha256",
        "matched_no_touch_system_id",
        "restoration_index",
        "restoration_mode",
        "semantic_version",
    }
)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return cast(int, value)


def _strict_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


@dataclass(frozen=True)
class TrialManifest:
    """One immutable system/condition cell for a paired task instance."""

    task: str
    initial_seed: int
    exogenous_seed: int
    condition: Condition
    base_system_id: str
    executed_system_id: str
    dataset_sha256: str
    base_system_manifest_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    action_spec: str
    fault_manifest_sha256: Optional[str]
    matched_no_touch_system_id: Optional[str]
    restoration_index: Optional[int]
    restoration_mode: Optional[RestorationMode]
    semantic_version: str = "1.0"

    def __post_init__(self) -> None:
        initial_seed = _strict_int(self.initial_seed, "initial_seed")
        exogenous_seed = _strict_int(self.exogenous_seed, "exogenous_seed")
        if not isinstance(self.condition, Condition):
            object.__setattr__(self, "condition", Condition(self.condition))
        if self.restoration_mode is not None and not isinstance(
            self.restoration_mode, RestorationMode
        ):
            object.__setattr__(
                self, "restoration_mode", RestorationMode(self.restoration_mode)
            )
        if initial_seed < 0 or exogenous_seed < 0:
            raise ValueError("trial seeds must be non-negative")
        if self.semantic_version != "1.0":
            raise ValueError("unsupported trial-manifest semantic version")
        for field_name in (
            "task",
            "base_system_id",
            "executed_system_id",
            "action_spec",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "dataset_sha256",
            "base_system_manifest_sha256",
            "checkpoint_sha256",
            "config_sha256",
        ):
            _validate_sha256(field_name, getattr(self, field_name))
        object.__setattr__(self, "action_spec", validate_action_spec(self.action_spec))
        if self.condition is not Condition.NO_TOUCH:
            expected_base_hash = system_manifest_hash(
                self.base_system_id,
                self.checkpoint_sha256,
                self.config_sha256,
                self.action_spec,
            )
            if self.base_system_manifest_sha256 != expected_base_hash:
                raise ValueError(
                    "base system manifest does not match checkpoint/config identity"
                )
        if self.fault_manifest_sha256 is not None:
            _validate_sha256("fault_manifest_sha256", self.fault_manifest_sha256)
        if self.condition is Condition.CLEAN and self.fault_manifest_sha256 is not None:
            raise ValueError("clean condition cannot reference a fault manifest")
        if (
            self.condition is Condition.CLEAN
            and self.executed_system_id != self.base_system_id
        ):
            raise ValueError("clean condition must execute the base system")
        if self.condition in {Condition.FAULTED, Condition.RESTORED}:
            if self.fault_manifest_sha256 is None:
                raise ValueError("faulted/restored condition requires a fault manifest")
            if self.executed_system_id != self.base_system_id:
                raise ValueError("faulted/restored must execute the base system")
        if self.condition is Condition.RESTORED:
            if self.restoration_index is None or self.restoration_index < 0:
                raise ValueError("restored condition requires a restoration index")
            if self.restoration_mode is None:
                raise ValueError("restored condition requires a restoration mode")
        elif self.restoration_index is not None or self.restoration_mode is not None:
            raise ValueError("restoration metadata is only valid for restored trials")
        if self.condition is Condition.NO_TOUCH:
            if not self.matched_no_touch_system_id:
                raise ValueError(
                    "no-touch condition requires a matched no-touch system"
                )
            if self.executed_system_id != self.matched_no_touch_system_id:
                raise ValueError(
                    "no-touch executed system must equal its matched control"
                )
            if self.fault_manifest_sha256 is not None:
                raise ValueError("no-touch condition cannot reference a fault manifest")
        elif self.matched_no_touch_system_id is not None:
            raise ValueError("matched no-touch ID is only valid for no-touch trials")
        object.__setattr__(self, "initial_seed", initial_seed)
        object.__setattr__(self, "exogenous_seed", exogenous_seed)

    @property
    def pair_key(self) -> str:
        """Hash condition-invariant fields used for matched pairing."""

        return canonical_hash(
            {
                "task": self.task,
                "initial_seed": self.initial_seed,
                "exogenous_seed": self.exogenous_seed,
                "base_system_id": self.base_system_id,
                "dataset_sha256": self.dataset_sha256,
                "base_system_manifest_sha256": self.base_system_manifest_sha256,
                "action_spec": self.action_spec,
            }
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize one trial without absolute paths or runtime state."""

        return {
            "task": self.task,
            "initial_seed": self.initial_seed,
            "exogenous_seed": self.exogenous_seed,
            "condition": self.condition.value,
            "base_system_id": self.base_system_id,
            "executed_system_id": self.executed_system_id,
            "dataset_sha256": self.dataset_sha256,
            "base_system_manifest_sha256": self.base_system_manifest_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "action_spec": self.action_spec,
            "fault_manifest_sha256": self.fault_manifest_sha256,
            "matched_no_touch_system_id": self.matched_no_touch_system_id,
            "restoration_index": self.restoration_index,
            "restoration_mode": (
                self.restoration_mode.value
                if self.restoration_mode is not None
                else None
            ),
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrialManifest:
        """Build a validated trial from machine-readable data."""

        if set(value) != _TRIAL_FIELDS:
            missing = sorted(_TRIAL_FIELDS - set(value))
            extra = sorted(set(value) - _TRIAL_FIELDS)
            raise ValueError(
                f"trial manifest fields mismatch: missing={missing}, extra={extra}"
            )
        fault_hash = value["fault_manifest_sha256"]
        no_touch_id = value["matched_no_touch_system_id"]
        restoration_index = value["restoration_index"]
        restoration_mode = value["restoration_mode"]
        for optional_string, name in (
            (fault_hash, "fault_manifest_sha256"),
            (no_touch_id, "matched_no_touch_system_id"),
            (restoration_mode, "restoration_mode"),
        ):
            if optional_string is not None and not isinstance(optional_string, str):
                raise TypeError(f"{name} must be a string or null")
        return cls(
            task=_strict_str(value["task"], "task"),
            initial_seed=_strict_int(value["initial_seed"], "initial_seed"),
            exogenous_seed=_strict_int(value["exogenous_seed"], "exogenous_seed"),
            condition=Condition(_strict_str(value["condition"], "condition")),
            base_system_id=_strict_str(value["base_system_id"], "base_system_id"),
            executed_system_id=_strict_str(
                value["executed_system_id"], "executed_system_id"
            ),
            dataset_sha256=_strict_str(value["dataset_sha256"], "dataset_sha256"),
            base_system_manifest_sha256=_strict_str(
                value["base_system_manifest_sha256"], "base_system_manifest_sha256"
            ),
            checkpoint_sha256=_strict_str(
                value["checkpoint_sha256"], "checkpoint_sha256"
            ),
            config_sha256=_strict_str(value["config_sha256"], "config_sha256"),
            action_spec=_strict_str(value["action_spec"], "action_spec"),
            fault_manifest_sha256=fault_hash,
            matched_no_touch_system_id=no_touch_id,
            restoration_index=(
                _strict_int(restoration_index, "restoration_index")
                if restoration_index is not None
                else None
            ),
            restoration_mode=(
                RestorationMode(restoration_mode)
                if restoration_mode is not None
                else None
            ),
            semantic_version=_strict_str(value["semantic_version"], "semantic_version"),
        )


def build_paired_trial_grid(
    clean: TrialManifest,
    faulted_manifest: FaultManifest,
    restored_manifest: FaultManifest,
    restoration_index: int,
    restoration_mode: RestorationMode,
    no_touch_system_id: str,
    no_touch_checkpoint_sha256: str,
    no_touch_config_sha256: str,
) -> Tuple[TrialManifest, ...]:
    """Build clean/faulted/no-touch/restored cells from one frozen pair key."""

    if clean.condition is not Condition.CLEAN:
        raise ValueError("paired grid must start from a clean trial")
    invariant_fields = (
        "operator_id",
        "severity_level",
        "operator_seed",
        "start_index",
        "sensor_slots",
        "observability",
        "semantic_version",
        "implementation_version",
        "severity_registry",
    )
    if any(
        getattr(faulted_manifest, field_name) != getattr(restored_manifest, field_name)
        for field_name in invariant_fields
    ):
        raise ValueError("faulted/restored manifests must differ only at stop_index")
    if static_parameter_view(faulted_manifest.parameters) != static_parameter_view(
        restored_manifest.parameters
    ):
        raise ValueError("faulted/restored manifests must share one operator instance")
    if restored_manifest.stop_index != restoration_index:
        raise ValueError("restoration index must equal the restored manifest stop")
    if faulted_manifest.stop_index <= restoration_index:
        raise ValueError("faulted manifest must persist beyond restoration index")
    faulted = replace(
        clean,
        condition=Condition.FAULTED,
        fault_manifest_sha256=faulted_manifest.sha256,
    )
    no_touch = replace(
        clean,
        condition=Condition.NO_TOUCH,
        executed_system_id=no_touch_system_id,
        checkpoint_sha256=no_touch_checkpoint_sha256,
        config_sha256=no_touch_config_sha256,
        matched_no_touch_system_id=no_touch_system_id,
    )
    restored = replace(
        clean,
        condition=Condition.RESTORED,
        fault_manifest_sha256=restored_manifest.sha256,
        restoration_index=restoration_index,
        restoration_mode=restoration_mode,
    )
    return clean, faulted, no_touch, restored
