"""Configuration contract for deterministic N0 fault campaign generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Tuple

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS,
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPTICAL_CONTACT_STRESS_REGISTRY_ID,
    SENSOR_SLOTS,
    SEVERITY_REGISTRY_ID,
    SUPPORTED_SEVERITY_REGISTRY_IDS,
    THREE_DOSE_STRESS_REGISTRY_IDS,
)
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.fault_timing import (
    DEFAULT_EARLY_ONSET_MAX_INDEX,
    EARLY_RANDOM_ONSET_MODE,
    FIXED_FAULT_ONSET_MODE,
)
from robotactile_benchmark.manifests import Observability

from .contracts import (
    N0FaultCampaignError,
    require_identifier,
    require_integer,
)

N0_FAULT_GENERATION_SPEC_VERSION = "1.0"


@dataclass(frozen=True)
class N0FaultCampaignGenerationSpec:
    """Frozen source requests and axes for one generated campaign bundle."""

    campaign_id: str
    base_clean_request_paths: Tuple[Path, ...]
    operator_ids: Tuple[str, ...]
    severity_levels: Tuple[int, ...]
    operator_seed_master: int
    fault_start_index: int
    fault_stop_index: int
    rest_reference_artifacts: Mapping[str, Path]
    fault_onset_mode: str = FIXED_FAULT_ONSET_MODE
    fault_onset_max_index: int = DEFAULT_EARLY_ONSET_MAX_INDEX
    sensor_slots: Tuple[str, ...] = SENSOR_SLOTS
    observability: Observability = Observability.BLIND
    severity_registry: str = SEVERITY_REGISTRY_ID
    semantic_version: str = N0_FAULT_GENERATION_SPEC_VERSION
    spatial_calibration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", require_identifier(self.campaign_id, "campaign_id")
        )
        paths = tuple(
            Path(path).expanduser().absolute() for path in self.base_clean_request_paths
        )
        if not paths or len(paths) != len(set(paths)):
            raise N0FaultCampaignError(
                "base Clean request paths must be non-empty and unique"
            )
        operators = tuple(self.operator_ids)
        levels = tuple(self.severity_levels)
        if (
            len(operators) != len(set(operators))
            or not operators
            or set(operators) - CORE_OPERATOR_IDS
        ):
            raise N0FaultCampaignError(
                "operator_ids must be unique canonical operators"
            )
        if (
            len(levels) != len(set(levels))
            or not levels
            or any(level not in range(1, 6) for level in levels)
        ):
            raise N0FaultCampaignError(
                "severity_levels must be unique levels in [1, 5]"
            )
        if self.severity_registry not in SUPPORTED_SEVERITY_REGISTRY_IDS:
            raise N0FaultCampaignError("unsupported severity registry")
        if self.severity_registry in THREE_DOSE_STRESS_REGISTRY_IDS and set(levels) - {
            1,
            3,
            5,
        }:
            raise N0FaultCampaignError("three-dose stress requires levels 1, 3, or 5")
        if self.spatial_calibration:
            if (
                self.severity_registry != OPTICAL_CONTACT_STRESS_REGISTRY_ID
                or "F3_persistent_surface_artifact" not in operators
            ):
                raise N0FaultCampaignError(
                    "spatial calibration requires contact-stress F3"
                )
            from robotactile_benchmark.optical.stress_templates import (
                validate_spatial_calibration,
            )

            calibrated = validate_spatial_calibration(self.spatial_calibration)
            object.__setattr__(self, "spatial_calibration", freeze_value(calibrated))
        if self.severity_registry in DIAGNOSTIC_LEVEL_FIVE_REGISTRY_IDS and levels != (
            5,
        ):
            raise N0FaultCampaignError(
                "diagnostic stress campaigns require severity level 5 only"
            )
        if self.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID:
            if operators != ("F1_global_response_drift",):
                raise N0FaultCampaignError(
                    "tactile-null campaign requires only F1 realization"
                )
            if self.fault_start_index != 0:
                raise N0FaultCampaignError(
                    "tactile-null campaign must start at observation zero"
                )
        if self.severity_registry == DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID:
            if operators != ("A1_stream_absence",):
                raise N0FaultCampaignError(
                    "observed-tactile absence requires only A1 realization"
                )
            if self.fault_start_index != 0:
                raise N0FaultCampaignError(
                    "observed-tactile absence must start at observation zero"
                )
        for name in ("operator_seed_master", "fault_start_index"):
            object.__setattr__(self, name, require_integer(getattr(self, name), name))
        object.__setattr__(
            self,
            "fault_stop_index",
            require_integer(self.fault_stop_index, "fault_stop_index", 1),
        )
        if self.fault_start_index >= self.fault_stop_index:
            raise N0FaultCampaignError("fault window must satisfy start < stop")
        if self.fault_onset_mode not in {
            FIXED_FAULT_ONSET_MODE,
            EARLY_RANDOM_ONSET_MODE,
        }:
            raise N0FaultCampaignError("unsupported fault onset mode")
        if (
            type(self.fault_onset_max_index) is not int
            or self.fault_onset_max_index < 1
        ):
            raise N0FaultCampaignError("fault_onset_max_index must be positive")
        if (
            self.fault_onset_mode == FIXED_FAULT_ONSET_MODE
            and self.fault_onset_max_index != DEFAULT_EARLY_ONSET_MAX_INDEX
        ):
            raise N0FaultCampaignError(
                "fault_onset_max_index requires early_random_onset_v1"
            )
        if self.fault_onset_mode == EARLY_RANDOM_ONSET_MODE:
            if self.fault_start_index != 0 or self.fault_stop_index < 4:
                raise N0FaultCampaignError(
                    "early random onset requires placeholder start 0 and horizon >= 4"
                )
            if self.severity_registry in {
                DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
                DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
            }:
                raise N0FaultCampaignError(
                    "full-horizon tactile absence/null ablations cannot randomize onset"
                )
        if tuple(self.sensor_slots) != SENSOR_SLOTS:
            raise N0FaultCampaignError("N0 fault campaign requires both tactile slots")
        observability = (
            self.observability
            if isinstance(self.observability, Observability)
            else Observability(self.observability)
        )
        if observability is not Observability.BLIND:
            raise N0FaultCampaignError("N0 primary robustness faults must be blind")
        bindings: dict[str, Path] = {}
        for task, path in self.rest_reference_artifacts.items():
            if not isinstance(task, str) or not task:
                raise N0FaultCampaignError("rest-reference task must be non-empty")
            bindings[task] = Path(path).expanduser().absolute()
        object.__setattr__(self, "base_clean_request_paths", paths)
        object.__setattr__(self, "operator_ids", tuple(sorted(operators)))
        object.__setattr__(self, "severity_levels", tuple(sorted(levels)))
        object.__setattr__(self, "sensor_slots", SENSOR_SLOTS)
        object.__setattr__(self, "observability", observability)
        object.__setattr__(self, "rest_reference_artifacts", MappingProxyType(bindings))
        if self.semantic_version != N0_FAULT_GENERATION_SPEC_VERSION:
            raise N0FaultCampaignError("generation spec version mismatch")
