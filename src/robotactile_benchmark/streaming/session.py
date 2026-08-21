"""Causal O(T) streaming engine for registered tactile faults."""

from __future__ import annotations

from typing import Any

from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators import get_operator
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.streaming.availability_context import (
    apply_availability,
    apply_context,
)
from robotactile_benchmark.streaming.fidelity import apply_fidelity
from robotactile_benchmark.streaming.temporal import (
    apply_temporal,
    maximum_source_age,
)


class StreamingFaultSession:
    """Deliver exactly one immutable output for each consecutive clean input."""

    def __init__(
        self,
        manifest: FaultManifest,
        rest_references: RestReferenceBundle | None = None,
    ) -> None:
        if not isinstance(manifest, FaultManifest):
            raise TypeError("streaming session requires a FaultManifest")
        self._manifest = manifest
        self._manifest_sha256 = manifest.sha256
        self._manifest_dict = manifest.to_dict()
        self._rest_references = rest_references
        if manifest.operator_id in REST_REFERENCE_OPERATOR_IDS:
            if rest_references is None:
                raise ValueError(
                    "this operator requires a frozen rest-reference bundle"
                )
            if manifest.parameters["rest_reference_sha256"] != rest_references.sha256:
                raise ValueError(
                    "rest-reference bundle does not match the fault manifest"
                )
        self._family = get_operator(manifest.operator_id).spec.family
        self._processed_count = 0
        self._histories: dict[str, Array] = {}
        self._source_records: dict[int, EvaluationRecord] = {}
        self._maximum_source_age = (
            maximum_source_age(manifest) if self._family == "temporal" else 0
        )
        self._affected_indices = _affected_indices(manifest)

    @property
    def processed_count(self) -> int:
        """Return the number of records handled without prefix replay."""

        return self._processed_count

    @property
    def retained_source_count(self) -> int:
        """Return the bounded temporal cache size for diagnostics and tests."""

        return len(self._source_records)

    @property
    def maximum_source_age(self) -> int:
        """Return the largest registered temporal source age in frames."""

        return self._maximum_source_age

    def deliver_one(self, clean_record: EvaluationRecord) -> EvaluationRecord:
        """Apply the frozen operator to the next dense clean record."""

        if not isinstance(clean_record, EvaluationRecord):
            raise TypeError("streaming delivery requires an EvaluationRecord")
        if (
            self._manifest.sha256 != self._manifest_sha256
            or self._manifest.to_dict() != self._manifest_dict
        ):
            raise ValueError("fault manifest changed during streaming delivery")
        index = clean_record.observation.step_index
        if index != self._processed_count:
            raise ValueError("streaming records must start at zero and remain dense")
        if index == 0:
            self._validate_rest_calibration(clean_record)
        if self._family == "temporal":
            self._source_records[index] = clean_record
        if self._family == "availability":
            delivered = apply_availability(
                clean_record, self._manifest, self._affected_indices
            )
        elif self._family == "fidelity":
            delivered = apply_fidelity(
                clean_record,
                self._manifest,
                self._rest_references,
                self._histories,
            )
        elif self._family == "temporal":
            delivered = apply_temporal(
                clean_record, self._manifest, self._lookup_source
            )
        elif self._family == "context":
            delivered = apply_context(
                clean_record,
                self._manifest,
                self._rest_references,
                self._affected_indices,
            )
        else:
            raise KeyError(f"unknown operator family: {self._family}")
        self._processed_count += 1
        self._trim_source_cache(index)
        return delivered

    def _validate_rest_calibration(self, clean_record: EvaluationRecord) -> None:
        if (
            self._manifest.operator_id not in REST_REFERENCE_OPERATOR_IDS
            or self._rest_references is None
        ):
            return
        for slot_id in self._manifest.sensor_slots:
            if (
                clean_record.provenance_for(slot_id).calibration_sha256
                != self._rest_references.calibration_sha256[slot_id]
            ):
                raise ValueError(
                    "rest-reference calibration does not match the clean episode"
                )

    def _lookup_source(self, source_index: int) -> EvaluationRecord:
        try:
            return self._source_records[source_index]
        except KeyError as error:
            raise ValueError(
                f"temporal source {source_index} is outside the retained causal history"
            ) from error

    def _trim_source_cache(self, current_index: int) -> None:
        if self._family != "temporal":
            return
        oldest_required = current_index - self._maximum_source_age
        expired = tuple(
            index for index in self._source_records if index < oldest_required
        )
        for index in expired:
            del self._source_records[index]


def _affected_indices(manifest: FaultManifest) -> frozenset[int]:
    schedule_key: str | None = None
    if manifest.operator_id in {"A1_stream_absence", "C1_sensor_identity_misrouting"}:
        schedule_key = "affected_offsets"
    elif manifest.operator_id == "A2_frame_erasure":
        schedule_key = "erased_offsets"
    if schedule_key is None:
        return frozenset()
    raw: Any = manifest.parameters[schedule_key]
    return frozenset(manifest.start_index + int(offset) for offset in raw)
