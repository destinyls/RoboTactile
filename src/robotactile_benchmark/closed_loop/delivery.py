"""Causal, correctness-first online delivery of registered tactile faults."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from robotactile_benchmark.closed_loop.validation import validate_online_delivery
from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.contracts import (
    EvaluationRecord,
    canonical_hash,
    delivered_hash,
    freeze_value,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.runtime import trace_hash
from robotactile_benchmark.streaming import StreamingFaultSession
from robotactile_benchmark.validators import ValidationReport

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CLEAN_TRACE_NAMESPACE = "robotactile_benchmark.closed_loop.clean_trace.v1"


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def _clean_trace_hash(records: Sequence[EvaluationRecord]) -> str:
    """Hash a clean trace without changing the historical runtime hash contract."""

    return canonical_hash(
        {
            "namespace": _CLEAN_TRACE_NAMESPACE,
            "records": [
                delivered_hash(record.observation, record.provenance)
                for record in records
            ],
        }
    )


def _fault_trace_hash(records: Sequence[EvaluationRecord], manifest_sha256: str) -> str:
    """Preserve the canonical runtime trace contract without a manifest object."""

    return canonical_hash(
        {
            "manifest_sha256": manifest_sha256,
            "records": [
                delivered_hash(record.observation, record.provenance)
                for record in records
            ],
        }
    )


def _validate_record_hashes(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
) -> None:
    """Validate cached hashes and the delivered-to-clean pairing for one trace."""

    for index, (clean_record, delivered_record) in enumerate(
        zip(clean_records, delivered_records)
    ):
        clean_content_hash = delivered_hash(
            clean_record.observation, clean_record.provenance
        )
        delivered_content_hash = delivered_hash(
            delivered_record.observation, delivered_record.provenance
        )
        if (
            clean_record.clean_record_sha256 != clean_content_hash
            or clean_record.delivered_record_sha256 != clean_content_hash
        ):
            raise ValueError(f"clean record {index} has stale cached hashes")
        if delivered_record.delivered_record_sha256 != delivered_content_hash:
            raise ValueError(f"delivered record {index} has a stale cached hash")
        if delivered_record.clean_record_sha256 != clean_content_hash:
            raise ValueError(f"delivered record {index} has a wrong clean reference")


def _freeze_validation_report(report: ValidationReport) -> ValidationReport:
    """Copy nested report metrics into the immutable contract value domain."""

    return ValidationReport(
        passed=report.passed,
        failure_codes=tuple(report.failure_codes),
        failures=tuple(report.failures),
        metrics=freeze_value(report.metrics),
    )


@dataclass(frozen=True)
class DeliveryFinalization:
    """Immutable audit receipt for one completed delivery session."""

    clean_records: Tuple[EvaluationRecord, ...]
    delivered_records: Tuple[EvaluationRecord, ...]
    validation: Optional[ValidationReport]
    clean_trace_sha256: str
    delivered_trace_sha256: str
    manifest_sha256: Optional[str]

    def __post_init__(self) -> None:
        clean_records = tuple(self.clean_records)
        delivered_records = tuple(self.delivered_records)
        if not clean_records or len(clean_records) != len(delivered_records):
            raise ValueError("finalization traces must be non-empty and equal-length")
        if not all(isinstance(record, EvaluationRecord) for record in clean_records):
            raise TypeError("clean trace must contain EvaluationRecord values")
        if not all(
            isinstance(record, EvaluationRecord) for record in delivered_records
        ):
            raise TypeError("delivered trace must contain EvaluationRecord values")
        _require_sha256(self.clean_trace_sha256, "clean trace hash")
        _require_sha256(self.delivered_trace_sha256, "delivered trace hash")
        _validate_record_hashes(clean_records, delivered_records)
        if self.clean_trace_sha256 != _clean_trace_hash(clean_records):
            raise ValueError("clean trace hash does not match its records")
        if self.validation is None:
            if self.manifest_sha256 is not None:
                raise ValueError("identity finalization cannot name a manifest")
            if self.delivered_trace_sha256 != _clean_trace_hash(delivered_records):
                raise ValueError("identity delivered trace hash does not match records")
            if any(
                canonical_hash(clean_record) != canonical_hash(delivered_record)
                for clean_record, delivered_record in zip(
                    clean_records, delivered_records
                )
            ):
                raise ValueError("identity delivery must preserve complete records")
        else:
            if not isinstance(self.validation, ValidationReport):
                raise TypeError("fault finalization requires a ValidationReport")
            if self.manifest_sha256 is None:
                raise ValueError("fault finalization requires a manifest hash")
            _require_sha256(self.manifest_sha256, "manifest hash")
            if self.delivered_trace_sha256 != _fault_trace_hash(
                delivered_records, self.manifest_sha256
            ):
                raise ValueError("fault delivered trace hash does not match records")
            object.__setattr__(
                self, "validation", _freeze_validation_report(self.validation)
            )
        object.__setattr__(self, "clean_records", clean_records)
        object.__setattr__(self, "delivered_records", delivered_records)


class _DeliverySession:
    """Shared clean-stream gate and lifecycle for identity and fault sessions."""

    def __init__(self) -> None:
        self._clean_records: list[EvaluationRecord] = []
        self._delivered_records: list[EvaluationRecord] = []
        self._stream_identity: Optional[tuple[object, ...]] = None
        self._finalized = False

    def _require_open(self) -> None:
        if self._finalized:
            raise ValueError("delivery session has already been finalized")

    def _validate_clean_record(self, clean_record: EvaluationRecord) -> None:
        if not isinstance(clean_record, EvaluationRecord):
            raise TypeError("delivery requires an EvaluationRecord")
        observation = clean_record.observation
        if observation.step_index != len(self._clean_records):
            raise ValueError(
                "clean stream step indices must start at zero and be dense"
            )
        content_hash = delivered_hash(observation, clean_record.provenance)
        if clean_record.clean_record_sha256 != content_hash:
            raise ValueError("clean record clean hash is stale")
        if clean_record.delivered_record_sha256 != content_hash:
            raise ValueError("clean record delivered hash is stale")
        if any(item.active_fault_ids for item in clean_record.provenance):
            raise ValueError("clean stream cannot contain active fault identifiers")
        identity = (
            observation.episode_id,
            observation.task,
            observation.seed,
            tuple(sensor.slot_id for sensor in observation.tactile),
            tuple(item.physical_source_id for item in clean_record.provenance),
            tuple(sensor.frame_id for sensor in observation.tactile),
            tuple(sensor.calibration_id for sensor in observation.tactile),
            tuple(item.calibration_sha256 for item in clean_record.provenance),
        )
        if self._stream_identity is None:
            self._stream_identity = identity
        elif identity != self._stream_identity:
            raise ValueError("clean stream identity changed during delivery")

    def _append_clean(self, clean_record: EvaluationRecord) -> None:
        self._validate_clean_record(clean_record)
        self._clean_records.append(clean_record)

    def _require_records(self) -> Tuple[EvaluationRecord, ...]:
        if not self._clean_records:
            raise ValueError("cannot finalize an empty delivery session")
        return tuple(self._clean_records)


class IdentityDeliverySession(_DeliverySession):
    """Direct clean/no-touch delivery that preserves record object identity."""

    def deliver(self, clean_record: EvaluationRecord) -> EvaluationRecord:
        self._require_open()
        self._append_clean(clean_record)
        self._delivered_records.append(clean_record)
        return clean_record

    def finalize(self) -> DeliveryFinalization:
        self._require_open()
        clean_records = self._require_records()
        delivered_records = tuple(self._delivered_records)
        clean_trace_sha256 = _clean_trace_hash(clean_records)
        finalization = DeliveryFinalization(
            clean_records=clean_records,
            delivered_records=delivered_records,
            validation=None,
            clean_trace_sha256=clean_trace_sha256,
            delivered_trace_sha256=clean_trace_sha256,
            manifest_sha256=None,
        )
        self._finalized = True
        return finalization


class OnlineFaultSession(_DeliverySession):
    """O(T) streaming delivery for one frozen fault manifest."""

    def __init__(
        self,
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> None:
        super().__init__()
        self._manifest = manifest
        self._rest_references = rest_references
        self._manifest_sha256 = manifest.sha256
        self._manifest_dict = manifest.to_dict()
        if operator_requires_rest_reference(
            manifest.operator_id,
            severity_registry=manifest.severity_registry,
        ):
            if rest_references is None:
                raise ValueError(
                    "this operator requires a frozen rest-reference bundle"
                )
            if manifest.parameters["rest_reference_sha256"] != rest_references.sha256:
                raise ValueError(
                    "rest-reference bundle does not match the fault manifest"
                )
        self._streaming = StreamingFaultSession(manifest, rest_references)

    @property
    def streaming_record_count(self) -> int:
        """Expose append-only work accounting without mutable operator state."""

        return self._streaming.processed_count

    @property
    def retained_temporal_source_count(self) -> int:
        """Expose the bounded temporal source cache size for qualification."""

        return self._streaming.retained_source_count

    @property
    def maximum_temporal_source_age(self) -> int:
        """Return the largest source age registered by a temporal manifest."""

        return self._streaming.maximum_source_age

    def _require_manifest_unchanged(self) -> None:
        if (
            self._manifest.sha256 != self._manifest_sha256
            or self._manifest.to_dict() != self._manifest_dict
        ):
            raise ValueError("fault manifest changed during delivery")

    def _validate_rest_calibration(self, clean_record: EvaluationRecord) -> None:
        if (
            not operator_requires_rest_reference(
                self._manifest.operator_id,
                severity_registry=self._manifest.severity_registry,
            )
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

    def deliver(self, clean_record: EvaluationRecord) -> EvaluationRecord:
        self._require_open()
        self._require_manifest_unchanged()
        if not self._clean_records:
            self._validate_rest_calibration(clean_record)
        self._append_clean(clean_record)
        current = self._streaming.deliver_one(clean_record)
        self._delivered_records.append(current)
        return current

    def finalize(self) -> DeliveryFinalization:
        self._require_open()
        self._require_manifest_unchanged()
        clean_records = self._require_records()
        delivered_records = tuple(self._delivered_records)
        validation = validate_online_delivery(
            clean_records,
            delivered_records,
            self._manifest,
            rest_references=self._rest_references,
        )
        finalization = DeliveryFinalization(
            clean_records=clean_records,
            delivered_records=delivered_records,
            validation=validation,
            clean_trace_sha256=_clean_trace_hash(clean_records),
            delivered_trace_sha256=trace_hash(delivered_records, self._manifest),
            manifest_sha256=self._manifest_sha256,
        )
        self._finalized = True
        return finalization
